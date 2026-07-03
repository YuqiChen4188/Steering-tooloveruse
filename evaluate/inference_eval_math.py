import argparse
import glob
import json
import os
import threading
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


SYS_PROMPT = """You are a helpful assistant to jusge whether the model's final response (might be word, phrase or sentence) and the given correct answer is same in value.
- If their intrinsic numerical value of the answer is not equal, please mark it as wrong.
- If they are just expressed in different format or wording or unit, but have the same main value, please mark it as correct.

Example:
- Model response: The final answer should be 4.123
- Ground truth: \sqrt{17}
- Judgment: correct

- Model response: 0.2687
- Ground truth: \frac{pi}{9}
- Judgment: wrong

- Model response: 40%
- Ground truth: 40
- Judgment: correct

- Model response: Therefore, the speed of the car is 25 miles per hour
- Ground truth: 25
- Judgment: correct

- Model response: The temperature of the metal ay noon will be 10938.893 T
- Ground truth: 1.983e4
- Judgment: wrong

- Model response: $1000.00
- Ground truth: 1000
- Judgment: correct"""

USER_PROMPT = """- Model response: <pd>
- Ground truth: <gt>
- Judgment: """


def build_client() -> OpenAI:
    secret_path = Path(__file__).resolve().parent.parent / "secret.json"
    if secret_path.exists():
        secret = json.load(secret_path.open("r", encoding="utf-8"))
        return OpenAI(api_key=secret["api_key"], base_url=secret["base_url"])
    return OpenAI(api_key="sk-...")


client = build_client()


def form_messages(msg: str, system_prompt: str = ""):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": msg},
    ]


def gpt_chatcompletion(messages, model="gpt-4o-mini"):
    rounds = 0
    while True:
        rounds += 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.000001,
                n=1,
            )
            content = response.choices[0].message.content
            return content.strip()
        except Exception as e:
            print(f"Chat Generation Error: {e}")
            time.sleep(5)
            if rounds > 3:
                raise Exception("Chat Completion failed too many times")


def evaluate_example(
    data: dict,
    answered_data: list[dict],
    hash_tab: set[str],
    save_path: Path,
    log: dict[str, int],
    judgment: dict[str, int],
    model: str,
    lock: threading.Lock,
) -> None:
    try:
        task = data["task"]
        with lock:
            if task in hash_tab:
                return

        pd = data["predict"][-1]["reasoning"].replace("\n", " ").strip()
        gt = data["ground_truth"].split("### Final Response")[1].strip()

        if pd.lower() == gt.lower():
            response = "correct"
        else:
            print("\n======================= Question ========================\n")
            print(task)
            user_prompt = USER_PROMPT.replace("<pd>", pd).replace("<gt>", gt)
            messages = form_messages(user_prompt, SYS_PROMPT)
            response = gpt_chatcompletion(messages, model=model).strip()
            print("\n======================= Response ========================\n")
            print(response)

        if response not in {"correct", "wrong"}:
            raise ValueError("Unknown judgment: " + response)

        data["judge"] = response

        with lock:
            judgment[response] += 1
            answered_data.append(data)
            hash_tab.add(task)
            log["success"] += 1
            if len(answered_data) % 50 == 0:
                with save_path.open("w", encoding="utf-8") as f:
                    json.dump(answered_data, f, indent=2, ensure_ascii=False)
    except Exception:
        with lock:
            log["fail"] += 1


def evaluate_file(data_path: Path, model: str, max_workers: int, overwrite: bool) -> None:
    all_data = json.load(data_path.open("r", encoding="utf-8"))
    save_path = data_path.with_name(data_path.stem + "_judge.json")

    if save_path.exists() and not overwrite:
        answered_data = json.load(save_path.open("r", encoding="utf-8"))
        hash_tab = {data["task"] for data in answered_data}
    else:
        answered_data = []
        hash_tab = set()

    print(f"\n===== Evaluating {data_path} =====")
    print(f"Existing data: {len(answered_data)}")

    log = {"success": 0, "fail": 0}
    judgment = {"correct": 0, "wrong": 0}
    for data in answered_data:
        if data.get("judge") in judgment:
            judgment[data["judge"]] += 1

    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                evaluate_example,
                data,
                answered_data,
                hash_tab,
                save_path,
                log,
                judgment,
                model,
                lock,
            )
            for data in tqdm(all_data, desc=data_path.name)
        ]
        for future in futures:
            future.result()

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(answered_data, f, indent=2, ensure_ascii=False)

    final_cw_dict = {"correct": 0, "wrong": 0}
    for data in answered_data:
        if data.get("judge") == "correct":
            final_cw_dict["correct"] += 1
        elif data.get("judge") == "wrong":
            final_cw_dict["wrong"] += 1

    total = final_cw_dict["correct"] + final_cw_dict["wrong"]
    correct_rate = round(final_cw_dict["correct"] / total, 4) if total else 0.0

    print(log)
    print(judgment)
    print(f"Correct rate: {correct_rate}")
    print(final_cw_dict)
    print(f"Saved to: {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate math inference JSON files with GPT judging.")
    parser.add_argument(
        "--data-path",
        action="append",
        default=[],
        help="Path to a single inference JSON file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default=None,
        help="Glob pattern for inference JSON files to evaluate.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Judge model name.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=50,
        help="Number of worker threads per file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_judge.json files instead of resuming.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(path) for path in args.data_path]
    if args.glob_pattern:
        paths.extend(Path(path) for path in glob.glob(args.glob_pattern))

    if not paths:
        raise ValueError("Please provide --data-path and/or --glob.")

    unique_paths = sorted({path.resolve() for path in paths})
    return unique_paths


if __name__ == "__main__":
    args = parse_args()
    for data_path in resolve_paths(args):
        evaluate_file(data_path, model=args.model, max_workers=args.max_workers, overwrite=args.overwrite)
