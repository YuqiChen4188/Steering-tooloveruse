"""Minimal AskUser helper copied for Steering-tooloveruse inference."""

from __future__ import annotations

import json
import time
from pathlib import Path

from openai import OpenAI


SYS_PROMPT = """You are given a very general task. Now, you should pretend to be a user and give answer to a related query. You should provide a response to the query to show your preferences or requirements.

Please directly provide a concise and coherent response."""


def _load_client() -> OpenAI:
    secret_path = Path(__file__).resolve().parents[1] / "secret.json"
    if secret_path.exists():
        secret = json.loads(secret_path.read_text())
        return OpenAI(api_key=secret["api_key"], base_url=secret["base_url"])
    return OpenAI(api_key="sk-...")


client = _load_client()


def form_messages(task: str, query: str) -> list[dict[str, str]]:
    user_prompt = f"### Task{task}\n\n### Query\n{query}\n\n### Response\n"
    return [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def gpt_chatcompletion(messages: list[dict[str, str]], model: str = "gpt-4o") -> str:
    rounds = 0
    while True:
        rounds += 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                n=1,
            )
            content = response.choices[0].message.content
            return content.replace("### Response", "").strip()
        except Exception as exc:
            print(f"Chat Generation Error: {exc}")
            time.sleep(5)
            if rounds > 3:
                raise RuntimeError("Chat Completion failed too many times") from exc


def simulate_user_response(task: str, query: str, model: str = "gpt-4o") -> str | None:
    try:
        messages = form_messages(task, query)
        return gpt_chatcompletion(messages, model=model)
    except Exception as exc:
        print(f"Error in simulate_user_response: {exc}")
        return None
