#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

from huggingface_hub import get_token, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError


MODEL_SPECS = {
    "mistral-7b": {
        "repo_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "local_dir": "Mistral-7B-Instruct-v0.3",
    },
    "llama-3.1-8b": {
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "local_dir": "Llama-3.1-8B-Instruct",
    },
    "mistral-nemo-12b": {
        "repo_id": "mistralai/Mistral-Nemo-Instruct-2407",
        "local_dir": "Mistral-Nemo-12B-Instruct",
    },
    "mistral-small-24b": {
        "repo_id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "local_dir": "Mistral-Small-24B-Instruct",
    },
    "llama-3.1-70b": {
        "repo_id": "meta-llama/Llama-3.1-70B-Instruct",
        "local_dir": "Llama-3.1-70B-Instruct",
    },
}


SNAPSHOT_DOWNLOAD_SUPPORTS_DRY_RUN = "dry_run" in inspect.signature(snapshot_download).parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download requested model weights into the local model_weights directory."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "model_weights",
        help="Destination root for model weights.",
    )
    parser.add_argument(
        "--token",
        default=(
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or get_token()
        ),
        help="Optional Hugging Face token. Needed for gated models such as Llama 3.1.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=sorted(MODEL_SPECS),
        default=sorted(MODEL_SPECS),
        help="Subset of models to download.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only check whether the repositories are reachable and list planned files.",
    )
    return parser.parse_args()


def download_one(root: Path, model_key: str, token: str | None, dry_run: bool) -> int:
    spec = MODEL_SPECS[model_key]
    local_dir = root / spec["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"[start] {model_key} -> {spec['repo_id']}")
    try:
        download_kwargs = dict(
            repo_id=spec["repo_id"],
            local_dir=local_dir,
            token=token,
        )
        if dry_run:
            if not SNAPSHOT_DOWNLOAD_SUPPORTS_DRY_RUN:
                print(
                    f"[failed] {model_key}: installed huggingface_hub does not support --dry-run.",
                    file=sys.stderr,
                )
                return 1
            download_kwargs["dry_run"] = True

        result = snapshot_download(**download_kwargs)
    except GatedRepoError:
        print(
            f"[failed] {model_key}: gated repository. Provide a Hugging Face token with access.",
            file=sys.stderr,
        )
        return 1
    except HfHubHTTPError as exc:
        print(f"[failed] {model_key}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[failed] {model_key}: unexpected error: {exc}", file=sys.stderr)
        return 1

    if dry_run:
        print(f"[ok] {model_key}: {len(result)} files would be downloaded")
    else:
        print(f"[ok] {model_key}: saved to {local_dir}")
    return 0


def main() -> int:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    failures = 0
    for model_key in args.models:
        failures += download_one(args.root, model_key, args.token, args.dry_run)

    if failures:
        print(f"[done] completed with {failures} failure(s)", file=sys.stderr)
        return 1

    print("[done] all requested models completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
