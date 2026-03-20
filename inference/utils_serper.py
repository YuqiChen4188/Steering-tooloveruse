"""Minimal Serper search helper copied for Steering-tooloveruse inference."""

from __future__ import annotations

import http.client
import json
import time
from pathlib import Path


def _load_secret() -> dict:
    secret_path = Path(__file__).resolve().parents[1] / "secret.json"
    if not secret_path.exists():
        raise FileNotFoundError(f"Missing secret.json at {secret_path}")
    return json.loads(secret_path.read_text())


def search_serper(query: str, link: bool = False, num: int = 10) -> str:
    api_key = _load_secret()["serper_key"]
    conn = http.client.HTTPSConnection("google.serper.dev")
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = json.dumps({"q": query, "tbs": "qdr:y"})

    try_time = 0
    while True:
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))
        if try_time > 10:
            return "Search Error: Timeout"
        if data.get("organic"):
            break
        try_time += 1
        time.sleep(5)

    try:
        output = ""
        index = 1
        answer_box = data.get("answerBox", "")
        if answer_box:
            if link:
                if {"title", "link", "snippet"} <= answer_box.keys():
                    output += (
                        f"{index}. {answer_box['title']}\n- Link: {answer_box['link']}\n"
                        f"- Snippet: {answer_box['snippet']}\n"
                    )
                    index += 1
            else:
                if {"title", "date", "snippet"} <= answer_box.keys():
                    output += (
                        f"{index}. {answer_box['title']}\n- Date: {answer_box['date']}\n"
                        f"- Snippet: {answer_box['snippet']}\n"
                    )
                    index += 1

        if index > num:
            return output.strip()

        for item in data.get("organic", []):
            if link:
                if {"title", "link", "snippet"} <= item.keys():
                    output += f"{index}. {item['title']}\n- Link: {item['link']}\n- Snippet: {item['snippet']}\n"
                    index += 1
            else:
                if {"title", "date", "snippet"} <= item.keys():
                    output += f"{index}. {item['title']}\n- Date: {item['date']}\n- Snippet: {item['snippet']}\n"
                    index += 1
            if index > num:
                return output.strip()

        return output.strip()
    except Exception as exc:
        error = f"Search Error: {exc}"
        print(error)
        return error
