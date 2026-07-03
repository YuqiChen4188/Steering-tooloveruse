import http.client
import json
import time
from pathlib import Path


def _load_secret():
    secret_path = Path(__file__).resolve().parents[1] / "secret.json"
    if not secret_path.exists():
        raise FileNotFoundError(f"Missing secret.json at {secret_path}")
    with secret_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


api_key = _load_secret()["serper_key"]

conn = http.client.HTTPSConnection("google.serper.dev")
headers = {
  'X-API-KEY': api_key,
  'Content-Type': 'application/json'
}


def _strip_wrapping_quotes(text):
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1].strip()
    return stripped


def _request_search(query, tbs=None):
    payload_dict = {"q": query}
    if tbs:
        payload_dict["tbs"] = tbs
    payload = json.dumps(payload_dict)
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = res.read()
    data = data.decode("utf-8")
    return json.loads(data)

def search_serper(query, link=False, num=10):
    quoted_query = query.strip()
    unquoted_query = _strip_wrapping_quotes(quoted_query)
    base_query = unquoted_query or quoted_query

    candidates = [(quoted_query, "qdr:y")]
    if unquoted_query and unquoted_query != quoted_query:
        candidates.append((unquoted_query, "qdr:y"))
    candidates.append((base_query, None))
    candidates.append((f"site:wikipedia.org {base_query}", None))

    data = None
    for query_text, tbs in candidates:
        try:
            data = _request_search(query_text, tbs=tbs)
        except Exception as e:
            error = f"Search Error: {e}"
            print(error)
            return error
        if data.get("organic", []):
            break
        data = None
        time.sleep(2)

    if data is None:
        return "Search Error: Timeout"
        
    try:
        output = ""
        index = 1
        answer_box = data.get("answerBox", "")
        if answer_box:
            if link:
                if 'title' in answer_box and 'link' in answer_box and 'snippet' in answer_box:
                    output += f"{str(index)}. {answer_box['title']}\n- Link: {answer_box['link']}\n- Snippet: {answer_box['snippet']}\n"
                    index += 1
            else:
                if 'title' in answer_box and 'date' in answer_box and 'snippet' in answer_box:
                    output += f"{str(index)}. {answer_box['title']}\n- Date: {answer_box['date']}\n- Snippet: {answer_box['snippet']}\n"
                    index += 1
        
        if index > num:
            return output.strip()
        
        for item in data.get("organic", []):
            if link:
                if 'title' in item and 'link' in item and 'snippet' in item:
                    output += f"{str(index)}. {item['title']}\n- Link: {item['link']}\n- Snippet: {item['snippet']}\n"
                    index += 1
            else:
                if 'title' in item and 'date' in item and 'snippet' in item:
                    output += f"{str(index)}. {item['title']}\n- Date: {item['date']}\n- Snippet: {item['snippet']}\n"
                    index += 1
            if index > num:
                return output.strip()
        
        return output.strip()
    
    except Exception as e:
        error = f"Search Error: {e}"
        print(error)
        return error
