"""Copy this file to D's project; only Python's standard library is required."""

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def recommend_frames(question, top_k=5, video_ids=None, representation="combined", use_llm=True,
                     *, base_url="http://127.0.0.1:8765", api_key=None, timeout=120):
    """Ask C's running local service for frame/time recommendations."""
    if not isinstance(base_url, str) or base_url != base_url.strip():
        raise ValueError("base_url must be an HTTP(S) server origin")
    parsed = urllib.parse.urlsplit(base_url)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or
            parsed.username is not None or parsed.password is not None or
            parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise ValueError("base_url must contain only an HTTP(S) host and optional port")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("Invalid server port") from exc
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or
            not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be a positive number of seconds")
    key = api_key or os.environ.get("CCTV_RAG_API_KEY")
    if key is not None and (not isinstance(key, str) or "\r" in key or "\n" in key):
        raise ValueError("api_key must be a single-line string")
    payload = {"question": question, "top_k": top_k, "video_ids": video_ids,
               "representation": representation, "use_llm": use_llm}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(base_url.rstrip("/") + "/recommend",
                                     data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                                     headers=headers, method="POST")
    # LAN traffic and API keys must not be forwarded through proxies or redirects.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    limit = 1024 * 1024
    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read(limit + 1)
    except urllib.error.HTTPError as exc:
        with exc:
            body = exc.read(limit + 1)
        try:
            error = json.loads(body.decode("utf-8"))["error"] if len(body) <= limit else {}
            code, message = error["code"], error["message"]
            if not isinstance(code, str) or not isinstance(message, str):
                raise ValueError("Invalid error response")
        except (ValueError, KeyError, TypeError):
            code, message = "HTTP_ERROR", f"RAG server returned HTTP {exc.code}"
        raise RuntimeError(f"{code}: {message}") from None
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise RuntimeError("SERVER_UNAVAILABLE: Could not reach the RAG server before the timeout") from exc
    if len(body) > limit:
        raise RuntimeError("INVALID_RESPONSE: RAG server response exceeds 1 MiB")
    try:
        result = json.loads(body.decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object")
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError("INVALID_RESPONSE: RAG server did not return a JSON object") from exc
    return result
