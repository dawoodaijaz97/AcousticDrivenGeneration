"""Helpers for LM Studio's OpenAI-compatible HTTP API (local inference server).

LM Studio base URL is typically ``http://<host>:1234/v1`` — see
https://lmstudio.ai/docs/developer/openai-compat
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def normalize_lm_studio_base_url(url: str) -> str:
    """Ensure the base URL ends with ``/v1`` (LM Studio OpenAI-compat root)."""
    u = url.strip().rstrip("/")
    if not u.endswith("/v1"):
        return f"{u}/v1"
    return u


def lm_studio_chat_completion(
    base_url: str,
    model: str,
    user_text: str,
    max_tokens: int,
    *,
    api_key: str = "lm-studio",
    temperature: float = 0.0,
    timeout_s: float = 600.0,
    seed: int | None = None,
) -> str:
    """POST ``/v1/chat/completions``; return assistant message content text."""
    root = normalize_lm_studio_base_url(base_url)
    url = f"{root.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_text}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    if seed is not None:
        payload["seed"] = int(seed)

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LM Studio HTTP {e.code} at {url}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"LM Studio request failed ({url}): {e.reason}") from e

    data = json.loads(raw)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"LM Studio response missing choices: {raw[:2000]}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        raise RuntimeError(f"LM Studio response missing message.content: {raw[:2000]}")
    return str(content).strip()
