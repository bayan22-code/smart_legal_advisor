"""
OpenRouter API client for LLM chat completions.
Uses OpenAI SDK with OpenRouter base URL (recommended by OpenRouter docs).
"""

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from config import OPENROUTER_API_BASE, OPENROUTER_API_KEY, OPENROUTER_MODEL

try:
    import streamlit as st
except Exception:  # pragma: no cover - streamlit may be unavailable in scripts
    st = None


def _build_client(key: str) -> OpenAI:
    return OpenAI(
        base_url=OPENROUTER_API_BASE,
        api_key=key,
    )


if st:
    _get_cached_client = st.cache_resource(show_spinner=False)(_build_client)
else:
    def _get_cached_client(key: str) -> OpenAI:
        return _build_client(key)


def chat_completion(
    messages: list[dict[str, str]],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    stream: bool = False,
) -> str:
    """
    Call OpenRouter chat completion API via OpenAI-compatible client.

    Args:
        messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
        api_key: Override env OPENROUTER_API_KEY
        model: Override default model
        temperature: Sampling temperature (lower = more deterministic)
        max_tokens: Maximum generated tokens for cost control
        stream: Whether to stream response (returns full text when stream=False)

    Returns:
        Assistant reply text
    """
    key = api_key or OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY not set. Add it to .env or pass api_key.")

    m = model or OPENROUTER_MODEL or os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
    client = _get_cached_client(key)

    resp = client.chat.completions.create(
        model=m,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        extra_headers={
            "HTTP-Referer": "http://localhost",
            "X-OpenRouter-Title": "Legal-RAG",
        },
        extra_body={
            "transforms": ["middle-out"]
        }
    )

    if stream:
        return "".join(chunk.choices[0].delta.content or "" for chunk in resp)
    return resp.choices[0].message.content or ""
