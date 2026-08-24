"""DeepSeek adapter — OpenAI-compatible chat completions.

DeepSeek uses the OpenAI ``/chat/completions`` wire format, so this adapter
reuses ``OpenAILLM`` and only supplies DeepSeek-specific defaults and API-key
lookup. ``deepseek-reasoner`` thinking is exposed by the inherited adapter as
``channel="reasoning"`` deltas.

    llm = DeepSeekLLM(model="deepseek-chat")
    reasoner = DeepSeekLLM(model="deepseek-reasoner")

Set ``DEEPSEEK_API_KEY`` or pass ``api_key=``.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from .openai import OpenAILLM


# ---- adapter ----------------------------------------------------------


class DeepSeekLLM(OpenAILLM):
    """DeepSeek provider using its OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        client: httpx.AsyncClient | None = None,
        timeout: float = 600.0,
        **defaults: Any,
    ):
        key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if client is None and not key:
            raise ValueError("no API key: pass api_key= or set DEEPSEEK_API_KEY")
        super().__init__(model=model, api_key=key, base_url=base_url,
                         client=client, timeout=timeout, **defaults)
