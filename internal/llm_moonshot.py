"""Moonshot AI adapter — OpenAI-compatible chat completions.

Moonshot uses the OpenAI ``/chat/completions`` wire format, so this adapter
reuses ``OpenAILLM`` and only supplies Moonshot-specific defaults and API-key
lookup. Kimi reasoning is exposed by the inherited adapter as
``channel="reasoning"`` deltas.

    llm = MoonshotLLM(model="kimi-k2.5")

Set ``MOONSHOT_API_KEY`` or pass ``api_key=``. The international API is used
by default; pass ``base_url="https://api.moonshot.cn/v1"`` for the China API.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from internal.llm_openai import OpenAILLM


class MoonshotLLM(OpenAILLM):
    """Moonshot/Kimi provider using its OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "kimi-k2.5",
        api_key: str | None = None,
        base_url: str = "https://api.moonshot.ai/v1",
        client: httpx.AsyncClient | None = None,
        timeout: float = 600.0,
        **defaults: Any,
    ):
        key = api_key or os.environ.get("MOONSHOT_API_KEY", "")
        if client is None and not key:
            raise ValueError("no API key: pass api_key= or set MOONSHOT_API_KEY")
        super().__init__(model=model, api_key=key, base_url=base_url,
                         client=client, timeout=timeout, **defaults)


def setup(ctx: Any) -> None:
    """Adapter shape (core/adapter.py); config = MoonshotLLM kwargs."""
    ctx.llm(MoonshotLLM(**ctx.config))
