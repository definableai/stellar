"""DeepSeek adapter — OpenAI-compatible chat completions.

DeepSeek uses the OpenAI ``/chat/completions`` wire format, so this adapter
reuses ``OpenAILLM`` and only supplies DeepSeek-specific defaults and API-key
lookup. ``deepseek-reasoner`` thinking is exposed by the inherited adapter as
``channel="reasoning"`` deltas.

    llm = DeepSeekLLM(model="deepseek-chat")
    reasoner = DeepSeekLLM(model="deepseek-reasoner")

Set ``DEEPSEEK_API_KEY`` or pass ``api_key=``.

Self-check (no network, fake transport):

    uv run python -m internal.llm.deepseek
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from core import LLMReply, Message, ToolCall, ToolSpec

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


# ---- self-check -------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def _selfcheck() -> None:
        configured = DeepSeekLLM(api_key="test")
        assert str(configured.client.base_url) == "https://api.deepseek.com"
        assert configured.client.headers["Authorization"] == "Bearer test"
        assert configured.model == "deepseek-chat"
        await configured.client.aclose()

        seen: dict[str, Any] = {}

        def event(payload: dict[str, Any]) -> bytes:
            return f"data: {json.dumps(payload)}\n\n".encode()

        sse = b"".join([
            event({"choices": [{"delta": {"reasoning_content": "think"},
                                "finish_reason": None}]}),
            event({"choices": [{"delta": {"content": "Hi"},
                                "finish_reason": None}]}),
            event({"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "c1", "function": {
                    "name": "search", "arguments": '{"q":"deepseek"}'}}]},
                "finish_reason": "tool_calls"}]}),
            event({"choices": [], "usage": {
                "prompt_tokens": 8, "completion_tokens": 5,
                "completion_tokens_details": {"reasoning_tokens": 2}}}),
            b"data: [DONE]\n\n",
        ])

        def respond(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["payload"] = json.loads(request.content)
            return httpx.Response(
                200, content=sse,
                headers={"content-type": "text/event-stream"})

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="http://fake")
        llm = DeepSeekLLM(model="deepseek-reasoner", client=client)
        got = [item async for item in llm.stream(
            [Message(role="user", content="hello")],
            [ToolSpec(name="search", parameters={"type": "object"})])]
        await client.aclose()

        *deltas, reply = got
        assert seen["path"] == "/chat/completions"
        assert seen["payload"]["model"] == "deepseek-reasoner"
        assert seen["payload"]["stream_options"] == {"include_usage": True}
        assert seen["payload"]["tools"][0]["function"]["name"] == "search"
        assert [(d.text, d.channel) for d in deltas] == [
            ("think", "reasoning"), ("Hi", "text"),
            ('{"q":"deepseek"}', "tool_args")]
        assert isinstance(reply, LLMReply)
        assert reply.message.content == "Hi"
        assert reply.message.tool_calls == [
            ToolCall(id="c1", name="search", arguments={"q": "deepseek"})]
        assert (reply.usage.input_tokens, reply.usage.output_tokens,
                reply.usage.reasoning_tokens) == (8, 5, 2)
        assert reply.stop_reason == "tool_calls"

        print("deepseek adapter self-check ok")

    asyncio.run(_selfcheck())
