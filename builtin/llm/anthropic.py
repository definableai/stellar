"""Raw Anthropic Messages API adapter — httpx + SSE, no SDK.

Contract (see core/llm.py): yield ``LLMDelta`` chunks, then exactly one
``LLMReply``. Thinking streams natively as ``channel="reasoning"`` deltas —
pass ``thinking={"type": "adaptive", "display": "summarized"}`` to see the
summary text (current models default to omitted/empty thinking text).

    agent = Agent(AnthropicLLM(model="claude-opus-5"), tools=[...])

Self-check (no network, fake transport):

    uv run python -m builtin.llm.anthropic
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Sequence

import httpx

from core import LLMDelta, LLMError, LLMReply, Message, ToolCall, ToolSpec, Usage


def _to_anthropic(messages: Sequence[Message]) -> tuple[str | None, list[dict[str, Any]]]:
    """-> (system, messages). Tool results coalesce into one user message —
    Anthropic requires all parallel tool_result blocks in a single turn."""
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            if m.content:
                system_parts.append(m.content)
        elif m.role == "tool" and m.tool_result:
            block = {
                "type": "tool_result",
                "tool_use_id": m.tool_result.call_id,
                "content": json.dumps(m.tool_result.content, default=str),
                **({"is_error": True} if m.tool_result.is_error else {}),
            }
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                    and out[-1]["content"] and out[-1]["content"][-1].get("type") == "tool_result":
                out[-1]["content"].append(block)      # coalesce consecutive results
            else:
                out.append({"role": "user", "content": [block]})
        elif m.role == "assistant":
            # thinking blocks must be echoed back unchanged (required when
            # thinking + tool use continue on the same model)
            blocks: list[dict[str, Any]] = list(m.meta.get("anthropic_thinking", []))
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            blocks += [{"type": "tool_use", "id": c.id, "name": c.name,
                        "input": c.arguments} for c in m.tool_calls]
            if blocks:
                out.append({"role": "assistant", "content": blocks})
            # else: empty assistant turn (stop before any text) — API rejects ""
        else:
            out.append({"role": m.role, "content": m.content or ""})
    return "\n\n".join(system_parts) or None, out


class AnthropicLLM:
    def __init__(
        self,
        model: str = "claude-opus-5",
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        client: httpx.AsyncClient | None = None,
        timeout: float = 600.0,           # thinking turns can run minutes
        max_tokens: int = 64000,
        **defaults: Any,
    ):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if client is None and not key:
            raise ValueError("no API key: pass api_key= or set ANTHROPIC_API_KEY")
        # ponytail: client lives for the process; `await llm.client.aclose()` to exit clean.
        self.client = client or httpx.AsyncClient(
            base_url=base_url,
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            timeout=timeout,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.defaults = defaults

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], **params: Any
    ) -> AsyncIterator[LLMDelta | LLMReply]:
        system, msgs = _to_anthropic(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": msgs,
            "stream": True,
            **{**self.defaults, **params},
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [{"name": s.name, "description": s.description,
                                 "input_schema": s.parameters} for s in tools]

        text_parts: list[str] = []
        blocks: dict[int, dict[str, Any]] = {}    # index -> {type, id, name, json}
        finish: str | None = None
        usage = Usage()

        async with self.client.stream("POST", "/messages", json=payload) as r:
            if r.status_code >= 400:
                body = (await r.aread()).decode("utf-8", "replace")
                raise LLMError(r.status_code, body)
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue                       # event:/ping/comment lines
                ev = json.loads(line[5:].strip())
                t = ev.get("type", "")
                if t == "error":                   # mid-stream overload etc.
                    err = ev.get("error") or {}
                    status = 529 if err.get("type") == "overloaded_error" else 500
                    raise LLMError(status, json.dumps(err))
                elif t == "message_start":
                    u = (ev.get("message") or {}).get("usage") or {}
                    usage.input_tokens = (u.get("input_tokens", 0)
                                          + u.get("cache_creation_input_tokens", 0)
                                          + u.get("cache_read_input_tokens", 0))
                elif t == "content_block_start":
                    cb = ev.get("content_block") or {}
                    blocks[ev["index"]] = {"type": cb.get("type"),
                                           "id": cb.get("id"), "name": cb.get("name"),
                                           "json": "", "thinking": "", "sig": ""}
                elif t == "content_block_delta":
                    d = ev.get("delta") or {}
                    slot = blocks.setdefault(ev["index"], {   # tolerate orphan delta
                        "type": None, "id": None, "name": None,
                        "json": "", "thinking": "", "sig": ""})
                    if d.get("type") == "text_delta" and d.get("text"):
                        text_parts.append(d["text"])
                        yield LLMDelta(text=d["text"])
                    elif d.get("type") == "thinking_delta" and d.get("thinking"):
                        slot["thinking"] += d["thinking"]
                        yield LLMDelta(text=d["thinking"], channel="reasoning")
                    elif d.get("type") == "signature_delta":
                        slot["sig"] += d.get("signature", "")
                    elif d.get("type") == "input_json_delta" and d.get("partial_json"):
                        slot["json"] += d["partial_json"]
                        yield LLMDelta(text=d["partial_json"], channel="tool_args",
                                       index=ev["index"])
                elif t == "message_delta":
                    finish = (ev.get("delta") or {}).get("stop_reason") or finish
                    u = ev.get("usage") or {}
                    usage.output_tokens = u.get("output_tokens", usage.output_tokens)

        tool_calls = []
        for _, b in sorted(blocks.items()):
            if b["type"] != "tool_use":
                continue
            try:
                args = json.loads(b["json"] or "{}")
            except json.JSONDecodeError:
                continue    # truncated by max_tokens — drop the incomplete call
            tool_calls.append(ToolCall(id=b["id"], name=b["name"], arguments=args))
        # ponytail: redacted_thinking blocks not round-tripped; add if hit
        thinking_blocks = [
            {"type": "thinking", "thinking": b["thinking"], "signature": b["sig"]}
            for _, b in sorted(blocks.items())
            if b["type"] == "thinking" and b["thinking"]
        ]
        yield LLMReply(
            message=Message(role="assistant",
                            content="".join(text_parts) or None,
                            tool_calls=tool_calls,
                            meta={"anthropic_thinking": thinking_blocks}
                            if thinking_blocks else {}),
            usage=usage,
            stop_reason=finish or "end",
        )


if __name__ == "__main__":
    import asyncio

    async def _selfcheck() -> None:
        sys, msgs = _to_anthropic([
            Message(role="system", content="be terse"),
            Message(role="user", content="add 1+2 and 3+4"),
        ])
        assert sys == "be terse" and msgs == [{"role": "user", "content": "add 1+2 and 3+4"}]

        from core import ToolResult
        _, msgs2 = _to_anthropic([
            Message(role="user", content="go"),
            Message(role="assistant", tool_calls=[
                ToolCall(id="t1", name="add", arguments={"a": 1}),
                ToolCall(id="t2", name="add", arguments={"a": 3})]),
            Message(role="tool", tool_result=ToolResult("t1", "add", 3)),
            Message(role="tool", tool_result=ToolResult("t2", "add", 7, is_error=True)),
        ])
        assert [b["type"] for b in msgs2[1]["content"]] == ["tool_use", "tool_use"]
        results = msgs2[2]["content"]                 # coalesced into ONE user turn
        assert msgs2[2]["role"] == "user" and len(results) == 2
        assert results[1]["is_error"] and "tool_use_id" in results[0]

        sse = b"".join([
            b'event: message_start\n',
            b'data: {"type":"message_start","message":{"usage":{"input_tokens":9}}}\n\n',
            b'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}\n\n',
            b'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}\n\n',
            b'data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"s1"}}\n\n',
            b'data: {"type":"content_block_stop","index":0}\n\n',
            b'data: {"type":"content_block_start","index":1,"content_block":{"type":"text"}}\n\n',
            b'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hi"}}\n\n',
            b'data: {"type":"content_block_start","index":2,"content_block":'
            b'{"type":"tool_use","id":"t9","name":"add"}}\n\n',
            b'data: {"type":"content_block_delta","index":2,"delta":'
            b'{"type":"input_json_delta","partial_json":"{\\"a\\": "}}\n\n',
            b'data: {"type":"content_block_delta","index":2,"delta":'
            b'{"type":"input_json_delta","partial_json":"5}"}}\n\n',
            b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},'
            b'"usage":{"output_tokens":11}}\n\n',
            b'data: {"type":"message_stop"}\n\n',
        ])
        transport = httpx.MockTransport(lambda req: httpx.Response(
            200, content=sse, headers={"content-type": "text/event-stream"}))
        llm = AnthropicLLM(api_key="test", client=httpx.AsyncClient(
            transport=transport, base_url="http://fake"))
        got = [x async for x in llm.stream([Message(role="user", content="hi")],
                                           [ToolSpec(name="add")])]
        *deltas, reply = got
        assert [(d.text, d.channel, d.index) for d in deltas] == [
            ("hmm", "reasoning", 0), ("Hi", "text", 0),
            ('{"a": ', "tool_args", 2), ("5}", "tool_args", 2)], deltas
        assert reply.message.content == "Hi"          # reasoning stays out
        assert reply.message.tool_calls == [ToolCall(id="t9", name="add",
                                                     arguments={"a": 5})]
        assert (reply.usage.input_tokens, reply.usage.output_tokens) == (9, 11)
        assert reply.stop_reason == "tool_use"
        assert reply.message.meta["anthropic_thinking"] == [
            {"type": "thinking", "thinking": "hmm", "signature": "s1"}]
        _, echo = _to_anthropic([Message(role="user", content="hi"), reply.message])
        assert echo[1]["content"][0]["type"] == "thinking"   # echoed back first
        _, dropped = _to_anthropic([Message(role="user", content="x"),
                                    Message(role="assistant")])
        assert len(dropped) == 1                             # empty assistant dropped

        err_t = httpx.MockTransport(lambda req: httpx.Response(
            401, content=b'{"error":{"type":"authentication_error"}}'))
        bad = AnthropicLLM(api_key="x", client=httpx.AsyncClient(
            transport=err_t, base_url="http://fake"))
        try:
            async for _ in bad.stream([Message(role="user", content="hi")], []):
                pass
            raise AssertionError("expected LLMError on 401")
        except LLMError as e:
            assert e.status == 401 and not e.retryable

        overload = b'data: {"type":"error","error":{"type":"overloaded_error"}}\n\n'
        ot = httpx.MockTransport(lambda req: httpx.Response(
            200, content=overload, headers={"content-type": "text/event-stream"}))
        ollm = AnthropicLLM(api_key="x", client=httpx.AsyncClient(
            transport=ot, base_url="http://fake"))
        try:
            async for _ in ollm.stream([Message(role="user", content="hi")], []):
                pass
            raise AssertionError("expected LLMError on overload")
        except LLMError as e:
            assert e.status == 529 and e.retryable

        print("anthropic adapter self-check ok")

    asyncio.run(_selfcheck())
