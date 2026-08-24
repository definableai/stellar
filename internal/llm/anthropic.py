"""Raw Anthropic Messages API adapter — httpx + SSE, no SDK.

Contract (see core/llm.py): yield ``LLMDelta`` chunks, then exactly one
``LLMReply``. Thinking streams natively as ``channel="reasoning"`` deltas —
pass ``thinking={"type": "adaptive", "display": "summarized"}`` to see the
summary text (current models default to omitted/empty thinking text).

    agent = Agent(AnthropicLLM(model="claude-opus-5"), tools=[...])
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Sequence

import httpx

from core import (LLMDelta, LLMError, LLMReply, Message, ReplyBuilder,
                  ToolResult, ToolSpec)
from internal.llm.common import dump_result, map_blocks

# ---- request: core -> wire --------------------------------------------


def _source(b: dict[str, Any], default_media: str) -> dict[str, Any]:
    """url or base64 source — images and documents share the shape."""
    if b.get("url"):
        return {"type": "url", "url": b["url"]}
    return {"type": "base64", "media_type": b.get("media_type", default_media),
            "data": b.get("data", "")}


def _blocks(content: Any) -> Any:
    return map_blocks(
        content,
        text=lambda b: {"type": "text", "text": b.get("text", "")},
        image=lambda b: {"type": "image", "source": _source(b, "image/png")},
        file=lambda b: {"type": "document",          # PDFs and documents
                        "source": _source(b, "application/pdf")},
    )


def _take_system(m: Message, parts: list[str]) -> None:
    """System is a top-level request field, not a message; block content
    keeps its text parts only."""
    if isinstance(m.content, list):
        parts += [b.get("text", "") for b in m.content
                  if b.get("type") != "image"]
    elif m.content:
        parts.append(m.content)


def _add_result(out: list[dict[str, Any]], tr: ToolResult) -> None:
    """Anthropic requires ALL parallel tool_result blocks coalesced into
    ONE user turn — consecutive results append to the same message."""
    block = {"type": "tool_result", "tool_use_id": tr.call_id,
             "content": dump_result(tr.content),
             **({"is_error": True} if tr.is_error else {})}
    last = out[-1] if out else None
    if (last and last["role"] == "user" and isinstance(last["content"], list)
            and last["content"]
            and last["content"][-1].get("type") == "tool_result"):
        last["content"].append(block)
    else:
        out.append({"role": "user", "content": [block]})


def _add_assistant(out: list[dict[str, Any]], m: Message) -> None:
    """Thinking blocks echo back unchanged and FIRST (required when
    thinking + tool use continue on the same model); an empty assistant
    turn is dropped — the API rejects ""."""
    blocks: list[dict[str, Any]] = list(m.meta.get("anthropic_thinking", []))
    if m.content:
        blocks.append({"type": "text", "text": m.content})
    blocks += [{"type": "tool_use", "id": c.id, "name": c.name,
                "input": c.arguments} for c in m.tool_calls]
    if blocks:
        out.append({"role": "assistant", "content": blocks})


def _to_anthropic(messages: Sequence[Message]) -> tuple[str | None, list[dict[str, Any]]]:
    """-> (system, messages)."""
    system: list[str] = []
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            _take_system(m, system)
        elif m.role == "tool" and m.tool_result:
            _add_result(out, m.tool_result)
        elif m.role == "assistant":
            _add_assistant(out, m)
        else:
            out.append({"role": m.role, "content": _blocks(m.content)})
    return "\n\n".join(system) or None, out


# ---- adapter ----------------------------------------------------------


class AnthropicLLM:
    def __init__(
        self,
        model: str = "claude-opus-5",
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        client: httpx.AsyncClient | None = None,
        timeout: float = 600.0,           # thinking turns can run minutes
        max_tokens: int = 64000,
        cache: bool = True,               # prompt caching: breakpoints on
        **defaults: Any,                  # system + last message (~90% input
    ):                                    # cost cut on long agent runs)
        # client= is the test seam: when passed, it wins wholesale —
        # api_key/base_url/timeout are ignored and the key guard is skipped.
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
        self.cache = cache
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
        if self.cache:
            # two ephemeral breakpoints: system (covers tools+system prefix)
            # and the last message (covers the whole conversation prefix, so
            # each turn only pays for what's new since the previous one)
            if system:
                payload["system"] = [{"type": "text", "text": system,
                                      "cache_control": {"type": "ephemeral"}}]
            if msgs:
                content = msgs[-1]["content"]
                if isinstance(content, str) and content:
                    msgs[-1]["content"] = [{"type": "text", "text": content,
                                            "cache_control": {"type": "ephemeral"}}]
                elif isinstance(content, list) and content \
                        and content[-1].get("type") != "thinking":
                    content[-1] = {**content[-1],
                                   "cache_control": {"type": "ephemeral"}}

        b = ReplyBuilder()
        thinking: dict[int, dict[str, str]] = {}  # index -> {thinking, sig}

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
                    b.usage(input_tokens=(u.get("input_tokens", 0)
                            + u.get("cache_creation_input_tokens", 0)
                            + u.get("cache_read_input_tokens", 0)))
                elif t == "content_block_start":
                    cb = ev.get("content_block") or {}
                    if cb.get("type") == "tool_use":
                        b.tool_call(ev["index"], id=cb.get("id") or "",
                                    name=cb.get("name") or "")
                elif t == "content_block_delta":
                    d = ev.get("delta") or {}
                    slot = thinking.setdefault(ev["index"],
                                               {"thinking": "", "sig": ""})
                    if d.get("type") == "text_delta" and d.get("text"):
                        yield b.text(d["text"])
                    elif d.get("type") == "thinking_delta" and d.get("thinking"):
                        slot["thinking"] += d["thinking"]
                        yield b.reasoning(d["thinking"])
                    elif d.get("type") == "signature_delta":
                        slot["sig"] += d.get("signature", "")
                    elif d.get("type") == "input_json_delta" and d.get("partial_json"):
                        yield b.tool_args(ev["index"], d["partial_json"])
                elif t == "message_delta":
                    b.finish((ev.get("delta") or {}).get("stop_reason"))
                    u = ev.get("usage") or {}
                    if u.get("output_tokens") is not None:
                        b.usage(output_tokens=u["output_tokens"])

        # ponytail: redacted_thinking blocks not round-tripped; add if hit
        thinking_blocks = [
            {"type": "thinking", "thinking": s["thinking"], "signature": s["sig"]}
            for _, s in sorted(thinking.items()) if s["thinking"]
        ]
        if thinking_blocks:
            b.meta["anthropic_thinking"] = thinking_blocks
        yield b.reply()
