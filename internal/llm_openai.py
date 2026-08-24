"""Raw OpenAI adapters — httpx + SSE, no SDK.

Two adapters, same contract (see core/llm.py: yield ``LLMDelta`` chunks,
then exactly one ``LLMReply``):

- ``OpenAILLM`` — /chat/completions. De-facto standard wire format:
  ``base_url`` points it at any compatible server (Groq, vLLM, Ollama,
  OpenRouter, LiteLLM...). NOTE: OpenAI hides reasoning here — no
  reasoning deltas, only ``usage.reasoning_tokens``.
- ``OpenAIResponsesLLM`` — /responses. OpenAI-only; streams reasoning
  summaries as ``channel="reasoning"`` deltas during the think:

      OpenAIResponsesLLM(model="gpt-5.6-luna",
                         reasoning={"summary": "auto", "effort": "low"})

Provider params pass through verbatim — constructor ``**defaults`` for
every call, ``Agent(params=...)`` per agent, ``agent.run(..., params=...)``
per run (dict values deep-merge one level). Chat Completions accepts:
``max_completion_tokens``, ``reasoning_effort`` ("minimal".."high"),
``verbosity``, ``temperature``/``top_p`` (non-reasoning models only),
``tool_choice`` ("auto"/"none"/"required"/forced), ``parallel_tool_calls``,
``response_format`` (json_object/json_schema), ``stop``, ``seed``,
``frequency_penalty``/``presence_penalty``, ``logprobs``/``logit_bias``,
``prediction``, ``user``/``metadata``/``store``/``service_tier``.
Responses renames: ``max_output_tokens``, ``instructions``,
``reasoning={"effort","summary"}``, ``text={"verbosity","format"}``,
``previous_response_id``/``store``, ``background``, ``truncation``.
Responses also takes OpenAI built-in tools via the ``tools`` param or
default — ``tools=[{"type": "web_search"}]`` — run on OpenAI's side;
the adapter appends the loop's function tools to them.

    OpenAILLM(model="gpt-5.6-luna", reasoning_effort="low",
              max_completion_tokens=8000)

Caveat: ``n > 1`` unsupported — the adapter reads ``choices[0]`` only.
Field list drifts; platform.openai.com/docs/api-reference is authoritative.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Sequence

import httpx

from core import (LLMDelta, LLMError, LLMReply, Message, ReplyBuilder,
                  ToolResult, ToolSpec)
from internal.llm_common import data_url, dump_result, map_blocks

# ---- request: core -> wire (chat completions) -------------------------


def _file_part(b: dict[str, Any]) -> dict[str, Any]:
    """chat completions takes file bytes (data URL) or a file_id — it
    cannot fetch URLs; failing loud beats sending an empty payload."""
    if b.get("url"):
        raise ValueError("chat completions cannot fetch file URLs — pass "
                         "base64 data, or upload and use a file_id")
    return {"type": "file", "file": {"filename": b.get("name", "file"),
                                     "file_data": data_url(b, "application/pdf")}}


def _blocks(content: Any) -> Any:
    return map_blocks(
        content,
        text=lambda b: {"type": "text", "text": b.get("text", "")},
        image=lambda b: {"type": "image_url",
                         "image_url": {"url": data_url(b, "image/png")}},
        file=_file_part,
    )


def _result_msg(tr: ToolResult) -> dict[str, Any]:
    """Tool results are their own role="tool" messages, keyed by call id."""
    return {"role": "tool", "tool_call_id": tr.call_id,
            "content": dump_result(tr.content)}


def _assistant_msg(m: Message) -> dict[str, Any]:
    """Tool calls ride ON the assistant message, arguments as JSON text."""
    return {"role": "assistant", "content": m.content,
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.name,
                                         "arguments": json.dumps(c.arguments)}}
                           for c in m.tool_calls]}


def _to_openai(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool" and m.tool_result:
            out.append(_result_msg(m.tool_result))
        elif m.role == "assistant" and m.tool_calls:
            out.append(_assistant_msg(m))
        else:
            out.append({"role": m.role, "content": _blocks(m.content)})
    return out


# ---- adapter: chat completions ----------------------------------------


class OpenAILLM:
    # the evolution seam: subclass and override to change what crosses the
    # wire (e.g. tool results carrying images) without copying the stream loop
    to_wire = staticmethod(_to_openai)

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        client: httpx.AsyncClient | None = None,
        timeout: float = 120.0,
        **defaults: Any,
    ):
        # client= is the test seam: when passed, it wins wholesale —
        # api_key/base_url/timeout are ignored and the key guard is skipped.
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if client is None and not key:
            raise ValueError("no API key: pass api_key= or set OPENAI_API_KEY")
        # ponytail: client lives for the process (connection reuse across
        # loop steps); `await llm.client.aclose()` if you need a clean exit.
        self.client = client or httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )
        self.model = model
        self.defaults = defaults

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], **params: Any
    ) -> AsyncIterator[LLMDelta | LLMReply]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self.to_wire(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
            **{**self.defaults, **params},
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": s.to_dict()} for s in tools
            ]

        b = ReplyBuilder()
        async with self.client.stream("POST", "/chat/completions", json=payload) as r:
            if r.status_code >= 400:
                body = (await r.aread()).decode("utf-8", "replace")
                raise LLMError(r.status_code, body)
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue                    # SSE comments, keep-alives, blanks
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    u = chunk["usage"]
                    b.usage(input_tokens=u.get("prompt_tokens", 0),
                            output_tokens=u.get("completion_tokens", 0),
                            reasoning_tokens=(u.get("completion_tokens_details")
                                              or {}).get("reasoning_tokens", 0))
                if not chunk.get("choices"):
                    continue
                choice = chunk["choices"][0]
                b.finish(choice.get("finish_reason"))
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):   # DeepSeek-style compat servers
                    yield b.reasoning(delta["reasoning_content"])
                if delta.get("content"):
                    yield b.text(delta["content"])
                for i, tc in enumerate(delta.get("tool_calls") or []):
                    # some proxies omit "index" — fall back to position in
                    # the delta so parallel calls don't merge into one slot
                    idx = tc.get("index", i)
                    fn = tc.get("function") or {}
                    b.tool_call(idx, id=tc.get("id") or "",
                                name=fn.get("name") or "")
                    if fn.get("arguments"):
                        yield b.tool_args(idx, fn["arguments"])
        yield b.reply()


# ---- request: core -> wire (responses) --------------------------------


def _r_blocks(content: Any) -> Any:
    return map_blocks(
        content,
        text=lambda b: {"type": "input_text", "text": b.get("text", "")},
        image=lambda b: {"type": "input_image",
                         "image_url": data_url(b, "image/png")},
        file=lambda b: ({"type": "input_file", "file_url": b["url"]}
                        if b.get("url") else
                        {"type": "input_file", "filename": b.get("name", "file"),
                         "file_data": data_url(b, "application/pdf")}),
    )


def _to_responses(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Responses flattens: tool calls and results are TOP-LEVEL typed
    items, not fields on messages."""
    items: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool" and m.tool_result:
            items.append({"type": "function_call_output",
                          "call_id": m.tool_result.call_id,
                          "output": dump_result(m.tool_result.content)})
        elif m.role == "assistant" and m.tool_calls:
            if m.content:
                items.append({"role": "assistant", "content": m.content})
            items += [{"type": "function_call", "call_id": c.id, "name": c.name,
                       "arguments": json.dumps(c.arguments)}
                      for c in m.tool_calls]
        else:
            items.append({"role": m.role, "content": _r_blocks(m.content)})
    return items


# ---- adapter: responses -----------------------------------------------


class OpenAIResponsesLLM(OpenAILLM):
    """/responses variant. Pass reasoning={"summary": "auto"} to stream
    reasoning summaries. ponytail: resends full transcript per turn
    (stateless); previous_response_id server-state if traffic matters."""

    to_wire = staticmethod(_to_responses)

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], **params: Any
    ) -> AsyncIterator[LLMDelta | LLMReply]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": self.to_wire(messages),
            "stream": True,
            **{**self.defaults, **params},
        }
        # a "tools" param/default carries OpenAI built-ins (web_search, ...)
        # run server-side; the loop's function tools are appended to them.
        # /responses defaults function tools to strict=true, which rejects
        # any schema with optional params. ToolSpec promises no such thing.
        builtin = payload.pop("tools", None)
        if tools or builtin:
            payload["tools"] = list(builtin or []) + [
                {"type": "function", "strict": False, **s.to_dict()} for s in tools]

        b = ReplyBuilder()
        saw_calls = False
        status = "end"
        async with self.client.stream("POST", "/responses", json=payload) as r:
            if r.status_code >= 400:
                body = (await r.aread()).decode("utf-8", "replace")
                raise LLMError(r.status_code, body)
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                t = ev.get("type", "")
                if t == "response.output_text.delta":
                    yield b.text(ev["delta"])
                elif t == "response.reasoning_summary_text.delta":
                    yield b.reasoning(ev["delta"])
                elif t == "response.function_call_arguments.delta" and ev.get("delta"):
                    # display only — the finalized JSON arrives on item.done
                    yield LLMDelta(text=ev["delta"], channel="tool_args",
                                   index=ev.get("output_index", 0))
                elif (t == "response.output_item.done"
                      and ev["item"].get("type") == "function_call"):
                    it = ev["item"]
                    idx = ev.get("output_index", 0)
                    saw_calls = True
                    b.tool_call(idx, id=it["call_id"], name=it["name"])
                    b.tool_args(idx, it.get("arguments") or "")
                elif (t == "response.output_item.done"
                      and ev["item"].get("type") == "web_search_call"):
                    # server-side tool: display-only, nothing to execute
                    q = (ev["item"].get("action") or {}).get("query") or ""
                    yield LLMDelta(text=f"[web search] {q}\n", channel="status")
                elif t in ("response.completed", "response.incomplete",
                           "response.failed"):
                    resp = ev.get("response") or {}
                    u = resp.get("usage") or {}
                    b.usage(input_tokens=u.get("input_tokens", 0),
                            output_tokens=u.get("output_tokens", 0),
                            reasoning_tokens=(u.get("output_tokens_details")
                                              or {}).get("reasoning_tokens", 0))
                    if t == "response.failed":
                        raise LLMError(500, json.dumps(resp.get("error") or {}))
                    if t == "response.incomplete":
                        status = (resp.get("incomplete_details") or {}
                                  ).get("reason", "incomplete")
        b.finish("tool_use" if saw_calls else status)
        yield b.reply()
