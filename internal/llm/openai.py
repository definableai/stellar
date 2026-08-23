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

    OpenAILLM(model="gpt-5.6-luna", reasoning_effort="low",
              max_completion_tokens=8000)

Caveat: ``n > 1`` unsupported — the adapter reads ``choices[0]`` only.
Field list drifts; platform.openai.com/docs/api-reference is authoritative.

Self-check (no network, fake transport):

    uv run python -m internal.llm.openai
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Sequence

import httpx

from core import (LLMDelta, LLMError, LLMReply, Message, ReplyBuilder,
                  ToolCall, ToolResult, ToolSpec)
from internal.llm.common import data_url, dump_result, map_blocks

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
            "messages": _to_openai(messages),
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

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], **params: Any
    ) -> AsyncIterator[LLMDelta | LLMReply]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": _to_responses(messages),
            "stream": True,
            **{**self.defaults, **params},
        }
        if tools:
            # /responses defaults function tools to strict=true, which rejects
            # any schema with optional params. ToolSpec promises no such thing.
            payload["tools"] = [{"type": "function", "strict": False, **s.to_dict()}
                                for s in tools]

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


if __name__ == "__main__":
    import asyncio

    async def _selfcheck() -> None:
        from core import file_block

        try:
            _blocks([file_block(url="http://x/d.pdf")])
            raise AssertionError("chat file URL must raise")
        except ValueError:
            pass
        assert _blocks([file_block(data="QUJD")])[0]["file"][
            "file_data"].startswith("data:application/pdf")
        assert _r_blocks([file_block(url="http://x/d.pdf")]) == [
            {"type": "input_file", "file_url": "http://x/d.pdf"}]

        sse_body = b"".join([
            b'data: {"choices":[{"delta":{"reasoning_content":"hmm"},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}\n\n',
            b': keep-alive comment\n\n',
            b'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1",'
            b'"function":{"name":"add","arguments":"{\\"a\\": "}}]},'
            b'"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            b'"function":{"arguments":"2}"}}]},"finish_reason":"tool_calls"}]}\n\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3,'
            b'"completion_tokens_details":{"reasoning_tokens":2}}}\n\n',
            b'data: [DONE]\n\n',
        ])
        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200, content=sse_body,
                headers={"content-type": "text/event-stream"}))
        llm = OpenAILLM(api_key="test", client=httpx.AsyncClient(
            transport=transport, base_url="http://fake"))
        got = [x async for x in llm.stream(
            [Message(role="user", content="hi")],
            [ToolSpec(name="add", parameters={"type": "object"})])]

        *deltas, reply = got
        assert [(d.text, d.channel) for d in deltas] == [
            ("hmm", "reasoning"), ("Hel", "text"), ("lo", "text"),
            ('{"a": ', "tool_args"), ("2}", "tool_args")], deltas
        assert isinstance(reply, LLMReply)
        assert reply.message.content == "Hello"       # reasoning stays out
        assert reply.message.tool_calls == [
            ToolCall(id="c1", name="add", arguments={"a": 2})]
        assert (reply.usage.input_tokens, reply.usage.output_tokens,
                reply.usage.reasoning_tokens) == (7, 3, 2)
        assert reply.stop_reason == "tool_calls"

        err_transport = httpx.MockTransport(
            lambda req: httpx.Response(401, content=b'{"error":"bad key"}'))
        bad = OpenAILLM(api_key="x", client=httpx.AsyncClient(
            transport=err_transport, base_url="http://fake"))
        try:
            async for _ in bad.stream([Message(role="user", content="hi")], []):
                pass
            raise AssertionError("expected LLMError on 401")
        except LLMError as e:
            assert e.status == 401 and "bad key" in e.body and not e.retryable

        resp_body = b"".join([
            b'data: {"type":"response.reasoning_summary_text.delta","delta":"think"}\n\n',
            b'data: {"type":"response.output_text.delta","delta":"Hi"}\n\n',
            b'data: {"type":"response.function_call_arguments.delta",'
            b'"delta":"{\\"a\\": 1}","output_index":1}\n\n',
            b'data: {"type":"response.output_item.done","item":{"type":"function_call",'
            b'"call_id":"c9","name":"add","arguments":"{\\"a\\": 1}"}}\n\n',
            b'data: {"type":"response.completed","response":{"usage":{"input_tokens":5,'
            b'"output_tokens":9,"output_tokens_details":{"reasoning_tokens":4}}}}\n\n',
        ])
        rt = httpx.MockTransport(lambda req: httpx.Response(
            200, content=resp_body,
            headers={"content-type": "text/event-stream"}))
        rllm = OpenAIResponsesLLM(api_key="test", client=httpx.AsyncClient(
            transport=rt, base_url="http://fake"))
        got2 = [x async for x in rllm.stream([Message(role="user", content="hi")], [])]
        *d2, rep2 = got2
        assert [(d.text, d.channel, d.index) for d in d2] == [
            ("think", "reasoning", 0), ("Hi", "text", 0),
            ('{"a": 1}', "tool_args", 1)], d2
        assert rep2.message.content == "Hi"
        assert rep2.message.tool_calls == [ToolCall(id="c9", name="add",
                                                    arguments={"a": 1})]
        assert (rep2.usage.input_tokens, rep2.usage.output_tokens,
                rep2.usage.reasoning_tokens) == (5, 9, 4)
        assert rep2.stop_reason == "tool_use"

        print("openai adapter self-check ok")

    asyncio.run(_selfcheck())
