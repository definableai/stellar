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
                         reasoning={"summary": "auto"})

Self-check (no network, fake transport):

    uv run python -m builtin.llm.openai
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Sequence

import httpx

from core import LLMDelta, LLMError, LLMReply, Message, ToolCall, ToolSpec, Usage


def _to_openai(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool" and m.tool_result:
            out.append({
                "role": "tool",
                "tool_call_id": m.tool_result.call_id,
                "content": json.dumps(m.tool_result.content, default=str),
            })
        elif m.role == "assistant" and m.tool_calls:
            out.append({
                "role": "assistant",
                "content": m.content,
                "tool_calls": [{
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name,
                                 "arguments": json.dumps(c.arguments)},
                } for c in m.tool_calls],
            })
        else:
            out.append({"role": m.role, "content": m.content or ""})
    return out


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

        text_parts: list[str] = []
        calls: dict[int, dict[str, str]] = {}   # index -> {id, name, args}
        finish: str | None = None
        usage = Usage()

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
                    usage = Usage(u.get("prompt_tokens", 0),
                                  u.get("completion_tokens", 0),
                                  (u.get("completion_tokens_details") or {})
                                  .get("reasoning_tokens", 0))
                if not chunk.get("choices"):
                    continue
                choice = chunk["choices"][0]
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):   # DeepSeek-style compat servers
                    yield LLMDelta(text=delta["reasoning_content"], channel="reasoning")
                if delta.get("content"):
                    text_parts.append(delta["content"])
                    yield LLMDelta(text=delta["content"])
                for tc in delta.get("tool_calls") or []:   # accumulate partial JSON
                    # some OpenAI-compatible proxies omit "index" (single call)
                    slot = calls.setdefault(tc.get("index", 0),
                                            {"id": "", "name": "", "args": ""})
                    fn = tc.get("function") or {}
                    slot["id"] = tc.get("id") or slot["id"]
                    slot["name"] += fn.get("name") or ""
                    slot["args"] += fn.get("arguments") or ""

        tool_calls = []
        for _, s in sorted(calls.items()):
            try:
                args = json.loads(s["args"] or "{}")
            except json.JSONDecodeError:
                continue    # truncated by length stop — drop the incomplete call
            tool_calls.append(ToolCall(id=s["id"], name=s["name"], arguments=args))
        yield LLMReply(
            message=Message(role="assistant",
                            content="".join(text_parts) or None,
                            tool_calls=tool_calls),
            usage=usage,
            stop_reason=finish or "end",
        )


def _to_responses(messages: Sequence[Message]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool" and m.tool_result:
            items.append({"type": "function_call_output",
                          "call_id": m.tool_result.call_id,
                          "output": json.dumps(m.tool_result.content, default=str)})
        elif m.role == "assistant" and m.tool_calls:
            if m.content:
                items.append({"role": "assistant", "content": m.content})
            for c in m.tool_calls:
                items.append({"type": "function_call", "call_id": c.id,
                              "name": c.name, "arguments": json.dumps(c.arguments)})
        else:
            items.append({"role": m.role, "content": m.content or ""})
    return items


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
            payload["tools"] = [{"type": "function", **s.to_dict()} for s in tools]

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage = Usage()
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
                    text_parts.append(ev["delta"])
                    yield LLMDelta(text=ev["delta"])
                elif t == "response.reasoning_summary_text.delta":
                    yield LLMDelta(text=ev["delta"], channel="reasoning")
                elif (t == "response.output_item.done"
                      and ev["item"].get("type") == "function_call"):
                    it = ev["item"]   # args arrive finalized here — no accumulation
                    tool_calls.append(ToolCall(
                        id=it["call_id"], name=it["name"],
                        arguments=json.loads(it.get("arguments") or "{}")))
                elif t in ("response.completed", "response.incomplete",
                           "response.failed"):
                    resp = ev.get("response") or {}
                    u = resp.get("usage") or {}
                    usage = Usage(u.get("input_tokens", 0),
                                  u.get("output_tokens", 0),
                                  (u.get("output_tokens_details") or {})
                                  .get("reasoning_tokens", 0))
                    if t == "response.failed":
                        raise LLMError(500, json.dumps(resp.get("error") or {}))
                    if t == "response.incomplete":
                        status = (resp.get("incomplete_details") or {}
                                  ).get("reason", "incomplete")

        yield LLMReply(
            message=Message(role="assistant",
                            content="".join(text_parts) or None,
                            tool_calls=tool_calls),
            usage=usage,
            stop_reason="tool_use" if tool_calls else status,
        )


if __name__ == "__main__":
    import asyncio

    async def _selfcheck() -> None:
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
            ("hmm", "reasoning"), ("Hel", "text"), ("lo", "text")], deltas
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
        assert [(d.text, d.channel) for d in d2] == [
            ("think", "reasoning"), ("Hi", "text")], d2
        assert rep2.message.content == "Hi"
        assert rep2.message.tool_calls == [ToolCall(id="c9", name="add",
                                                    arguments={"a": 1})]
        assert (rep2.usage.input_tokens, rep2.usage.output_tokens,
                rep2.usage.reasoning_tokens) == (5, 9, 4)
        assert rep2.stop_reason == "tool_use"

        print("openai adapter self-check ok")

    asyncio.run(_selfcheck())
