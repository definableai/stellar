"""Reference LLM adapter: OpenAI Chat Completions (streaming).

UNTESTED SKETCH — a map of where provider translation lives, so you
can write your own (Anthropic, Groq, local vLLM, LiteLLM...) in ~60
lines. The contract to satisfy: yield LLMDelta chunks, then exactly
one LLMReply. Everything provider-specific stays in this file; the
loop never sees it.

    pip install openai
    agent = Agent(OpenAILLM(model="gpt-4o"), tools=[...])
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Sequence

from agentcore import LLMDelta, LLMReply, Message, ToolCall, ToolSpec, Usage


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
    def __init__(self, model: str = "gpt-4o", client: Any = None, **defaults: Any):
        from openai import AsyncOpenAI
        self.client = client or AsyncOpenAI()
        self.model = model
        self.defaults = defaults

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], **params: Any
    ) -> AsyncIterator[LLMDelta | LLMReply]:
        oai_tools = [
            {"type": "function", "function": s.to_dict()} for s in tools
        ] or None
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=_to_openai(messages),
            tools=oai_tools,
            stream=True,
            stream_options={"include_usage": True},
            **{**self.defaults, **params},
        )

        text_parts: list[str] = []
        calls: dict[int, dict[str, str]] = {}   # index -> {id, name, args}
        finish, usage = None, Usage()

        async for chunk in response:
            if chunk.usage:
                usage = Usage(chunk.usage.prompt_tokens,
                              chunk.usage.completion_tokens)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish = choice.finish_reason or finish
            delta = choice.delta
            if delta.content:
                text_parts.append(delta.content)
                yield LLMDelta(text=delta.content)
            for tc in delta.tool_calls or []:      # accumulate partial JSON
                slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments

        tool_calls = [
            ToolCall(id=s["id"], name=s["name"],
                     arguments=json.loads(s["args"] or "{}"))
            for _, s in sorted(calls.items())
        ]
        yield LLMReply(
            message=Message(role="assistant",
                            content="".join(text_parts) or None,
                            tool_calls=tool_calls),
            usage=usage,
            stop_reason=finish or "end",
        )
