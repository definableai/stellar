"""The Messages API, in three steps: build the request, send it, read the reply.

to_provider and to_core are pure. This file is the only place that knows what
a Part holds, and the only place that knows what the wire looks like.
"""

import asyncio
import os

import httpx

from core import Message, Part, ProviderModel, ToolCall

VERSION = "2023-06-01"


def flat(message) -> str:
    """A message's content as plain text."""
    if isinstance(message.content, str):
        return message.content
    return "".join(p.data for p in message.content if p.type == "text")


def source(data: dict) -> dict:
    """Where a picture lives: behind a link, or right here as base64."""
    if "url" in data:
        return {"type": "url", "url": data["url"]}
    return {"type": "base64", "media_type": data["media_type"], "data": data["data"]}


def block(part: Part) -> dict:
    """One Part, opened up into one content block."""
    if part.type == "text":
        return {"type": "text", "text": part.data}
    if part.type == "image":
        return {"type": "image", "source": source(part.data)}
    return {"type": part.type, **part.data}     # a block we never opened, handed back


def blocks(message) -> list[dict]:
    """Everything one message says: what it holds, then what it asks for."""
    parts = message.content
    if isinstance(parts, str):
        parts = [Part("text", parts)] if parts else []
    return [block(p) for p in parts] + [
        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.args}
        for c in message.tool_calls
    ]


class Anthropic(ProviderModel):
    """Claude over HTTP. The key comes from you or from ANTHROPIC_API_KEY."""

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        max_tokens: int = 4096,
        **params,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.params = params            # temperature, stop_sequences, whatever else

    def to_provider(self, agent) -> dict:
        """The notebook and the toolbox, as one request body."""
        system, turns, merging = [], [], False
        for m in agent.messages:
            if m.role == "system":
                system.append(flat(m))
            elif m.role == "tool":
                result = {"type": "tool_result", "tool_use_id": m.tool_call_id,
                          "content": flat(m)}
                if merging:             # results in a row make one user turn
                    turns[-1]["content"].append(result)
                else:
                    turns.append({"role": "user", "content": [result]})
                    merging = True
            else:
                turns.append({"role": m.role, "content": blocks(m)})
                merging = False
        body = {"model": self.model, "max_tokens": self.max_tokens,
                "messages": turns, **self.params}
        if system:
            body["system"] = "\n\n".join(system)
        if agent.tools:
            body["tools"] = [
                {"name": t.name, "description": t.description,
                 "input_schema": t.parameters}
                for t in agent.tools.values()
            ]
        return body

    async def send(self, body) -> dict:
        """POST it. Three tries; a 429, a 5xx or a timeout earns another one."""
        headers = {"x-api-key": self.api_key, "anthropic-version": VERSION}
        async with httpx.AsyncClient(timeout=60) as http:   # no pool to close
            for wait in (1, 2, 0):                          # 0 means last try
                try:
                    answer = await http.post(
                        f"{self.base_url}/v1/messages", json=body, headers=headers
                    )
                except httpx.TimeoutException:
                    if not wait:
                        raise
                    await asyncio.sleep(wait)
                    continue
                if answer.status_code < 400:
                    return answer.json()
                again = answer.status_code == 429 or answer.status_code >= 500
                if not (again and wait):
                    raise RuntimeError(
                        f"anthropic {answer.status_code}: {answer.text[:200]}"
                    )
                await asyncio.sleep(wait)

    def to_core(self, raw) -> Message:
        """The reply, as one line for the notebook."""
        calls, parts = [], []
        for b in raw.get("content", []):
            if b["type"] == "tool_use":
                calls.append(ToolCall(b["id"], b["name"], b.get("input") or {}))
            elif b["type"] == "text":
                parts.append(Part("text", b["text"]))
            else:
                parts.append(Part(b["type"], b))    # thinking, and whatever comes next
        text_only = all(p.type == "text" for p in parts)
        used = raw.get("usage") or {}
        return Message(
            "assistant",
            "".join(p.data for p in parts) if text_only else parts,
            calls,
            meta={
                "usage": {k: used.get(k) for k in ("input_tokens", "output_tokens")},
                "stop_reason": raw.get("stop_reason"),
            },
        )
