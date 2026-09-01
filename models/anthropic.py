"""The Messages API, in two steps: build the request, send it back as Parts.

encode is pure. Core never opens a Part and never sees the wire — an adapter
does both, and here the wire is the Messages API. This adapter does not
stream off the socket yet: send POSTs once and yields the Parts of that one
reply, which is all the Part protocol asks for.
"""

import asyncio
import os
from typing import cast

import httpx

from core import ContractError, Message, Part, ProviderModel, Run, ToolCall

VERSION = "2023-06-01"


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


def blocks(message: Message) -> list[dict]:
    """Everything one message says: what it holds, then what it asks for."""
    return [block(p) for p in cast(list[Part], message.content)] + [
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
        self.api_key = api_key if api_key else os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ContractError("no API key: pass api_key= or set ANTHROPIC_API_KEY")
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.params = params            # temperature, stop_sequences, whatever else

    def encode(self, run: Run) -> dict:
        """This run's notebook and the agent's toolbox, as one request body."""
        system: list[str] = []
        turns: list[dict] = []
        merging = False
        for m in run.messages:
            if m.role == "system":
                system.append(m.text)
            elif m.role == "tool":
                result = {"type": "tool_result", "tool_use_id": m.tool_call_id,
                          "content": m.text}
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
        if run.agent.tools:
            body["tools"] = [
                {"name": t.name, "description": t.description,
                 "input_schema": t.parameters}
                for t in run.agent.tools.values()
            ]
        return body

    async def send(self, run, body: dict):
        """POST once, then hand the reply over as Parts."""
        raw = await self.post(body)
        for b in raw.get("content", []):
            if b["type"] == "tool_use":
                yield Part("tool_call",
                           ToolCall(b["id"], b["name"], b.get("input") or {}))
            elif b["type"] == "text":
                yield Part("text", b["text"])
            else:
                yield Part(b["type"], b)    # thinking, and whatever comes next
        used = raw.get("usage") or {}
        yield Part("meta", {
            "usage": {k: used.get(k) for k in ("input_tokens", "output_tokens")},
            "stop_reason": raw.get("stop_reason"),
        })

    async def post(self, body: dict) -> dict:
        """Three tries; a 429, a 5xx or a timeout earns another one."""
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
        raise AssertionError("unreachable: the last try returns or raises")
