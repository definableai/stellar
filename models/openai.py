"""OpenAI Chat Completions — the wire format the compatible vendors clone.

Three steps, like any ProviderModel: build the body, POST it, read the reply.
The quirk worth remembering is that a tool call's arguments travel as a JSON
*string*: dumped on the way out, parsed on the way back in.
"""

import asyncio
import json
import os

import httpx

from core import ContractError, Message, ProviderModel, ToolCall


def part(piece) -> dict:
    """One Part as one content part. Text stays text; an image becomes a URL."""
    if piece.type == "text":
        return {"type": "text", "text": piece.data}
    if piece.type == "image":                # {"url": …} or {"media_type": …, "data": …}
        d = piece.data
        url = d["url"] if "url" in d else f"data:{d['media_type']};base64,{d['data']}"
        return {"type": "image_url", "image_url": {"url": url}}
    raise ContractError(
        f"Chat Completions has no part type {piece.type!r}; say 'text' or 'image'"
    )


def line(message) -> dict:
    """One notebook line as one message: the role, the words, the plumbing."""
    body = message.content
    said = {
        "role": message.role,
        "content": body if isinstance(body, str) else [part(p) for p in body],
    }
    if message.tool_calls:
        said["tool_calls"] = [
            {"id": call.id, "type": "function",
             "function": {"name": call.name, "arguments": json.dumps(call.args)}}
            for call in message.tool_calls
        ]
    if message.tool_call_id:
        said["tool_call_id"] = message.tool_call_id
    return said


def args_of(function: dict) -> tuple[dict, str | None]:
    """The arguments string as a dict — or an empty one and the string back."""
    try:
        args = json.loads(function.get("arguments") or "{}")
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
    except ValueError:                       # a broken parse is a ValueError too
        return {}, function.get("arguments")
    return args, None


class OpenAI(ProviderModel):
    """The brain behind api.openai.com — or anything that copied its format.

    Point base_url somewhere else and the same class talks to the clone.
    """

    def __init__(
        self,
        model: str = "gpt-5.6-luna",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        **params,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.params = params                 # temperature, max_tokens, whatever else

    def to_provider(self, agent) -> dict:
        """The whole notebook and the whole toolbox, as one request body."""
        body = {
            "model": self.model,
            "messages": [line(m) for m in agent.messages],
            **self.params,
        }
        if agent.tools:
            body["tools"] = [
                {"type": "function",
                 "function": {"name": t.name, "description": t.description,
                              "parameters": t.parameters}}
                for t in agent.tools.values()
            ]
        return body

    async def send(self, body) -> dict:
        """POST it. Three tries for the failures worth retrying, then give up.

        A fresh client per call: no connection reuse, no lifecycle to own.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=60) as http:
            for attempt in range(3):
                try:
                    reply = await http.post(
                        f"{self.base_url}/chat/completions", json=body, headers=headers
                    )
                except httpx.TimeoutException as ex:
                    problem = f"timeout: {ex!r}"
                else:
                    if reply.status_code < 400:
                        return reply.json()
                    problem = f"{reply.status_code}: {reply.text[:200]}"
                    if reply.status_code != 429 and reply.status_code < 500:
                        break                # our mistake; asking again won't fix it
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
        raise RuntimeError(f"chat/completions failed — {problem}")

    def to_core(self, raw) -> Message:
        """Choice zero, as one assistant line. The rest of the reply is meta."""
        choice = raw["choices"][0]
        said = choice["message"]
        calls, invalid = [], {}
        for asked in said.get("tool_calls") or []:
            args, junk = args_of(asked["function"])
            if junk is not None:
                invalid[asked["id"]] = junk
            calls.append(ToolCall(asked["id"], asked["function"]["name"], args))
        used = raw.get("usage") or {}
        meta = {"finish_reason": choice.get("finish_reason")}
        if used:
            meta["usage"] = {"input_tokens": used.get("prompt_tokens"),
                             "output_tokens": used.get("completion_tokens")}
        if invalid:
            meta["invalid_args"] = invalid
        return Message("assistant", said.get("content") or "", calls, meta=meta)
