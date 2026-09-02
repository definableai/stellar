"""The Messages API, in two steps: build the request, send it back as Parts.

encode is pure. Core never opens a Part and never sees the wire — an adapter
does both, and here the wire is the Messages API. send has two ways home: the
event stream when the profile has one, else one POST and the same parse. Off
the stream every block but text is held open at its index until
content_block_stop, so a thinking block lands whole — the shape block()
replays next turn.
"""

import json
from typing import cast

from core import Message, Part, Profile, Provider, ProviderError, Run, ToolCall

VERSION = "2023-06-01"

# Every current Claude does all nine — pictures, PDFs, tools, thinking, the lot;
# redacted_thinking is a Part type of its own, and one it must read back unchanged
DOES = frozenset({"image", "document", "tools", "stream", "tool_stream",
                  "thinking", "redacted_thinking", "json", "system"})


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


def args_of(raw: str) -> tuple[dict, str | None]:
    """The fragments as one dict — or an empty one and the string back, unread."""
    try:
        args = json.loads(raw or "{}")
        if not isinstance(args, dict):
            raise ValueError("tool arguments must be a JSON object")
    except ValueError:                  # a broken parse is a ValueError too
        return {}, raw
    return args, None


class Anthropic(Provider):
    """Claude over HTTP. The key comes from you or from ANTHROPIC_API_KEY."""

    url, env = "https://api.anthropic.com", "ANTHROPIC_API_KEY"
    # context and max output off the models overview; the words off the feature
    # pages — vision, PDF support, tool use, thinking, structured outputs
    PROFILES = {
        "claude-fable-5-1": Profile("claude-fable-5-1", 1_000_000, 128_000, DOES),
        "claude-opus-5": Profile("claude-opus-5", 1_000_000, 128_000, DOES),
        "claude-sonnet-5": Profile("claude-sonnet-5", 1_000_000, 128_000, DOES),
        "claude-haiku-4-5": Profile("claude-haiku-4-5", 200_000, 64_000, DOES),
    }

    def headers(self) -> dict:
        """The key, and the wire version this file was written against."""
        return {"x-api-key": self.key, "anthropic-version": VERSION}

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
        body: dict = {"model": self.model, "max_tokens": self.profile.max_output,
                      "messages": turns}
        if "stream" in self.profile:
            body["stream"] = True
        body.update(self.params)        # yours last: max_tokens, stream, whatever
        if system:
            body["system"] = "\n\n".join(system)
        if run.agent.tools:
            body["tools"] = [
                {"name": t["name"], "description": t["description"],
                 "input_schema": t["parameters"]}
                for t in self.tool_schemas(run)
            ]
        return body

    async def send(self, run: Run, body: dict):
        """The reply as Parts — one POST, or the event stream, block by block."""
        if "stream" not in self.profile:
            raw = await self.post("/v1/messages", body)
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
            return

        held: dict[int, dict] = {}      # the blocks still open, by index
        fragments: dict[int, str] = {}  # a tool call's arguments, still in pieces
        invalid: dict[str, str] = {}
        usage: dict[str, int | None] = {"input_tokens": None, "output_tokens": None}
        stop: str | None = None
        async for piece in self.sse("/v1/messages", body):
            kind, index = piece.get("type"), piece.get("index")
            if kind == "message_start":
                used = piece["message"].get("usage") or {}
                usage["input_tokens"] = used.get("input_tokens")
            elif kind == "content_block_start":
                if piece["content_block"]["type"] != "text":
                    held[index] = dict(piece["content_block"])   # text holds nothing
            elif kind == "content_block_delta":
                delta = piece["delta"]
                if delta["type"] == "text_delta":
                    yield Part("text", delta["text"])
                elif delta["type"] == "input_json_delta":
                    more = delta["partial_json"]
                    fragments[index] = fragments.get(index, "") + more
                    if "tool_stream" in self.profile:
                        ask = held[index]
                        run.emit("tool.args", {"id": ask["id"], "name": ask["name"],
                                               "delta": more},
                                 source=f"model:{self.model}")
                elif index in held:     # thinking, signature, whatever comes next
                    key = delta["type"].removesuffix("_delta")
                    if key in delta:    # a *_delta is named for the field it grows
                        held[index][key] = held[index].get(key, "") + delta[key]
            elif kind == "content_block_stop":
                slot = held.pop(index, None)
                if slot is None:
                    continue                        # a text block: already yielded
                if slot["type"] != "tool_use":
                    yield Part(slot["type"], slot)  # whole, exactly as it came
                    continue
                args, junk = args_of(fragments.pop(index, ""))
                if junk is not None:
                    invalid[slot["id"]] = junk
                yield Part("tool_call", ToolCall(slot["id"], slot["name"], args))
            elif kind == "message_delta":
                stop = piece["delta"].get("stop_reason") or stop
                used = piece.get("usage") or {}     # cumulative: the last one wins
                usage["output_tokens"] = used.get(
                    "output_tokens", usage["output_tokens"])
            elif kind == "error":
                raise ProviderError(None, str(piece.get("error")))
        meta: dict = {"usage": usage, "stop_reason": stop}
        if invalid:
            meta["invalid_args"] = invalid
        yield Part("meta", meta)        # once, and merged: fold's merge is shallow
