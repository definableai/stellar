"""OpenAI Chat Completions — the wire format the compatible vendors clone.

Two steps, like any Provider: build the body, then hand the reply back as
Parts — off the event stream when the profile has one, off a single POST when
it does not. The quirk worth remembering is that a tool call's arguments
travel as a JSON *string*: dumped on the way out, parsed on the way back in,
and on the stream they arrive a fragment at a time.
"""

import json
from typing import cast

from core import (
    ContractError, Message, Part, Profile, Provider, ProviderError, Run, ToolCall,
)
from core.llm import args_of

# what every chat model below can do; the rows differ only in their numbers
CHAT = frozenset({"image", "tools", "stream", "tool_stream", "json", "system"})


def part(piece: Part) -> dict:
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


def line(message: Message) -> dict:
    """One notebook line as one message: the role, the words, the plumbing."""
    parts = cast(list[Part], message.content)   # always parts after construction
    text_only = all(p.type == "text" for p in parts)
    said = {
        "role": message.role,
        "content": message.text if text_only else [part(p) for p in parts],
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


class OpenAI(Provider):
    """The brain behind api.openai.com — or anything that copied its format.

    Point base_url somewhere else and the same class talks to the clone. A
    clone that cannot hold a stream open gets a profile without "stream" and
    takes the one-POST path instead; everything else is the same wire.
    """

    url, env = "https://api.openai.com/v1", "OPENAI_API_KEY"
    PROFILES = {                    # developers.openai.com/api/docs/models
        "gpt-5.6-sol": Profile("gpt-5.6-sol", 1_050_000, 128_000, CHAT),
        "gpt-5.6-terra": Profile("gpt-5.6-terra", 1_050_000, 128_000, CHAT),
        "gpt-5.6-luna": Profile("gpt-5.6-luna", 1_050_000, 128_000, CHAT),
        "gpt-5.5": Profile("gpt-5.5", 1_050_000, 128_000, CHAT),
    }

    def headers(self) -> dict:
        """The key, carried as a bearer token — the only header this wire wants."""
        return {"Authorization": f"Bearer {self.key}"}

    def encode(self, run: Run) -> dict:
        """This run's whole notebook and the whole toolbox, as one request body."""
        body = {
            "model": self.model,
            "messages": [line(m) for m in run.messages],
            "max_completion_tokens": self.profile.max_output,
        }
        if run.agent.tools:
            body["tools"] = [{"type": "function", "function": schema}
                             for schema in self.tool_schemas(run)]
        if "stream" in self.profile:
            body |= {"stream": True, "stream_options": {"include_usage": True}}
        return body | self.params            # a param of yours wins over all of it

    async def send(self, run: Run, body: dict):
        """The reply as Parts: chunk by chunk when the profile streams, else once."""
        if "stream" in self.profile:
            async for piece in self.streamed(run, body):
                yield piece
            return
        raw = await self.post("/chat/completions", body)
        choice = raw["choices"][0]
        said = choice["message"]
        if said.get("content"):
            yield Part("text", said["content"])
        invalid = {}
        for asked in said.get("tool_calls") or []:
            args, junk = args_of(asked["function"].get("arguments") or "")
            if junk is not None:
                invalid[asked["id"]] = junk
            yield Part("tool_call",
                       ToolCall(asked["id"], asked["function"]["name"], args))
        meta = {"stop_reason": choice.get("finish_reason")}
        used = raw.get("usage") or {}
        if used:
            meta["usage"] = {"input_tokens": used.get("prompt_tokens"),
                             "output_tokens": used.get("completion_tokens")}
        if invalid:
            meta["invalid_args"] = invalid
        yield Part("meta", meta)

    async def streamed(self, run: Run, body: dict):
        """The same reply off the socket: text now, tool calls when the turn ends.

        A call's arguments arrive in pieces, filed by index until the stream
        ends — nothing half-parsed leaves here, and the fragments ride the bus
        as tool.args when the profile allows it.
        """
        calls: dict[int, list] = {}          # index → [id, name, fragments]
        invalid: dict[str, str] = {}
        used: dict = {}
        stop: str | None = None
        async for chunk in self.sse("/chat/completions", body):
            if chunk.get("error"):
                raise ProviderError(None, str(chunk["error"])[:200])
            used = chunk.get("usage") or used
            if not chunk.get("choices"):     # the usage chunk carries none of them
                continue
            choice = chunk["choices"][0]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                yield Part("text", delta["content"])
            for asked in delta.get("tool_calls") or ():
                slot = calls.setdefault(asked.get("index", 0), [None, None, []])
                fn = asked.get("function") or {}
                slot[0] = asked.get("id") or slot[0]
                slot[1] = fn.get("name") or slot[1]
                fragment = fn.get("arguments") or ""
                slot[2].append(fragment)
                if fragment and "tool_stream" in self.profile:
                    run.emit("tool.args",
                             {"id": slot[0], "name": slot[1], "delta": fragment},
                             source=f"model:{self.model}")
            stop = choice.get("finish_reason") or stop
        for call_id, name, fragments in calls.values():   # the turn is over now
            args, junk = args_of("".join(fragments))
            if junk is not None:
                invalid[call_id] = junk
            yield Part("tool_call", ToolCall(call_id, name, args))
        meta: dict = {"usage": {"input_tokens": used.get("prompt_tokens"),
                                "output_tokens": used.get("completion_tokens")},
                      "stop_reason": stop}
        if invalid:
            meta["invalid_args"] = invalid
        yield Part("meta", meta)             # one meta: fold merges it shallowly
