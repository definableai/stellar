"""OpenAI Chat Completions — the wire format the compatible vendors clone.

Two steps, like any Provider: build the body, then hand the reply back as
Parts — off one parse, because a POST reply is a stream of one chunk. The
quirk worth remembering is that a tool call's arguments travel as a JSON
*string*: dumped on the way out, parsed on the way back in, and on the stream
they arrive a fragment at a time.

core                  Chat Completions
Part("text")          {"type": "text", "text"}
Part("image")         {"type": "image_url", "image_url": {"url": url | data-url}}
Part(other)           a user turn: ContractError, the wire cannot say it
                      an assistant turn: dropped — its own past, off another wire
ToolCall              message.tool_calls[{"id", "type": "function",
                      "function": {"name", "arguments": JSON string}}]
Message("tool")       {"role": "tool", "tool_call_id", "content"}
Message("system")     as it is
meta.usage            usage.prompt_tokens / completion_tokens
meta.model            model
meta.stop_reason      choices[0].finish_reason
"""

import json
from typing import cast

from core import (
    ContractError, Message, Part, Profile, Provider, ProviderError, Run, ToolCall,
)
from core.llm import args_of

# what every chat model below can do; the rows differ only in their numbers
CHAT = frozenset({"image", "tools", "stream", "tool_stream", "json", "system"})
SAID = frozenset({"text", "image"})     # the part types this wire has a shape for


# ---- out: core -> wire.  block() makes a block, line() makes a message ----


def block(part: Part) -> dict:
    """One Part as one content part. Text stays text; an image becomes a URL."""
    if part.type == "text":
        return {"type": "text", "text": part.data}
    if part.type == "image":            # {"url": …} or {"media_type": …, "data": …}
        d = part.data
        url = d["url"] if "url" in d else f"data:{d['media_type']};base64,{d['data']}"
        return {"type": "image_url", "image_url": {"url": url}}
    raise ContractError(
        f"Chat Completions has no part type {part.type!r}; say 'text' or 'image'"
    )


def line(message: Message) -> dict:
    """One notebook line as one message: the role, the words, the plumbing."""
    parts = cast(list[Part], message.content)   # always parts after construction
    if message.role == "assistant":
        # its own past may be off another wire — a thinking block from a Claude
        # turn — so drop what this one cannot say; a part you authored still
        # raises in block(), because a profile word promised it
        parts = [p for p in parts if p.type in SAID]
    text_only = all(p.type == "text" for p in parts)
    said = {
        "role": message.role,
        "content": message.text if text_only else [block(p) for p in parts],
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


# ---- in: wire -> core.  part() makes a Part, meta() makes the meta Part ----


def part(call: dict, invalid: dict) -> Part:
    """One tool call, in the POST shape both paths end up holding.

    Arguments that will not parse are filed in invalid under the call's id,
    and the ToolCall goes out with none — the loop answers the model itself.
    """
    args, junk = args_of(call["function"].get("arguments") or "")
    if junk is not None:
        invalid[call["id"]] = junk
    return Part("tool_call", ToolCall(call["id"], call["function"]["name"], args))


def meta(seen: dict, asked: str, invalid: dict) -> Part:
    """The one meta Part: what the turn cost, who answered, why it stopped."""
    used = seen.get("usage") or {}
    said = {"usage": {"input_tokens": used.get("prompt_tokens"),
                      "output_tokens": used.get("completion_tokens")},
            "model": seen.get("model") or asked,
            "stop_reason": seen.get("finish_reason")}
    if invalid:
        said["invalid_args"] = invalid
    return Part("meta", said)


async def once(reply):
    """One POST reply as a stream of one chunk — so both paths share a parse."""
    yield await reply


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
        """The reply as Parts, off one parse: a POST reply is a stream of one chunk.

        Text yields as it comes. A tool call's arguments arrive a fragment at a
        time, grown in place by index until the turn ends — nothing half-parsed
        leaves here, and the fragments ride the bus as tool.args when the
        profile allows it. One meta last.
        """
        path = "/chat/completions"
        chunks = (self.sse(path, body) if "stream" in self.profile
                  else once(self.post(path, body)))
        calls: dict[int, dict] = {}     # index -> the call, POST-shaped, still growing
        seen: dict = {}                 # model, usage, finish_reason: the last word
        async for chunk in chunks:
            if chunk.get("error"):
                raise ProviderError(None, str(chunk["error"])[:200])
            seen |= {k: chunk[k] for k in ("model", "usage") if chunk.get(k)}
            if not chunk.get("choices"):    # the usage chunk carries none of them
                continue
            choice = chunk["choices"][0]
            # a piece off the stream, or the whole message off the POST
            said = choice.get("delta") or choice.get("message") or {}
            if said.get("content"):
                yield Part("text", said["content"])
            # a POST's tool calls carry no index; their position is their index
            for n, asked in enumerate(said.get("tool_calls") or ()):
                call = calls.setdefault(asked.get("index", n), {
                    "id": None, "function": {"name": None, "arguments": ""}})
                fn = asked.get("function") or {}
                call["id"] = asked.get("id") or call["id"]
                call["function"]["name"] = fn.get("name") or call["function"]["name"]
                call["function"]["arguments"] += fn.get("arguments") or ""
                if fn.get("arguments") and "tool_stream" in self.profile:
                    run.emit("tool.args",
                             {"id": call["id"], "name": call["function"]["name"],
                              "delta": fn["arguments"]},
                             source=f"model:{self.model}")
            seen["finish_reason"] = (choice.get("finish_reason")
                                     or seen.get("finish_reason"))
        invalid: dict = {}
        for call in calls.values():         # the turn is over now
            yield part(call, invalid)
        yield meta(seen, self.model, invalid)   # one meta: fold merges it shallowly
