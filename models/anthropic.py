"""The Messages API, in two steps: build the request, send it back as Parts.

encode is pure. Core never opens a Part and never sees the wire — an adapter
does both, and here the wire is the Messages API. send has two ways home: one
POST, or the event stream assembling the message that POST would have said.
Off the stream every block but text is held open at its index until
content_block_stop, so a thinking block lands whole — the shape block()
replays next turn.

core                  Messages API
Part("text")          {"type": "text", "text"}       streams as deltas
Part("image")         {"type": "image", "source": url | base64}
Part("document")      {"type": "document", "source": …}
Part(other)           {"type": other, **data}        thinking etc., replayed as it came
ToolCall              {"type": "tool_use", "id", "name", "input"}
Message("tool")       a user turn holding {"type": "tool_result",
                      "tool_use_id", "content"}
Message("system")     top-level "system"
meta.usage            usage.input_tokens / output_tokens
meta.model            model
meta.stop_reason      stop_reason
"""

from typing import cast

from core import Message, Part, Profile, Provider, ProviderError, Run, ToolCall
from core.llm import args_of

VERSION = "2023-06-01"

# Every current Claude does all eight — pictures, PDFs, tools, thinking, the lot
DOES = frozenset({"image", "document", "tools", "stream", "tool_stream",
                  "thinking", "json", "system"})


# ---- out: core -> wire.  block() makes a block, blocks() a whole turn ----


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


# ---- in: wire -> core.  part() makes a Part, meta() makes the meta Part ----


def part(block: dict) -> Part:
    """One content block, as the one Part core folds — whole, off either path."""
    if block["type"] == "text":
        return Part("text", block["text"])
    if block["type"] == "tool_use":
        return Part("tool_call",
                    ToolCall(block["id"], block["name"], block.get("input") or {}))
    return Part(block["type"], block)   # thinking, and whatever comes next


def meta(raw: dict, asked: str, invalid: dict) -> Part:
    """The one meta Part: what the turn cost, who answered, why it stopped."""
    used = raw.get("usage") or {}
    said = {"usage": {k: used.get(k) for k in ("input_tokens", "output_tokens")},
            "model": raw.get("model") or asked,
            "stop_reason": raw.get("stop_reason")}
    if invalid:
        said["invalid_args"] = invalid
    return Part("meta", said)


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
        """The reply as Parts: one POST, or the stream assembling what it would say.

        Off the stream, text yields as it comes and every other block is held
        open at its index until content_block_stop, so a thinking block lands
        whole. A tool call's input arrives as partial_json on its block and is
        parsed once, at the stop. One meta last, on either path.
        """
        invalid: dict[str, str] = {}
        if "stream" not in self.profile:
            raw = await self.post("/v1/messages", body)
            for b in raw.get("content", []):
                yield part(b)
        else:
            raw, held = {}, {}      # the message so far; blocks still open, by index
            async for ev in self.sse("/v1/messages", body):
                kind, i = ev.get("type"), ev.get("index")
                if kind == "message_start":
                    raw = dict(ev["message"])           # model, usage, stop_reason
                elif kind == "content_block_start":
                    if ev["content_block"]["type"] != "text":
                        held[i] = dict(ev["content_block"])   # text streams, never held
                elif kind == "content_block_delta":
                    delta = ev["delta"]
                    if delta["type"] == "text_delta":
                        yield Part("text", delta["text"])
                    elif i in held:                 # thinking, signature, partial_json
                        # a string delta grows the field it names; a dict is not ours
                        for field, more in delta.items():
                            if field != "type" and isinstance(more, str):
                                held[i][field] = held[i].get(field, "") + more
                        if "partial_json" in delta and "tool_stream" in self.profile:
                            run.emit("tool.args",
                                     {"id": held[i]["id"], "name": held[i]["name"],
                                      "delta": delta["partial_json"]},
                                     source=f"model:{self.model}")
                elif kind == "content_block_stop" and i in held:   # text: already out
                    done = held.pop(i)
                    if done["type"] == "tool_use":    # its arguments, done arriving
                        done["input"], junk = args_of(done.pop("partial_json", ""))
                        if junk is not None:
                            invalid[done["id"]] = junk
                    yield part(done)
                elif kind == "message_delta":     # cumulative: the last word wins
                    raw["stop_reason"] = (ev["delta"].get("stop_reason")
                                          or raw.get("stop_reason"))
                    raw["usage"] = (raw.get("usage") or {}) | (ev.get("usage") or {})
                elif kind == "error":
                    raise ProviderError(None, str(ev.get("error")))
        yield meta(raw, self.model, invalid)    # once, and merged: fold's is shallow
