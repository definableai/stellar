"""The Messages API in one class: the request's own keys, the reply as Parts.

encode is pure. Core never opens a Part and never sees the wire — an adapter
does both. This file decides the body's own keys and how the event stream
assembles the message a POST would return; what crosses the wire is
mapping.py's. send has two ways home: one POST, or that assembly. Off the
stream every block but text is held open at its index until
content_block_stop, so a thinking block lands whole — the shape passthrough
replays next turn.
"""

from core import Part, Profile, Provider, ProviderError, Run
from core.llm import args_of

from .mapping import IN, meta, notebook, tool, whole

VERSION = "2023-06-01"

# Every current Claude does all eight — pictures, PDFs, tools, thinking, the lot
DOES = frozenset({"image", "document", "tools", "stream", "tool_stream",
                  "thinking", "json", "system"})


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
        system, turns = notebook(run.messages)
        body: dict = {"model": self.model, "max_tokens": self.profile.max_output,
                      "messages": turns}
        if "stream" in self.profile:
            body["stream"] = True
        body.update(self.params)        # yours last: max_tokens, stream, whatever
        if system:
            body["system"] = system
        if run.agent.tools:
            body["tools"] = [tool(t) for t in self.tool_schemas(run)]
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
                yield IN.get(b["type"], whole)(b)
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
                    yield IN.get(done["type"], whole)(done)
                elif kind == "message_delta":     # cumulative: the last word wins
                    raw["stop_reason"] = (ev["delta"].get("stop_reason")
                                          or raw.get("stop_reason"))
                    raw["usage"] = (raw.get("usage") or {}) | (ev.get("usage") or {})
                elif kind == "error":
                    raise ProviderError(None, str(ev.get("error")))
        yield meta(raw, self.model, invalid)    # once, and merged: fold's is shallow
