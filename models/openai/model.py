"""OpenAI Chat Completions — the wire format the compatible vendors clone.

Two steps, like any Provider: build the body, then hand the reply back as
Parts — off one parse, because a POST reply is a stream of one chunk. This
file decides the body's own keys and how the chunks assemble the message a
POST would return; what crosses the wire is mapping.py's. The quirk worth
remembering is that a tool call's arguments travel as a JSON *string*, and on
the stream they arrive a fragment at a time.
"""

from core import Part, Profile, Provider, ProviderError, Run

from .mapping import meta, notebook, tool, tool_call

# what every chat model below can do; the rows differ only in their numbers
CHAT = frozenset({"image", "tools", "stream", "tool_stream", "json", "system"})


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
            "messages": notebook(run.messages),
            "max_completion_tokens": self.profile.max_output,
        }
        if run.agent.tools:
            body["tools"] = [tool(t) for t in self.tool_schemas(run)]
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
            yield tool_call(call, invalid)
        yield meta(seen, self.model, invalid)   # one meta: fold merges it shallowly
