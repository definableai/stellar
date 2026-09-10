"""OpenAI's Responses API — the wire that hands the reasoning back.

Two steps, like any Provider: build the body, then hand the reply back as
Parts. What differs from Chat Completions is the shape of a turn — a list of
items, not a message — and what is in it: the model's own reasoning is an item
of its own, and replaying it keeps a thought alive across a tool call. Nothing
is kept on their side (store: false), so it travels encrypted in the notebook.
"""

from core import Part, Profile, Provider, ProviderError, Run

from .mapping import meta, notebook, parts, tool

# the chat set plus the one this wire adds: reasoning, kept between the steps
THINKS = frozenset({"image", "tools", "stream", "tool_stream", "json", "system",
                    "thinking"})


class Responses(Provider):
    """The brain behind /v1/responses — the same models, thinking out loud.

    Same key and base_url as Chat Completions; the reasoning a tool call ends
    on that wire survives here.
    """

    url, env = "https://api.openai.com/v1", "OPENAI_API_KEY"
    PROFILES = {                    # developers.openai.com/api/docs/models
        "gpt-5.6-sol": Profile("gpt-5.6-sol", 1_050_000, 128_000, THINKS),
        "gpt-5.6-terra": Profile("gpt-5.6-terra", 1_050_000, 128_000, THINKS),
        "gpt-5.6-luna": Profile("gpt-5.6-luna", 1_050_000, 128_000, THINKS),
        "gpt-5.5": Profile("gpt-5.5", 1_050_000, 128_000, THINKS),
    }

    def headers(self) -> dict:
        """The key, carried as a bearer token — the only header this wire wants."""
        return {"Authorization": f"Bearer {self.key}"}

    def encode(self, run: Run) -> dict:
        """The notebook and the toolbox as one body; system lines are the
        instructions, one string above the input, not items.
        """
        instructions, items = notebook(run.messages)
        body = {
            "model": self.model,
            "input": items,
            "max_output_tokens": self.profile.max_output,
            "store": False,                             # nothing kept on their side
            "include": ["reasoning.encrypted_content"],  # … so it comes back to us
        }
        if instructions:
            body["instructions"] = instructions
        if run.agent.tools:
            body["tools"] = [tool(t) for t in self.tool_schemas(run)]
        if "stream" in self.profile:
            body["stream"] = True
        return body | self.params            # a param of yours wins over all of it

    async def send(self, run: Run, body: dict):
        """The reply as Parts: every output item, in the order it arrived.

        The POST path walks response["output"]; the stream takes the same items
        off output_item.done — the one event carrying a finished item, and the
        only place a reasoning item's encrypted_content is whole. The words
        already went by as deltas, so that item alone is skipped; argument
        fragments ride the bus as tool.args. One meta last.
        """
        path, invalid, seen = "/responses", {}, {}
        if "stream" not in self.profile:
            seen = await self.post(path, body)
            for item in seen.get("output") or ():
                for part in parts(item, invalid):
                    yield part
        else:
            calls: dict[str, dict] = {}     # item id -> the call it opened
            async for event in self.sse(path, body):
                name = event.get("type")
                if name == "error":
                    raise ProviderError(None, str(event.get("message"))[:200])
                seen = event.get("response") or seen    # created … completed: last wins
                if name == "response.output_text.delta":
                    yield Part("text", event["delta"])
                elif name == "response.output_item.added":
                    calls[event["item"]["id"]] = event["item"]
                elif (name == "response.function_call_arguments.delta"
                      and "tool_stream" in self.profile):
                    call = calls.get(event["item_id"]) or {}
                    run.emit("tool.args",
                             {"id": call.get("call_id"), "name": call.get("name"),
                              "delta": event["delta"]},
                             source=f"model:{self.model}")
                elif (name == "response.output_item.done"
                      and event["item"]["type"] != "message"):
                    for part in parts(event["item"], invalid):
                        yield part
        yield meta(seen, self.model, invalid)   # one meta: fold merges it shallowly
