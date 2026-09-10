"""OpenAI Responses API as a stellar Provider: reasoning with tools on gpt-5.6-luna.

Chained: each call sends only what the server has not seen (previous_response_id),
so reasoning carries across turns and the input stays small. Two pockets in
run.extra: "responses.last" (the id to chain from) and "responses.sent" (how
much of the notebook the server already holds). Bench-only for now.
"""

from __future__ import annotations

import json
from typing import cast

from core import Message, Part, Provider, ProviderError, Run, ToolCall
from models.openai import OpenAI
from models.openai.mapping import args_of


def _image(p: Part) -> dict:
    d = p.data
    url = d["url"] if "url" in d else f"data:{d['media_type']};base64,{d['data']}"
    return {"type": "input_image", "image_url": url}


class Responses(Provider):
    """POST /v1/responses. Same key, same profiles as the Chat Completions class."""

    url, env = OpenAI.url, OpenAI.env
    PROFILES = OpenAI.PROFILES

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.key}"}

    def items(self, m: Message, chained: bool) -> list[dict]:
        """One notebook line as Responses input items. Pictures a tool returned ride in a user line."""
        parts = cast(list[Part], m.content)
        if m.role == "system":
            return [{"type": "message", "role": "system", "content": m.text}]
        if m.role == "user":
            if all(p.type == "text" for p in parts):
                return [{"type": "message", "role": "user", "content": m.text}]
            return [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": p.data} if p.type == "text" else _image(p) for p in parts]}]
        if m.role == "assistant":
            if chained:                         # the server already holds its own turn
                return []
            out: list[dict] = []
            if m.text:
                out.append({"type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": m.text}]})
            out += [{"type": "function_call", "call_id": c.id, "name": c.name,
                     "arguments": json.dumps(c.args)} for c in m.tool_calls]
            return out
        out = [{"type": "function_call_output", "call_id": m.tool_call_id, "output": m.text or "(no output)"}]
        images = [p for p in parts if p.type == "image"]
        if images:
            out.append({"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": f"[image returned by tool call {m.tool_call_id}]"},
                *map(_image, images)]})
        return out

    def encode(self, run: Run) -> dict:
        msgs = run.messages
        prev = run.extra.get("responses.last")
        sent = run.extra.get("responses.sent", 0) if prev else 0
        body: dict = {"model": self.model, "store": True, "max_output_tokens": self.profile.max_output}
        if msgs and msgs[0].role == "system":
            body["instructions"] = msgs[0].text
        items: list[dict] = []
        for i in range(max(sent, 1 if "instructions" in body else 0), len(msgs)):
            items += self.items(msgs[i], chained=bool(prev))
        body["input"] = items
        if prev:
            body["previous_response_id"] = prev
        if run.agent.tools:
            body["tools"] = [{"type": "function", **s} for s in self.tool_schemas(run)]
        return body | self.params

    async def send(self, run: Run, body: dict):
        reply = await self.post("/responses", body)
        if reply.get("error"):
            raise ProviderError(None, str(reply["error"])[:200])
        invalid: dict = {}
        for item in reply.get("output") or []:
            kind = item.get("type")
            if kind == "message":
                for c in item.get("content") or []:
                    if c.get("type") == "output_text" and c.get("text"):
                        yield Part("text", c["text"])
            elif kind == "function_call":
                args, junk = args_of(item.get("arguments") or "")
                if junk is not None:
                    invalid[item["call_id"]] = junk
                yield Part("tool_call", ToolCall(item["call_id"], item["name"], args))
        u = reply.get("usage") or {}
        said = {"usage": {"input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
                          "cached_tokens": (u.get("input_tokens_details") or {}).get("cached_tokens", 0),
                          "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)},
                "model": reply.get("model") or self.model, "stop_reason": reply.get("status"),
                "response_id": reply.get("id")}
        if invalid:
            said["invalid_args"] = invalid
        yield Part("meta", said)
        run.extra["responses.last"] = reply.get("id")
        run.extra["responses.sent"] = len(run.messages)
