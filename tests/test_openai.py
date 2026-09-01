"""The OpenAI adapter: three canned replies, and the translation both ways.

Run: uv run python tests/test_openai.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from core import (  # noqa: E402
    Agent, ContractError, Message, Part, Run, ToolCall, check_model, tool,
)
from models.openai import OpenAI  # noqa: E402

PLAIN = {
    "id": "chatcmpl-1", "object": "chat.completion", "created": 1735689600,
    "model": "gpt-5.6-luna",
    "choices": [{"index": 0, "logprobs": None, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "ok", "refusal": None}}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 1, "total_tokens": 12},
}
ASKS = {
    "id": "chatcmpl-2", "object": "chat.completion", "created": 1735689601,
    "model": "gpt-5.6-luna",
    "choices": [{"index": 0, "logprobs": None, "finish_reason": "tool_calls",
                 "message": {"role": "assistant", "content": None, "tool_calls": [
                     {"id": "call_abc123", "type": "function",
                      "function": {"name": "echo",
                                   "arguments": '{"text": "hi"}'}}]}}],
    "usage": {"prompt_tokens": 42, "completion_tokens": 17, "total_tokens": 59},
}
DONE = {
    "id": "chatcmpl-3", "object": "chat.completion", "created": 1735689602,
    "model": "gpt-5.6-luna",
    "choices": [{"index": 0, "logprobs": None, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "done",
                             "refusal": None}}],
    "usage": {"prompt_tokens": 60, "completion_tokens": 2, "total_tokens": 62},
}


@tool
def echo(text: str) -> str:
    """Repeat the text back."""
    return text


class Canned(OpenAI):
    """The same adapter, with the POST replaced by a list of replies."""

    def __init__(self, script) -> None:
        super().__init__()
        self.script, self.sent = list(script), []

    async def post(self, body) -> dict:
        self.sent.append(body)
        return self.script.pop(0)


def encoded(model, messages, tools=()) -> dict:
    """One request body, off a Run built by hand: encode needs no loop."""
    return model.encode(Run(Agent(model, tools), "rid", list(messages)))


def decoded(body: dict) -> Message:
    """One canned reply through the real send-and-fold path."""
    model = Canned([body])
    return asyncio.run(model.invoke(Run(Agent(model), "rid", [])))


def test_three_canned_replies_pass_the_check() -> None:
    model = Canned([PLAIN, ASKS, DONE])
    check_model(model)
    assert len(model.sent) == 3                  # plain, tool call, final


def test_the_body_carries_the_notebook_and_the_toolbox() -> None:
    body = encoded(OpenAI(temperature=0),
                   [Message("system", "be brief"), Message("user", "hi")],
                   [echo])
    assert body["model"] == "gpt-5.6-luna"
    assert body["temperature"] == 0              # **params ride along
    assert body["messages"] == [                 # system stays a message
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert body["tools"] == [{"type": "function", "function": {
        "name": "echo", "description": "Repeat the text back.",
        "parameters": echo.parameters}}]
    assert "tools" not in encoded(OpenAI(), [])                   # empty toolbox


def test_tool_calls_round_trip() -> None:
    asked = decoded(ASKS)
    assert asked.tool_calls == [ToolCall("call_abc123", "echo", {"text": "hi"})]

    answered = Message("tool", "hi", tool_call_id="call_abc123")
    said = encoded(OpenAI(), [asked, answered])["messages"]
    assert said[0]["tool_calls"] == [{
        "id": "call_abc123", "type": "function",
        "function": {"name": "echo", "arguments": '{"text": "hi"}'},   # a string
    }]
    assert said[1] == {"role": "tool", "content": "hi",
                       "tool_call_id": "call_abc123"}


def test_usage_and_finish_reason_land_in_meta() -> None:
    said = decoded(PLAIN)
    assert said.text == "ok"
    assert said.meta["finish_reason"] == "stop"
    assert said.meta["usage"] == {"input_tokens": 11, "output_tokens": 1}
    assert decoded(ASKS).meta["finish_reason"] == "tool_calls"


def test_unparseable_arguments_keep_the_raw_string() -> None:
    for junk in ('{"text": ', '"just a string"'):
        broken = {"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "tool_calls": [
                {"id": "call_bad", "type": "function",
                 "function": {"name": "echo", "arguments": junk}}]}}]}
        said = decoded(broken)
        assert said.tool_calls[0].args == {}          # the loop can still call it
        assert said.meta["invalid_args"] == {"call_bad": junk}
    assert "invalid_args" not in decoded(ASKS).meta


def test_parts_open_into_content_parts() -> None:
    body = encoded(OpenAI(), [Message("user", [
        Part("text", "what is this?"),
        Part("image", {"url": "https://example.com/cat.png"}),
        Part("image", {"media_type": "image/png", "data": "aGk="}),
    ])])
    assert body["messages"][0]["content"] == [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url",
         "image_url": {"url": "https://example.com/cat.png"}},
        {"type": "image_url",
         "image_url": {"url": "data:image/png;base64,aGk="}},
    ]


def test_a_part_type_the_wire_never_heard_of_says_so() -> None:
    try:
        encoded(OpenAI(), [Message("user", [Part("sound", b"...")])])
    except ContractError as e:
        assert "'sound'" in str(e)
    else:
        raise AssertionError("an unknown part type must not go on the wire")


if __name__ == "__main__":
    for test in (
        test_three_canned_replies_pass_the_check,
        test_the_body_carries_the_notebook_and_the_toolbox,
        test_tool_calls_round_trip,
        test_usage_and_finish_reason_land_in_meta,
        test_unparseable_arguments_keep_the_raw_string,
        test_parts_open_into_content_parts,
        test_a_part_type_the_wire_never_heard_of_says_so,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_openai: all ok")
