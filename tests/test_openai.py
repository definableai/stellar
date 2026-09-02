"""The OpenAI adapter: canned replies, scripted streams, translation both ways.

Run: uv run python tests/test_openai.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from core import (  # noqa: E402
    Agent, ContractError, Event, Message, Part, Profile, ProviderError, Run,
    ToolCall, tool,
)
from core.conformance import check_model  # noqa: E402
from models.openai import OpenAI  # noqa: E402

ONCE = Profile("once", 8_000, 512, frozenset({"tools"}))          # no stream: POST
LIVE = Profile("live", 8_000, 512, frozenset({"tools", "stream", "tool_stream"}))
QUIET = Profile("quiet", 8_000, 512, frozenset({"tools", "stream"}))   # no fragments

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

# the same three exchanges, chunk by chunk, in the shapes the socket sends
SAYS = [
    {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
    {"choices": [{"index": 0, "delta": {"content": "o"}}]},
    {"choices": [{"index": 0, "delta": {"content": "k"}}]},
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 1,
                              "total_tokens": 12}},
]
CALLS = [
    {"choices": [{"index": 0, "delta": {"role": "assistant", "content": None,
                                        "tool_calls": [
        {"index": 0, "id": "call_abc123", "type": "function",
         "function": {"name": "echo", "arguments": ""}}]}}]},
    {"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "function": {"arguments": '{"te'}}]}}]},
    {"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "function": {"arguments": 'xt": '}}]}}]},
    {"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "function": {"arguments": '"hi"}'}}]}}]},
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    {"choices": [], "usage": {"prompt_tokens": 42, "completion_tokens": 17,
                              "total_tokens": 59}},
]
BOTH = [                                     # two calls at once, told apart by index
    {"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "id": "call_a", "type": "function",
         "function": {"name": "echo", "arguments": '{"text": "one"}'}},
        {"index": 1, "id": "call_b", "type": "function",
         "function": {"name": "echo", "arguments": '{"text": '}}]}}]},
    {"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 1, "function": {"arguments": '"two"}'}}]}}]},
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
]
BROKEN = [
    {"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "id": "call_bad", "type": "function",
         "function": {"name": "echo", "arguments": '{"text": '}}]}}]},
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
]
CUT = [                                      # the stream stops without saying why
    {"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "id": "call_cut", "type": "function",
         "function": {"name": "echo", "arguments": '{"text": "hi"}'}}]}}]},
]
BYE = [
    {"choices": [{"index": 0, "delta": {"content": "done"}}]},
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    {"choices": [], "usage": {"prompt_tokens": 60, "completion_tokens": 2,
                              "total_tokens": 62}},
]


@tool
def echo(text: str) -> str:
    """Repeat the text back."""
    return text


class Canned(OpenAI):
    """The same adapter, with the POST replaced by a list of replies."""

    def __init__(self, script) -> None:
        super().__init__("gpt-5.6-luna", profile=ONCE)   # no stream: one POST
        self.script, self.sent = list(script), []

    async def post(self, path, body) -> dict:
        self.sent.append(body)
        return self.script.pop(0)


class Streamed(OpenAI):
    """The same adapter, with the socket replaced by a script of chunk lists."""

    def __init__(self, *script, profile=LIVE) -> None:
        super().__init__("gpt-5.6-luna", profile=profile)
        self.script, self.sent = list(script), []

    async def sse(self, path, body):
        self.sent.append(body)
        for chunk in self.script.pop(0):
            yield chunk


def encoded(model, messages, tools=()) -> dict:
    """One request body, off a Run built by hand: encode needs no loop."""
    return model.encode(Run(Agent(model, tools), "rid", list(messages)))


def decoded(body: dict) -> Message:
    """One canned reply through the real send-and-fold path."""
    model = Canned([body])
    return asyncio.run(model.invoke(Run(Agent(model), "rid", [])))


def played(script, profile=LIVE) -> tuple[Message, list[Event]]:
    """One scripted stream through the real invoke: what folded, what rang."""
    model = Streamed(script, profile=profile)
    agent = Agent(model, [echo])
    heard: list[Event] = []
    agent.events.listen(heard.append)
    said = asyncio.run(model.invoke(Run(agent, "rid", [Message("user", "hi")])))
    return said, heard


def deltas(heard: list[Event]) -> list[Part]:
    """Every Part the run rang on model.delta, in order."""
    return [e.data for e in heard if e.name == "model.delta"]


def test_three_canned_replies_pass_the_check() -> None:
    model = Canned([PLAIN, ASKS, DONE])
    check_model(model)
    assert len(model.sent) == 3                  # plain, tool call, final


def test_the_body_carries_the_notebook_and_the_toolbox() -> None:
    body = encoded(OpenAI("gpt-5.6-luna", temperature=0),
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
    assert "tools" not in encoded(OpenAI("gpt-5.6-luna"), [])      # empty toolbox


def test_the_body_streams_when_the_profile_does() -> None:
    body = encoded(OpenAI("gpt-5.6-luna"), [Message("user", "hi")])
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["max_completion_tokens"] == 128_000               # off the profile
    quiet = encoded(OpenAI("gpt-5.6-luna", profile=ONCE, max_completion_tokens=9),
                    [Message("user", "hi")])
    assert "stream" not in quiet and "stream_options" not in quiet
    assert quiet["max_completion_tokens"] == 9                    # a param wins


def test_tool_calls_round_trip() -> None:
    asked = decoded(ASKS)
    assert asked.tool_calls == [ToolCall("call_abc123", "echo", {"text": "hi"})]

    answered = Message("tool", "hi", tool_call_id="call_abc123")
    said = encoded(OpenAI("gpt-5.6-luna"), [asked, answered])["messages"]
    assert said[0]["tool_calls"] == [{
        "id": "call_abc123", "type": "function",
        "function": {"name": "echo", "arguments": '{"text": "hi"}'},   # a string
    }]
    assert said[1] == {"role": "tool", "content": "hi",
                       "tool_call_id": "call_abc123"}


def test_usage_and_stop_reason_land_in_meta() -> None:
    said = decoded(PLAIN)
    assert said.text == "ok"
    assert said.meta["stop_reason"] == "stop"
    assert said.meta["usage"] == {"input_tokens": 11, "output_tokens": 1}
    assert decoded(ASKS).meta["stop_reason"] == "tool_calls"


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
    body = encoded(OpenAI("gpt-5.6-luna"), [Message("user", [
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
        encoded(OpenAI("gpt-5.6-luna"), [Message("user", [Part("sound", b"...")])])
    except ContractError as e:
        assert "'sound'" in str(e)
    else:
        raise AssertionError("an unknown part type must not go on the wire")


def test_the_models_own_past_off_another_wire_replays_without_it() -> None:
    thought = Part("thinking", {"type": "thinking", "thinking": "Two halves.",
                                "signature": "EqQBCgIYAhgCIkDrtM1kM+2v"})
    said = encoded(OpenAI("gpt-5.6-luna"), [
        Message("user", "hi"),
        Message("assistant", [thought, Part("text", "half")],
                [ToolCall("call_abc123", "echo", {"text": "hi"})]),
        Message("tool", "hi", tool_call_id="call_abc123"),
    ])["messages"]
    assert said[1] == {                          # the thinking block is dropped
        "role": "assistant", "content": "half",
        "tool_calls": [{"id": "call_abc123", "type": "function",
                        "function": {"name": "echo",
                                     "arguments": '{"text": "hi"}'}}],
    }
    assert said[2]["role"] == "tool"             # and its result still answers it
    try:
        encoded(OpenAI("gpt-5.6-luna"), [Message("user", [thought])])
    except ContractError as e:
        assert "'thinking'" in str(e)            # yours, though: a promise broken
    else:
        raise AssertionError("a part you authored must not be dropped quietly")


def test_text_arrives_one_chunk_at_a_time() -> None:
    said, heard = played(SAYS)
    assert [p.data for p in deltas(heard) if p.type == "text"] == ["o", "k"]
    assert said.text == "ok"                     # and folds back into one reply


def test_a_tool_call_is_assembled_from_its_fragments() -> None:
    said, heard = played(CALLS)
    assert said.tool_calls == [ToolCall("call_abc123", "echo", {"text": "hi"})]
    assert [p.type for p in deltas(heard)] == ["tool_call", "meta"]  # nothing early

    both, _ = played(BOTH)
    assert both.tool_calls == [ToolCall("call_a", "echo", {"text": "one"}),
                              ToolCall("call_b", "echo", {"text": "two"})]

    cut, _ = played(CUT)                         # no finish_reason, no lost call
    assert cut.tool_calls == [ToolCall("call_cut", "echo", {"text": "hi"})]
    assert cut.meta["stop_reason"] is None


def test_the_stream_ends_with_one_meta() -> None:
    said, heard = played(CALLS)
    assert said.meta["usage"] == {"input_tokens": 42, "output_tokens": 17}
    assert said.meta["stop_reason"] == "tool_calls"
    assert sum(p.type == "meta" for p in deltas(heard)) == 1   # fold merges shallow

    broken, _ = played(BROKEN)
    assert broken.tool_calls[0].args == {}       # the loop can still call it
    assert broken.meta["invalid_args"] == {"call_bad": '{"text": '}
    assert "invalid_args" not in said.meta


def test_meta_says_which_model_answered() -> None:
    assert decoded(PLAIN).meta["model"] == "gpt-5.6-luna"      # the body's own word
    said, _ = played(SAYS)
    assert said.meta["model"] == "gpt-5.6-luna"                # none said: the id
    dated, _ = played([dict(c, model="gpt-5.6-luna-2026-03-01") if i == 1 else c
                       for i, c in enumerate(SAYS)])
    assert dated.meta["model"] == "gpt-5.6-luna-2026-03-01"    # the wire's word wins


def test_argument_fragments_ride_the_bus_when_the_profile_says_so() -> None:
    _, heard = played(CALLS)
    args = [e for e in heard if e.name == "tool.args"]
    assert [e.data["delta"] for e in args] == ['{"te', 'xt": ', '"hi"}']
    assert {(e.data["id"], e.data["name"], e.source) for e in args} == {
        ("call_abc123", "echo", "model:gpt-5.6-luna")}
    _, quiet = played(CALLS, profile=QUIET)
    assert not [e for e in quiet if e.name == "tool.args"]


def test_three_scripted_streams_pass_the_check() -> None:
    model = Streamed(SAYS, CALLS, BYE)
    check_model(model)
    assert [body["stream"] for body in model.sent] == [True] * 3


def test_an_error_chunk_ends_the_stream() -> None:
    try:
        played([{"error": {"message": "boom", "type": "server_error"}}])
    except ProviderError as ex:
        assert ex.status is None and "boom" in str(ex)
    else:
        raise AssertionError("an error chunk must not fold into a reply")


if __name__ == "__main__":
    for test in (
        test_three_canned_replies_pass_the_check,
        test_the_body_carries_the_notebook_and_the_toolbox,
        test_the_body_streams_when_the_profile_does,
        test_tool_calls_round_trip,
        test_usage_and_stop_reason_land_in_meta,
        test_unparseable_arguments_keep_the_raw_string,
        test_parts_open_into_content_parts,
        test_a_part_type_the_wire_never_heard_of_says_so,
        test_the_models_own_past_off_another_wire_replays_without_it,
        test_text_arrives_one_chunk_at_a_time,
        test_a_tool_call_is_assembled_from_its_fragments,
        test_the_stream_ends_with_one_meta,
        test_meta_says_which_model_answered,
        test_argument_fragments_ride_the_bus_when_the_profile_says_so,
        test_three_scripted_streams_pass_the_check,
        test_an_error_chunk_ends_the_stream,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_openai: all ok")
