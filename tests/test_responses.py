"""The Responses adapter: canned replies, scripted streams, reasoning kept whole.

Run: uv run python tests/test_responses.py
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
from models.responses import Responses  # noqa: E402
from models.responses import mapping  # noqa: E402

ONCE = Profile("once", 8_000, 512, frozenset({"tools"}))          # no stream: POST
LIVE = Profile("live", 8_000, 512, frozenset({"tools", "stream", "tool_stream"}))
QUIET = Profile("quiet", 8_000, 512, frozenset({"tools", "stream"}))   # no fragments

THINKS = {"id": "rs_1", "type": "reasoning", "summary": [],
          "encrypted_content": "gAAAAABo..."}
CALL = {"id": "fc_1", "type": "function_call", "call_id": "call_abc123",
        "name": "echo", "arguments": '{"text": "hi"}', "status": "completed"}


def reply(number: int, output: list[dict], **rest) -> dict:
    """One response object, the shape a POST returns."""
    return {"id": f"resp_{number}", "object": "response", "model": "gpt-5.6-luna",
            "status": "completed", "incomplete_details": None, "output": output,
            "usage": {"input_tokens": 11, "output_tokens": 1, "total_tokens": 12},
            **rest}


def message(said: str) -> dict:
    return {"id": "msg_1", "type": "message", "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": said, "annotations": []}]}


PLAIN = reply(1, [message("ok")])
ASKS = reply(2, [THINKS, CALL])
DONE = reply(3, [message("done")])

# the same three exchanges, event by event, in the shapes the socket sends
SAYS = [
    {"type": "response.created", "response": dict(PLAIN, status="in_progress",
                                                  output=[], usage=None)},
    {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "o"},
    {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "k"},
    {"type": "response.output_item.done", "output_index": 0, "item": message("ok")},
    {"type": "response.completed", "response": PLAIN},
]
CALLS = [
    {"type": "response.output_item.done", "output_index": 0, "item": THINKS},
    {"type": "response.output_item.added", "output_index": 1,
     "item": dict(CALL, arguments="", status="in_progress")},
    {"type": "response.function_call_arguments.delta", "item_id": "fc_1",
     "delta": '{"te'},
    {"type": "response.function_call_arguments.delta", "item_id": "fc_1",
     "delta": 'xt": '},
    {"type": "response.function_call_arguments.delta", "item_id": "fc_1",
     "delta": '"hi"}'},
    {"type": "response.output_item.done", "output_index": 1, "item": CALL},
    {"type": "response.completed", "response": ASKS},
]
BYE = [
    {"type": "response.output_text.delta", "item_id": "msg_2", "delta": "done"},
    {"type": "response.completed", "response": DONE},
]


@tool
def echo(text: str) -> str:
    """Repeat the text back."""
    return text


class Canned(Responses):
    """The same adapter, with the POST replaced by a list of replies."""

    def __init__(self, script) -> None:
        super().__init__("gpt-5.6-luna", profile=ONCE)   # no stream: one POST
        self.script, self.sent = list(script), []

    async def post(self, path, body) -> dict:
        self.sent.append(body)
        return self.script.pop(0)


class Streamed(Responses):
    """The same adapter, with the socket replaced by a script of event lists."""

    def __init__(self, *script, profile=LIVE) -> None:
        super().__init__("gpt-5.6-luna", profile=profile)
        self.script, self.sent = list(script), []

    async def sse(self, path, body):
        self.sent.append(body)
        for event in self.script.pop(0):
            yield event


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
    body = encoded(Responses("gpt-5.6-luna", temperature=0),
                   [Message("system", "be brief"), Message("user", "hi")],
                   [echo])
    assert body["model"] == "gpt-5.6-luna"
    assert body["temperature"] == 0              # **params ride along
    assert body["instructions"] == "be brief"    # system is not an input item
    assert body["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]
    assert body["store"] is False                # nothing kept on their side …
    assert body["include"] == ["reasoning.encrypted_content"]     # … it comes here
    assert body["tools"] == [{                   # flat: no nested "function" key
        "type": "function", "name": "echo",
        "description": "Repeat the text back.", "parameters": echo.parameters}]
    empty = encoded(Responses("gpt-5.6-luna"), [Message("user", "hi")])
    assert "tools" not in empty and "instructions" not in empty


def test_the_body_streams_when_the_profile_does() -> None:
    body = encoded(Responses("gpt-5.6-luna"), [Message("user", "hi")])
    assert body["stream"] is True
    assert body["max_output_tokens"] == 128_000                   # off the profile
    quiet = encoded(Responses("gpt-5.6-luna", profile=ONCE, max_output_tokens=9),
                    [Message("user", "hi")])
    assert "stream" not in quiet
    assert quiet["max_output_tokens"] == 9                        # a param wins


def test_tool_calls_round_trip() -> None:
    asked = decoded(ASKS)
    assert asked.tool_calls == [ToolCall("call_abc123", "echo", {"text": "hi"})]

    answered = Message("tool", "hi", tool_call_id="call_abc123")
    said = encoded(Responses("gpt-5.6-luna"), [asked, answered])["input"]
    assert said[-2] == {"type": "function_call", "call_id": "call_abc123",
                        "name": "echo", "arguments": '{"text": "hi"}'}  # a string
    assert said[-1] == {"type": "function_call_output",
                        "call_id": "call_abc123", "output": "hi"}


def test_the_reasoning_goes_back_whole_and_first() -> None:
    asked = decoded(ASKS)
    assert asked.content[0] == Part("reasoning", THINKS)     # kept as it came
    said = encoded(Responses("gpt-5.6-luna"), [
        Message("user", "hi"), asked,
        Message("tool", "hi", tool_call_id="call_abc123"),
    ])["input"]
    assert said[1] is not None and said[1] == THINKS         # before the call it led to
    assert [i.get("type") for i in said[1:]] == [
        "reasoning", "function_call", "function_call_output"]

    thought = Part("thinking", {"type": "thinking", "thinking": "Two halves."})
    other = encoded(Responses("gpt-5.6-luna"), [
        Message("assistant", [thought, Part("text", "half")])])["input"]
    assert other == [{"role": "assistant", "content": "half"}]   # another wire's: dropped


def test_a_tools_pictures_follow_the_run_of_tool_lines() -> None:
    shot = Part("image", {"url": "https://example.com/shot.png"})
    seen = {"type": "input_image", "image_url": "https://example.com/shot.png",
            "detail": "auto"}
    asks = Message("assistant", "", [ToolCall("t1", "echo", {"text": "a"}),
                                     ToolCall("t2", "echo", {"text": "b"})])
    two = encoded(Responses("gpt-5.6-luna"), [
        Message("user", "look twice"),
        asks,
        Message("tool", [shot], tool_call_id="t1"),          # a picture and no words
        Message("tool", [Part("text", "b"), shot], tool_call_id="t2"),
        Message("user", "thanks"),
    ])["input"]
    assert [i.get("type") or i["role"] for i in two] == [
        "user", "function_call", "function_call", "function_call_output",
        "function_call_output", "user", "user"]
    assert two[3] == {"type": "function_call_output", "call_id": "t1", "output": ""}
    assert two[5] == {"role": "user", "content": [   # one item for the whole run
        {"type": "input_text", "text": "[image returned by tool call t1]"}, seen,
        {"type": "input_text", "text": "[image returned by tool call t2]"}, seen]}


def test_usage_and_stop_reason_land_in_meta() -> None:
    said = decoded(PLAIN)
    assert said.text == "ok"
    assert said.meta["stop_reason"] == "completed"
    assert said.meta["usage"] == {"input_tokens": 11, "output_tokens": 1}
    assert said.meta["model"] == "gpt-5.6-luna"
    cut = decoded(reply(4, [message("half")], status="incomplete",
                        incomplete_details={"reason": "max_output_tokens"}))
    assert cut.meta["stop_reason"] == "max_output_tokens"     # the reason underneath


def test_unparseable_arguments_keep_the_raw_string() -> None:
    for junk in ('{"text": ', '"just a string"'):
        said = decoded(reply(5, [dict(CALL, call_id="call_bad", arguments=junk)]))
        assert said.tool_calls[0].args == {}          # the loop can still call it
        assert said.meta["invalid_args"] == {"call_bad": junk}
    assert "invalid_args" not in decoded(ASKS).meta


def test_parts_open_into_content_parts() -> None:
    body = encoded(Responses("gpt-5.6-luna"), [Message("user", [
        Part("text", "what is this?"),
        Part("image", {"url": "https://example.com/cat.png"}),
        Part("image", {"media_type": "image/png", "data": "aGk="}),
    ])])
    assert body["input"][0]["content"] == [
        {"type": "input_text", "text": "what is this?"},
        {"type": "input_image", "image_url": "https://example.com/cat.png",
         "detail": "auto"},
        {"type": "input_image", "image_url": "data:image/png;base64,aGk=",
         "detail": "auto"},
    ]


def test_a_part_type_the_wire_never_heard_of_says_so() -> None:
    try:
        encoded(Responses("gpt-5.6-luna"), [Message("user", [Part("sound", b"...")])])
    except ContractError as e:
        assert "'sound'" in str(e)
    else:
        raise AssertionError("an unknown part type must not go on the wire")


def test_text_arrives_one_delta_at_a_time() -> None:
    said, heard = played(SAYS)
    assert [p.data for p in deltas(heard) if p.type == "text"] == ["o", "k"]
    assert said.text == "ok"                     # and folds back into one reply
    assert [p.type for p in deltas(heard)] == ["text", "text", "meta"]  # no message


def test_the_stream_yields_finished_items_whole() -> None:
    said, heard = played(CALLS)
    assert said.tool_calls == [ToolCall("call_abc123", "echo", {"text": "hi"})]
    assert said.content[0] == Part("reasoning", THINKS)   # encrypted_content and all
    assert [p.type for p in deltas(heard)] == ["reasoning", "tool_call", "meta"]
    assert said.meta["usage"] == {"input_tokens": 11, "output_tokens": 1}
    assert said.meta["stop_reason"] == "completed"


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
    assert model.sent[2]["input"][1] == THINKS   # the thought rode back with it


def test_an_error_event_ends_the_stream() -> None:
    try:
        played([{"type": "error", "code": "ERR", "message": "boom", "param": None}])
    except ProviderError as ex:
        assert ex.status is None and "boom" in str(ex)
    else:
        raise AssertionError("an error event must not fold into a reply")


def test_the_mapping_names_its_rows() -> None:
    assert set(mapping.OUT) == {"text", "image"}      # Part.type -> content part
    assert set(mapping.IN) == {"message", "function_call", "reasoning"}
    assert mapping.USAGE == ("input_tokens", "output_tokens")   # named as core does


if __name__ == "__main__":
    for test in (
        test_three_canned_replies_pass_the_check,
        test_the_body_carries_the_notebook_and_the_toolbox,
        test_the_body_streams_when_the_profile_does,
        test_tool_calls_round_trip,
        test_the_reasoning_goes_back_whole_and_first,
        test_a_tools_pictures_follow_the_run_of_tool_lines,
        test_usage_and_stop_reason_land_in_meta,
        test_unparseable_arguments_keep_the_raw_string,
        test_parts_open_into_content_parts,
        test_a_part_type_the_wire_never_heard_of_says_so,
        test_text_arrives_one_delta_at_a_time,
        test_the_stream_yields_finished_items_whole,
        test_argument_fragments_ride_the_bus_when_the_profile_says_so,
        test_three_scripted_streams_pass_the_check,
        test_an_error_event_ends_the_stream,
        test_the_mapping_names_its_rows,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_responses: all ok")
