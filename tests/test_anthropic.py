"""The Anthropic adapter: what goes on the wire, and what comes back off it.

The canned bodies and the scripted streams are real Messages API traffic. No
network here — Canned swaps the POST out, Streamed swaps the socket out, and
the reply still travels the real Part path either way.

Run: uv run python tests/test_anthropic.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from core import (  # noqa: E402
    Agent, Message, Part, Profile, ProviderError, Run, ToolCall, tool,
)
from core.conformance import check_model  # noqa: E402
from models.anthropic import DOES, Anthropic  # noqa: E402

# a clone behind a proxy that cannot SSE: the same model, one POST at a time
ONCE = Profile("claude-sonnet-5", 1_000_000, 4096, DOES - {"stream"})
SIGNED = "EqQBCgIYAhgCIkDrtM1kM+2vN9pQ=="
ASKED = "toolu_01T1x1fJ34qAmk2tNTrN7Up6"

BODIES = [
    {
        "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 14, "cache_creation_input_tokens": 0,
                  "cache_read_input_tokens": 0, "output_tokens": 5},
    },
    {
        "id": "msg_01Aq9w938a90dw8q",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [
            {"type": "text", "text": "I'll call the echo tool."},
            {"type": "tool_use", "id": "toolu_01A09q90qw90lq917835lq9",
             "name": "echo", "input": {"text": "hi"}},
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 380, "cache_creation_input_tokens": 0,
                  "cache_read_input_tokens": 0, "output_tokens": 64},
    },
    {
        "id": "msg_01RkYPBmqLqcCbKZHm3wJ2vN",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 452, "cache_creation_input_tokens": 0,
                  "cache_read_input_tokens": 0, "output_tokens": 9},
    },
]

MIXED = {
    "id": "msg_01Rk2FzTaFcHsq5w9vqCq3zP",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5",
    "content": [
        {"type": "thinking", "thinking": "Two halves will do.",
         "signature": SIGNED},
        {"type": "text", "text": "half one "},
        {"type": "text", "text": "half two"},
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 42, "cache_creation_input_tokens": 0,
              "cache_read_input_tokens": 0, "output_tokens": 18},
}

TEXT = [
    {"type": "message_start", "message": {
        "id": "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY", "type": "message",
        "role": "assistant", "content": [], "model": "claude-sonnet-5",
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 25, "output_tokens": 1}}},
    {"type": "content_block_start", "index": 0,
     "content_block": {"type": "text", "text": ""}},
    {"type": "ping"},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "text_delta", "text": "Hello"}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "text_delta", "text": "!"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn",
                                        "stop_sequence": None},
     "usage": {"output_tokens": 15}},
    {"type": "message_stop"},
]

TOOL = [
    {"type": "message_start", "message": {
        "id": "msg_014p7gG3wDgGV9EUtLvnow3U", "type": "message",
        "role": "assistant", "content": [], "model": "claude-sonnet-5",
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 472, "output_tokens": 2}}},
    {"type": "content_block_start", "index": 0,
     "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "text_delta", "text": "Okay,"}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "text_delta", "text": " calling echo."}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1, "content_block": {
        "type": "tool_use", "id": ASKED, "name": "echo", "input": {}}},
    {"type": "content_block_delta", "index": 1,
     "delta": {"type": "input_json_delta", "partial_json": ""}},
    {"type": "content_block_delta", "index": 1,
     "delta": {"type": "input_json_delta", "partial_json": '{"text":'}},
    {"type": "content_block_delta", "index": 1,
     "delta": {"type": "input_json_delta", "partial_json": ' "hi"}'}},
    {"type": "content_block_stop", "index": 1},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use",
                                        "stop_sequence": None},
     "usage": {"output_tokens": 89}},
    {"type": "message_stop"},
]

THINKS = [
    {"type": "message_start", "message": {
        "id": "msg_01Rk2FzTaFcHsq5w9vqCq3zP", "type": "message",
        "role": "assistant", "content": [], "model": "claude-sonnet-5",
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 42, "output_tokens": 1}}},
    {"type": "content_block_start", "index": 0,
     "content_block": {"type": "redacted_thinking", "data": "EroBCkYIBRgCKkC"}},
    {"type": "content_block_stop", "index": 0},          # no deltas, ever
    {"type": "content_block_start", "index": 1,
     "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
    {"type": "content_block_delta", "index": 1,
     "delta": {"type": "thinking_delta", "thinking": "Two halves "}},
    {"type": "content_block_delta", "index": 1,
     "delta": {"type": "thinking_delta", "thinking": "will do."}},
    {"type": "content_block_delta", "index": 1,
     "delta": {"type": "signature_delta", "signature": SIGNED}},
    {"type": "content_block_stop", "index": 1},
    {"type": "content_block_start", "index": 2,
     "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 2,
     "delta": {"type": "text_delta", "text": "half one "}},
    {"type": "content_block_delta", "index": 2,
     "delta": {"type": "text_delta", "text": "half two"}},
    {"type": "content_block_stop", "index": 2},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn",
                                        "stop_sequence": None},
     "usage": {"output_tokens": 18}},
    {"type": "message_stop"},
]

BROKEN = [
    {"type": "message_start", "message": {
        "id": "msg_01Aq9w938a90dw8q", "type": "message", "role": "assistant",
        "content": [], "model": "claude-sonnet-5", "stop_reason": None,
        "stop_sequence": None, "usage": {"input_tokens": 9, "output_tokens": 1}}},
    {"type": "content_block_start", "index": 0,
     "content_block": {"type": "text", "text": ""}},
    {"type": "error", "error": {"type": "overloaded_error",
                                "message": "Overloaded"}},
]


@tool
def echo(text: str) -> str:
    """Repeat the text back."""
    return text


class Canned(Anthropic):
    """The real adapter with the POST swapped for a list of replies."""

    def __init__(self, bodies) -> None:
        super().__init__("claude-sonnet-5", api_key="test", profile=ONCE)
        self.bodies, self.sent = list(bodies), []

    async def post(self, path, body) -> dict:
        self.sent.append(body)
        return self.bodies[len(self.sent) - 1]


class Streamed(Anthropic):
    """The real adapter with the socket swapped for scripted event streams."""

    def __init__(self, *scripts) -> None:
        super().__init__("claude-sonnet-5", api_key="test")
        self.scripts, self.sent = list(scripts), []

    async def sse(self, path, body):
        self.sent.append(body)
        for event in self.scripts[len(self.sent) - 1]:
            yield event


def encoded(model, messages, tools=()) -> dict:
    """One request body, off a Run built by hand: encode needs no loop."""
    return model.encode(Run(Agent(model, tools), "rid", list(messages)))


def decoded(body: dict) -> Message:
    """One canned reply through the real send-and-fold path."""
    model = Canned([body])
    return asyncio.run(model.invoke(Run(Agent(model), "rid", [])))


def replied(script) -> tuple[Message, list]:
    """One scripted stream through invoke: what folded, and every event it rang."""
    model = Streamed(script)
    agent, heard = Agent(model), []
    agent.events.listen(heard.append)              # the whole bus, in order
    said = asyncio.run(model.invoke(Run(agent, "rid", [Message("user", "hi")])))
    return said, heard


def deltas(heard: list) -> list[Part]:
    """Every Part the run rang on model.delta — send's own order, kept."""
    return [event.data for event in heard if event.name == "model.delta"]


def test_canned_replies_pass_check_model() -> None:
    model = Canned(BODIES)
    check_model(model)
    assert len(model.sent) == 3
    assert "tools" not in model.sent[0]               # no toolbox, no tools field
    assert "stream" not in model.sent[0]              # nor a stream this one cannot
    assert model.sent[2]["messages"][-1]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
         "content": "hi"},
    ]


def test_system_messages_go_to_the_top() -> None:
    body = encoded(Anthropic("claude-sonnet-5", api_key="x", max_tokens=64,
                             stop_sequences=["END"]),
                   [Message("system", "Be brief."),
                    Message("user", "hi"),
                    Message("system", "Be kind too.")])
    assert body["system"] == "Be brief.\n\nBe kind too."
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ]
    assert body["model"] == "claude-sonnet-5"
    assert body["max_tokens"] == 64 and body["stop_sequences"] == ["END"]


def test_tool_results_in_a_row_become_one_turn() -> None:
    body = encoded(
        Anthropic("claude-sonnet-5", api_key="x"),
        [
            Message("user", "echo a and b"),
            Message("assistant", "", [ToolCall("t1", "echo", {"text": "a"}),
                                      ToolCall("t2", "echo", {"text": "b"})]),
            Message("tool", "a", tool_call_id="t1"),
            Message("tool", "b", tool_call_id="t2"),
            Message("user", "thanks"),
        ],
        [echo],
    )
    assert [t["role"] for t in body["messages"]] == [
        "user", "assistant", "user", "user"
    ]
    assert body["messages"][1]["content"] == [
        {"type": "tool_use", "id": "t1", "name": "echo", "input": {"text": "a"}},
        {"type": "tool_use", "id": "t2", "name": "echo", "input": {"text": "b"}},
    ]
    assert body["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "a"},
        {"type": "tool_result", "tool_use_id": "t2", "content": "b"},
    ]
    assert body["tools"] == [{
        "name": "echo",
        "description": "Repeat the text back.",
        "input_schema": {"type": "object",
                         "properties": {"text": {"type": "string"}},
                         "required": ["text"]},
    }]


def test_parts_become_content_blocks() -> None:
    body = encoded(Anthropic("claude-sonnet-5", api_key="x"), [Message("user", [
        Part("text", "what is this?"),
        Part("image", {"media_type": "image/png", "data": "aGk="}),
        Part("image", {"url": "https://example.com/cat.png"}),
    ])])
    assert body["messages"][0]["content"] == [
        {"type": "text", "text": "what is this?"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": "aGk="}},
        {"type": "image", "source": {"type": "url",
                                     "url": "https://example.com/cat.png"}},
    ]


def test_the_body_says_max_tokens_and_stream_the_profile_way() -> None:
    live = Anthropic("claude-sonnet-5", api_key="x")
    body = encoded(live, [Message("user", "hi")])
    assert body["max_tokens"] == live.profile.max_output == 128_000
    assert body["stream"] is True                    # this profile has a socket
    quiet = Anthropic("claude-sonnet-5", api_key="x", profile=ONCE)
    assert "stream" not in encoded(quiet, [Message("user", "hi")])
    assert encoded(quiet, [Message("user", "hi")])["max_tokens"] == 4096
    mine = Anthropic("claude-sonnet-5", api_key="x", max_tokens=64, stream=False)
    assert encoded(mine, [Message("user", "hi")])["max_tokens"] == 64
    assert encoded(mine, [Message("user", "hi")])["stream"] is False


def test_send_yields_text_tool_use_and_usage() -> None:
    answer = decoded(BODIES[1])
    assert answer.role == "assistant"
    assert answer.text == "I'll call the echo tool."
    assert answer.content == [Part("text", "I'll call the echo tool.")]
    assert answer.tool_calls == [
        ToolCall("toolu_01A09q90qw90lq917835lq9", "echo", {"text": "hi"})
    ]
    assert answer.meta == {
        "usage": {"input_tokens": 380, "output_tokens": 64},
        "stop_reason": "tool_use",
    }


def test_text_blocks_fold_into_one_part() -> None:
    two = dict(BODIES[2], content=[{"type": "text", "text": "all "},
                                   {"type": "text", "text": "done"}])
    assert decoded(two).content == [Part("text", "all done")]


def test_a_block_we_do_not_know_passes_through_untouched() -> None:
    answer = decoded(MIXED)
    assert answer.content == [
        Part("thinking", MIXED["content"][0]),
        Part("text", "half one half two"),           # halves folded into one
    ]
    body = encoded(Anthropic("claude-sonnet-5", api_key="x"),
                   [Message("user", "hi"), answer])
    assert body["messages"][1]["content"] == [
        MIXED["content"][0],
        {"type": "text", "text": "half one half two"},
    ]


def test_text_deltas_ring_one_each_and_add_up_to_the_reply() -> None:
    said, heard = replied(TEXT)
    assert [(p.type, p.data) for p in deltas(heard)][:2] == [
        ("text", "Hello"), ("text", "!"),            # one bell per chunk
    ]
    assert said.content == [Part("text", "Hello!")]  # and one Part when folded


def test_a_tool_call_arrives_whole_when_its_block_ends_and_not_before() -> None:
    said, heard = replied(TOOL)
    rang = deltas(heard)
    assert [p.type for p in rang] == ["text", "text", "tool_call", "meta"]
    assert rang[2].data == ToolCall(ASKED, "echo", {"text": "hi"})
    assert said.tool_calls == [ToolCall(ASKED, "echo", {"text": "hi"})]
    fragments = [e for e in heard if e.name == "tool.args"]
    assert [e.data["delta"] for e in fragments] == ["", '{"text":', ' "hi"}']
    assert {(e.data["id"], e.data["name"], e.source) for e in fragments} == {
        (ASKED, "echo", "model:claude-sonnet-5")
    }


def test_arguments_that_are_not_json_are_remembered_not_raised() -> None:
    junk = [dict(e, delta={"type": "input_json_delta", "partial_json": "{oops"})
            if e.get("delta", {}).get("type") == "input_json_delta" else e
            for e in TOOL]
    said, _ = replied(junk)
    assert said.tool_calls == [ToolCall(ASKED, "echo", {})]
    assert said.meta["invalid_args"] == {ASKED: "{oops{oops{oops"}


def test_a_thinking_block_lands_whole_and_replays_as_itself() -> None:
    said, _ = replied(THINKS)
    sealed = {"type": "redacted_thinking", "data": "EroBCkYIBRgCKkC"}
    thought = {"type": "thinking", "thinking": "Two halves will do.",
               "signature": SIGNED}
    assert said.content == [
        Part("redacted_thinking", sealed),           # held from start to stop
        Part("thinking", thought),                   # deltas folded into the block
        Part("text", "half one half two"),
    ]
    body = encoded(Anthropic("claude-sonnet-5", api_key="x"),
                   [Message("user", "hi"), said])
    assert body["messages"][1]["content"][:2] == [sealed, thought]


def test_usage_is_merged_once_from_both_ends_of_the_stream() -> None:
    said, heard = replied(TEXT)
    assert said.meta == {
        "usage": {"input_tokens": 25, "output_tokens": 15},   # start, then delta
        "stop_reason": "end_turn",
    }
    assert [p.type for p in deltas(heard)].count("meta") == 1  # merge is shallow


def test_an_error_event_stops_the_stream() -> None:
    try:
        replied(BROKEN)
    except ProviderError as ex:
        assert ex.status is None and "overloaded_error" in str(ex)
    else:
        raise AssertionError("an error event must not read as a finished stream")


def test_scripted_streams_pass_check_model() -> None:
    model = Streamed(TEXT, TOOL, TEXT)
    check_model(model)
    assert len(model.sent) == 3
    assert model.sent[0]["stream"] is True
    assert model.sent[2]["messages"][-1]["content"] == [
        {"type": "tool_result", "tool_use_id": ASKED, "content": "hi"},
    ]


if __name__ == "__main__":
    for test in (
        test_canned_replies_pass_check_model,
        test_system_messages_go_to_the_top,
        test_tool_results_in_a_row_become_one_turn,
        test_parts_become_content_blocks,
        test_the_body_says_max_tokens_and_stream_the_profile_way,
        test_send_yields_text_tool_use_and_usage,
        test_text_blocks_fold_into_one_part,
        test_a_block_we_do_not_know_passes_through_untouched,
        test_text_deltas_ring_one_each_and_add_up_to_the_reply,
        test_a_tool_call_arrives_whole_when_its_block_ends_and_not_before,
        test_arguments_that_are_not_json_are_remembered_not_raised,
        test_a_thinking_block_lands_whole_and_replays_as_itself,
        test_usage_is_merged_once_from_both_ends_of_the_stream,
        test_an_error_event_stops_the_stream,
        test_scripted_streams_pass_check_model,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_anthropic: all ok")
