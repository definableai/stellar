"""The Anthropic adapter: what goes on the wire, and what comes back off it.

The three canned bodies are real Messages API responses. No network here.

Run: uv run python tests/test_anthropic.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Agent, Message, Part, ToolCall, check_model, tool  # noqa: E402
from models.anthropic import Anthropic  # noqa: E402

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
         "signature": "EqQBCgIYAhgCIkDrtM1kM+2vN9pQ=="},
        {"type": "text", "text": "half one "},
        {"type": "text", "text": "half two"},
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 42, "cache_creation_input_tokens": 0,
              "cache_read_input_tokens": 0, "output_tokens": 18},
}


@tool
def echo(text: str) -> str:
    """Repeat the text back."""
    return text


class Canned(Anthropic):
    """The three replies check_model expects, off the shelf."""

    def __init__(self) -> None:
        super().__init__(api_key="test")
        self.sent = []

    async def send(self, body) -> dict:
        self.sent.append(body)
        return BODIES[len(self.sent) - 1]


def test_canned_replies_pass_check_model() -> None:
    model = Canned()
    check_model(model)
    assert len(model.sent) == 3
    assert "tools" not in model.sent[0]               # no toolbox, no tools field
    assert model.sent[2]["messages"][-1]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
         "content": "hi"},
    ]


def test_system_messages_go_to_the_top() -> None:
    agent = Agent(Anthropic(api_key="x", max_tokens=64, stop_sequences=["END"]),
                  messages=[Message("system", "Be brief."),
                            Message("user", "hi"),
                            Message("system", "Be kind too.")])
    body = agent.model.to_provider(agent)
    assert body["system"] == "Be brief.\n\nBe kind too."
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ]
    assert body["model"] == "claude-sonnet-5"
    assert body["max_tokens"] == 64 and body["stop_sequences"] == ["END"]


def test_tool_results_in_a_row_become_one_turn() -> None:
    agent = Agent(
        Anthropic(api_key="x"),
        {"echo": echo},
        messages=[
            Message("user", "echo a and b"),
            Message("assistant", "", [ToolCall("t1", "echo", {"text": "a"}),
                                      ToolCall("t2", "echo", {"text": "b"})]),
            Message("tool", "a", tool_call_id="t1"),
            Message("tool", "b", tool_call_id="t2"),
            Message("user", "thanks"),
        ],
    )
    body = agent.model.to_provider(agent)
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
    agent = Agent(Anthropic(api_key="x"), messages=[Message("user", [
        Part("text", "what is this?"),
        Part("image", {"media_type": "image/png", "data": "aGk="}),
        Part("image", {"url": "https://example.com/cat.png"}),
    ])])
    assert agent.model.to_provider(agent)["messages"][0]["content"] == [
        {"type": "text", "text": "what is this?"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": "aGk="}},
        {"type": "image", "source": {"type": "url",
                                     "url": "https://example.com/cat.png"}},
    ]


def test_to_core_reads_text_tool_use_and_usage() -> None:
    answer = Anthropic(api_key="x").to_core(BODIES[1])
    assert answer.role == "assistant"
    assert answer.content == "I'll call the echo tool."
    assert answer.tool_calls == [
        ToolCall("toolu_01A09q90qw90lq917835lq9", "echo", {"text": "hi"})
    ]
    assert answer.meta == {
        "usage": {"input_tokens": 380, "output_tokens": 64},
        "stop_reason": "tool_use",
    }


def test_text_blocks_join_into_one_string() -> None:
    two = dict(BODIES[2], content=[{"type": "text", "text": "all "},
                                   {"type": "text", "text": "done"}])
    assert Anthropic(api_key="x").to_core(two).content == "all done"


def test_a_block_we_do_not_know_turns_the_content_into_parts() -> None:
    model = Anthropic(api_key="x")
    answer = model.to_core(MIXED)
    assert answer.content == [
        Part("thinking", MIXED["content"][0]),
        Part("text", "half one "),
        Part("text", "half two"),
    ]
    agent = Agent(model, messages=[Message("user", "hi"), answer])
    assert model.to_provider(agent)["messages"][1]["content"] == MIXED["content"]


if __name__ == "__main__":
    for test in (
        test_canned_replies_pass_check_model,
        test_system_messages_go_to_the_top,
        test_tool_results_in_a_row_become_one_turn,
        test_parts_become_content_blocks,
        test_to_core_reads_text_tool_use_and_usage,
        test_text_blocks_join_into_one_string,
        test_a_block_we_do_not_know_turns_the_content_into_parts,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_anthropic: all ok")
