"""Prompt caching on Claude: where the two breakpoints land, and what they cost.

A long tool loop re-reads the same prefix every step. Two cache_control marks
— one on the system block, which tools sit in front of in the prefix, one on
the newest block of the newest turn — let every step after the first read that
prefix at a tenth of the price. Block shapes off the prompt caching page:
https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching

No network here: encode is pure, and the one reply is canned.

Run: uv run python tests/test_anthropic_cache.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from core import Agent, Message, Profile, Run, ToolCall, tool  # noqa: E402
from models.anthropic import DOES, Anthropic  # noqa: E402

MARK = {"type": "ephemeral"}                       # the only cache_control there is
ONCE = Profile("claude-sonnet-5", 1_000_000, 4096, DOES - {"stream"})

# the usage a cached step reports — the four names off the docs page
CACHED_REPLY = {
    "id": "msg_01Q9pKmz1hVvXH4b8s6Tn2Rd",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5",
    "content": [{"type": "text", "text": "still here"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 21, "cache_creation_input_tokens": 248,
              "cache_read_input_tokens": 100_000, "output_tokens": 503},
}

LOOP = [
    Message("system", "Be brief."),
    Message("user", "echo a and b"),
    Message("assistant", "on it", [ToolCall("t1", "echo", {"text": "a"}),
                                   ToolCall("t2", "echo", {"text": "b"})]),
    Message("tool", "a", tool_call_id="t1"),
    Message("tool", "b", tool_call_id="t2"),
]


@tool
def echo(text: str) -> str:
    """Repeat the text back."""
    return text


class Canned(Anthropic):
    """The real adapter with the POST swapped for one reply."""

    def __init__(self, body) -> None:
        super().__init__("claude-sonnet-5", api_key="test", profile=ONCE)
        self.body = body

    async def post(self, path, body) -> dict:
        return self.body


def encoded(model, messages, tools=()) -> dict:
    """One request body, off a Run built by hand: encode needs no loop."""
    return model.encode(Run(Agent(model, tools), "rid", list(messages)))


def marks(node) -> int:
    """Every cache_control anywhere in the body, however deep it is buried."""
    if isinstance(node, dict):
        return ("cache_control" in node) + sum(marks(v) for v in node.values())
    if isinstance(node, list):
        return sum(marks(v) for v in node)
    return 0


def test_two_breakpoints_and_not_one_more() -> None:
    body = encoded(Anthropic("claude-sonnet-5", api_key="x"),
                   [Message("system", "Be brief."), Message("user", "hi"),
                    Message("assistant", "hello"), Message("user", "again")],
                   [echo])
    assert body["system"] == [{"type": "text", "text": "Be brief.",
                               "cache_control": MARK}]
    assert body["messages"][-1]["content"] == [{"type": "text", "text": "again",
                                                "cache_control": MARK}]
    assert marks(body) == 2                     # and nowhere else in the body
    assert marks(body["tools"]) == 0            # the system mark covers tools
    assert marks(body["messages"][:-1]) == 0    # nor any turn but the newest


def test_the_newest_breakpoint_rides_a_merged_tool_result_turn() -> None:
    body = encoded(Anthropic("claude-sonnet-5", api_key="x"), LOOP, [echo])
    last = body["messages"][-1]
    assert last["role"] == "user"               # both results, merged into one turn
    assert last["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "a"},
        {"type": "tool_result", "tool_use_id": "t2", "content": "b",
         "cache_control": MARK},                # the last block of the turn, marked
    ]
    assert marks(body) == 2


def test_a_turn_with_nothing_in_it_takes_no_breakpoint() -> None:
    body = encoded(Anthropic("claude-sonnet-5", api_key="x"),
                   [Message("user", "hi"), Message("assistant", "")])
    assert body["messages"][-1]["content"] == []
    assert marks(body) == 0                     # no system either: nothing to mark


def test_cache_off_sends_the_body_of_before() -> None:
    off = Anthropic("claude-sonnet-5", api_key="x")
    off.cache = False                           # by assignment, never a kwarg
    body = encoded(off, LOOP, [echo])
    assert body["system"] == "Be brief."        # a plain string, as it always was
    assert marks(body) == 0


def test_meta_says_what_the_cache_read_and_what_it_wrote() -> None:
    model = Canned(CACHED_REPLY)
    said = asyncio.run(model.invoke(Run(Agent(model), "rid", [])))
    assert said.meta["usage"] == {
        "input_tokens": 21, "output_tokens": 503,
        "cache_read_input_tokens": 100_000, "cache_creation_input_tokens": 248,
    }
    cold = Canned(dict(CACHED_REPLY, usage={"input_tokens": 21, "output_tokens": 5,
                                            "cache_creation_input_tokens": 0,
                                            "cache_read_input_tokens": 0}))
    said = asyncio.run(cold.invoke(Run(Agent(cold), "rid", [])))
    assert said.meta["usage"] == {"input_tokens": 21, "output_tokens": 5}


if __name__ == "__main__":
    for test in (
        test_two_breakpoints_and_not_one_more,
        test_the_newest_breakpoint_rides_a_merged_tool_result_turn,
        test_a_turn_with_nothing_in_it_takes_no_breakpoint,
        test_cache_off_sends_the_body_of_before,
        test_meta_says_what_the_cache_read_and_what_it_wrote,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_anthropic_cache: all ok")
