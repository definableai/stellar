"""Compact: the notebook folded when the last reply's prompt outgrew the window.

Run: uv run python tests/test_compact.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Agent, FakeModel, Message, ToolCall, tool  # noqa: E402
from hooks.compact import Compact  # noqa: E402


class Spy(FakeModel):
    """A FakeModel that keeps every prompt it was handed, summary calls included."""

    def __init__(self, script) -> None:
        super().__init__(script)
        self.asked = []
        self.armed = []           # the toolbox each ask carried

    def encode(self, run) -> None:
        self.asked.append(run.messages[-1].text)
        self.armed.append(sorted(run.agent.tools))


def priced(text: str, cost: int) -> Message:
    """An assistant line that says what its own prompt cost."""
    return Message("assistant", text, meta={"usage": {"input_tokens": cost}})


def asks(text: str, *calls: ToolCall) -> Message:
    """An assistant line that asked for tools."""
    return Message("assistant", text, list(calls))


def roles(run) -> list[str]:
    """The notebook, read as roles."""
    return [m.role for m in run.messages]


def test_a_big_prompt_folds_the_older_notebook() -> None:
    history = [Message("system", "be terse"), Message("user", "one"),
               priced("first", 10), Message("user", "two"), priced("second", 5000)]
    agent = Agent(Spy(["the story so far", "done"]))
    agent.hooks.attach(Compact(1000, keep=2))
    run = asyncio.run(agent.run("three", messages=history))
    assert roles(run) == ["system", "user", "assistant", "user", "assistant"]
    assert run.messages[0].text == "be terse"
    assert run.messages[1].text == ("Summary of the conversation so far:\n"
                                    "the story so far")
    assert [m.text for m in run.messages[2:]] == ["second", "three", "done"]
    assert not agent.model.script          # both entries: the summary, then the turn
    assert "user: one" in agent.model.asked[0]      # the dropped lines were the ask
    assert "assistant: first" in agent.model.asked[0]


def test_a_tool_batch_in_the_tail_is_not_split() -> None:
    batch = asks("", ToolCall("c1", "grep", {"for": "x"}), ToolCall("c2", "read"))
    history = [
        Message("system", "be terse"), Message("user", "a"),
        asks("", ToolCall("c0", "look", {"at": "x"})),
        Message("tool", "old", tool_call_id="c0"), batch,
        Message("tool", "hit", tool_call_id="c1"),
        Message("tool", "text", tool_call_id="c2"), priced("big", 5000),
    ]
    agent = Agent(Spy(["the story so far", "done"]))
    agent.hooks.attach(Compact(1000, keep=4))
    run = asyncio.run(agent.run("next", messages=history))
    assert roles(run) == ["system", "user", "assistant", "tool", "tool",
                          "assistant", "user", "assistant"]
    assert run.messages[2] is batch      # the batch kept the turn that asked for it
    assert [m.tool_call_id for m in run.messages[3:5]] == ["c1", "c2"]
    assert "look({'at': 'x'})" in agent.model.asked[0]   # a dropped call, name(args)


def test_a_run_under_the_limit_is_untouched() -> None:
    history = [Message("system", "be terse"), Message("user", "one"),
               priced("first", 10)]
    agent = Agent(Spy(["done"]))
    agent.hooks.attach(Compact(1000, keep=2))
    run = asyncio.run(agent.run("two", messages=history))
    assert [m.text for m in run.messages] == ["be terse", "one", "first", "two", "done"]
    assert len(agent.model.asked) == 1          # one call: nothing was summarised


@tool
def grep(for_: str) -> str:
    """Look for it."""
    return "found"


def test_the_summary_call_carries_no_toolbox() -> None:
    history = [Message("user", "one"), priced("first", 10),
               Message("user", "two"), priced("second", 5000)]
    agent = Agent(Spy(["the story so far", "done"]), [grep])
    agent.hooks.attach(Compact(1000, keep=2))
    asyncio.run(agent.run("three", messages=history))
    assert agent.model.armed == [[], ["grep"]]   # the summary ask, then the turn


if __name__ == "__main__":
    for test in (
        test_a_big_prompt_folds_the_older_notebook,
        test_a_tool_batch_in_the_tail_is_not_split,
        test_a_run_under_the_limit_is_untouched,
        test_the_summary_call_carries_no_toolbox,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_compact: all ok")
