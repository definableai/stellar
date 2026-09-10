"""Output: the final answer is parsed, and a bad one is asked for again.

Run: uv run python tests/test_output.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Agent, FakeModel, Message, ToolCall, tool  # noqa: E402
from hooks.output import Output  # noqa: E402


@tool
def ping() -> str:
    """Answers pong."""
    return "pong"


def carded(script, card) -> Agent:
    """One agent reading off a script, one rule card on it."""
    agent = Agent(FakeModel(script), [ping])
    agent.hooks.attach(card)
    return agent


def test_a_bad_answer_is_asked_for_again() -> None:
    agent = carded(["not json", '{"a": 1}'], Output(json.loads))
    r = asyncio.run(agent.run("go"))
    assert r.extra["output"] == {"a": 1}
    assert [m.role for m in r.messages] == ["user", "assistant", "user", "assistant"]
    assert r.messages[2].text.startswith("Not valid: ")   # the error, verbatim
    assert r.messages[-1].text == '{"a": 1}'              # the good reply is kept
    assert r.step == 1                                    # the re-ask is not a turn


def test_it_gives_up_after_the_last_try() -> None:
    agent = carded(["nope", "still nope"], Output(json.loads, tries=1))
    r = asyncio.run(agent.run("go"))
    assert "output" not in r.extra                        # the caller checks
    assert [m.role for m in r.messages] == ["user", "assistant", "user", "assistant"]
    assert r.messages[-1].text == "still nope"            # unparsed, but handed back
    assert agent.model.script == []                       # one try plus one re-ask


def test_a_reply_that_asks_for_tools_is_left_alone() -> None:
    asks = Message("assistant", "", [ToolCall("c1", "ping", {})])
    agent = carded([asks, '{"a": 1}'], Output(json.loads))
    r = asyncio.run(agent.run("go"))
    assert r.extra["output"] == {"a": 1}                  # only the last word parsed
    assert [m.role for m in r.messages] == [
        "user", "assistant", "tool", "assistant"]         # no re-ask in between


if __name__ == "__main__":
    for test in (
        test_a_bad_answer_is_asked_for_again,
        test_it_gives_up_after_the_last_try,
        test_a_reply_that_asks_for_tools_is_left_alone,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_output: all ok")
