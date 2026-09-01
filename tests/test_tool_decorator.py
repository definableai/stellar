"""@tool and on(): a plain function becomes a Tool, or a Hook.

Run: uv run python tests/test_tool_decorator.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (  # noqa: E402
    Agent, ContractError, FakeModel, Message, Part, ToolCall, check, on, tool,
)
from core.contracts import EVENTS  # noqa: E402


@tool
def shout(word: str, times: int = 1) -> str:
    """Shout a word."""
    return (word.upper() + " ") * times


@tool
def kinds(a: str, b: int, c: float, d: bool, e: list, f: dict[str, int],
          g: list[str], h) -> None:
    """Every hint the schema knows, plus one with no hint at all."""


@tool
def counter(agent) -> int:
    """Count the notebook."""
    return len(agent.messages)


@tool
async def slowly(word: str) -> str:
    """Shout, but async."""
    await asyncio.sleep(0)
    return word.upper()


@tool
async def told(word: str):
    """Shout one letter at a time — a streaming tool."""
    for letter in word:
        yield Part("text", letter.upper())


def test_schema_comes_from_the_signature() -> None:
    assert shout.name == "shout"
    assert shout.description == "Shout a word."
    assert shout.parameters == {
        "type": "object",
        "properties": {"word": {"type": "string"}, "times": {"type": "integer"}},
        "required": ["word"],                        # times has a default
    }


def test_every_hint_has_a_type() -> None:
    assert kinds.parameters["properties"] == {
        "a": {"type": "string"},
        "b": {"type": "integer"},
        "c": {"type": "number"},
        "d": {"type": "boolean"},
        "e": {"type": "array"},
        "f": {"type": "object"},
        "g": {"type": "array"},
        "h": {},                                     # no hint, no promise
    }
    assert kinds.parameters["required"] == list("abcdefgh")


def test_the_agent_is_injected_only_when_declared() -> None:
    agent = Agent(FakeModel([]), messages=[Message("user", "hi")])
    assert counter.parameters["properties"] == {}    # the model never sees it
    assert asyncio.run(counter.execute(agent)) == 1
    assert asyncio.run(shout.execute(agent, word="hi")) == "HI "    # not passed on


def test_sync_and_async_functions_both_work() -> None:
    assert asyncio.run(slowly.execute(None, word="hi")) == "HI"
    assert slowly.parameters == {
        "type": "object",
        "properties": {"word": {"type": "string"}},
        "required": ["word"],
    }
    check(Agent(FakeModel([]), [shout, slowly, told]))   # all three pass


def test_a_decorated_tool_runs_end_to_end() -> None:
    asked = Message("assistant", "", [ToolCall("c1", "shout", {"word": "hi"})])
    agent = Agent(FakeModel([asked, "done"]), [shout])
    asyncio.run(agent.run())
    assert agent.messages[1].text == "HI "
    assert agent.messages[1].tool_call_id == "c1"


def test_an_async_generator_streams_end_to_end() -> None:
    asked = Message("assistant", "", [ToolCall("c1", "told", {"word": "hi"})])
    agent = Agent(FakeModel([asked, "done"]), [told])
    asyncio.run(agent.run())
    assert agent.messages[1].text == "HI"            # letters folded into one
    assert agent.messages[1].tool_call_id == "c1"


def test_on_rejects_a_typo() -> None:
    try:
        on("run_pre_", lambda agent: None)
    except ContractError as e:
        for event in EVENTS:
            assert event in str(e), f"{event} is missing from the message"
    else:
        raise AssertionError("a misspelt event name is a typo, not a new event")


def test_on_builds_a_hook_that_passes_check_and_fires() -> None:
    log = []

    async def later(agent) -> None:
        log.append("run_post")

    agent = Agent(
        FakeModel(["hi"]),
        hooks=[on("run_pre", lambda agent: log.append("run_pre")),
               on("run_post", later)],
    )
    check(agent)                                     # __post_init__ said so too
    asyncio.run(agent.run())
    assert log == ["run_pre", "run_post"]


if __name__ == "__main__":
    for test in (
        test_schema_comes_from_the_signature,
        test_every_hint_has_a_type,
        test_the_agent_is_injected_only_when_declared,
        test_sync_and_async_functions_both_work,
        test_a_decorated_tool_runs_end_to_end,
        test_an_async_generator_streams_end_to_end,
        test_on_rejects_a_typo,
        test_on_builds_a_hook_that_passes_check_and_fires,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_tool_decorator: all ok")
