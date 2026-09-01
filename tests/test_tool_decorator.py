"""@tool: a plain function becomes a Tool.

Run: uv run python tests/test_tool_decorator.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (  # noqa: E402
    Agent, FakeModel, Message, Part, Run, ToolCall, check, tool,
)


@tool
def shout(word: str, times: int = 1) -> str:
    """Shout a word."""
    return (word.upper() + " ") * times


@tool
def kinds(a: str, b: int, c: float, d: bool, e: list, f: dict[str, int],
          g: list[str], h) -> None:
    """Every hint the schema knows, plus one with no hint at all."""


@tool
def quoted(a: "int", b: "list[str]") -> "str":
    """Hints written as strings — what the future-import leaves in every module."""
    return f"{a}{b}"


@tool
def counter(run) -> int:
    """Count the notebook."""
    return len(run.messages)


@tool
def later(when: str, run: bool = False) -> str:
    """A run that is not the Run: only the first parameter is the loop's."""
    return f"{when}:{run}"


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


def test_string_hints_are_read_as_hints() -> None:
    assert quoted.parameters["properties"] == {      # a module that says
        "a": {"type": "integer"},                    # from __future__ import
        "b": {"type": "array"},                      # annotations has only these
    }


def test_the_run_is_injected_only_when_declared() -> None:
    run = Run(Agent(FakeModel([])), "r1", [Message("user", "hi")])
    assert counter.parameters["properties"] == {}    # the model never sees it
    assert asyncio.run(counter.execute(run)) == 1
    assert asyncio.run(shout.execute(run, word="hi")) == "HI "    # not passed on


def test_only_the_first_parameter_answers_to_run() -> None:
    assert later.parameters["properties"] == {       # second in line, so it is
        "when": {"type": "string"},                  # an argument like any other
        "run": {"type": "boolean"},
    }
    assert asyncio.run(later.execute(None, when="now", run=True)) == "now:True"


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
    said = asyncio.run(Agent(FakeModel([asked, "done"]), [shout]).run("go")).messages
    assert said[2].text == "HI "                     # no run reached it, none asked
    assert said[2].tool_call_id == "c1"


def test_the_run_reaches_a_tool_that_asked_for_it() -> None:
    asked = Message("assistant", "", [ToolCall("c1", "counter", {})])
    said = asyncio.run(Agent(FakeModel([asked, "done"]), [counter]).run("go")).messages
    assert said[2].text == "2"                       # the notebook, live: ask + reply
    assert said[2].tool_call_id == "c1"


def test_an_async_generator_streams_end_to_end() -> None:
    asked = Message("assistant", "", [ToolCall("c1", "told", {"word": "hi"})])
    agent = Agent(FakeModel([asked, "done"]), [told])
    heard: list = []
    agent.events.listen(lambda e: heard.append((e.source, e.data.data)), "tool.delta")
    said = asyncio.run(agent.run("go")).messages
    assert heard == [("told", "H"), ("told", "I")]   # every letter rang, by name
    assert said[2].text == "HI"                      # letters folded into one
    assert said[2].tool_call_id == "c1"


if __name__ == "__main__":
    for test in (
        test_schema_comes_from_the_signature,
        test_every_hint_has_a_type,
        test_string_hints_are_read_as_hints,
        test_the_run_is_injected_only_when_declared,
        test_only_the_first_parameter_answers_to_run,
        test_sync_and_async_functions_both_work,
        test_a_decorated_tool_runs_end_to_end,
        test_the_run_reaches_a_tool_that_asked_for_it,
        test_an_async_generator_streams_end_to_end,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_tool_decorator: all ok")
