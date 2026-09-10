"""Truncate: one big tool result cut down, the Parts beside it kept.

Run: uv run python tests/test_truncate.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Agent, FakeModel, Message, Part, ToolCall, tool  # noqa: E402
from hooks.truncate import Truncate  # noqa: E402

PICTURE = Part("image", {"url": "http://x/y.png"})


def dumping(result) -> Agent:
    """One agent whose only tool hands back `result`, carded with Truncate(10)."""

    @tool
    async def dump(run):
        """Hand back whatever the test wrote."""
        return result

    model = FakeModel([Message("assistant", "", [ToolCall("c1", "dump")]), "done"])
    agent = Agent(model, [dump])
    agent.hooks.attach(Truncate(10))
    return agent


def told(agent: Agent) -> Message:
    """The tool line the model was left with."""
    r = asyncio.run(agent.run("go"))
    return [m for m in r.messages if m.role == "tool"][0]


def test_a_long_result_is_cut_with_the_marker() -> None:
    reply = told(dumping("x" * 50))
    assert reply.text == "x" * 10 + "\n… [truncated, 40 more chars]"
    assert len(reply.content) == 1


def test_a_short_result_is_untouched() -> None:
    reply = told(dumping("x" * 10))         # exactly the cap still fits
    assert reply.text == "x" * 10
    assert len(reply.content) == 1


def test_an_image_survives_beside_the_cut_text() -> None:
    reply = told(dumping([Part("text", "x" * 30), PICTURE, Part("text", "y" * 20)]))
    assert reply.text == "x" * 10 + "\n… [truncated, 40 more chars]"
    assert reply.content[1] is PICTURE      # the picture rode along whole
    assert len(reply.content) == 2          # both text Parts became the one


if __name__ == "__main__":
    for test in (
        test_a_long_result_is_cut_with_the_marker,
        test_a_short_result_is_untouched,
        test_an_image_survives_beside_the_cut_text,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_truncate: all ok")
