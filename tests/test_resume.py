"""Resume at the tool: a notebook that still owes tool answers pays first.

Run: uv run python tests/test_resume.py
"""

import asyncio
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Agent, FakeModel, Message, Stop, ToolCall, hook, tool  # noqa: E402


def gated(ran: list[str]) -> Agent:
    """An agent whose one tool waits behind an approval hook at tool.pre."""

    @tool
    async def rm(path: str) -> str:
        """Delete a path."""
        ran.append(path)
        return f"deleted {path}"

    @hook("tool.pre")
    async def approve(call, run) -> None:
        if not run.agent.extra.get("approved"):
            raise Stop
    agent = Agent(
        FakeModel([Message("assistant", "", [ToolCall("c1", "rm", {"path": "/x"})]),
                   "done"]),
        [rm],
    )
    agent.hooks.attach(approve)
    return agent


def test_stop_at_the_gate_leaves_the_call_unanswered() -> None:
    ran: list[str] = []
    agent = gated(ran)
    first = asyncio.run(agent.run("clean up", run_id="r1"))
    assert ran == []
    assert [m.role for m in first.messages] == ["user", "assistant"]
    assert first.messages[-1].tool_calls[0].id == "c1"


def test_a_resume_runs_the_pending_call_once_then_carries_on() -> None:
    ran: list[str] = []
    agent = gated(ran)
    first = asyncio.run(agent.run("clean up", run_id="r1"))
    state = pickle.loads(pickle.dumps(first))
    agent.extra["approved"] = True
    again = asyncio.run(agent.run(messages=state.messages, run_id=state.id))
    assert ran == ["/x"]
    assert [m.role for m in again.messages] == [
        "user", "assistant", "tool", "assistant"]
    assert again.messages[-1].text == "done"
    assert again.id == state.id
    assert agent.model.script == []          # exactly the one reply was consumed


def test_run_pre_still_rings_on_the_last_line() -> None:
    heard: list[str] = []

    @hook("run.pre")
    async def watch(message) -> None:
        heard.append(message.role)
    ran: list[str] = []
    agent = gated(ran)
    agent.hooks.attach(watch)
    first = asyncio.run(agent.run("clean up", run_id="r1"))
    agent.extra["approved"] = True
    asyncio.run(agent.run(messages=first.messages, run_id=first.id))
    assert heard == ["user", "assistant"]    # the resume rings on the half-run answer


def test_only_the_unanswered_half_of_a_batch_runs_again() -> None:
    ran: list[str] = []

    @tool(parallel=True)
    async def touch(path: str) -> str:
        """Touch a path."""
        ran.append(path)
        return f"touched {path}"
    answer = Message("assistant", "", [ToolCall("c1", "touch", {"path": "/a"}),
                                       ToolCall("c2", "touch", {"path": "/b"})])
    agent = Agent(FakeModel(["done"]), [touch])
    said = [Message("user", "go"), answer,
            Message("tool", "touched /a", tool_call_id="c1")]
    r = asyncio.run(agent.run(messages=said))
    assert ran == ["/b"]
    assert [m.tool_call_id for m in r.messages if m.role == "tool"] == ["c1", "c2"]
    assert answer.tool_calls[0].id == "c1"   # the notebook's own answer is untouched


def test_an_answered_notebook_goes_straight_to_the_model() -> None:
    ran: list[str] = []

    @tool
    async def rm(path: str) -> str:
        """Delete a path."""
        ran.append(path)
        return "gone"
    agent = Agent(FakeModel(["done"]), [rm])
    said = [Message("user", "go"),
            Message("assistant", "", [ToolCall("c1", "rm", {"path": "/x"})]),
            Message("tool", "gone", tool_call_id="c1")]
    r = asyncio.run(agent.run(messages=said))
    assert ran == []
    assert r.messages[-1].text == "done"


def test_a_notebook_ending_in_a_user_line_goes_straight_to_the_model() -> None:
    agent = Agent(FakeModel(["hi"]))
    r = asyncio.run(agent.run("go"))
    assert [m.role for m in r.messages] == ["user", "assistant"]


if __name__ == "__main__":
    for test in (
        test_stop_at_the_gate_leaves_the_call_unanswered,
        test_a_resume_runs_the_pending_call_once_then_carries_on,
        test_run_pre_still_rings_on_the_last_line,
        test_only_the_unanswered_half_of_a_batch_runs_again,
        test_an_answered_notebook_goes_straight_to_the_model,
        test_a_notebook_ending_in_a_user_line_goes_straight_to_the_model,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_resume: all ok")
