"""The four rule cards: Steps, Budget, Permission on the control plane, Log on
the data plane.

Run: uv run python tests/test_hooks.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (  # noqa: E402
    Agent, FakeModel, Message, Part, Run, Tool, ToolCall, loop,
)
from hooks.budget import Budget  # noqa: E402
from hooks.logging import Log  # noqa: E402
from hooks.permission import Permission  # noqa: E402
from hooks.steps import Steps  # noqa: E402


def asks(name: str, cid: str = "c1", **args) -> Message:
    """An assistant line that asks for one tool."""
    return Message("assistant", "", [ToolCall(cid, name, args)])


def priced(cid: str, cost: int) -> Message:
    """A tool-asking reply that says what it cost: cost in, cost out."""
    return Message("assistant", "", [ToolCall(cid, "shout", {"word": "hi"})],
                   meta={"usage": {"input_tokens": cost, "output_tokens": cost}})


def told(run) -> list[str]:
    """What the model was told about the calls it asked for."""
    return [m.text for m in run.messages if m.role == "tool"]


class Shout(Tool):
    """Uppercases a word, and remembers every word it was handed."""

    name = "shout"
    description = "Shout a word."
    parameters = {"type": "object", "properties": {"word": {"type": "string"}}}

    def __init__(self) -> None:
        self.said = []

    async def execute(self, run, word) -> str:
        self.said.append(word)
        return word.upper()


class Whisper(Shout):
    """The same tool under a quieter name."""

    name = "whisper"

    async def execute(self, run, word) -> str:
        self.said.append(word)
        return word.lower()


class Counted(FakeModel):
    """A FakeModel that remembers how many times it was asked."""

    def __init__(self, script) -> None:
        super().__init__(script)
        self.asked = 0

    async def invoke(self, run) -> Message:
        self.asked += 1
        return await super().invoke(run)


def scripted_ask(answers: list[str], heard: list):
    """A ui.ask channel that answers off a list and writes down the questions."""

    async def ask(question, options=None) -> str:
        heard.append((question, options))
        return answers.pop(0)

    return ask


def carded(model, tools, card) -> Agent:
    """One agent, one rule card on it."""
    agent = Agent(model, tools)
    agent.hooks.attach(card)
    return agent


def test_steps_buys_exactly_three_replies() -> None:
    model = Counted([asks("shout", f"c{n}", word="hi") for n in range(5)])
    r = asyncio.run(carded(model, [Shout()], Steps(3)).run("go"))
    assert model.asked == 3
    assert r.step == 4                     # the fourth turn ended at model.pre
    assert len(model.script) == 2          # two replies were never asked for


def test_every_run_gets_its_own_ration() -> None:
    agent = carded(Counted([asks("shout", word="hi")] * 6), [Shout()], Steps(2))

    async def both() -> list[Run]:
        return list(await asyncio.gather(agent.run("a"), agent.run("b")))

    one, two = asyncio.run(both())
    assert (one.step, two.step) == (3, 3)  # neither run spent the other's turns
    assert agent.model.asked == 4


def test_budget_stops_the_run_past_the_limit() -> None:
    model = Counted([priced("c1", 30), priced("c2", 30), priced("c3", 30)])
    r = asyncio.run(carded(model, [Shout()], Budget(100)).run("go"))
    assert r.extra["budget.tokens"] == 120         # 30+30 twice: both ends counted
    assert model.asked == 2
    # the over-budget reply was paid for: Budget keeps it, then stops the run
    assert [m.role for m in r.messages] == ["user", "assistant", "tool", "assistant"]


def test_budget_counts_nothing_it_cannot_read() -> None:
    blank = Message("assistant", "done")           # no usage meta at all
    half = Message("assistant", "", [ToolCall("c1", "shout", {"word": "hi"})],
                   meta={"usage": {"input_tokens": None, "output_tokens": None}})
    agent = carded(FakeModel([half, blank]), [Shout()], Budget(1))
    r = asyncio.run(agent.run("go"))
    assert r.extra["budget.tokens"] == 0
    assert r.messages[-1].text == "done"           # the run reached the end


def test_permission_denies_by_name() -> None:
    shout = Shout()
    agent = carded(FakeModel([asks("shout", word="hi"), "done"]), [shout],
                   Permission(deny={"shout"}))
    r = asyncio.run(agent.run("go"))
    assert shout.said == []                        # the tool never ran
    assert told(r) == ["denied: shout"]


def test_permission_takes_a_predicate() -> None:
    shout = Shout()
    two = Message("assistant", "", [ToolCall("c1", "shout", {"word": "no"}),
                                    ToolCall("c2", "shout", {"word": "yes"})])
    agent = carded(FakeModel([two, "done"]), [shout],
                   Permission(deny=lambda run, call: call.args["word"] == "no"))
    r = asyncio.run(agent.run("go"))
    assert shout.said == ["yes"]
    assert told(r) == ["denied: shout", "YES"]


def test_an_allow_list_settles_it_without_asking() -> None:
    shout, whisper = Shout(), Whisper()
    two = Message("assistant", "", [ToolCall("c1", "shout", {"word": "HI"}),
                                    ToolCall("c2", "whisper", {"word": "HI"})])
    agent = carded(FakeModel([two, "done"]), [shout, whisper],
                   Permission(allow={"whisper"}, ask=True))   # no ui.ask needed
    r = asyncio.run(agent.run("go"))
    assert shout.said == [] and whisper.said == ["HI"]
    assert told(r) == ["denied: shout", "hi"]


def test_ask_lets_a_human_decide() -> None:
    shout, heard = Shout(), []
    two = Message("assistant", "", [ToolCall("c1", "shout", {"word": "one"}),
                                    ToolCall("c2", "shout", {"word": "two"})])
    agent = carded(FakeModel([two, "done"]), [shout], Permission(ask=True))
    agent.extra["ui.ask"] = scripted_ask(["allow", "nope"], heard)
    r = asyncio.run(agent.run("go"))
    assert shout.said == ["one"]                   # anything but "allow" denies
    assert told(r) == ["ONE", "denied: shout"]
    assert heard[0] == ("Allow shout({'word': 'one'})?", ["allow", "deny"])


def test_one_run_can_bring_its_own_channel() -> None:
    shout, heard = Shout(), []
    agent = carded(FakeModel([asks("shout", word="hi"), "done"]), [shout],
                   Permission(ask=True))
    r = Run(agent, "r1", [Message("user", "go")])
    r.extra["ui.ask"] = scripted_ask(["allow"], heard)   # this conversation's UI
    asyncio.run(loop.run(r))
    assert shout.said == ["hi"] and told(r) == ["HI"]
    assert len(heard) == 1 and agent.extra == {}


def test_ask_with_no_channel_denies() -> None:
    shout = Shout()
    agent = carded(FakeModel([asks("shout", word="hi"), "done"]), [shout],
                   Permission(ask=True))
    r = asyncio.run(agent.run("go"))
    assert shout.said == []                        # a guard fails closed
    assert told(r) == ["denied: no ui.ask channel"]


def test_log_says_what_happened() -> None:
    lines = []
    agent = Agent(FakeModel([asks("shout", word="hi"), "done"]), [Shout()])
    agent.events.listen(Log(lines.append))         # a listener: it changes nothing
    asyncio.run(agent.run("go"))
    assert lines == [
        "run.pre: go",
        "model.pre: 1 message",
        "model.delta: <tool_call>",
        "model.post: shout({'word': 'hi'})",
        "tool.pre: shout({'word': 'hi'})",
        "tool.post: HI",
        "model.pre: 3 messages",
        "model.delta: done",
        "model.post: done",
        "run.post: done",
    ]


def test_log_hears_a_streaming_tool() -> None:
    lines = []

    class Teller(Tool):
        name = "tell"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, run):
            yield Part("text", "hi ")
            yield Part("text", "there")

    agent = Agent(FakeModel([asks("tell"), "done"]), [Teller()])
    agent.events.listen(Log(lines.append), "tool")      # one prefix, one lane
    asyncio.run(agent.run("go"))
    assert lines == ["tool.pre: tell({})", "tool.delta: hi ",
                     "tool.delta: there", "tool.post: hi there"]


if __name__ == "__main__":
    for test in (
        test_steps_buys_exactly_three_replies,
        test_every_run_gets_its_own_ration,
        test_budget_stops_the_run_past_the_limit,
        test_budget_counts_nothing_it_cannot_read,
        test_permission_denies_by_name,
        test_permission_takes_a_predicate,
        test_an_allow_list_settles_it_without_asking,
        test_ask_lets_a_human_decide,
        test_one_run_can_bring_its_own_channel,
        test_ask_with_no_channel_denies,
        test_log_says_what_happened,
        test_log_hears_a_streaming_tool,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_hooks: all ok")
