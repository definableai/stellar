"""The four rule cards: Steps, Budget, Permission, Log.

Run: uv run python tests/test_hooks.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Agent, FakeModel, Message, Tool, ToolCall  # noqa: E402
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


def told(agent) -> list[str]:
    """What the model was told about the calls it asked for."""
    return [m.content for m in agent.messages if m.role == "tool"]


class Shout(Tool):
    """Uppercases a word, and remembers every word it was handed."""

    name = "shout"
    description = "Shout a word."
    parameters = {"type": "object", "properties": {"word": {"type": "string"}}}

    def __init__(self) -> None:
        self.said = []

    def execute(self, agent, word) -> str:
        self.said.append(word)
        return word.upper()


class Whisper(Shout):
    """The same tool under a quieter name."""

    name = "whisper"

    def execute(self, agent, word) -> str:
        self.said.append(word)
        return word.lower()


class Counted(FakeModel):
    """A FakeModel that remembers how many times it was asked."""

    def __init__(self, script) -> None:
        super().__init__(script)
        self.asked = 0

    async def ainvoke(self, agent) -> Message:
        self.asked += 1
        return await super().ainvoke(agent)


def scripted_ask(answers: list[str], heard: list):
    """A ui.ask channel that answers off a list and writes down the questions."""

    async def ask(question, options=None) -> str:
        heard.append((question, options))
        return answers.pop(0)

    return ask


def test_steps_buys_exactly_three_replies() -> None:
    model = Counted([asks("shout", f"c{n}", word="hi") for n in range(5)])
    agent = Agent(model, {"shout": Shout()}, [Steps(3)])
    asyncio.run(agent.run())
    assert model.asked == 3
    assert agent.step == 4                 # the fourth turn ended at model_pre
    assert len(model.script) == 2          # two replies were never asked for


def test_budget_stops_the_run_past_the_limit() -> None:
    model = Counted([priced("c1", 30), priced("c2", 30), priced("c3", 30)])
    agent = Agent(model, {"shout": Shout()}, [Budget(100)])
    asyncio.run(agent.run())
    assert agent.extra["budget.tokens"] == 120     # 30+30 twice: both ends counted
    assert model.asked == 2
    # the over-budget reply was paid for: Budget keeps it, then stops the run
    assert [m.role for m in agent.messages] == ["assistant", "tool", "assistant"]


def test_budget_counts_nothing_it_cannot_read() -> None:
    blank = Message("assistant", "done")           # no usage meta at all
    half = Message("assistant", "", [ToolCall("c1", "shout", {"word": "hi"})],
                   meta={"usage": {"input_tokens": None, "output_tokens": None}})
    agent = Agent(FakeModel([half, blank]), {"shout": Shout()}, [Budget(1)])
    asyncio.run(agent.run())
    assert agent.extra["budget.tokens"] == 0
    assert agent.messages[-1].content == "done"    # the run reached the end


def test_permission_denies_by_name() -> None:
    shout = Shout()
    agent = Agent(FakeModel([asks("shout", word="hi"), "done"]), {"shout": shout},
                  [Permission(deny={"shout"})])
    asyncio.run(agent.run())
    assert shout.said == []                        # the tool never ran
    assert told(agent) == ["denied: shout"]


def test_permission_takes_a_predicate() -> None:
    shout = Shout()
    two = Message("assistant", "", [ToolCall("c1", "shout", {"word": "no"}),
                                    ToolCall("c2", "shout", {"word": "yes"})])
    agent = Agent(FakeModel([two, "done"]), {"shout": shout},
                  [Permission(deny=lambda agent, call: call.args["word"] == "no")])
    asyncio.run(agent.run())
    assert shout.said == ["yes"]
    assert told(agent) == ["denied: shout", "YES"]


def test_an_allow_list_settles_it_without_asking() -> None:
    shout, whisper = Shout(), Whisper()
    two = Message("assistant", "", [ToolCall("c1", "shout", {"word": "HI"}),
                                    ToolCall("c2", "whisper", {"word": "HI"})])
    agent = Agent(FakeModel([two, "done"]), {"shout": shout, "whisper": whisper},
                  [Permission(allow={"whisper"}, ask=True)])   # no ui.ask needed
    asyncio.run(agent.run())
    assert shout.said == [] and whisper.said == ["HI"]
    assert told(agent) == ["denied: shout", "hi"]


def test_ask_lets_a_human_decide() -> None:
    shout, heard = Shout(), []
    two = Message("assistant", "", [ToolCall("c1", "shout", {"word": "one"}),
                                    ToolCall("c2", "shout", {"word": "two"})])
    agent = Agent(FakeModel([two, "done"]), {"shout": shout}, [Permission(ask=True)])
    agent.extra["ui.ask"] = scripted_ask(["allow", "nope"], heard)
    asyncio.run(agent.run())
    assert shout.said == ["one"]                   # anything but "allow" denies
    assert told(agent) == ["ONE", "denied: shout"]
    assert heard[0] == ("Allow shout({'word': 'one'})?", ["allow", "deny"])


def test_ask_with_no_channel_denies() -> None:
    shout = Shout()
    agent = Agent(FakeModel([asks("shout", word="hi"), "done"]), {"shout": shout},
                  [Permission(ask=True)])
    asyncio.run(agent.run())
    assert shout.said == []                        # a guard fails closed
    assert told(agent) == ["denied: no ui.ask channel"]


def test_log_hands_every_bell_to_one_callable() -> None:
    heard = []

    async def emit(name, payload) -> None:         # an async emit works too
        heard.append((name, payload))

    agent = Agent(FakeModel([asks("shout", word="hi"), "done"]),
                  {"shout": Shout()}, [Log(emit)])
    asyncio.run(agent.run())
    assert [name for name, _ in heard] == [
        "run_pre", "model_pre", "model_post", "tool_pre", "tool_post",
        "model_pre", "model_post", "run_post",
    ]
    assert heard[3][1] == {"id": "c1", "name": "shout", "args": {"word": "hi"}}
    assert heard[4][1] == {"id": "c1", "content": "HI"}
    assert heard[-1][1] == {"step": 2, "messages": 3}


if __name__ == "__main__":
    for test in (
        test_steps_buys_exactly_three_replies,
        test_budget_stops_the_run_past_the_limit,
        test_budget_counts_nothing_it_cannot_read,
        test_permission_denies_by_name,
        test_permission_takes_a_predicate,
        test_an_allow_list_settles_it_without_asking,
        test_ask_lets_a_human_decide,
        test_ask_with_no_channel_denies,
        test_log_hands_every_bell_to_one_callable,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_hooks: all ok")
