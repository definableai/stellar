"""The loop's promises: order, control signals, coercion, teardown.

Run: uv run python tests/test_loop.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (  # noqa: E402
    Agent, ContractError, FakeModel, Hook, Message, Model, Stop, Tool, ToolCall, add,
)
from core.contracts import EVENTS  # noqa: E402


def asks(name: str, cid: str = "c1", **args) -> Message:
    """An assistant line that asks for one tool."""
    return Message("assistant", "", [ToolCall(cid, name, args)])


class Shout(Tool):
    """Uppercases a word."""

    name = "shout"
    description = "Shout a word."
    parameters = {"type": "object", "properties": {"word": {"type": "string"}}}

    def execute(self, agent, word) -> str:
        return word.upper()


class Recorder(Hook):
    """Writes down every bell it hears."""

    def __init__(self, log: list) -> None:
        self.log = log

    def run_pre(self, agent) -> None:
        self.log.append(("a", "run_pre"))

    def model_pre(self, agent) -> None:
        self.log.append(("a", "model_pre"))

    def model_post(self, agent) -> None:
        self.log.append(("a", "model_post"))

    def tool_pre(self, agent) -> None:
        self.log.append(("a", "tool_pre"))

    def tool_post(self, agent) -> None:
        self.log.append(("a", "tool_post"))

    def run_post(self, agent) -> None:
        self.log.append(("a", "run_post"))


class Echo(Hook):
    """A second hook, async, so fire() has to await something."""

    def __init__(self, log: list) -> None:
        self.log = log

    async def run_pre(self, agent) -> None:
        self.log.append(("b", "run_pre"))


def test_hook_order() -> None:
    log = []
    agent = Agent(
        FakeModel([asks("shout", word="hi"), "done"]),
        {"shout": Shout()},
        [Recorder(log), Echo(log)],
    )
    asyncio.run(agent.run())
    assert [event for tag, event in log if tag == "a"] == [
        "run_pre", "model_pre", "model_post", "tool_pre", "tool_post",
        "model_pre", "model_post", "run_post",
    ]
    assert log[:2] == [("a", "run_pre"), ("b", "run_pre")]


def test_deny_path() -> None:
    ran = []

    class Watched(Shout):
        def execute(self, agent, word) -> str:
            ran.append(word)
            return word.upper()

    class Deny(Hook):
        def tool_pre(self, agent) -> None:
            agent.result = "denied"

    agent = Agent(
        FakeModel([asks("shout", word="hi"), "done"]), {"shout": Watched()}, [Deny()]
    )
    asyncio.run(agent.run())
    assert ran == []
    results = [m for m in agent.messages if m.role == "tool"]
    assert [(m.content, m.tool_call_id) for m in results] == [("denied", "c1")]


def test_stop_from_a_hook() -> None:
    log = []

    class Quit(Hook):
        def model_pre(self, agent) -> None:
            raise Stop

        def run_post(self, agent) -> None:
            log.append("run_post")

    agent = Agent(FakeModel(["never asked for"]), hooks=[Quit()])
    asyncio.run(agent.run())
    assert log == ["run_post"]
    assert agent.messages == []


def test_stop_from_a_tool() -> None:
    log = []

    class Quit(Shout):
        def execute(self, agent, word) -> str:
            raise Stop

    class Watch(Hook):
        def run_post(self, agent) -> None:
            log.append("run_post")

    agent = Agent(FakeModel([asks("shout", word="hi")]), {"shout": Quit()}, [Watch()])
    asyncio.run(agent.run())
    assert log == ["run_post"]
    assert [m.role for m in agent.messages] == ["assistant"]


def test_a_broken_tool_becomes_a_result() -> None:
    class Boom(Shout):
        def execute(self, agent, word) -> str:
            raise ValueError("nope")

    agent = Agent(FakeModel([asks("shout", word="hi"), "done"]), {"shout": Boom()})
    asyncio.run(agent.run())
    assert agent.messages[1].content == "error: ValueError: nope"
    assert agent.messages[2].content == "done"       # the model got another turn


def test_an_unknown_tool_becomes_a_result() -> None:
    agent = Agent(FakeModel([asks("ghost"), "done"]))
    asyncio.run(agent.run())
    assert agent.messages[1].content == "error: unknown tool: ghost"
    assert agent.messages[2].content == "done"


def test_the_model_must_return_a_message() -> None:
    class Junk(Model):
        async def ainvoke(self, agent) -> dict:
            return {"role": "assistant"}

    try:
        asyncio.run(Agent(Junk()).run())
    except ContractError as e:
        assert "Junk.ainvoke returned dict, expected Message" in str(e)
    else:
        raise AssertionError("a dict should not pass for a Message")


def test_a_hook_must_listen_for_something() -> None:
    class Silent(Hook):
        pass

    try:
        Agent(FakeModel([]), hooks=[Silent()])
    except ContractError as e:
        for event in EVENTS:
            assert event in str(e), f"{event} is missing from the message"
    else:
        raise AssertionError("a hook that overrides nothing is a typo")


def test_a_bare_function_is_not_a_model() -> None:
    try:
        Agent(lambda agent: Message("assistant", "hi"))
    except ContractError as e:
        assert "wrap it in a Model subclass" in str(e)
    else:
        raise AssertionError("a function should not pass for a Model")


def test_a_hook_can_add_a_hook() -> None:
    log = []

    class Late(Hook):
        def run_pre(self, agent) -> None:
            log.append("late:run_pre")

        def model_pre(self, agent) -> None:
            log.append(f"late:model_pre:{agent.step}")

    class Grower(Hook):
        def run_pre(self, agent) -> None:
            agent.hooks.append(Late())

    agent = Agent(FakeModel(["done"]), hooks=[Grower()])
    asyncio.run(agent.run())
    assert log == ["late:model_pre:1"]               # not this bell, the next one


def test_agents_do_not_share_memory() -> None:
    model, tools, hooks = FakeModel(["hi"]), {}, []
    one, two = Agent(model, tools, hooks), Agent(model, tools, hooks)
    asyncio.run(one.run())
    one.extra["k"] = 1
    assert one.model is two.model
    assert one.tools is two.tools and one.hooks is two.hooks
    assert two.messages == [] and two.extra == {} and two.step == 0
    assert one.step == 1 and one.extra == {"k": 1}


def test_a_sync_model_and_a_sync_tool_both_work() -> None:
    class Turns(Model):
        def __init__(self) -> None:
            self.turn = 0

        def invoke(self, agent) -> Message:
            self.turn += 1
            if self.turn == 1:
                return asks("shout", word="hi")
            return Message("assistant", "done")

    agent = Agent(Turns(), {"shout": Shout()})
    asyncio.run(agent.run())
    assert agent.messages[1].content == "HI"
    assert agent.messages[2].content == "done"


def test_every_result_becomes_a_tool_message() -> None:
    class Line(Tool):
        name = "line"

        def execute(self, agent) -> str:
            return "plain"

    class Bag(Tool):
        name = "bag"

        def execute(self, agent) -> dict:
            return {"n": 1}

    class Note(Tool):
        name = "note"

        def execute(self, agent) -> Message:
            return Message("assistant", "mine")

    three = Message(
        "assistant", "", [ToolCall("c1", "line"), ToolCall("c2", "bag"),
                          ToolCall("c3", "note")]
    )
    agent = Agent(
        FakeModel([three, "done"]), {"line": Line(), "bag": Bag(), "note": Note()}
    )
    asyncio.run(agent.run())
    results = [m for m in agent.messages if m.role == "tool"]
    assert [m.tool_call_id for m in results] == ["c1", "c2", "c3"]
    assert [m.content for m in results] == ["plain", '{"n": 1}', "mine"]


def test_run_post_fires_after_a_crash() -> None:
    log = []

    class Boom(Model):
        async def ainvoke(self, agent) -> Message:
            raise RuntimeError("kaboom")

    class Watch(Hook):
        def run_post(self, agent) -> None:
            log.append("run_post")

    try:
        asyncio.run(Agent(Boom(), hooks=[Watch()]).run())
    except RuntimeError as e:
        assert str(e) == "kaboom"                    # and it still reaches us
    else:
        raise AssertionError("a broken model should not be swallowed")
    assert log == ["run_post"]


def test_the_wiring_is_checked_every_step() -> None:
    """Hot-swapping is safe because check() runs again at the top of each step."""

    class Swap(Hook):
        def model_post(self, agent) -> None:
            agent.model = lambda agent: Message("assistant", "hi")

    agent = Agent(
        FakeModel([asks("shout", word="hi"), "done"]), {"shout": Shout()}, [Swap()]
    )
    try:
        asyncio.run(agent.run())
    except ContractError as e:
        assert "wrap it in a Model subclass" in str(e)
    else:
        raise AssertionError("a broken swap should not survive into the next step")
    assert agent.step == 1


def test_run_gives_back_the_agent() -> None:
    agent = Agent(FakeModel(["hi"]))
    assert asyncio.run(agent.run()) is agent


def test_stop_inside_run_post_stays_inside() -> None:
    """Stop is a control signal; from run_post it is too late and ignored."""

    class LateStop(Hook):
        def run_post(self, agent) -> None:
            raise Stop

    agent = asyncio.run(Agent(FakeModel(["hi"]), hooks=[LateStop()]).run())
    assert agent.messages[-1].content == "hi"        # run ended normally


def test_add_refuses_a_name_collision() -> None:
    agent = Agent(FakeModel(["hi"]), {"shout": Shout()})
    try:
        add(agent, Shout())
    except ContractError as e:
        assert "shout" in str(e)
    else:
        raise AssertionError("two tools with one name must not register")


if __name__ == "__main__":
    for test in (
        test_hook_order,
        test_deny_path,
        test_stop_from_a_hook,
        test_stop_from_a_tool,
        test_a_broken_tool_becomes_a_result,
        test_an_unknown_tool_becomes_a_result,
        test_the_model_must_return_a_message,
        test_a_hook_must_listen_for_something,
        test_a_bare_function_is_not_a_model,
        test_a_hook_can_add_a_hook,
        test_agents_do_not_share_memory,
        test_a_sync_model_and_a_sync_tool_both_work,
        test_every_result_becomes_a_tool_message,
        test_run_post_fires_after_a_crash,
        test_run_gives_back_the_agent,
        test_the_wiring_is_checked_every_step,
        test_stop_inside_run_post_stays_inside,
        test_add_refuses_a_name_collision,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_loop: all ok")
