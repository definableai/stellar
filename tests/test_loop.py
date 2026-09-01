"""The loop's promises: order, control signals, coercion, deltas, teardown.

Run: uv run python tests/test_loop.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (  # noqa: E402
    Agent, ContractError, FakeModel, Hook, Message, Model, Part, Stop, Tool,
    ToolCall, check,
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

    async def execute(self, agent, word) -> str:
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
        [Shout()],
        [Recorder(log), Echo(log)],
    )
    asyncio.run(agent.run())
    assert [event for tag, event in log if tag == "a"] == [
        "run_pre", "model_pre", "model_post", "tool_pre", "tool_post",
        "model_pre", "model_post", "run_post",
    ]
    assert log[:2] == [("a", "run_pre"), ("b", "run_pre")]


def test_model_deltas_fire_per_part_and_fold_as_they_go() -> None:
    heard = []

    class Ear(Hook):
        def model_delta(self, agent) -> None:
            heard.append((agent.delta.type, agent.response.text))

    scripted = Message("assistant", [Part("text", "hel"), Part("text", "lo")],
                       meta={"usage": {"input_tokens": 1, "output_tokens": 1}})
    agent = Agent(FakeModel([scripted]), hooks=[Ear()])
    asyncio.run(agent.run())
    assert heard == [("text", "hel"), ("text", "hello"), ("meta", "hello")]
    assert agent.messages[-1].content == [Part("text", "hello")]   # folded
    assert agent.delta is None                     # nothing left in flight


def test_a_streaming_tool_rings_tool_delta_and_folds() -> None:
    heard = []

    class Ear(Hook):
        def tool_delta(self, agent) -> None:
            heard.append((agent.delta.type, agent.result.text))

    class Teller(Tool):
        name = "tell"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, agent):
            yield Part("text", "once upon ")
            yield Part("text", "a time")
            yield Part("meta", {"lines": 2})

    agent = Agent(FakeModel([asks("tell"), "done"]), [Teller()], [Ear()])
    asyncio.run(agent.run())
    assert heard == [("text", "once upon "), ("text", "once upon a time"),
                     ("meta", "once upon a time")]
    told = agent.messages[1]
    assert told.role == "tool" and told.tool_call_id == "c1"
    assert told.text == "once upon a time" and told.meta == {"lines": 2}


def test_a_streaming_tool_that_raises_still_answers_the_model() -> None:
    class Half(Tool):
        name = "half"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, agent):
            yield Part("text", "so far so good")
            raise ValueError("nope")

    agent = Agent(FakeModel([asks("half"), "done"]), [Half()])
    asyncio.run(agent.run())
    assert agent.messages[1].text == "error: ValueError: nope"
    assert agent.delta is None


def test_deny_path() -> None:
    ran = []

    class Watched(Shout):
        async def execute(self, agent, word) -> str:
            ran.append(word)
            return word.upper()

    class Deny(Hook):
        def tool_pre(self, agent) -> None:
            agent.result = "denied"

    agent = Agent(
        FakeModel([asks("shout", word="hi"), "done"]), [Watched()], [Deny()]
    )
    asyncio.run(agent.run())
    assert ran == []
    results = [m for m in agent.messages if m.role == "tool"]
    assert [(m.text, m.tool_call_id) for m in results] == [("denied", "c1")]


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
        async def execute(self, agent, word) -> str:
            raise Stop

    class Watch(Hook):
        def run_post(self, agent) -> None:
            log.append("run_post")

    agent = Agent(FakeModel([asks("shout", word="hi")]), [Quit()], [Watch()])
    asyncio.run(agent.run())
    assert log == ["run_post"]
    assert [m.role for m in agent.messages] == ["assistant"]


def test_a_broken_tool_becomes_a_result() -> None:
    class Boom(Shout):
        async def execute(self, agent, word) -> str:
            raise ValueError("nope")

    agent = Agent(FakeModel([asks("shout", word="hi"), "done"]), [Boom()])
    asyncio.run(agent.run())
    assert agent.messages[1].text == "error: ValueError: nope"
    assert agent.messages[2].text == "done"        # the model got another turn


def test_an_unknown_tool_becomes_a_result() -> None:
    agent = Agent(FakeModel([asks("ghost"), "done"]))
    asyncio.run(agent.run())
    assert agent.messages[1].text == "error: unknown tool: ghost"
    assert agent.messages[2].text == "done"


def test_the_model_must_return_a_message() -> None:
    class Junk(Model):
        async def invoke(self, agent) -> dict:
            return {"role": "assistant"}

    try:
        asyncio.run(Agent(Junk()).run())
    except ContractError as e:
        assert "Junk.invoke returned dict, expected Message" in str(e)
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
        assert "subclass Model and" in str(e)
    else:
        raise AssertionError("a function should not pass for a Model")


def test_a_look_alike_with_an_async_invoke_passes() -> None:
    class Duck:                              # no Model base — nobody checks ancestry
        async def invoke(self, agent) -> Message:
            return Message("assistant", "hi")

    assert asyncio.run(Agent(Duck()).run()).text == "hi"


def test_a_sync_invoke_is_refused_up_front() -> None:
    class DuckSync:
        def invoke(self, agent) -> Message:
            return Message("assistant", "hi")

    try:
        Agent(DuckSync())
    except ContractError as e:
        assert "must be async" in str(e) and "to_thread" in str(e)
    else:
        raise AssertionError("a sync invoke cannot be awaited; check must say so")


def test_a_sync_execute_is_refused_up_front() -> None:
    class OldShout(Shout):
        def execute(self, agent, word) -> str:   # the pre-async spelling
            return word.upper()

    try:
        Agent(FakeModel([]), [OldShout()])
    except ContractError as e:
        assert "must be async" in str(e) and "to_thread" in str(e)
    else:
        raise AssertionError("a sync execute cannot be awaited; check must say so")


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
    model, tools, hooks = FakeModel(["hi"]), [], []
    one, two = Agent(model, tools, hooks), Agent(model, tools, hooks)
    asyncio.run(one.run())
    one.extra["k"] = 1
    assert one.model is two.model
    assert one.tools is two.tools and one.hooks is two.hooks
    assert two.messages == [] and two.extra == {} and two.step == 0
    assert one.step == 1 and one.extra == {"k": 1}


def test_sync_code_bridges_with_one_to_thread_line() -> None:
    def my_sdk(messages) -> str:                 # the sync world
        return "done" if any(m.role == "tool" for m in messages) else "ask"

    class Bridged(Model):
        async def invoke(self, agent) -> Message:
            said = await asyncio.to_thread(my_sdk, agent.messages)
            return asks("shout", word="hi") if said == "ask" else Message(
                "assistant", said)

    class SyncTool(Shout):
        async def execute(self, agent, word) -> str:
            return await asyncio.to_thread(str.upper, word)

    agent = Agent(Bridged(), [SyncTool()])
    asyncio.run(agent.run())
    assert agent.messages[1].text == "HI"
    assert agent.messages[2].text == "done"


def test_every_result_becomes_a_tool_message() -> None:
    class Line(Tool):
        name = "line"

        async def execute(self, agent) -> str:
            return "plain"

    class Bag(Tool):
        name = "bag"

        async def execute(self, agent) -> dict:
            return {"n": 1}

    class Note(Tool):
        name = "note"

        async def execute(self, agent) -> Message:
            return Message("assistant", "mine")

    three = Message(
        "assistant", "", [ToolCall("c1", "line"), ToolCall("c2", "bag"),
                          ToolCall("c3", "note")]
    )
    agent = Agent(FakeModel([three, "done"]), [Line(), Bag(), Note()])
    asyncio.run(agent.run())
    results = [m for m in agent.messages if m.role == "tool"]
    assert [m.tool_call_id for m in results] == ["c1", "c2", "c3"]
    assert [m.text for m in results] == ["plain", '{"n": 1}', "mine"]


def test_run_post_fires_after_a_crash() -> None:
    log = []

    class Boom(Model):
        async def invoke(self, agent) -> Message:
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
        FakeModel([asks("shout", word="hi"), "done"]), [Shout()], [Swap()]
    )
    try:
        asyncio.run(agent.run())
    except ContractError as e:
        assert "subclass Model and" in str(e)
    else:
        raise AssertionError("a broken swap should not survive into the next step")
    assert agent.step == 1


def test_run_gives_back_the_last_message() -> None:
    agent = Agent(FakeModel(["hi"]))
    assert asyncio.run(agent.run()) is agent.messages[-1]


def test_run_wraps_a_str_into_a_user_message() -> None:
    agent = Agent(FakeModel(["hi"]))
    answer = asyncio.run(agent.run("hello"))
    assert agent.messages[0].role == "user"
    assert agent.messages[0].text == "hello"
    assert answer.text == "hi"


def test_run_appends_a_message_as_is() -> None:
    agent = Agent(FakeModel(["hi"]))
    ask = Message("user", "hello")
    asyncio.run(agent.run(ask))
    assert agent.messages[0] is ask


def test_stop_inside_run_post_stays_inside() -> None:
    """Stop is a control signal; from run_post it is too late and ignored."""

    class LateStop(Hook):
        def run_post(self, agent) -> None:
            raise Stop

    answer = asyncio.run(Agent(FakeModel(["hi"]), hooks=[LateStop()]).run())
    assert answer.text == "hi"                       # run ended normally


def test_check_refuses_a_name_collision() -> None:
    agent = Agent(FakeModel(["hi"]), [Shout()])
    agent.tools.append(Shout())                      # nobody guards an append
    try:
        check(agent)
    except ContractError as e:
        assert "two tools answer to 'shout'" in str(e)
    else:
        raise AssertionError("two tools with one name must not pass check")
    try:
        asyncio.run(agent.run())                     # and check() runs every step
    except ContractError:
        pass
    else:
        raise AssertionError("a dup added mid-run must not survive the next step")


if __name__ == "__main__":
    for test in (
        test_hook_order,
        test_model_deltas_fire_per_part_and_fold_as_they_go,
        test_a_streaming_tool_rings_tool_delta_and_folds,
        test_a_streaming_tool_that_raises_still_answers_the_model,
        test_deny_path,
        test_stop_from_a_hook,
        test_stop_from_a_tool,
        test_a_broken_tool_becomes_a_result,
        test_an_unknown_tool_becomes_a_result,
        test_the_model_must_return_a_message,
        test_a_hook_must_listen_for_something,
        test_a_bare_function_is_not_a_model,
        test_a_look_alike_with_an_async_invoke_passes,
        test_a_sync_invoke_is_refused_up_front,
        test_a_sync_execute_is_refused_up_front,
        test_a_hook_can_add_a_hook,
        test_agents_do_not_share_memory,
        test_sync_code_bridges_with_one_to_thread_line,
        test_every_result_becomes_a_tool_message,
        test_run_post_fires_after_a_crash,
        test_the_wiring_is_checked_every_step,
        test_run_gives_back_the_last_message,
        test_run_wraps_a_str_into_a_user_message,
        test_run_appends_a_message_as_is,
        test_stop_inside_run_post_stays_inside,
        test_check_refuses_a_name_collision,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_loop: all ok")
