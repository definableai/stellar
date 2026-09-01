"""The loop's promises: order, control signals, coercion, deltas, teardown.

Run: uv run python tests/test_loop.py
"""

import asyncio
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (  # noqa: E402
    Agent, ContractError, FakeModel, Message, Model, Part, Run, Stop, Tool,
    ToolCall, check, hook, loop,
)


def asks(name: str, cid: str = "c1", **args) -> Message:
    """An assistant line that asks for one tool."""
    return Message("assistant", "", [ToolCall(cid, name, args)])


class Shout(Tool):
    """Uppercases a word."""

    name = "shout"
    description = "Shout a word."
    parameters = {"type": "object", "properties": {"word": {"type": "string"}}}

    async def execute(self, run, word) -> str:
        return word.upper()


def test_every_stage_reaches_the_bus_in_order() -> None:
    seen = []
    agent = Agent(FakeModel([asks("shout", word="hi"), "done"]), [Shout()])
    agent.events.listen(lambda e: seen.append((e.name, e.source)))
    r = asyncio.run(agent.run("go"))
    assert [name for name, _ in seen] == [
        "run.pre", "model.pre", "model.delta", "model.post",
        "tool.pre", "tool.post",
        "model.pre", "model.delta", "model.post", "run.post",
    ]
    assert {source for _, source in seen} == {"loop"}
    assert {e.run_id for e in agent.events.log} == {r.id}
    assert [e.seq for e in agent.events.log] == list(range(10))


def test_a_hook_rings_before_the_event() -> None:
    heard = []

    @hook("model.post")
    async def card(answer) -> None:
        heard.append("hook")

    agent = Agent(FakeModel(["hi"]))
    agent.hooks.attach(card)
    agent.events.listen(lambda e: heard.append(e.name), "model.post")
    asyncio.run(agent.run("go"))
    assert heard == ["hook", "model.post"]


def test_the_agents_cards_ring_before_the_runs() -> None:
    heard = []

    @hook("run.pre")
    async def mine(message) -> None:
        heard.append("agent")

    @hook("run.pre")
    async def yours(message) -> None:
        heard.append("run")

    agent = Agent(FakeModel(["hi"]))
    agent.hooks.attach(mine)
    r = Run(agent, "r1", [Message("user", "go")])       # the run's own lane
    r.hooks.attach(yours)
    asyncio.run(loop.run(r))
    assert heard == ["agent", "run"]
    assert agent.hooks.fns["run.pre"] == [(mine, False)]   # nothing leaked back


def test_model_deltas_ring_per_part_and_the_hook_lands_before_the_fold() -> None:
    heard = []

    @hook("model.delta")
    async def louder(part) -> Part | None:
        heard.append(part.type)
        return Part("text", part.data.upper()) if part.type == "text" else None

    scripted = Message("assistant", [Part("text", "hel"), Part("text", "lo")],
                       meta={"usage": {"input_tokens": 1, "output_tokens": 1}})
    agent = Agent(FakeModel([scripted]))
    agent.hooks.attach(louder)
    r = asyncio.run(agent.run("go"))
    assert heard == ["text", "text", "meta"]
    assert r.messages[-1].content == [Part("text", "HELLO")]   # folded, transformed
    assert r.messages[-1].meta == {"usage": {"input_tokens": 1, "output_tokens": 1}}


def test_a_streaming_tool_rings_tool_delta_and_folds() -> None:
    heard = []

    @hook("tool.delta")
    async def louder(part) -> Part | None:
        heard.append(part.data)
        return Part("text", part.data.upper()) if part.type == "text" else None

    class Teller(Tool):
        name = "tell"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, run):
            yield Part("text", "once upon ")
            yield Part("text", "a time")
            yield Part("meta", {"lines": 2})

    agent = Agent(FakeModel([asks("tell"), "done"]), [Teller()])
    agent.hooks.attach(louder)
    said = []
    agent.events.listen(lambda e: said.append((e.data.type, e.source)), "tool.delta")
    r = asyncio.run(agent.run("go"))
    assert heard == ["once upon ", "a time", {"lines": 2}]
    told = r.messages[2]
    assert told.role == "tool" and told.tool_call_id == "c1"
    assert told.text == "ONCE UPON A TIME" and told.meta == {"lines": 2}
    assert said == [("text", "tell"), ("text", "tell"), ("meta", "tell")]


def test_a_streaming_tool_that_raises_still_answers_the_model() -> None:
    class Half(Tool):
        name = "half"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, run):
            yield Part("text", "so far so good")
            raise ValueError("nope")

    r = asyncio.run(Agent(FakeModel([asks("half"), "done"]), [Half()]).run("go"))
    assert r.messages[2].text == "error: ValueError: nope"


def test_a_hook_can_deny_a_tool_by_returning_its_result() -> None:
    ran = []

    class Watched(Shout):
        async def execute(self, run, word) -> str:
            ran.append(word)
            return word.upper()

    @hook("tool.pre")
    async def deny(call) -> str:
        return "denied"

    agent = Agent(FakeModel([asks("shout", word="hi"), "done"]), [Watched()])
    agent.hooks.attach(deny)
    r = asyncio.run(agent.run("go"))
    assert ran == []
    results = [m for m in r.messages if m.role == "tool"]
    assert [(m.text, m.tool_call_id) for m in results] == [("denied", "c1")]


def test_a_hook_can_replace_the_call() -> None:
    @hook("tool.pre")
    async def reroute(call) -> ToolCall:
        return ToolCall(call.id, "shout", call.args)

    agent = Agent(FakeModel([asks("ghost", word="hi"), "done"]), [Shout()])
    agent.hooks.attach(reroute)
    said = []
    agent.events.listen(lambda e: said.append(e.data.name), "tool.pre")
    r = asyncio.run(agent.run("go"))
    assert r.messages[2].text == "HI"       # the replacement is what ran
    assert said == ["shout"]                # and what the bus was told about


def test_stop_from_a_hook() -> None:
    log = []

    @hook("model.pre")
    async def quit(messages) -> None:
        raise Stop

    @hook("run.post")
    async def watch(message) -> None:
        log.append(message.text)

    agent = Agent(FakeModel(["never asked for"]))
    agent.hooks.attach(quit).attach(watch)
    r = asyncio.run(agent.run("go"))
    assert log == ["go"]                    # run.post still heard the last message
    assert [m.role for m in r.messages] == ["user"]


def test_stop_from_a_tool() -> None:
    log = []

    class Quit(Shout):
        async def execute(self, run, word) -> str:
            raise Stop

    @hook("run.post")
    async def watch(message) -> None:
        log.append("run.post")

    agent = Agent(FakeModel([asks("shout", word="hi")]), [Quit()])
    agent.hooks.attach(watch)
    r = asyncio.run(agent.run("go"))
    assert log == ["run.post"]
    assert [m.role for m in r.messages] == ["user", "assistant"]


def test_a_broken_tool_becomes_a_result() -> None:
    class Boom(Shout):
        async def execute(self, run, word) -> str:
            raise ValueError("nope")

    r = asyncio.run(
        Agent(FakeModel([asks("shout", word="hi"), "done"]), [Boom()]).run("go"))
    assert r.messages[2].text == "error: ValueError: nope"
    assert r.messages[3].text == "done"     # the model got another turn


def test_an_unknown_tool_becomes_a_result() -> None:
    r = asyncio.run(Agent(FakeModel([asks("ghost"), "done"])).run("go"))
    assert r.messages[2].text == "error: unknown tool: ghost"
    assert r.messages[3].text == "done"


def test_the_model_must_return_a_message() -> None:
    class Junk(Model):
        async def invoke(self, run) -> dict:
            return {"role": "assistant"}

    try:
        asyncio.run(Agent(Junk()).run("go"))
    except ContractError as e:
        assert "Junk.invoke returned dict, expected Message" in str(e)
    else:
        raise AssertionError("a dict should not pass for a Message")


def test_a_bare_function_is_not_a_model() -> None:
    try:
        Agent(lambda run: Message("assistant", "hi"))
    except ContractError as e:
        assert "subclass Model and" in str(e)
    else:
        raise AssertionError("a function should not pass for a Model")


def test_a_look_alike_with_an_async_invoke_passes() -> None:
    class Duck:                              # no Model base — nobody checks ancestry
        async def invoke(self, run) -> Message:
            return Message("assistant", "hi")

    assert asyncio.run(Agent(Duck()).run("go")).messages[-1].text == "hi"


def test_a_sync_invoke_is_refused_up_front() -> None:
    class DuckSync:
        def invoke(self, run) -> Message:
            return Message("assistant", "hi")

    try:
        Agent(DuckSync())
    except ContractError as e:
        assert "must be async" in str(e) and "to_thread" in str(e)
    else:
        raise AssertionError("a sync invoke cannot be awaited; check must say so")


def test_a_sync_execute_is_refused_up_front() -> None:
    class OldShout(Shout):
        def execute(self, run, word) -> str:     # the pre-async spelling
            return word.upper()

    try:
        Agent(FakeModel([]), [OldShout()])
    except ContractError as e:
        assert "must be async" in str(e) and "to_thread" in str(e)
    else:
        raise AssertionError("a sync execute cannot be awaited; check must say so")


def test_a_hook_can_add_a_hook() -> None:
    log = []

    @hook("model.pre")
    async def late(messages, run) -> None:
        log.append(f"late:{run.step}")

    @hook("run.pre")
    async def grower(message, run) -> None:
        run.agent.hooks.attach(late)

    agent = Agent(FakeModel(["done"]))
    agent.hooks.attach(grower)
    asyncio.run(agent.run("go"))
    assert log == ["late:1"]                 # attached at run.pre, heard at model.pre


def test_the_wiring_is_checked_every_step() -> None:
    """Hot-swapping is safe because check() runs again at the top of each step."""

    @hook("model.post")
    async def swap(answer, run) -> None:
        run.agent.model = lambda run: Message("assistant", "hi")

    agent = Agent(FakeModel([asks("shout", word="hi"), "done"]), [Shout()])
    agent.hooks.attach(swap)
    r = Run(agent, "r1", [Message("user", "go")])
    try:
        asyncio.run(loop.run(r))
    except ContractError as e:
        assert "subclass Model and" in str(e)
    else:
        raise AssertionError("a broken swap should not survive into the next step")
    assert r.step == 1


def test_the_tools_are_a_mapping_keyed_by_name() -> None:
    shout = Shout()
    assert Agent(FakeModel([]), [shout]).tools == {"shout": shout}
    assert Agent(FakeModel([]), {"shout": shout}).tools == {"shout": shout}
    try:
        Agent(FakeModel([]), [shout, Shout()])
    except ContractError as e:
        assert "two tools answer to 'shout'" in str(e)
    else:
        raise AssertionError("two tools with one name must not pass check")
    try:
        Agent(FakeModel([]), {"yell": shout})
    except ContractError as e:
        assert "filed under 'yell'" in str(e)
    else:
        raise AssertionError("a key that is not the name must not pass check")


def test_a_tool_swapped_in_mid_run_is_checked_too() -> None:
    agent = Agent(FakeModel([asks("shout", word="hi"), "done"]), [Shout()])
    agent.tools["yell"] = Shout()            # nobody guards an assignment
    try:
        check(agent)
    except ContractError as e:
        assert "filed under 'yell'" in str(e)
    else:
        raise AssertionError("a mis-keyed tool must not pass check")
    try:
        asyncio.run(agent.run("go"))         # and check() runs every step
    except ContractError:
        pass
    else:
        raise AssertionError("a bad key added mid-run must not survive the step")


def test_sync_code_bridges_with_one_to_thread_line() -> None:
    def my_sdk(messages) -> str:                 # the sync world
        return "done" if any(m.role == "tool" for m in messages) else "ask"

    class Bridged(Model):
        async def invoke(self, run) -> Message:
            said = await asyncio.to_thread(my_sdk, run.messages)
            return asks("shout", word="hi") if said == "ask" else Message(
                "assistant", said)

    class SyncTool(Shout):
        async def execute(self, run, word) -> str:
            return await asyncio.to_thread(str.upper, word)

    r = asyncio.run(Agent(Bridged(), [SyncTool()]).run("go"))
    assert r.messages[2].text == "HI"
    assert r.messages[3].text == "done"


def test_every_result_becomes_a_tool_message() -> None:
    class Line(Tool):
        name = "line"

        async def execute(self, run) -> str:
            return "plain"

    class Bag(Tool):
        name = "bag"

        async def execute(self, run) -> dict:
            return {"n": 1}

    class Note(Tool):
        name = "note"

        async def execute(self, run) -> Message:
            return Message("assistant", "mine")

    three = Message(
        "assistant", "", [ToolCall("c1", "line"), ToolCall("c2", "bag"),
                          ToolCall("c3", "note")]
    )
    agent = Agent(FakeModel([three, "done"]), [Line(), Bag(), Note()])
    results = [m for m in asyncio.run(agent.run("go")).messages if m.role == "tool"]
    assert [m.tool_call_id for m in results] == ["c1", "c2", "c3"]
    assert [m.text for m in results] == ["plain", '{"n": 1}', "mine"]


def test_run_post_fires_after_a_crash() -> None:
    log = []

    class Boom(Model):
        async def invoke(self, run) -> Message:
            raise RuntimeError("kaboom")

    @hook("run.post")
    async def watch(message) -> None:
        log.append("run.post")

    agent = Agent(Boom())
    agent.hooks.attach(watch)
    agent.events.listen(lambda e: log.append(e.name), "run.post")
    try:
        asyncio.run(agent.run("go"))
    except RuntimeError as e:
        assert str(e) == "kaboom"                    # and it still reaches us
    else:
        raise AssertionError("a broken model should not be swallowed")
    assert log == ["run.post", "run.post"]           # hook, then the bus


def test_stop_inside_run_post_stays_inside() -> None:
    """Stop is a control signal; from run.post it is too late and ignored."""

    @hook("run.post")
    async def late(message) -> None:
        raise Stop

    agent = Agent(FakeModel(["hi"]))
    agent.hooks.attach(late)
    r = asyncio.run(agent.run("go"))
    assert r.messages[-1].text == "hi"               # the run ended normally


def test_a_run_needs_a_prompt_or_messages() -> None:
    try:
        asyncio.run(Agent(FakeModel(["hi"])).run())
    except ContractError as e:
        assert "prompt or messages" in str(e)
    else:
        raise AssertionError("a run with nothing to say is a typo")


def test_a_str_becomes_a_user_message_and_a_message_is_kept() -> None:
    r = asyncio.run(Agent(FakeModel(["hi"])).run("hello"))
    assert (r.messages[0].role, r.messages[0].text) == ("user", "hello")
    assert r.messages[-1].text == "hi"

    ask = Message("user", "hello")
    said = [Message("system", "be brief")]
    r = asyncio.run(Agent(FakeModel(["hi"])).run(ask, messages=said))
    assert r.messages[:2] == said + [ask] and r.messages[1] is ask
    assert said == [Message("system", "be brief")]   # the caller's list is its own


def test_two_runs_share_one_agent_and_nothing_else() -> None:
    class Parrot(Model):
        async def invoke(self, run) -> Message:
            return Message("assistant", run.messages[-1].text.upper())

    agent = Agent(Parrot())

    async def both() -> list[Run]:
        return list(await asyncio.gather(agent.run("a"), agent.run("b")))

    one, two = asyncio.run(both())
    assert one.agent is two.agent is agent
    assert one.id != two.id and one.hooks is not two.hooks
    assert [m.text for m in one.messages] == ["a", "A"]
    assert [m.text for m in two.messages] == ["b", "B"]
    assert (one.step, two.step) == (1, 1)
    assert {e.run_id for e in agent.events.log} == {one.id, two.id}
    fresh = Agent(Parrot())                             # nothing is class-wide
    assert fresh.events is not agent.events and fresh.hooks is not agent.hooks
    assert fresh.extra == {} and fresh.extra is not agent.extra


def test_a_finished_run_pickles_down_to_its_checkpoint() -> None:
    agent = Agent(FakeModel(["hi"]))
    r = asyncio.run(agent.run("hello", run_id="r1"))
    r.extra["seen"] = 1
    back = pickle.loads(pickle.dumps(r))
    assert (back.id, back.step, back.extra) == ("r1", 1, {"seen": 1})
    assert [(m.role, m.text) for m in back.messages] == [
        ("user", "hello"), ("assistant", "hi")]
    assert not hasattr(back, "agent") and not hasattr(back, "hooks")


def test_a_run_can_be_resumed_under_its_own_id() -> None:
    agent = Agent(FakeModel(["hi", "again"]))
    first = asyncio.run(agent.run("hello"))
    again = asyncio.run(agent.run("more", messages=first.messages,
                                  run_id=first.id))
    assert again.id == first.id and again.step == 1        # a fresh Run, same id
    assert [m.text for m in again.messages[:3]] == ["hello", "hi", "more"]
    assert len(first.messages) == 2                        # the old list is untouched


def test_a_broken_delta_hook_stays_loud() -> None:
    class Teller(Tool):
        name = "tell"
        description = "Tells a story."
        parameters = {"type": "object", "properties": {}}

        async def execute(self, run):
            yield Part("text", "once")

    @hook("tool.delta")
    async def bag(part) -> dict:
        return {"nope": True}

    agent = Agent(FakeModel([asks("tell"), "done"]), [Teller()])
    agent.hooks.attach(bag)
    try:
        asyncio.run(agent.run("go"))
    except ContractError as e:
        assert "bag at tool.delta" in str(e)
    else:
        raise AssertionError("a broken contract must not become a tool result")


def test_the_model_pre_event_does_not_grow_with_the_run() -> None:
    agent = Agent(FakeModel([asks("shout", word="hi"), "done"]), [Shout()])
    r = asyncio.run(agent.run("go"))
    first = next(e for e in agent.events.log if e.name == "model.pre")
    assert len(first.data) == 1                # what the model saw at step one
    assert len(r.messages) == 4                # not what the run grew into


if __name__ == "__main__":
    for test in (
        test_every_stage_reaches_the_bus_in_order,
        test_a_hook_rings_before_the_event,
        test_the_agents_cards_ring_before_the_runs,
        test_model_deltas_ring_per_part_and_the_hook_lands_before_the_fold,
        test_a_streaming_tool_rings_tool_delta_and_folds,
        test_a_streaming_tool_that_raises_still_answers_the_model,
        test_a_hook_can_deny_a_tool_by_returning_its_result,
        test_a_hook_can_replace_the_call,
        test_stop_from_a_hook,
        test_stop_from_a_tool,
        test_a_broken_tool_becomes_a_result,
        test_an_unknown_tool_becomes_a_result,
        test_the_model_must_return_a_message,
        test_a_bare_function_is_not_a_model,
        test_a_look_alike_with_an_async_invoke_passes,
        test_a_sync_invoke_is_refused_up_front,
        test_a_sync_execute_is_refused_up_front,
        test_a_hook_can_add_a_hook,
        test_the_wiring_is_checked_every_step,
        test_the_tools_are_a_mapping_keyed_by_name,
        test_a_tool_swapped_in_mid_run_is_checked_too,
        test_sync_code_bridges_with_one_to_thread_line,
        test_every_result_becomes_a_tool_message,
        test_run_post_fires_after_a_crash,
        test_stop_inside_run_post_stays_inside,
        test_a_run_needs_a_prompt_or_messages,
        test_a_str_becomes_a_user_message_and_a_message_is_kept,
        test_two_runs_share_one_agent_and_nothing_else,
        test_a_finished_run_pickles_down_to_its_checkpoint,
        test_a_run_can_be_resumed_under_its_own_id,
        test_a_broken_delta_hook_stays_loud,
        test_the_model_pre_event_does_not_grow_with_the_run,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_loop: all ok")
