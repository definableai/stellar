"""The control plane: eight stages, @hook, two registries, one bell.

Run: uv run python tests/test_stages.py
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.contracts import ContractError, Hooks, fire, hook  # noqa: E402
from core.types import Message, Part, ToolCall  # noqa: E402


class Run:
    """A run, as much of one as contracts.py can see: two registries, one emit."""

    def __init__(self) -> None:
        self.agent = SimpleNamespace(hooks=Hooks())
        self.hooks = Hooks()

    def emit(self, name, data=None, source=None) -> None:
        """The data plane. Nothing in here listens."""


def refuses(what, *words) -> None:
    """Call it and read the complaint: it says the words, and shows the shape."""
    try:
        what()
    except ContractError as e:
        for word in words:
            assert word in str(e), f"{word!r} is missing from: {e}"
        assert '@hook("tool.pre")' in str(e), "the skeleton did not come along"
    else:
        raise AssertionError(f"{words[0]!r} should not have been allowed")


def test_the_tag_travels_to_attach() -> None:
    heard = []

    @hook("run.pre", "run.post")
    async def note(message) -> None:
        heard.append(message.text)

    assert note.stages == ("run.pre", "run.post")     # tagged, and still a function
    run = Run()
    assert run.hooks.attach(note) is run.hooks        # attach chains
    asyncio.run(run.hooks.fire("run.pre", run, Message("user", "hi")))
    asyncio.run(run.hooks.fire("run.post", run, Message("assistant", "bye")))
    assert heard == ["hi", "bye"]


def test_a_stage_named_at_attach_beats_the_tag() -> None:
    heard = []

    async def plain(message) -> None:
        heard.append(("plain", message.text))

    @hook("run.pre")
    async def tagged(message) -> None:
        heard.append(("tagged", message.text))

    run = Run()
    run.hooks.attach(plain, "model.post").attach(tagged, "model.post")
    asyncio.run(run.hooks.fire("run.pre", run, Message("user", "hi")))
    assert heard == []                                # neither listens there anymore
    asyncio.run(run.hooks.fire("model.post", run, Message("assistant", "yo")))
    assert heard == [("plain", "yo"), ("tagged", "yo")]


def test_a_typo_never_attaches() -> None:
    heard = []

    async def note(message) -> None:
        ...

    def sync(message) -> None:
        ...

    @hook("run.pre")
    async def greedy(message, run, spare) -> None:
        ...

    @hook("run.pre", "tool.post")
    async def half(message) -> None:            # fits run.pre, not tool.post
        heard.append("half")

    refuses(lambda: hook("run.start"), "not a stage", "run.pre")
    refuses(hook, "listens for nothing")
    refuses(lambda: Hooks().attach(note, "run.start"), "not a stage")
    refuses(lambda: Hooks().attach(note), "listens for nothing")
    refuses(lambda: Hooks().attach(sync, "run.pre"), "must be async", "to_thread")
    refuses(lambda: Hooks().attach(greedy), "does not fit run.pre", "Message")
    hooks = Hooks()
    refuses(lambda: hooks.attach(half), "does not fit tool.post", "ToolCall, Message")
    asyncio.run(hooks.fire("run.pre", Run(), Message("user", "hi")))
    assert heard == []                          # a refused attach files nothing


def test_arity_decides_whether_run_comes_along() -> None:
    seen = []

    @hook("run.pre")
    async def bare(message) -> None:
        seen.append(("bare", message.text))

    @hook("run.pre")
    async def with_run(message, run) -> None:
        seen.append(("run", run))

    run = Run()
    run.hooks.attach(bare).attach(with_run)
    asyncio.run(run.hooks.fire("run.pre", run, Message("user", "hi")))
    assert seen == [("bare", "hi"), ("run", run)]


def test_tool_post_hands_over_two() -> None:
    seen = []

    @hook("tool.post")
    async def pair(call, result) -> None:
        seen.append((call.name, result.text))

    @hook("tool.post")
    async def trio(call, result, run) -> Message:
        seen.append((call.name, result.text, run.hooks is not None))
        return Message("tool", "louder")

    run = Run()
    run.hooks.attach(pair).attach(trio)
    call = ToolCall("c1", "shout")
    out = asyncio.run(run.hooks.fire("tool.post", run, call, Message("tool", "hi")))
    assert seen == [("shout", "hi"), ("shout", "hi", True)]
    assert out[0] is call and out[1].text == "louder"   # only the result is replaced


def test_none_keeps_the_payload_and_the_right_type_replaces_it() -> None:
    heard = []

    @hook("run.pre")
    async def quiet(message) -> None:
        return None

    @hook("run.pre")
    async def louder(message) -> Message:
        return Message("user", message.text + "!")

    @hook("run.pre")
    async def last(message) -> None:
        heard.append(message.text)

    run = Run()
    run.hooks.attach(quiet).attach(louder).attach(last)
    (out,) = asyncio.run(run.hooks.fire("run.pre", run, Message("user", "hi")))
    assert out.text == "hi!" and heard == ["hi!"]      # the next hook heard the new one


def test_a_delta_is_a_part_at_both_delta_stages() -> None:
    @hook("model.delta", "tool.delta")
    async def shout(part) -> Part:
        return Part(part.type, part.data.upper())

    run = Run()
    run.hooks.attach(shout)
    for stage in ("model.delta", "tool.delta"):
        (out,) = asyncio.run(run.hooks.fire(stage, run, Part("text", "hi")))
        assert out == Part("text", "HI")


def test_model_pre_takes_a_list_and_nothing_else() -> None:
    @hook("model.pre")
    async def trim(messages) -> list:
        return messages[-1:]

    @hook("model.pre")
    async def junk(messages) -> Message:
        return Message("user", "not a list")

    run = Run()
    run.hooks.attach(trim)
    (out,) = asyncio.run(run.hooks.fire(
        "model.pre", run, [Message("user", "a"), Message("user", "b")]))
    assert [m.text for m in out] == ["b"]
    run.hooks.detach(trim).attach(junk)
    refuses(lambda: asyncio.run(run.hooks.fire("model.pre", run, [])),
            "junk at model.pre returned Message", "return list")


def test_a_wrong_return_is_a_contract_error() -> None:
    @hook("model.post")
    async def stringly(message) -> str:
        return "just text"

    run = Run()
    run.hooks.attach(stringly)
    refuses(
        lambda: asyncio.run(
            run.hooks.fire("model.post", run, Message("assistant", "hi"))),
        "stringly at model.post returned str", "return Message",
    )


def test_tool_pre_replaces_the_call_or_answers_for_it() -> None:
    heard = []

    @hook("tool.pre")
    async def rename(call) -> ToolCall:
        return ToolCall(call.id, "shout", call.args)

    @hook("tool.pre")
    async def deny(call) -> str:
        heard.append(call.name)
        return "denied"

    @hook("tool.pre")
    async def never(call) -> None:
        heard.append("never")

    run = Run()
    run.agent.hooks.attach(rename).attach(deny)
    run.hooks.attach(never)
    assert asyncio.run(fire(run, "tool.pre", ToolCall("c1", "rm"))) == "denied"
    assert heard == ["shout"]        # deny saw the rename; the run's card never rang


def test_tool_pre_refuses_anything_else() -> None:
    @hook("tool.pre")
    async def bag(call) -> dict:
        return {"name": call.name}

    run = Run()
    run.hooks.attach(bag)
    refuses(lambda: asyncio.run(fire(run, "tool.pre", ToolCall("c1", "rm"))),
            "returned dict", "ToolCall, str or Message")


def test_the_agent_rings_before_the_run() -> None:
    heard = []

    @hook("run.pre")
    async def shared(message) -> Message:
        heard.append(message.text)
        return Message("user", message.text + "!")

    run = Run()
    run.agent.hooks.attach(shared)
    run.hooks.attach(shared)
    (out,) = asyncio.run(fire(run, "run.pre", Message("user", "hi")))
    assert heard == ["hi", "hi!"] and out.text == "hi!!"   # threaded through both


def test_a_denial_by_the_agent_skips_the_runs_cards() -> None:
    heard = []

    @hook("tool.pre")
    async def deny(call) -> str:
        return "denied"

    @hook("tool.pre")
    async def never(call) -> None:
        heard.append("rung")

    run = Run()
    run.agent.hooks.attach(deny)
    run.hooks.attach(never)
    out = asyncio.run(fire(run, "tool.pre", ToolCall("c1", "rm", {})))
    assert out == "denied" and heard == []      # the run's cards missed the bell


def test_detach_lands_on_the_next_bell() -> None:
    heard = []

    @hook("run.pre")
    async def first(message, run) -> None:
        heard.append("first")
        run.hooks.detach(second)

    @hook("run.pre")
    async def second(message) -> None:
        heard.append("second")

    run = Run()
    run.hooks.attach(first).attach(second)
    for _ in range(2):
        asyncio.run(run.hooks.fire("run.pre", run, Message("user", "hi")))
    assert heard == ["first", "second", "first"]      # not this bell, the next one


if __name__ == "__main__":
    for test in (
        test_the_tag_travels_to_attach,
        test_a_stage_named_at_attach_beats_the_tag,
        test_a_typo_never_attaches,
        test_arity_decides_whether_run_comes_along,
        test_tool_post_hands_over_two,
        test_none_keeps_the_payload_and_the_right_type_replaces_it,
        test_a_delta_is_a_part_at_both_delta_stages,
        test_model_pre_takes_a_list_and_nothing_else,
        test_a_wrong_return_is_a_contract_error,
        test_tool_pre_replaces_the_call_or_answers_for_it,
        test_tool_pre_refuses_anything_else,
        test_the_agent_rings_before_the_run,
        test_a_denial_by_the_agent_skips_the_runs_cards,
        test_detach_lands_on_the_next_bell,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_stages: all ok")
