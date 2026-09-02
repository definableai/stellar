"""check_model: the little harness that grades an adapter.

Run: uv run python tests/test_conformance.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (  # noqa: E402
    ContractError, FakeModel, Message, Model, Part, ProviderModel, ToolCall,
)
from core.conformance import check_model  # noqa: E402

SCRIPT = [
    {"text": "ok", "call": None},
    {"text": "", "call": {"id": "c1", "name": "echo", "input": {"text": "hi"}}},
    {"text": "done", "call": None},
]
ASKS = Message("assistant", "", [ToolCall("c1", "echo", {"text": "hi"})])


class Scripted(ProviderModel):
    """A tiny provider whose wire format is a dict off a list."""

    def __init__(self) -> None:
        self.turn = 0

    def encode(self, run) -> dict:
        return {"messages": len(run.messages)}

    async def send(self, run, body):
        self.turn += 1
        raw = SCRIPT[self.turn - 1]
        if raw["text"]:
            yield Part("text", raw["text"])
        if raw["call"]:
            call = raw["call"]
            yield Part("tool_call", ToolCall(call["id"], call["name"],
                                             call["input"]))


class Deaf(Scripted):
    """The same wire, but it forgets to yield the tool calls."""

    async def send(self, run, body):
        self.turn += 1
        raw = SCRIPT[self.turn - 1]
        yield Part("text", raw["text"] or "thinking about it")


class Liar(Model):
    """Rings deltas that do not add up to the message it hands back."""

    async def invoke(self, run) -> Message:
        run.emit("model.delta", Part("text", "no"), source="loop")
        return Message("assistant", "ok")


class Thinks(Scripted):
    """The same wire, with a thinking Part in front of every reply."""

    async def send(self, run, body):
        yield Part("thinking", {"thinking": "hmm"})
        async for part in super().send(run, body):
            yield part


def test_a_three_line_script_passes() -> None:
    check_model(FakeModel(["ok", ASKS, "done"]))


def test_a_two_step_provider_passes() -> None:
    model = Scripted()
    check_model(model)
    assert model.turn == 3                      # plain, tool call, final


def test_a_dropped_tool_call_names_exchange_2() -> None:
    try:
        check_model(Deaf())
    except ContractError as e:
        assert "exchange 2" in str(e)
        assert "tool_call" in str(e)
    else:
        raise AssertionError("an adapter that never yields tool_calls must fail")


def test_an_empty_first_reply_names_exchange_1() -> None:
    try:
        check_model(FakeModel([""]))
    except ContractError as e:
        assert "exchange 1" in str(e)
    else:
        raise AssertionError("a reply with no content is not a reply")


def test_deltas_that_do_not_add_up_name_exchange_1() -> None:
    try:
        check_model(Liar())
    except ContractError as e:
        assert "exchange 1" in str(e)
        assert "'no'" in str(e) and "'ok'" in str(e)   # streamed, then folded
    else:
        raise AssertionError("deltas that miss the message must fail")


def test_a_second_tool_round_names_exchange_3() -> None:
    try:
        check_model(FakeModel(["ok", ASKS, ASKS, "done"]))
    except ContractError as e:
        assert "exchange 3" in str(e)
        assert "stop after the tool result" in str(e)
    else:
        raise AssertionError("check_model allows exactly one tool round")


def test_a_yielded_type_needs_no_profile() -> None:
    check_model(Thinks())                    # no profile attribute, nothing to grade


def test_a_profile_that_admits_the_type_passes() -> None:
    model = Thinks()
    model.profile = frozenset({"thinking"})  # duck-typed: anything `in` works on
    check_model(model)


def test_a_profile_missing_a_yielded_type_names_it() -> None:
    model = Thinks()
    model.profile = frozenset({"image"})
    try:
        check_model(model)
    except ContractError as e:
        assert "exchange 1" in str(e)
        assert "the profile to admit 'thinking' parts" in str(e)
    else:
        raise AssertionError("a Part type the profile will not read back must fail")


if __name__ == "__main__":
    for test in (
        test_a_three_line_script_passes,
        test_a_two_step_provider_passes,
        test_a_dropped_tool_call_names_exchange_2,
        test_an_empty_first_reply_names_exchange_1,
        test_deltas_that_do_not_add_up_name_exchange_1,
        test_a_second_tool_round_names_exchange_3,
        test_a_yielded_type_needs_no_profile,
        test_a_profile_that_admits_the_type_passes,
        test_a_profile_missing_a_yielded_type_names_it,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_conformance: all ok")
