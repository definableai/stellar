"""check_model: the little harness that grades an adapter.

Run: uv run python tests/test_conformance.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (  # noqa: E402
    ContractError, FakeModel, Message, ProviderModel, ToolCall, check_model,
)

SCRIPT = [
    {"text": "ok", "call": None},
    {"text": "", "call": {"id": "c1", "name": "echo", "input": {"text": "hi"}}},
    {"text": "done", "call": None},
]


class Scripted(ProviderModel):
    """A tiny provider whose wire format is a dict off a list."""

    def __init__(self) -> None:
        self.turn = 0

    def to_provider(self, agent) -> dict:
        return {"messages": len(agent.messages)}

    async def send(self, body) -> dict:
        self.turn += 1
        return SCRIPT[self.turn - 1]

    def to_core(self, raw) -> Message:
        call = raw["call"]
        return Message(
            "assistant",
            raw["text"],
            [ToolCall(call["id"], call["name"], call["input"])] if call else [],
        )


class Deaf(Scripted):
    """The same wire, but it forgets to map tool_calls."""

    def to_core(self, raw) -> Message:
        return Message("assistant", raw["text"] or "thinking about it")


def test_a_three_line_script_passes() -> None:
    check_model(FakeModel([
        "ok",
        Message("assistant", "", [ToolCall("c1", "echo", {"text": "hi"})]),
        "done",
    ]))


def test_a_three_step_provider_passes() -> None:
    model = Scripted()
    check_model(model)
    assert model.turn == 3                      # plain, tool call, final


def test_a_dropped_tool_call_names_exchange_2() -> None:
    try:
        check_model(Deaf())
    except ContractError as e:
        assert "exchange 2" in str(e)
        assert "tool_calls" in str(e)
    else:
        raise AssertionError("an adapter that never maps tool_calls must fail")


def test_an_empty_first_reply_names_exchange_1() -> None:
    try:
        check_model(FakeModel([""]))
    except ContractError as e:
        assert "exchange 1" in str(e)
    else:
        raise AssertionError("a reply with no content is not a reply")


if __name__ == "__main__":
    for test in (
        test_a_three_line_script_passes,
        test_a_three_step_provider_passes,
        test_a_dropped_tool_call_names_exchange_2,
        test_an_empty_first_reply_names_exchange_1,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_conformance: all ok")
