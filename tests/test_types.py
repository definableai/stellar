"""Vocabulary self-checks: the three data nouns, the contracts, the fold.

Run: uv run python tests/test_types.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (  # noqa: E402
    ContractError, Message, Model, Part, ProviderModel, Stop, Tool, ToolCall,
)
from core.contracts import PAYLOAD, SKELETONS, STAGES, fold  # noqa: E402


def test_part() -> None:
    assert Part("text").data is None
    assert Part("image", b"png bytes").data == b"png bytes"


def test_tool_call() -> None:
    assert ToolCall("c1", "shout").args == {}
    assert ToolCall("c1", "shout", {"word": "hi"}).args == {"word": "hi"}


def test_message() -> None:
    bare = Message("user")
    assert bare.content == []                      # "" coerces to no parts
    assert bare.tool_calls == []
    assert bare.tool_call_id is None
    assert bare.meta == {}
    assert bare.meta is not Message("user").meta   # each one gets its own
    assert bare.content is not Message("user").content

    call = ToolCall("c1", "shout")
    full = Message("assistant", [Part("text", "hi")], [call], "c1", {"raw": 1})
    assert full.content == [Part("text", "hi")]
    assert full.tool_calls == [call]
    assert full.tool_call_id == "c1"
    assert full.meta == {"raw": 1}


def test_a_str_coerces_to_one_text_part() -> None:
    said = Message("user", "hi")
    assert said.content == [Part("text", "hi")]
    assert said.text == "hi"
    assert Message("user", "hi") == Message("user", [Part("text", "hi")])


def test_text_joins_the_text_parts() -> None:
    said = Message("assistant", [Part("text", "a"), Part("thinking", {"t": "…"}),
                                 Part("text", "b")])
    assert said.text == "ab"
    assert Message("assistant").text == ""


def test_fold_concatenates_text_deltas() -> None:
    said = Message("assistant")
    for piece in ("hel", "lo"):
        fold(said, Part("text", piece))
    assert said.content == [Part("text", "hello")]


def test_fold_keeps_interrupted_text_apart() -> None:
    said = Message("assistant")
    fold(said, Part("text", "a"))
    fold(said, Part("thinking", {"t": "hm"}))      # passes through untouched
    fold(said, Part("text", "b"))
    assert said.content == [Part("text", "a"), Part("thinking", {"t": "hm"}),
                            Part("text", "b")]


def test_fold_takes_a_whole_tool_call() -> None:
    said = Message("assistant")
    call = ToolCall("c1", "shout", {"word": "hi"})
    fold(said, Part("tool_call", call))
    assert said.tool_calls == [call] and said.content == []


def test_fold_refuses_a_tool_call_that_is_not_one() -> None:
    try:
        fold(Message("assistant"), Part("tool_call", {"id": "c1"}))
    except ContractError as e:
        assert "whole ToolCall" in str(e)
    else:
        raise AssertionError("a dict must not pass for a ToolCall")


def test_fold_merges_meta_later_keys_winning() -> None:
    said = Message("assistant")
    fold(said, Part("meta", {"a": 1, "b": 1}))
    fold(said, Part("meta", {"b": 2}))
    assert said.meta == {"a": 1, "b": 2}


def test_fold_never_mutates_the_part_it_was_handed() -> None:
    delta = Part("text", "hi")
    said = Message("assistant")
    fold(said, delta)
    fold(said, Part("text", "!"))
    assert delta.data == "hi"                      # the fold grew its own copy


def test_the_contracts_are_async_only() -> None:
    missing = [
        ("invoke", lambda: asyncio.run(Model().invoke(None))),
        ("execute", lambda: asyncio.run(Tool().execute(None))),
        ("encode", lambda: ProviderModel().encode(None)),
        ("send", lambda: ProviderModel().send(None, None)),
    ]
    for name, call in missing:
        try:
            call()
        except NotImplementedError as e:
            assert str(e).startswith(name), f"{name} raised {e!s}"
            if name in ("encode", "send"):                 # template methods
                assert "class MyProvider" in str(e)        # carry the template
        else:
            raise AssertionError(f"{name} should have raised")


def test_every_stage_hands_over_something() -> None:
    assert len(STAGES) == 8
    assert set(PAYLOAD) == set(STAGES)
    assert all(stage.count(".") == 1 for stage in STAGES)
    assert PAYLOAD["tool.post"] == (ToolCall, Message)   # the only pair


def test_skeletons() -> None:
    assert set(SKELETONS) == {"model", "provider", "tool", "hook"}
    assert "Model" in SKELETONS["model"]
    assert "ProviderModel" in SKELETONS["provider"]
    assert "Tool" in SKELETONS["tool"]
    assert "@hook" in SKELETONS["hook"]
    assert "async def" in SKELETONS["model"]
    assert "async def" in SKELETONS["tool"]


def test_signals_are_exceptions() -> None:
    assert issubclass(ContractError, Exception)
    assert issubclass(Stop, Exception)


if __name__ == "__main__":
    for test in (
        test_part,
        test_tool_call,
        test_message,
        test_a_str_coerces_to_one_text_part,
        test_text_joins_the_text_parts,
        test_fold_concatenates_text_deltas,
        test_fold_keeps_interrupted_text_apart,
        test_fold_takes_a_whole_tool_call,
        test_fold_refuses_a_tool_call_that_is_not_one,
        test_fold_merges_meta_later_keys_winning,
        test_fold_never_mutates_the_part_it_was_handed,
        test_the_contracts_are_async_only,
        test_every_stage_hands_over_something,
        test_skeletons,
        test_signals_are_exceptions,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_types: all ok")
