"""Vocabulary self-checks: the three data nouns and the contract classes.

Run: uv run python tests/test_types.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (  # noqa: E402
    ContractError, Hook, Message, Model, Part, ProviderModel, Stop, Tool,
    ToolCall,
)
from core.contracts import EVENTS, SKELETONS  # noqa: E402


def test_part() -> None:
    assert Part("text").data is None
    assert Part("image", b"png bytes").data == b"png bytes"


def test_tool_call() -> None:
    assert ToolCall("c1", "shout").args == {}
    assert ToolCall("c1", "shout", {"word": "hi"}).args == {"word": "hi"}


def test_message() -> None:
    bare = Message("user")
    assert bare.content == ""
    assert bare.tool_calls == []
    assert bare.tool_call_id is None
    assert bare.meta == {}
    assert bare.meta is not Message("user").meta   # each one gets its own

    call = ToolCall("c1", "shout")
    full = Message("assistant", [Part("text", "hi")], [call], "c1", {"raw": 1})
    assert full.content == [Part("text", "hi")]
    assert full.tool_calls == [call]
    assert full.tool_call_id == "c1"
    assert full.meta == {"raw": 1}


def test_sync_model_is_bridged() -> None:
    class MyModel(Model):
        def invoke(self, agent) -> Message:
            return Message("assistant", "hello")

    assert asyncio.run(MyModel().ainvoke(None)).content == "hello"


def test_sync_tool_is_bridged() -> None:
    class MyTool(Tool):
        name = "shout"

        def execute(self, agent, word) -> str:
            return word.upper()

    assert asyncio.run(MyTool().aexecute(None, word="hi")) == "HI"


def test_missing_methods_say_their_name() -> None:
    def bare_provider_model():
        return asyncio.run(ProviderModel().ainvoke(None))

    missing = [
        ("invoke", lambda: Model().invoke(None)),
        ("execute", lambda: Tool().execute(None)),
        ("to_provider", bare_provider_model),
        ("send", lambda: asyncio.run(ProviderModel().send(None))),
        ("to_core", lambda: ProviderModel().to_core(None)),
    ]
    for name, call in missing:
        try:
            call()
        except NotImplementedError as e:
            assert str(e).startswith(name), f"{name} raised {e!s}"
            if name in ("to_provider", "send", "to_core"):     # template methods
                assert "class MyProvider" in str(e)            # carry the template
        else:
            raise AssertionError(f"{name} should have raised")


def test_hook_methods_are_the_events() -> None:
    hook = Hook()
    for name in EVENTS:
        assert getattr(hook, name)(None) is None


def test_skeletons() -> None:
    assert set(SKELETONS) == {"model", "provider", "tool", "hook"}
    assert "Model" in SKELETONS["model"]
    assert "ProviderModel" in SKELETONS["provider"]
    assert "Tool" in SKELETONS["tool"]
    assert "Hook" in SKELETONS["hook"]


def test_signals_are_exceptions() -> None:
    assert issubclass(ContractError, Exception)
    assert issubclass(Stop, Exception)


if __name__ == "__main__":
    test_part()
    test_tool_call()
    test_message()
    test_sync_model_is_bridged()
    test_sync_tool_is_bridged()
    test_missing_methods_say_their_name()
    test_hook_methods_are_the_events()
    test_skeletons()
    test_signals_are_exceptions()
    print("test_types: all ok")
