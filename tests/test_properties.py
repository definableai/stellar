"""Property tests (hypothesis): the invariants that simple examples miss.

Run: uv run python tests/test_properties.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hypothesis import given, settings, strategies as st  # noqa: E402

from core import (  # noqa: E402
    Message, ReplyBuilder, Session, ToolCall, ToolResult,
    validate_args,
)

texts = st.text(max_size=80)
names = st.text(alphabet="abcdefgh_", min_size=1, max_size=10)
metas = st.dictionaries(names, st.integers() | texts, max_size=3)
argdicts = st.dictionaries(names, st.integers() | texts | st.booleans(),
                           max_size=4)


@st.composite
def transcripts(draw) -> list[Message]:
    """Valid 'closed' transcripts: every tool call is answered."""
    msgs: list[Message] = []
    for turn in range(draw(st.integers(0, 5))):
        kind = draw(st.sampled_from(["user", "assistant", "tools"]))
        if kind == "user":
            msgs.append(Message(role="user", content=draw(texts)))
        elif kind == "assistant":
            msgs.append(Message(role="assistant", content=draw(texts),
                                meta=draw(metas)))
        else:
            calls = [ToolCall(f"c{turn}_{i}", draw(names), draw(argdicts))
                     for i in range(draw(st.integers(1, 3)))]
            msgs.append(Message(role="assistant", tool_calls=calls))
            for c in calls:
                msgs.append(Message(role="tool", tool_result=ToolResult(
                    c.id, c.name, draw(texts), draw(st.booleans()))))
    return msgs


def _dicts(msgs: list[Message]) -> list[dict]:
    return [m.to_dict() for m in msgs]


@settings(max_examples=60)
@given(transcripts())
def test_session_round_trip(msgs: list[Message]) -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.jsonl"
        with Session(p) as s:
            for m in msgs:
                s.append(m)
        loaded = Session.load(p).messages()
    assert _dicts(loaded) == _dicts(msgs)   # closed transcript: repair adds nothing


@settings(max_examples=60)
@given(transcripts(), st.data())
def test_crash_cut_always_recovers(msgs: list[Message], data) -> None:
    """Truncate the file at ANY byte past the header: load() must succeed,
    keep an exact prefix, and leave no tool call unanswered."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.jsonl"
        with Session(p) as s:
            for m in msgs:
                s.append(m)
        raw = p.read_bytes()
        header_end = raw.index(b"\n") + 1
        cut = data.draw(st.integers(header_end, len(raw)))
        p.write_bytes(raw[:cut])

        loaded = Session.load(p).messages()
        real = [m for m in loaded if not m.meta.get("repaired")]
        assert _dicts(real) == _dicts(msgs[:len(real)])     # exact prefix
        for extra in loaded[len(real):]:                    # closers only
            assert extra.role == "tool" and extra.tool_result.is_error
        answered = {m.tool_result.call_id for m in loaded if m.role == "tool"}
        for m in loaded:                                    # nothing dangling
            for c in m.tool_calls:
                assert c.id in answered
        # and the repaired log reloads clean
        assert _dicts(Session.load(p).messages()) == _dicts(loaded)


@settings(max_examples=100)
@given(argdicts, st.data())
def test_builder_reassembles_any_fragmentation(args: dict, data) -> None:
    s = json.dumps(args)
    b = ReplyBuilder()
    b.tool_call(0, id="c1", name="t")
    i = 0
    while i < len(s):
        j = data.draw(st.integers(i + 1, len(s)))
        b.tool_args(0, s[i:j])
        i = j
    reply = b.reply()
    assert reply.message.tool_calls[0].arguments == args
    assert "invalid_tool_args" not in reply.message.meta


@settings(max_examples=100)
@given(st.dictionaries(st.sampled_from(["path", "limit", "ok"]),
                       st.integers() | texts | st.booleans(), max_size=3))
def test_validate_args_never_crashes(args: dict) -> None:
    schema = {"type": "object",
              "properties": {"path": {"type": "string"},
                             "limit": {"type": "integer"},
                             "ok": {"type": "boolean"}},
              "required": ["path"]}
    problems = validate_args(schema, args)   # list of strings, never raises
    assert all(isinstance(p, str) for p in problems)
    conforms = (isinstance(args.get("path"), str)
                and (("limit" not in args) or (isinstance(args["limit"], int)
                                               and not isinstance(args["limit"], bool)))
                and (("ok" not in args) or isinstance(args["ok"], bool)))
    assert (problems == []) == conforms


if __name__ == "__main__":
    test_session_round_trip()
    test_crash_cut_always_recovers()
    test_builder_reassembles_any_fragmentation()
    test_validate_args_never_crashes()
    print("property tests: all ok")
