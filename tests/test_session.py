"""Session self-checks: round-trip, repair, corruption handling.

Run: uv run python tests/test_session.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Message, Session, SessionError, ToolCall, ToolResult  # noqa: E402


def test_in_memory() -> None:
    s = Session()
    s.append(Message(role="user", content="hi"))
    s.append(Message(role="assistant", content="hello"))
    assert len(s) == 2
    assert [m.content for m in s.messages()] == ["hi", "hello"]
    assert s.messages() is not s.messages()  # fresh list each call


def test_file_round_trip(dir: Path) -> None:
    path = dir / "a.jsonl"
    with Session(path) as s:
        sid = s.id
        s.append(Message(role="user", content="hi"))
        s.append(Message(
            role="assistant", content="calling",
            tool_calls=[ToolCall("c1", "echo", {"x": "1"})]))
        s.append(Message(role="tool", tool_result=ToolResult("c1", "echo", "1")))

    s2 = Session.load(path)
    assert s2.id == sid
    assert len(s2) == 3
    assert s2.messages()[1].tool_calls[0].arguments == {"x": "1"}
    assert s2.messages()[2].tool_result.content == "1"

    s2.append(Message(role="assistant", content="done"))  # continues after load
    s2.close()
    assert [m.content for m in Session.load(path).messages()][-1] == "done"


def test_repair(dir: Path) -> None:
    path = dir / "b.jsonl"
    with Session(path) as s:
        s.append(Message(role="user", content="go"))
        s.append(Message(
            role="assistant",
            tool_calls=[ToolCall("c1", "echo", {}), ToolCall("c2", "echo", {})]))
        s.append(Message(role="tool", tool_result=ToolResult("c1", "echo", "ok")))
        # crash before c2's result

    s2 = Session.load(path)
    assert len(s2) == 4
    fixed = s2.messages()[-1]
    assert fixed.meta.get("repaired") and fixed.tool_result.call_id == "c2"
    assert fixed.tool_result.is_error
    s2.close()
    # repair is durable and idempotent: a second load adds nothing
    assert len(Session.load(path)) == 4


def test_torn_tail(dir: Path) -> None:
    path = dir / "c.jsonl"
    with Session(path) as s:
        s.append(Message(role="user", content="hi"))
    with path.open("a") as f:
        f.write('{"seq": 1, "type": "message", "data": {"role": "assis')  # torn write
    s2 = Session.load(path)
    assert len(s2) == 1  # tail dropped, rest intact
    s2.append(Message(role="assistant", content="ok"))  # must not corrupt the file
    s2.close()
    assert [m.content for m in Session.load(path).messages()] == ["hi", "ok"]


def test_unicode_line_separators(dir: Path) -> None:
    path = dir / "u.jsonl"
    # LS, PS, NEL split str.splitlines() but must not split session records
    tricky = "a\u2028b\u2029c\u0085d\nnewline"
    with Session(path) as s:
        s.append(Message(role="user", content=tricky))
        s.append(Message(role="assistant", content="ok"))
    assert [m.content for m in Session.load(path).messages()] == [tricky, "ok"]


def test_torn_mid_codepoint(dir: Path) -> None:
    path = dir / "cp.jsonl"
    with Session(path) as s:
        s.append(Message(role="user", content="hi"))
        s.append(Message(role="assistant", content="emoji \U0001f600 end"))
    size = path.stat().st_size
    with path.open("r+b") as f:
        f.truncate(size - 8)  # torn inside the second record's bytes
    s2 = Session.load(path)
    assert [m.content for m in s2.messages()] == ["hi"]
    s2.append(Message(role="assistant", content="ok"))
    s2.close()
    assert [m.content for m in Session.load(path).messages()] == ["hi", "ok"]


def test_append_is_atomic(dir: Path) -> None:
    # closed session: loud, nothing lost silently
    path = dir / "closed.jsonl"
    s = Session(path)
    s.append(Message(role="user", content="hi"))
    s.close()
    try:
        s.append(Message(role="user", content="lost"))
        raise AssertionError("append on closed session must raise")
    except SessionError:
        pass
    assert len(s) == 1

    # non-serializable content: neither memory nor disk mutated
    path2 = dir / "atomic.jsonl"
    with Session(path2) as s2:
        s2.append(Message(role="user", content="hi"))
        try:
            s2.append(Message(role="tool", tool_result=ToolResult("c1", "t", object())))
            raise AssertionError("non-serializable append must raise")
        except TypeError:
            pass
        s2.append(Message(role="assistant", content="ok"))
    assert [m.content for m in Session.load(path2).messages()] == ["hi", "ok"]


def test_corruption_is_loud(dir: Path) -> None:
    # corrupt middle line
    path = dir / "d.jsonl"
    with Session(path) as s:
        s.append(Message(role="user", content="a"))
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], "garbage", lines[1]]) + "\n")
    try:
        Session.load(path)
        raise AssertionError("corrupt middle line must raise")
    except SessionError:
        pass

    # bad header / version
    path2 = dir / "e.jsonl"
    path2.write_text('{"v": 99, "id": "x"}\n')
    try:
        Session.load(path2)
        raise AssertionError("bad version must raise")
    except SessionError:
        pass

    # seq gap
    path3 = dir / "f.jsonl"
    path3.write_text('{"v": 0, "id": "x"}\n'
                     '{"seq": 5, "type": "message", "data": {"role": "user"}}\n')
    try:
        Session.load(path3)
        raise AssertionError("seq gap must raise")
    except SessionError:
        pass

    # constructor refuses an existing file
    try:
        Session(path)
        raise AssertionError("existing file must raise")
    except SessionError:
        pass


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        test_in_memory()
        test_file_round_trip(d)
        test_repair(d)
        test_torn_tail(d)
        test_unicode_line_separators(d)
        test_torn_mid_codepoint(d)
        test_append_is_atomic(d)
        test_corruption_is_loud(d)
    print("test_session: all ok")
