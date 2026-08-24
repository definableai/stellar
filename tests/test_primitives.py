"""Primitive adapter self-checks: the file jail, the line window, edit
uniqueness, and bash's process-group semantics (timeout kills the tree,
an escaped child does not wedge the call).

Run: uv run python tests/test_primitives.py
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Agent  # noqa: E402
from internal.tool_bash import MAX_OUTPUT, bash_tool  # noqa: E402
from internal.tool_fs import fs_tools  # noqa: E402


class StubLLM:
    """Satisfies the Agent's LLM contract; these tests call handlers directly."""

    async def stream(self, messages, tools, **params):
        raise AssertionError("stub LLM must not be streamed")


def build(tmp: Path) -> Agent:
    return Agent(StubLLM(), tools=[*fs_tools(tmp), bash_tool(tmp)])


def fails(fn, *a, **kw) -> str:
    """Call fn expecting a rejection; hand back the message."""
    try:
        fn(*a, **kw)
    except (TypeError, ValueError) as ex:
        return str(ex)
    raise AssertionError(f"{getattr(fn, '__name__', fn)}{a} should have raised")


async def afails(coro) -> str:
    try:
        await coro
    except ValueError as ex:
        return str(ex)
    raise AssertionError(f"{coro} should have raised")


# handlers take a ToolCallContext but none of these use it
CTX = None


def test_fs(tmp: Path) -> None:
    agent = build(tmp)
    read = agent.tools["read_file"].handler
    write = agent.tools["write_file"].handler
    edit = agent.tools["edit_file"].handler
    ls = agent.tools["list_files"].handler

    # the jail: relative escape and absolute path, on read and on write
    outside = tmp.parent / "escaped.txt"
    for path in ("../escaped.txt", str(outside)):
        assert "stay under" in fails(read, CTX, path)
        assert "stay under" in fails(write, CTX, path, "nope")
    assert not outside.exists(), "a rejected write still wrote"

    # write -> read roundtrip, parents created, receipt counts lines
    assert write(CTX, "sub/app.py", "a = 1\nb = 2\na = 1\n") == \
        "Wrote 3 lines to sub/app.py"
    assert read(CTX, "sub/app.py") == "a = 1\nb = 2\na = 1"
    assert write(CTX, "one.txt", "hi\n") == "Wrote 1 line to one.txt"

    # the window, and its "more lines" tail
    assert read(CTX, "sub/app.py", offset=2, limit=1) == "b = 2\n… 1 more lines"
    assert read(CTX, "sub/app.py", offset=2) == "b = 2\na = 1"
    assert "past the end" in fails(read, CTX, "sub/app.py", 99)
    assert "greater than 0" in fails(read, CTX, "sub/app.py", 0, 0)

    # \f is text, not a line break — cat -n and every editor agree
    write(CTX, "ff.txt", "a\fb\nc\n")
    assert read(CTX, "ff.txt") == "a\fb\nc"

    # binary is a readable error, not mojibake
    (tmp / "b.bin").write_bytes(b"\x89PNG\x00\xff")
    assert "not UTF-8" in fails(read, CTX, "b.bin")

    # edit: uniqueness, empty old, old == new, no hits — none of them touch the file
    assert "not unique" in fails(edit, CTX, "sub/app.py", "a = 1", "a = 9")
    assert "(2 matches)" in fails(edit, CTX, "sub/app.py", "a = 1", "a = 9")
    assert "must not be empty" in fails(edit, CTX, "sub/app.py", "", "#")
    assert "must differ" in fails(edit, CTX, "sub/app.py", "a = 1", "a = 1")
    assert "0 matches" in fails(edit, CTX, "sub/app.py", "nope", "x")
    assert (tmp / "sub/app.py").read_text() == "a = 1\nb = 2\na = 1\n"

    assert edit(CTX, "sub/app.py", "a = 1", "a = 9", replace_all=True) == \
        "Edited sub/app.py (2 replacements)"
    assert edit(CTX, "sub/app.py", "b = 2", "b = 3") == \
        "Edited sub/app.py (1 replacement)"
    assert (tmp / "sub/app.py").read_text() == "a = 9\nb = 3\na = 9\n"

    # line endings survive the round trip
    (tmp / "crlf.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    edit(CTX, "crlf.py", "x = 1", "x = 7")
    assert (tmp / "crlf.py").read_bytes() == b"x = 7\r\ny = 2\r\n"

    # list: sorted, root-relative, skips .git and __pycache__ anywhere in the path
    (tmp / ".git").mkdir()
    (tmp / ".git" / "config").write_text("x")
    (tmp / "sub" / "__pycache__").mkdir()
    (tmp / "sub" / "__pycache__" / "app.pyc").write_text("x")
    assert ls(CTX).splitlines() == [
        "b.bin", "crlf.py", "ff.txt", "one.txt", "sub/app.py"]
    assert ls(CTX, "sub") == "sub/app.py"
    assert ls(CTX, ".", "*.py") == "crlf.py"
    assert ls(CTX, "sub", "nothing*") == "(no files)"
    assert "not a directory" in fails(ls, CTX, "crlf.py")

    # a glob escapes the jail too: judged by where its hits land
    assert ls(CTX, ".", "../*") == "(no files)"
    assert "bad glob" in fails(ls, CTX, ".", "/etc/*")

    # cap, with the tail line naming the remainder
    (tmp / "many").mkdir()
    for i in range(505):
        (tmp / "many" / f"f{i:03}.txt").write_text("")
    listed = ls(CTX, "many").splitlines()
    assert len(listed) == 501 and listed[0] == "many/f000.txt"
    assert listed[499] == "many/f499.txt" and listed[500] == "… 5 more"


async def test_bash(tmp: Path) -> None:
    bash = build(tmp).tools["bash"].handler
    clock = asyncio.get_running_loop().time

    assert await bash(CTX, "printf hello") == "hello"
    assert "(exit code 3)" in await bash(CTX, "exit 3")
    assert await bash(CTX, "true") == "(no output)"
    assert "must not be empty" in await afails(bash(CTX, "  "))
    assert "greater than 0" in await afails(bash(CTX, "true", 0))

    # cwd is the root: a bare relative path lands there
    await bash(CTX, "printf inside > made-here.txt")
    assert (tmp / "made-here.txt").read_text() == "inside"

    # timeout: keeps the output already printed, and kills the whole tree
    out = await bash(CTX, "printf early; (sleep 1.5; touch canary) & wait", timeout=0.3)
    assert out.startswith("early") and "timed out after 0.3s" in out, out
    await asyncio.sleep(2)
    assert not (tmp / "canary").exists(), "child survived the timeout kill"

    # a process that left the group still holds stdout: return anyway, don't wedge
    t0 = clock()
    out = await bash(CTX, f'{sys.executable} -c "import subprocess,time;'
                          "subprocess.Popen(['sleep','5'],start_new_session=True);"
                          'time.sleep(5)"', timeout=0.2)
    assert "timed out" in out and clock() - t0 < 2.5, (out, clock() - t0)

    # output cap, with the truncation note
    out = await bash(CTX, "head -c 40000 /dev/zero | tr '\\0' x")
    assert out.startswith("x" * 100) and out.endswith("… truncated (10000 more chars)")
    assert len(out) == MAX_OUTPUT + len("\n… truncated (10000 more chars)")


def test_factory_contract(tmp: Path) -> None:
    # root is a required argument, not a guess
    for factory in (fs_tools, bash_tool):
        assert "root" in fails(factory)

    # the root is created if it is missing
    made = tmp / "brand" / "new"
    fs_tools(made)
    assert made.is_dir()

    agent = build(tmp)
    assert sorted(agent.tools) == [
        "bash", "edit_file", "list_files", "read_file", "write_file"]

    # a real model call goes through the schema: every declared property has
    # to be a kwarg the handler can actually take
    for name, t in agent.tools.items():
        takes = set(inspect.signature(t.handler).parameters) - {"cctx"}
        assert set(t.spec.parameters["properties"]) <= takes, name
        assert set(t.spec.parameters.get("required", ())) <= takes, name
        assert t.spec.description, name


async def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        test_fs(Path(d) / "fs")
        await test_bash(Path(d) / "bash")
        test_factory_contract(Path(d) / "contract")
    print("test_primitives: all ok")


if __name__ == "__main__":
    asyncio.run(main())
