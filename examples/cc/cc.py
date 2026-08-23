"""Claude Code, rebuilt on this core — same prompt, same tool schemas, GPT brain.

``cc.json`` is a captured Claude Code request. Everything the model sees
is read straight out of it (system blocks, tool names, descriptions and
JSON Schemas) — nothing is retyped, so the replica cannot drift from the
capture. Only the *handlers* are ours, and only for the tools that do
something local:

    Read / Write / Edit / Bash

The rest of cc.json's tools (Agent, Artifact, Skill, Workflow, ToolSearch,
...) are harness features that do not exist here; they are left out rather
than faked, and calls to them come back as UnknownTool for the model to
recover from. The system prompt still describes them — that is what "same
prompt" means.

Where the handlers knowingly fall short of what their (verbatim) descriptions
promise: Bash is NOT sandboxed — ``dangerouslyDisableSandbox`` is accepted and
ignored, every command runs with this process's full permissions — its cwd
does not persist between calls, and ``run_in_background`` is refused; Read is
text-only, so ``pages`` is ignored and binary files error out.

    export OPENAI_API_KEY=...
    uv run python -m examples.cc.cc "add a healthcheck route to app.py"
    uv run python -m examples.cc.cc --session cc.jsonl "big refactor task"
    uv run python -m examples.cc.cc --session cc.jsonl      # REPL (Worker)
    uv run python -m examples.cc.cc selfcheck               # no network

With ``--session`` the whole conversation is durable: Ctrl-C stops the
run gracefully (partial reply persisted), kill -9 loses at most the
step in flight — rerun the same command and it repairs the log and
continues where it died. That one flag is the core's whole pitch.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

from internal.llm.openai import OpenAIResponsesLLM
from internal.worker import Worker
from core import (Agent, LLMReply, Message, Session, StepKind, StepPhase,
                  Tool, ToolCall, ToolSpec)

CC = json.loads((Path(__file__).parent / "cc.json").read_text())
# [0] is an x-anthropic-billing header, not prompt text.
SYSTEM = "\n\n".join(b["text"] for b in CC["system"][1:])
SCHEMAS = {t["name"]: t for t in CC["tools"]}

MAX_LINES = 2000        # Read default, per the tool description
MAX_OUTPUT = 30_000     # chars of Bash output kept before truncation
TOOLS: list[Tool] = []


def handler(name: str, parallel_safe: bool = True):
    """Bind a function to cc.json's spec for ``name`` — name, description
    and schema all come from the capture."""

    def wrap(fn):
        spec = SCHEMAS[name]
        TOOLS.append(Tool(
            spec=ToolSpec(
                name=name,
                description=spec["description"],
                # $schema is an Anthropic-side annotation; other providers reject it.
                parameters={k: v for k, v in spec["input_schema"].items()
                            if k != "$schema"},
            ),
            handler=fn,
            parallel_safe=parallel_safe,
        ))
        return fn

    return wrap


# Files Read so far, by resolved path. Edit/Write refuse to touch anything else.
# ponytail: process-global, so it is one conversation per process — the real
# harness scopes this per session. Move onto ctx.run.state once a run's state
# is threaded across turns.
SEEN: set[str] = set()


def _key(file_path: str) -> str:
    """`sub/../a.py`, `a.py` and a symlink to it are the same file."""
    return str(Path(file_path).resolve())


def _slurp(file_path: str) -> str:
    """Read as text, preserving line endings. Binary is an error, not mojibake."""
    try:
        with open(file_path, encoding="utf-8", newline="") as f:
            return f.read()
    except UnicodeDecodeError:
        raise ValueError(f"{file_path} is not UTF-8 text — this Read is text-only") from None


@handler("Read")
def read(ctx, file_path: str, offset: int = 0, limit: int = MAX_LINES,
         pages: str | None = None) -> str:
    lines = _slurp(file_path).split("\n")   # not splitlines(): \f, \v, U+2028 are
    if lines[-1] == "":                     # not line breaks to cat -n or any editor
        lines.pop()                         # a trailing newline terminates, not adds
    start = max(offset - 1, 0) if offset else 0
    if lines and start >= len(lines):
        raise ValueError(f"offset {offset} is past the end of the file ({len(lines)} lines)")
    if limit <= 0:
        raise ValueError("limit must be greater than 0")
    SEEN.add(_key(file_path))               # only a Read that succeeded opens the gate
    if not lines:
        return "<system-reminder>File exists but is empty.</system-reminder>"
    window = lines[start:start + min(limit, MAX_LINES)]
    body = "\n".join(f"{start + i + 1:>6}\t{ln}" for i, ln in enumerate(window))
    tail = len(lines) - (start + len(window))
    return body + (f"\n… {tail} more lines" if tail > 0 else "")
    # ponytail: text only, `pages` ignored. Images/PDF/notebooks need a
    # multimodal ToolResult; add when the core carries non-text content blocks.


@handler("Write", parallel_safe=False)
def write(ctx, file_path: str, content: str) -> str:
    p = Path(file_path)
    if p.exists() and _key(file_path) not in SEEN:
        raise ValueError(f"{file_path} exists and was not read this session — Read it first")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    SEEN.add(_key(file_path))
    n = len(content.splitlines())
    return f"Wrote {n} line{'s' * (n != 1)} to {file_path}"


@handler("Edit", parallel_safe=False)
def edit(ctx, file_path: str, old_string: str, new_string: str,
         replace_all: bool = False) -> str:
    if _key(file_path) not in SEEN:
        raise ValueError(f"{file_path} was not read this session — Read it first")
    if not old_string:
        raise ValueError("old_string must not be empty")
    if old_string == new_string:
        raise ValueError("new_string must differ from old_string")
    src = _slurp(file_path)
    hits = src.count(old_string)
    if hits == 0:
        raise ValueError("old_string not found in file")
    if hits > 1 and not replace_all:
        raise ValueError(f"old_string is not unique ({hits} matches) — "
                         "add surrounding context or pass replace_all")
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        f.write(src.replace(old_string, new_string, -1 if replace_all else 1))
    return f"Edited {file_path} ({hits if replace_all else 1} replacement(s))"


@handler("Bash", parallel_safe=False)
async def bash(ctx, command: str, timeout: float = 120_000,
               description: str | None = None, run_in_background: bool = False,
               dangerouslyDisableSandbox: bool = False) -> str:
    if run_in_background:
        raise ValueError("run_in_background is not supported here — run it in the foreground")
    proc = await asyncio.create_subprocess_shell(
        command, cwd=os.getcwd(), start_new_session=True,  # own group, so we can kill children
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    # ponytail: one subprocess per call — cwd/env do not persist across calls,
    # unlike the real Bash tool. A long-lived shell + sentinel framing if it matters.
    chunks: list[bytes] = []

    async def drain() -> None:
        while chunk := await proc.stdout.read(65536):
            chunks.append(chunk)

    reader = asyncio.ensure_future(drain())
    note = ""
    try:
        await asyncio.wait_for(asyncio.shield(reader), min(timeout, 600_000) / 1000)
    except asyncio.TimeoutError:
        try:
            os.killpg(proc.pid, signal.SIGKILL)   # the whole tree, not just sh
        except ProcessLookupError:
            pass
        note = f"\n(command timed out after {timeout:.0f}ms and was killed)"
        await asyncio.wait({reader}, timeout=1.0)  # a process that escaped the group
        reader.cancel()                            # can still hold stdout — don't wait
        # ponytail: private, but asyncio exposes no public way to drop our
        # end of a pipe an escapee still holds open. Leaks an fd otherwise.
        proc._transport.close()
    await proc.wait()
    text = b"".join(chunks).decode(errors="replace")   # keep whatever it printed first
    if len(text) > MAX_OUTPUT:
        text = text[:MAX_OUTPUT] + f"\n… truncated ({len(text) - MAX_OUTPUT} more chars)"
    if proc.returncode and not note:
        text += f"\n(exit code {proc.returncode})"
    return (text + note) or "(no output)"


class CliTracer:
    """Streams the run to the terminal, Claude-Code style. A tracer, so
    one-shot runs and Worker turns render through the same code."""

    def __init__(self) -> None:
        self._channel: str | None = None

    async def on_event(self, e: Any) -> None:
        p = e.payload
        if e.kind is StepKind.TEXT and e.phase is StepPhase.DELTA:
            if p["channel"] != self._channel:
                self._channel = p["channel"]
                print(f"\n\n[{self._channel}] ", end="", flush=True)
            print(p["text"], end="", flush=True)
            return
        self._channel = None
        if e.kind is StepKind.TOOL and e.phase is StepPhase.START:
            print(f"\n\n● {p['name']}({json.dumps(p['arguments'])[:120]})", flush=True)
        elif e.kind is StepKind.TOOL and e.phase is StepPhase.END:
            head = str(p["result"]).splitlines()[:3]
            print("  ⎿ " + " / ".join(head)[:200], flush=True)
        elif e.kind is StepKind.RUN and e.phase is StepPhase.END:
            print(f"\n\n[{p['status']}]", flush=True)


def build(model: str = "gpt-5.6-sol", tracers: Any = (), **params: Any) -> Agent:
    return Agent(
        llm=OpenAIResponsesLLM(model=model, reasoning={"effort": "high",
                                                       "summary": "auto"}),
        tools=TOOLS,
        system=SYSTEM,
        tracers=tracers,
        max_steps=40,
        parallel_tools=True,
        params={"max_output_tokens": CC["max_tokens"], **params},
    )


def _open_session(path: str | None) -> Session:
    if path is None:
        return Session()                        # in-memory: no resume
    if Path(path).exists():
        return Session.load(path)               # repair + continue
    return Session(path)


async def main(prompt: str, session_path: str | None = None) -> None:
    with _open_session(session_path) as s:
        handle = build(tracers=[CliTracer()]).run(input=prompt, session=s)
        hits = 0

        def on_int() -> None:
            nonlocal hits
            hits += 1
            if hits == 1:
                print("\n(stopping — Ctrl-C again to force)", flush=True)
                handle.stop("user")
            else:
                sys.exit(130)

        asyncio.get_running_loop().add_signal_handler(signal.SIGINT, on_int)
        await handle


async def repl(session_path: str) -> None:
    """A long-lived Worker on a durable session: Claude Code as a REPL."""
    with _open_session(session_path) as s:
        worker = Worker(build(tracers=[CliTracer()]), s)
        serve = asyncio.create_task(worker.serve())
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT,
                                lambda: worker.stop_turn("user"))
        print(f"[{s.id}] {len(s)} prior messages — Ctrl-D exits, "
              "Ctrl-C stops the current turn")
        while True:
            try:
                line = await loop.run_in_executor(None, input, "\n> ")
            except EOFError:
                break
            if line.strip():
                worker.send(line)
                await worker.idle()
        await worker.close()
        await serve


async def _selfcheck() -> None:
    """Every handler, plus the read-before-write gate, over a fake LLM."""
    import tempfile

    class _FakeLLM:
        def __init__(self, script): self.script = list(script)
        async def stream(self, messages, tools, **params):
            yield self.script.pop(0)

    assert {t.spec.name for t in TOOLS} == {"Read", "Write", "Edit", "Bash"}
    assert SYSTEM.startswith("You are Claude Code")
    assert SCHEMAS["Read"]["input_schema"]["required"] == ["file_path"]
    assert TOOLS[0].spec.parameters == {
        k: v for k, v in SCHEMAS["Read"]["input_schema"].items() if k != "$schema"}
    assert "$schema" not in TOOLS[0].spec.parameters

    def fails(fn, *a, **kw) -> str:
        try:
            fn(*a, **kw)
        except ValueError as ex:
            return str(ex)
        raise AssertionError(f"{fn.__name__}{a} should have raised")

    ctx = None  # handlers take a ToolCallContext but none of them use it
    with tempfile.TemporaryDirectory() as d:
        target = f"{d}/app.py"
        Path(target).write_text("a = 1\nb = 2\na = 1\n")

        # gates fire before anything has been Read
        assert "not read" in fails(edit, ctx, target, "a = 1", "a = 9")
        assert "not read" in fails(write, ctx, target, "clobber")
        assert Path(target).read_text() == "a = 1\nb = 2\na = 1\n"   # untouched

        # end-to-end through the loop: the LLM asks for Read, gets the file back
        agent = Agent(llm=_FakeLLM([
            LLMReply(Message(role="assistant", tool_calls=[
                ToolCall(id="c1", name="Read", arguments={"file_path": target})])),
            LLMReply(Message(role="assistant", content="done")),
        ]), tools=TOOLS, system=SYSTEM)
        result = await agent.run(input="read it")
        assert result.status == "completed" and result.output == "done"
        tool_msg = next(m for m in result.messages if m.role == "tool")
        assert not tool_msg.tool_result.is_error
        assert "\ta = 1" in tool_msg.tool_result.content

        assert read(ctx, target).splitlines()[0] == "     1\ta = 1"
        assert read(ctx, target, offset=2, limit=1) == "     2\tb = 2\n… 1 more lines"
        assert "past the end" in fails(read, ctx, target, offset=99)
        assert "greater than 0" in fails(read, ctx, target, limit=0)

        # a Read that failed must NOT open the write gate
        Path(f"{d}/unseen.py").write_text("k = 1\n")
        assert "past the end" in fails(read, ctx, f"{d}/unseen.py", offset=9)
        assert "not read" in fails(edit, ctx, f"{d}/unseen.py", "k = 1", "k = 2")

        # cat -n counts \n only: \f and U+2028 are text, not line breaks
        Path(f"{d}/ff.txt").write_text("a\fb\nc\n")
        assert read(ctx, f"{d}/ff.txt") == "     1\ta\fb\n     2\tc"

        assert "not unique" in fails(edit, ctx, target, "a = 1", "a = 9")
        assert "must not be empty" in fails(edit, ctx, target, "", "#")
        assert "must differ" in fails(edit, ctx, target, "a = 1", "a = 1")
        assert "not found" in fails(edit, ctx, target, "nope", "x")
        assert Path(target).read_text() == "a = 1\nb = 2\na = 1\n"   # still untouched

        edit(ctx, target, "a = 1", "a = 9", replace_all=True)
        edit(ctx, target, "b = 2", "b = 3")
        assert Path(target).read_text() == "a = 9\nb = 3\na = 9\n"

        # a Read of one spelling authorises an Edit of any other
        crlf = f"{d}/sub/../crlf.py"
        Path(f"{d}/crlf.py").write_bytes(b"x = 1\r\ny = 2\r\n")
        Path(f"{d}/sub").mkdir()
        read(ctx, f"{d}/crlf.py")
        edit(ctx, crlf, "x = 1", "x = 7")
        assert Path(f"{d}/crlf.py").read_bytes() == b"x = 7\r\ny = 2\r\n"  # endings kept

        Path(f"{d}/b.bin").write_bytes(b"\x89PNG\x00\xff")
        assert "not UTF-8" in fails(read, ctx, f"{d}/b.bin")
        assert write(ctx, f"{d}/new/x.txt", "hi\n") == f"Wrote 1 line to {d}/new/x.txt"
        assert Path(f"{d}/new/x.txt").read_text() == "hi\n"

        assert (await bash(ctx, "printf hello")) == "hello"
        assert "exit code 3" in await bash(ctx, "exit 3")
        # timeout: keeps the output already printed, and kills the whole tree
        canary = f"{d}/canary"
        out = await bash(ctx, f"printf early; (sleep 1.5; touch {canary}) &  wait",
                         timeout=300)
        assert out.startswith("early") and "timed out" in out, out
        await asyncio.sleep(2)
        assert not Path(canary).exists(), "child survived the timeout kill"

        # a process that left the group still holds stdout: return anyway, don't wedge
        clock = asyncio.get_running_loop().time
        t0 = clock()
        out = await bash(ctx, "python3 -c \"import subprocess,time;"
                              "subprocess.Popen(['sleep','5'],start_new_session=True);"
                              "time.sleep(5)\"", timeout=200)
        assert "timed out" in out and clock() - t0 < 2.5, (out, clock() - t0)

        # durable session: "crash" after the tool ran, reload, finish
        spath = f"{d}/cc-session.jsonl"
        with Session(spath) as s1:
            a1 = Agent(llm=_FakeLLM([
                LLMReply(Message(role="assistant", tool_calls=[
                    ToolCall(id="r1", name="Read",
                             arguments={"file_path": target})]))]),
                tools=TOOLS, system=SYSTEM, max_steps=1)
            r1 = await a1.run(input="check app.py", session=s1)
            assert r1.status == "truncated"      # died mid-task
        with Session.load(spath) as s2:          # new process: load + go on
            a2 = Agent(llm=_FakeLLM([
                LLMReply(Message(role="assistant", content="resumed fine"))]),
                tools=TOOLS, system=SYSTEM)
            r2 = await a2.run(input="continue", session=s2)
        assert r2.status == "completed" and r2.output == "resumed fine"
        logged = Session.load(spath).messages()
        assert [m.role for m in logged] == [
            "user", "assistant", "tool", "user", "assistant"]
        assert "\ta = 9" in logged[2].tool_result.content   # history intact

    print("cc replica self-check ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    session = None
    if "--session" in args:
        i = args.index("--session")
        if i + 1 >= len(args):
            sys.exit("usage: --session <path.jsonl> [prompt]")
        session = args[i + 1]
        del args[i:i + 2]
    prompt = " ".join(args)
    if prompt == "selfcheck":
        asyncio.run(_selfcheck())
    elif not prompt and session:
        asyncio.run(repl(session))
    else:
        asyncio.run(main(prompt, session))
