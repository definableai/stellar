"""Claude Code, rebuilt on this core — same prompt, same tool schemas, GPT brain.

    export OPENAI_API_KEY=...
    uv run python -m examples.cc.cc "add a healthcheck route to app.py"
    uv run python -m examples.cc.cc --session 7b2f "big refactor task"
    uv run python -m examples.cc.cc --session 7b2f      # REPL (Worker)

Everything the model sees is read verbatim from cc.json (a captured Claude
Code request) — only the handlers are ours: Read / Write / Edit / Bash. The
other captured tools (Agent, Skill, Workflow, ...) stay in the prompt but are
not faked; calls come back as UnknownTool. Bash is NOT sandboxed — commands
run with this process's full permissions. ``--session <id>`` logs to
``.cc-sessions/<id>.jsonl``: Ctrl-C stops gracefully, kill -9 loses at most
the step in flight — rerun the same command and it resumes.
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
from core import Agent, Session, StepKind, StepPhase, Tool, ToolSpec

# ---- capture: prompt + schemas straight from cc.json -----------------------

CC = json.loads((Path(__file__).parent / "cc.json").read_text())
SYSTEM = "\n\n".join(b["text"] for b in CC["system"][1:])   # [0] is a billing header
SCHEMAS = {t["name"]: t for t in CC["tools"]}

MAX_LINES = 2000        # Read default, per the tool description
MAX_OUTPUT = 30_000     # chars of Bash output kept before truncation
TOOLS: list[Tool] = []


def handler(name: str, parallel_safe: bool = True):
    # bind fn to cc.json's spec — name, description and schema from the capture
    def wrap(fn):
        spec = SCHEMAS[name]
        TOOLS.append(Tool(
            spec=ToolSpec(
                name=name,
                description=spec["description"],
                # $schema is an Anthropic-side annotation; other providers reject it
                parameters={k: v for k, v in spec["input_schema"].items()
                            if k != "$schema"},
            ),
            handler=fn,
            parallel_safe=parallel_safe,
        ))
        return fn
    return wrap


# ---- tool handlers ---------------------------------------------------------

# Files Read so far, by resolved path; Edit/Write refuse to touch anything else.
# ponytail: process-global, so one conversation per process — the real harness
# scopes this per session. Move onto ctx.run.state once run state spans turns.
SEEN: set[str] = set()


def _key(file_path: str) -> str:
    return str(Path(file_path).resolve())   # `sub/../a.py` == `a.py` == symlink


def _slurp(file_path: str) -> str:
    # text only, line endings preserved; binary is an error, not mojibake
    try:
        with open(file_path, encoding="utf-8", newline="") as f:
            return f.read()
    except UnicodeDecodeError:
        raise ValueError(f"{file_path} is not UTF-8 text — this Read is text-only") from None


@handler("Read")
def read(ctx, file_path: str, offset: int = 0, limit: int = MAX_LINES,
         pages: str | None = None) -> str:
    # ponytail: text only, `pages` ignored — images/PDF/notebooks need a
    # multimodal ToolResult; add when the core carries non-text content blocks.
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
    # ponytail: one subprocess per call — cwd/env do not persist across calls,
    # unlike the real Bash tool. A long-lived shell + sentinel framing if it matters.
    if run_in_background:
        raise ValueError("run_in_background is not supported here — run it in the foreground")
    proc = await asyncio.create_subprocess_shell(
        command, cwd=os.getcwd(), start_new_session=True,  # own group, so we can kill children
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
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


# ---- tracer ----------------------------------------------------------------

class CliTracer:
    # streams the run to the terminal, Claude-Code style; a tracer, so
    # one-shot runs and Worker turns render through the same code
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


# ---- agent -----------------------------------------------------------------

def build(model: str = "gpt-5.6-luna", tracers: Any = (), **params: Any) -> Agent:
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


# ---- entrypoints -----------------------------------------------------------

SESSIONS_DIR = Path(".cc-sessions")             # <cwd>/.cc-sessions/<id>.jsonl


def _open_session(sid: str | None) -> Session:
    if sid is None:
        return Session()                        # in-memory: no resume
    path = SESSIONS_DIR / f"{sid}.jsonl"
    return Session.load(path) if path.exists() else Session(path, id=sid)


async def main(prompt: str, session_id: str | None = None) -> None:
    with _open_session(session_id) as s:
        handle = build(tracers=[CliTracer()]).run(input=prompt, session=s)
        hits = 0

        def on_int() -> None:                   # first Ctrl-C graceful, second hard
            nonlocal hits
            hits += 1
            if hits == 1:
                print("\n(stopping — Ctrl-C again to force)", flush=True)
                handle.stop("user")
            else:
                sys.exit(130)

        asyncio.get_running_loop().add_signal_handler(signal.SIGINT, on_int)
        await handle


async def repl(session_id: str) -> None:
    # a long-lived Worker on a durable session: Claude Code as a REPL
    with _open_session(session_id) as s:
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


# ---- cli -------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]
    session = None
    if "--session" in args:
        i = args.index("--session")
        if i + 1 >= len(args):
            sys.exit("usage: --session <id> [prompt]")
        session = args[i + 1]
        del args[i:i + 2]
    prompt = " ".join(args)
    if not prompt and session:
        asyncio.run(repl(session))
    else:
        asyncio.run(main(prompt, session))
