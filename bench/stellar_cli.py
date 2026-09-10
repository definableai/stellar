#!/usr/bin/env python
"""stellar on Harness-Bench: the generic_cli side. Imports core, never edits it.

  stellar_cli.py run --workspace W --prompt-file P --session-id S

Worker agent does the task; a reviewer agent (fresh context, same model) audits
the deliverables; the worker fixes what the reviewer found. Every LLM call is
gpt-5.6-luna through the Responses API (reasoning + tools). The judge trace is
written in the proxy's own format, as the bench's codex adapter does.

Bench env: HARNESSBENCH_LLM_PROXY_ROUTES (trace dir), HARNESSBENCH_SANDBOX, HARNESSBENCH_TASK_ID.
Knobs: STELLAR_MODEL, STELLAR_EFFORT, STELLAR_MAX_OUT, STELLAR_MAX_STEPS, STELLAR_MAX_TOKENS,
STELLAR_REVIEWS, STELLAR_CAP_SEC, STELLAR_DEADLINE (epoch), STELLAR_FAKE.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench"), str(ROOT / "bench/harness-bench/src")]

from core import Agent, FakeModel, Message, Part, ProviderError, ToolCall, hook, tool  # noqa: E402
from hooks.budget import Budget  # noqa: E402
from hooks.logging import Log  # noqa: E402
from hooks.steps import Steps  # noqa: E402
from responses import Responses  # noqa: E402

WORKSPACE = Path.cwd()
CLIP = 2_000                                    # chars of tool output the model sees; the judge reads a 24K-char trace
VENV_BIN = str(ROOT / "bench/.venv/bin")        # python3 with docx/openpyxl/PIL/pypdf/pytest
MODEL_ID = "stellar-luna"

SYSTEM = """You are stellar, an autonomous agent finishing one task inside a workspace on this machine.

Workspace: {ws}
Relative paths resolve there. Deliverables go under out/ unless the task names another path.

How to work:
- First reply, before any tool call: a short checklist of the deliverables and hard constraints (exact paths, names, formats, fields, counts, exclusions, ordering). Work against it; check it at the end.
- Read what the task points at, do the work, then verify: read the output back, run the test, check the exact format. Fix what the check finds.
- Follow the task's exact paths, file names, formats and counts. Do not add files it did not ask for. Exact sets mean exactly those items: nothing missing, nothing extra.
- Every tool call has a purpose. Before each call, say in one short line what you are doing and why. Do not repeat a call whose result you already have.
- Keep tool output small: head, tail, grep, -q flags, short test output. Read only the parts of a file you need.
- Stay inside the workspace. Never read, modify or delete anything outside it (the parent directory holds harness internals and prompt files: off limits), never touch inputs under in/ unless told, never use rm -rf on paths you did not create.
- No network unless the task hands you a URL. Prefer python3 for data work; python-docx, openpyxl, PIL, pypdf and pytest are installed.
- Tests: run them with `python3 -m pytest -q` whenever the task involves tests or code changes.
- Nobody is watching: never ask questions. When something is ambiguous, take the most reasonable reading and say so at the end. When the inputs do not support an answer, say so instead of inventing one.
- There is enough time to do the job properly. Verify before you claim; when a check fails, fix the cause and re-run it.
- This conversation may span several rounds. Earlier rounds are real: keep what they said in mind.

When done, reply with a short summary of what you produced and where, then stop."""

REVIEW = """Sub-Agent: reviewer. You audit another agent's finished work with fresh eyes. Workspace: {ws}

You get the task it was given. Re-derive the outputs yourself: write ONE python3 script under {scratch} (never inside the workspace) that reads the task's inputs, applies every rule in the task text literally, and prints a compact comparison (at most 12 lines) with what the worker wrote under the workspace. You have only a shell tool and at most 5 commands; each command's output must stay under 10 lines (head, wc, diff -q, python summaries). Say in one short line what you are checking before each command. Never modify the workspace.

Report only defects you are sure of: a rule in the task text, quoted, that the output violates. When the task allows several readings, prefer the worker's.

Reply with exactly one of:
- NO DEFECTS
- a numbered list of concrete defects: file, what is wrong, the quoted rule it violates."""

FIX = """A reviewer audited the workspace against the task and reports:

{defects}

For each point: re-read the task text it quotes. Apply the fix only if the task text confirms the point; otherwise answer why not in one line. After any change, re-run your verification, then give the final summary and stop."""

CHECK = """Final check before you stop: compare the workspace against your checklist and the task statement. For each required output confirm path, format, fields and counts with the simplest check: read it back, or run the test or parser the task provides. Fix what is missing or wrong, then give the final summary. If everything already matches, say so in one line."""


class Patient:
    """Mixin: a 429 or a 5xx is waited out instead of ending the run."""

    async def invoke(self, run):
        for attempt in range(12):
            try:
                return await super().invoke(run)
            except ProviderError as ex:
                if ex.status not in (429, 500, 502, 503, 529) or attempt == 11:
                    raise
                await asyncio.sleep(min(45, 2 ** attempt) + random.random() * 3)


class PatientResponses(Patient, Responses):
    pass


def _path(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else WORKSPACE / p


PHASE_CLIP = [CLIP]                             # the reviewer phase swaps a smaller number in


def _clip(text: str) -> str:
    n = PHASE_CLIP[0]
    return text if len(text) <= n else text[:n * 2 // 5] + "\n...[clipped]...\n" + text[-n * 3 // 5:]


@tool
def bash(
    command: Annotated[str, "shell command; runs with the workspace as cwd"],
    timeout: Annotated[int, "seconds before the command is killed"] = 120,
) -> str:
    """Run a shell command in the workspace. Returns stdout, stderr and the exit code."""
    env = os.environ | {"PATH": VENV_BIN + os.pathsep + os.environ.get("PATH", "")}
    try:
        r = subprocess.run(command, shell=True, cwd=WORKSPACE, capture_output=True,
                           text=True, errors="replace", timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return f"error: timed out after {timeout}s"
    out = r.stdout + (f"\nstderr:\n{r.stderr}" if r.stderr.strip() else "")
    return _clip(out.strip()) + f"\n[exit {r.returncode}]"


@tool(parallel=True)
def read_file(
    path: Annotated[str, "file path, relative to the workspace or absolute"],
    offset: Annotated[int, "first line to show, 1-based"] = 1,
    limit: Annotated[int, "how many lines to show"] = 100,
) -> str:
    """Read a text file with line numbers."""
    lines = _path(path).read_text(errors="replace").splitlines()
    chunk = lines[offset - 1: offset - 1 + limit]
    body = "\n".join(f"{i}: {line[:300]}" for i, line in enumerate(chunk, offset))
    rest = len(lines) - (offset - 1 + len(chunk))
    return _clip(body) + (f"\n... {rest} more lines" if rest > 0 else "")


@tool
def write_file(
    path: Annotated[str, "file path, relative to the workspace or absolute"],
    content: Annotated[str, "the whole file content"],
) -> str:
    """Create or overwrite a text file. Parent directories are created."""
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} chars to {p}"


@tool
def edit_file(
    path: Annotated[str, "file path, relative to the workspace or absolute"],
    old: Annotated[str, "exact text to replace; must occur exactly once"],
    new: Annotated[str, "replacement text"],
) -> str:
    """Replace one exact occurrence of `old` with `new` in a text file."""
    p = _path(path)
    text = p.read_text()
    n = text.count(old)
    if n != 1:
        return f"error: `old` occurs {n} times in {p}; make it unique"
    p.write_text(text.replace(old, new, 1))
    return f"edited {p}"


@tool(parallel=True)
def view_image(path: Annotated[str, "png/jpg/gif/webp file, relative or absolute"]) -> list[Part]:
    """Look at an image file. The picture itself is shown to you."""
    p = _path(path)
    media = mimetypes.guess_type(p.name)[0] or "image/png"
    data = base64.b64encode(p.read_bytes()).decode()
    return [Part("text", f"image {p.name} ({media}, {p.stat().st_size} bytes)"),
            Part("image", {"media_type": media, "data": data})]


TOOLS = [bash, read_file, write_file, edit_file, view_image]


def model(script=None):
    """gpt-5.6-luna on the Responses API, reasoning on. STELLAR_FAKE=1 scripts a FakeModel instead."""
    if os.environ.get("STELLAR_FAKE"):
        return FakeModel(script or ["NO DEFECTS"])
    return PatientResponses(os.environ.get("STELLAR_MODEL", "gpt-5.6-luna"),
                            reasoning={"effort": os.environ.get("STELLAR_EFFORT", "high")},
                            max_output_tokens=int(os.environ.get("STELLAR_MAX_OUT", 16_000)))


FAKE_WORKER = [
    Message("assistant", "Checklist: out/linecount.txt with one integer.", tool_calls=[ToolCall(
        "c1", "bash", {"command": "mkdir -p out && python3 -c \"print(sum(1 for _ in open('in/input.txt')))\" > out/linecount.txt && cat out/linecount.txt"})]),
    "Done: out/linecount.txt holds the line count."]


def deadline() -> float:
    """Seconds this round may take: the task's own timeout, the cap, the wall, minus a margin."""
    cap = float(os.environ.get("STELLAR_CAP_SEC", 1500))
    tasks = Path(os.environ.get("STELLAR_TASKS_DIR", ROOT / "bench/harness-bench/tasks"))
    spec = tasks / os.environ.get("HARNESSBENCH_TASK_ID", "-") / "task.yaml"
    if spec.is_file() and (m := re.search(r"^timeout_sec:\s*(\d+)", spec.read_text(), re.M)):
        cap = min(cap, int(m.group(1)))
    if wall := os.environ.get("STELLAR_DEADLINE"):
        cap = min(cap, float(wall) - time.time())
    return max(cap - 45, 20)


def session_dir() -> Path:
    d = Path(os.environ.get("HARNESSBENCH_SANDBOX", WORKSPACE.parent)) / "stellar-sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load(sid: str) -> tuple[list[Message], dict]:
    f = session_dir() / f"{sid}.json"
    if not f.is_file():
        return [], {}
    d = json.loads(f.read_text())
    msgs = [Message(m["role"], [Part(**p) for p in m["content"]],
                    [ToolCall(**c) for c in m["tool_calls"]], m["tool_call_id"], m["meta"]) for m in d["messages"]]
    return msgs, d.get("extra", {})


def save(sid: str, messages: list[Message], extra: dict) -> None:
    while messages and (messages[-1].role == "tool" or messages[-1].tool_calls):
        messages.pop()                          # a cut-off tool batch is not resumed
    keep = {k: v for k, v in extra.items() if isinstance(v, (str, int, float, bool))}
    (session_dir() / f"{sid}.json").write_text(json.dumps({"messages": [asdict(m) for m in messages], "extra": keep}))


# ---- the judge's trace: proxy-format records, one per assistant turn ------------

def _chat(m: Message) -> dict:
    d: dict = {"role": m.role, "content": m.text}
    if m.tool_calls:
        d["tool_calls"] = [{"id": c.id, "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.args, ensure_ascii=False)}}
                           for c in m.tool_calls]
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    return d


def write_trace(messages: list[Message], state: dict, agent: str, sid: str) -> None:
    """Append this agent's unwritten assistant turns to the proxy dir, in the proxy's own format."""
    routes = os.environ.get("HARNESSBENCH_LLM_PROXY_ROUTES")
    if not routes:
        return
    proxy = Path(routes).parent
    (proxy / "responses").mkdir(parents=True, exist_ok=True)
    tid = os.environ.get("HARNESSBENCH_TASK_ID", "")
    done, seq = state.get(f"trace.{agent}", 0), state.get("trace.seq", 0)
    n, rows = 0, []
    for i, m in enumerate(messages):
        if m.role != "assistant":
            continue
        n += 1
        if n <= done:
            continue
        seq += 1
        path = proxy / "responses" / f"stellar-{seq:04d}.json"
        u = m.meta.get("usage") or {}
        inp, out, cached = u.get("input_tokens") or 0, u.get("output_tokens") or 0, u.get("cached_tokens") or 0
        common = {"task_id": tid, "session_id": sid, "model_id": MODEL_ID, "framework": "stellar", "provider": f"stellar-{agent}"}
        path.write_text(json.dumps(common | {
            "request_body": json.dumps({"messages": [_chat(x) for x in messages[:i]]}, ensure_ascii=False),
            "response_json": {"model": m.meta.get("model", ""), "choices": [{"message": _chat(m)}]},
        }, ensure_ascii=False, indent=2))
        rows.append(json.dumps(common | {
            "raw_response_file": str(path), "input_tokens": inp - cached, "output_tokens": out,
            "cache_read_tokens": cached, "cache_write_tokens": 0, "total_tokens": inp + out,
            "response_model": m.meta.get("model", "")}, ensure_ascii=False))
    if rows:
        with (proxy / "requests.jsonl").open("a", encoding="utf-8") as fh:
            fh.write("".join(r + "\n" for r in rows))
    state[f"trace.{agent}"], state["trace.seq"] = n, seq


# ---- the run ------------------------------------------------------------------

async def main(a: argparse.Namespace) -> dict:
    global WORKSPACE
    WORKSPACE = Path(a.workspace).resolve()
    os.chdir(WORKSPACE)
    prior, extra = load(a.session_id)
    steps, budget = int(os.environ.get("STELLAR_MAX_STEPS", 60)), int(os.environ.get("STELLAR_MAX_TOKENS", 1_200_000))
    worker = Agent(model(list(FAKE_WORKER)), TOOLS)
    reviewer = Agent(model(), [bash])
    worker.hooks.attach(Steps(steps))
    reviewer.hooks.attach(Steps(int(os.environ.get("STELLAR_REVIEW_STEPS", 6))))
    for agent in (worker, reviewer):
        agent.hooks.attach(Budget(budget))
    log = open(session_dir() / f"{a.session_id}.log", "a")
    for agent in (worker, reviewer):
        agent.events.listen(Log(lambda line: log.write(line + "\n")))
    seen: list = []

    @hook("run.pre")
    async def grab(message, run) -> None:
        seen.append(run)                         # the Run survives a timeout this way
    worker.hooks.attach(grab)

    prompt = Path(a.prompt_file).read_text()
    scratch = session_dir().parent / "review-scratch"
    scratch.mkdir(exist_ok=True)
    limit, t0 = deadline(), time.time()
    left = lambda: limit - (time.time() - t0)   # noqa: E731
    status, run, reviews = "done", None, []
    try:
        opened = prior or [Message("system", SYSTEM.format(ws=WORKSPACE))]
        run = await asyncio.wait_for(worker.run(prompt, messages=opened, run_id=a.session_id), timeout=limit)
        run.extra.update({k: v for k, v in extra.items() if k not in run.extra})
        write_trace(run.messages, run.extra, "worker", a.session_id)
        for _ in range(int(os.environ.get("STELLAR_REVIEWS", 0))):
            if left() < 150:
                break
            PHASE_CLIP[0] = int(os.environ.get("STELLAR_REVIEW_CLIP", 600))
            try:
                rv = await asyncio.wait_for(reviewer.run(f"Task given to the worker:\n\n{prompt}",
                                                         messages=[Message("system", REVIEW.format(ws=WORKSPACE, scratch=scratch))]),
                                            timeout=min(left() - 90, 420))
            finally:
                PHASE_CLIP[0] = CLIP
            verdict = rv.messages[-1].text.strip()
            write_trace(rv.messages, run.extra, "reviewer", a.session_id)
            reviews.append(verdict[:400])
            if not verdict or verdict.upper().startswith("NO DEFECTS") or "NO DEFECTS" in verdict.upper()[:40]:
                break
            run = await asyncio.wait_for(worker.run(Message("system", FIX.format(defects=verdict)),
                                                    messages=run.messages, run_id=a.session_id), timeout=left() - 30)
            write_trace(run.messages, run.extra, "worker", a.session_id)
        if int(os.environ.get("STELLAR_REVIEWS", 0)) == 0 and left() > 60 and not os.environ.get("STELLAR_FAKE"):
            run = await asyncio.wait_for(worker.run(Message("system", CHECK), messages=run.messages, run_id=a.session_id), timeout=left() - 30)
            write_trace(run.messages, run.extra, "worker", a.session_id)
    except asyncio.TimeoutError:
        status = "timeout"
    except Exception as ex:                      # a crash still leaves the workspace to grade
        status = f"error: {type(ex).__name__}: {str(ex)[:300]}"
    run = run or (seen[-1] if seen else None)
    if run is not None:
        write_trace(run.messages, run.extra, "worker", a.session_id)
        if reviews:                              # the judge reads the last record's reply: make it the worker's summary
            run.extra["trace.worker"] -= 1
            write_trace(run.messages, run.extra, "worker", a.session_id)
        save(a.session_id, run.messages, run.extra)
    log.close()
    tokens = sum(r.extra.get("budget.tokens", 0) for r in seen)
    return {"status": status, "steps": sum(r.step for r in seen), "elapsed": round(time.time() - t0, 1),
            "limit": round(limit), "tokens": tokens, "reviews": reviews,
            "final": (run.messages[-1].text[:400] if run and run.messages else "")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--workspace", required=True)
    r.add_argument("--prompt-file", required=True)
    r.add_argument("--session-id", required=True)
    print(json.dumps(asyncio.run(main(ap.parse_args())), ensure_ascii=False))
