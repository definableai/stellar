"""examples/web — the bare agent behind a browser.

    export OPENAI_API_KEY=...
    uv run python -m examples.web.server        # http://127.0.0.1:8765

One ``Agent(OpenAILLM)`` with two spaces. ``workspace/`` is the scratch
root: ``internal/tool_fs`` and ``internal/tool_bash`` are mounted there,
so read/write/edit/list and a shell all land in one directory.
``external/`` is the adapter manifest ``boot()`` mounts — the agent's own
parts — reached through ``adapter_write`` (drop a .py in) and the
kernel's ``adapter_load``/``unload``/``reload``/``list``;
``internal_load`` mounts a shipped ``internal/`` adapter instead. That is
enough to watch it grow: it writes an adapter, loads it, calls the tool
it just registered, and the right-hand panel shows the new part. Steps
stream over SSE; every message lands in ``.web-sessions/<id>.jsonl``, so
a reload resumes rather than restarts.

A server restart re-execs every ``external/*.py``, so an adapter's
``setup()`` must stay side-effect-free — a one-shot script parked there
runs again on every boot, forever. Script-shaped files belong in the
scratch workspace (``workspace/attic/`` is where strays go); the SYSTEM
prompt states the rule and ``adapter_write``'s ``llm_*``/``tool_*``/
``hook_*`` gate keeps the manifest readable.

Framework-free on purpose. ``core/transport.py`` is a plain async-iterator
serializer, so the "web framework" here is ``asyncio.start_server`` plus
fifty lines of HTTP/1.1 — swap in FastAPI and only this file changes.

NOT a sandbox, exactly like examples/cc: adapter code runs in-process with
full privileges, ``bash`` is a shell, and nothing is gated. Bind it to
localhost, and wire ``internal/hook_approval`` onto ``bash``,
``adapter_write`` and ``adapter_*`` before letting anyone near this port
you would not hand a shell.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import uuid
from contextlib import suppress
from importlib import import_module
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

from core import Agent, Session, Tool, ToolSpec, boot, sse
from internal import tool_bash, tool_fs
from internal.llm_openai import OpenAIResponsesLLM

# ---- knobs (env-overridable at import so a test can point them at a tmpdir) --

EXTERNAL = os.environ.get("STELLAR_WEB_EXTERNAL", "external")     # the manifest
SCRATCH = os.environ.get("STELLAR_WEB_SCRATCH", "workspace")      # where work happens
SESSIONS_DIR = Path(os.environ.get("STELLAR_WEB_SESSIONS", ".web-sessions"))
PORT = int(os.environ.get("STELLAR_WEB_PORT", "8765"))
# a reasoning model that actually CALLS tools — gpt-4o-class models tend to
# print imitation tool-call JSON as prose instead (observed), which reads as
# the agent "refusing" to grow itself
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
SEARCH = os.environ.get("STELLAR_WEB_SEARCH", "1") != "0"   # OpenAI web_search
MAX_STEPS = 20

HERE = Path(__file__).resolve().parent
INTERNAL = HERE.parents[1] / "internal"     # the package, not a cwd guess
WS = Path(EXTERNAL).resolve()               # external/ — adapters only
SCRATCH_ROOT = Path(SCRATCH).resolve()      # workspace/ — files, clones, builds
ROOTS = {"external": WS, "workspace": SCRATCH_ROOT}   # the two names /api/file takes

SYSTEM = """You are a stellar agent — a small Python agent loop that can
rewrite itself while it runs. This conversation is a browser tab: every
step you take streams into it live.

You have two spaces and they are not interchangeable.

workspace/ is where the work happens: read_file, write_file, edit_file,
list_files and bash all operate there (bash runs with cwd=workspace).
Scratch files, clones, downloads, installs, generated documents, scripts
you run once with `python thing.py` — all of it lives here.

external/ holds your parts: reusable capabilities, nothing else. Write
one with adapter_write, mount it with adapter_load and it is callable on
your very next step; adapter_reload picks up an edit, adapter_unload
drops it, adapter_list is the mirror of what you are made of right now.
Never park a one-shot script in external/: the server re-executes every
file there on each boot, so a script left in it runs again forever. Does
it do a job once? That is bash, in the workspace. Does it add a
capability you will call again? That is an adapter.

An adapter is a .py defining setup(ctx). setup only composes —
ctx.tool(...), ctx.hook(point, fn), ctx.llm(...), ctx.effect(undo), each
recording its own inverse — it never does IO or side effects. A tool
handler takes the call context FIRST, then the schema's properties as
keyword arguments:

    # tool_calc.py
    from core import Tool, ToolSpec

    def setup(ctx):
        def add(cctx, a=0, b=0):        # cctx first — always
            return a + b
        ctx.tool(Tool(spec=ToolSpec("add", "Add two numbers.", {
            "type": "object", "required": ["a", "b"],
            "properties": {"a": {"type": "number"},
                           "b": {"type": "number"}}}), handler=add))

Name an adapter for what it contributes: tool_<name>.py registers tools,
hook_<name>.py hooks, llm_<name>.py an LLM — the same convention the
shipped internal/ catalog uses. adapter_write rejects any other name.
internal_load("hook_compaction", {...}) mounts one of those shipped
adapters instead of writing your own.

Make real tool calls — never print imitation tool-call JSON as text.
Investigate with bash (ls, cat, grep, git clone, uv pip install) before
guessing, and verify your own result — run the thing, read the file back
— before calling it done. Prefer writing the tool you are missing over
apologising for not having it. Keep answers short.""" + ("""

web_search is built in (runs on OpenAI's side): use it for current
facts, documentation and repositories.""" if SEARCH else "")


# ---- the agent's two extra tools (both trust boundaries: the model is input) -

INTERNAL_NAME = re.compile(r"^(llm|tool|hook)_\w+$")   # the naming convention


def adapter_write(cctx: Any, path: str = "", content: str = "") -> str:
    """Write an adapter .py into external/ — for agent parts only; scratch
    files belong to write_file. Nothing is live until adapter_load mounts it."""
    p = (WS / path).resolve()   # the kernel's jail, same three checks
    if not (p.is_relative_to(WS) and p.suffix == ".py"):
        raise ValueError(f"external/ holds .py files under {WS}: {path!r}")
    if not INTERNAL_NAME.match(p.stem):
        raise ValueError(
            f"adapters are named for what they contribute — llm_*.py, tool_*.py "
            f"or hook_*.py (subdirectories are fine, the file name still "
            f"conforms), e.g. 'tool_calc.py': {path!r}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content.splitlines())} lines to {p.relative_to(WS).as_posix()}"


def internal_load(cctx: Any, name: str = "", config: dict | None = None) -> str:
    """Mount a shipped internal/ adapter by module name (llm_*, tool_*,
    hook_*); config is passed to its setup(ctx) as ctx.config."""
    if not INTERNAL_NAME.match(name):
        raise ValueError(f"internal adapters are named llm_*/tool_*/hook_*: {name!r}")
    setup = getattr(import_module(f"internal.{name}"), "setup", None)
    if not callable(setup):
        raise ValueError(f"internal.{name} defines no setup(ctx) — it is a shared "
                         "helper, not an adapter")
    scope = AGENT.use(setup, **(config or {}))
    return f"mounted {scope.name!r}: {', '.join(scope.notes) or 'nothing registered'}"


TOOLS = [
    Tool(spec=ToolSpec("adapter_write", adapter_write.__doc__ or "", {
        "type": "object", "required": ["path", "content"], "properties": {
            "path": {"type": "string", "description": "Path relative to "
                     "external/, named llm_*.py, tool_*.py or hook_*.py "
                     "(subdirectories allowed)."},
            "content": {"type": "string", "description": "The whole file — an "
                        "adapter defines setup(ctx)."}}}),
         handler=adapter_write, parallel_safe=False),
    Tool(spec=ToolSpec("internal_load", internal_load.__doc__ or "", {
        "type": "object", "required": ["name"], "properties": {
            "name": {"type": "string", "description": "Module under internal/, "
                     "e.g. 'hook_compaction'. See adapter_list for what is on."},
            "config": {"type": "object", "description": "Keyword arguments for "
                       "that adapter's setup(ctx), read as ctx.config."}}}),
         handler=internal_load, parallel_safe=False),
]

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("OPENAI_API_KEY is not set — export it and rerun")

AGENT = Agent(llm=OpenAIResponsesLLM(
                  model=MODEL, reasoning={"effort": "medium", "summary": "auto"},
                  **({"tools": [{"type": "web_search"}]} if SEARCH else {})),
              tools=TOOLS, system=SYSTEM, max_steps=MAX_STEPS)
boot(AGENT, EXTERNAL)                          # the kernel + its own parts
AGENT.use(tool_fs.setup, root=SCRATCH)          # …and hands to work with,
AGENT.use(tool_bash.setup, root=SCRATCH)        # rooted in the scratch space


# ---- state: one process, one writer per session ------------------------------

SESSIONS: dict[str, Session] = {}   # held open for the process lifetime
RUNNING: set[str] = set()           # sessions with a run in flight
# ponytail: never evicted — one open fd per session ever touched, and one
# agent for every tab, so concurrent runs share (and mutate) one toolset.
# An LRU close() and an agent per session if this ever leaves localhost.


def _open(sid: str) -> Session:
    """A held-open session, or one loaded from disk. KeyError -> 404."""
    if not sid.isalnum():           # request input: never build a path from it raw
        raise KeyError(sid)
    if sid not in SESSIONS:
        p = SESSIONS_DIR / f"{sid}.jsonl"
        if not p.exists():
            raise KeyError(sid)
        SESSIONS[sid] = Session.load(p)
    return SESSIONS[sid]


def _create() -> Session:
    sid = uuid.uuid4().hex[:8]
    SESSIONS[sid] = Session(SESSIONS_DIR / f"{sid}.jsonl", id=sid)
    return SESSIONS[sid]


def _sessions() -> list[dict[str, Any]]:
    """Held-open sessions merged with the log directory, newest first."""
    out = []
    for p in sorted(SESSIONS_DIR.glob("*.jsonl"),
                    key=lambda f: f.stat().st_mtime, reverse=True):
        held = SESSIONS.get(p.stem)   # in memory: exact, and no re-read
        out.append({"id": p.stem, "messages": len(held) if held else
                    max(0, sum(1 for _ in p.open("rb")) - 1)})   # minus the header
    return out


def _text(content: Any) -> str:
    if isinstance(content, list):   # multimodal blocks: the text parts
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content if isinstance(content, str) else ""


def _pretty(value: Any, n: int = 4000) -> str:
    s = value if isinstance(value, str) else json.dumps(value, indent=2,
                                                        default=str)
    return s if len(s) <= n else s[:n] + "…"


def _history(s: Session) -> list[dict[str, Any]]:
    """Render for the browser. Tool entries pair a call with its result
    by call_id: {"role": "tool", "name", "args", "result", "error"}."""
    out: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}
    for m in s.messages():
        if m.role == "tool" and m.tool_result:
            r = m.tool_result
            e = pending.pop(r.call_id, None)
            if e is None:                     # repaired log: result, no call
                e = {"role": "tool", "name": r.name, "args": ""}
                out.append(e)
            e["result"], e["error"] = _pretty(r.content), bool(r.is_error)
        elif m.role in ("user", "assistant"):
            if _text(m.content):
                out.append({"role": m.role, "content": _text(m.content)})
            for c in m.tool_calls:
                e = {"role": "tool", "name": c.name,
                     "args": _pretty(c.arguments), "result": "", "error": False}
                pending[c.id] = e
                out.append(e)
    return out


def _internal() -> list[str]:
    """internal/ modules that are adapters — a helper defines no setup."""
    names = []
    for p in sorted(INTERNAL.glob("*.py")):
        if not INTERNAL_NAME.match(p.stem):
            continue
        try:
            mod = import_module(f"internal.{p.stem}")
        except Exception:   # optional dep missing: not mountable, so not listed
            continue
        if callable(getattr(mod, "setup", None)):
            names.append(p.stem)
    return names


LIST_CAP = 300          # entries a panel shows before the "… N more" tail
MAX_FILE = 100_000      # chars /api/file returns before truncating
_SKIP = {".git", "__pycache__", ".venv", "node_modules"}


def _files(root: Path, pattern: str = "*") -> list[str]:
    """Files under ``root``, root-relative and sorted, noise directories
    dropped. ponytail: walks the whole tree per refresh, then caps — fine
    for a panel, not for a workspace with a linux checkout in it."""
    found = []
    for p in root.rglob(pattern):                 # rglob does not follow symlinks
        rel = p.relative_to(root)
        if p.is_file() and not _SKIP & set(rel.parts):
            found.append(rel.as_posix())
    found.sort()
    return (found if len(found) <= LIST_CAP else
            found[:LIST_CAP] + [f"… {len(found) - LIST_CAP} more"])


def _adapters() -> dict[str, list[Any]]:
    return {"mounted": [{"name": s.name, "source": s.source, "notes": s.notes}
                        for s in AGENT.adapters.values()],
            "internal": _internal(),
            "external": _files(WS, "*.py"),      # the manifest: adapters only
            "workspace": _files(SCRATCH_ROOT)}   # the scratch space: everything


def _file(query: dict[str, str]) -> dict[str, str]:
    """One file under one of the two named roots, for the viewer. Text
    only; a binary or oversized file reads as a placeholder, never bytes.
    KeyError (bad root, escape, missing) -> 404."""
    if (root := ROOTS.get(query.get("root", ""))) is None:
        raise KeyError(f"root is one of {sorted(ROOTS)}: {query.get('root', '')!r}")
    rel = query.get("path", "")
    p = (root / rel).resolve()                   # judged by where it lands
    if not (p.is_relative_to(root) and p.is_file()):
        raise KeyError(f"no such file under {query['root']}/: {rel!r}")
    try:
        content = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = f"binary file ({p.stat().st_size} bytes)"
    else:
        if len(content) > MAX_FILE:
            content = (content[:MAX_FILE]
                       + f"\n… truncated ({len(content) - MAX_FILE} more chars)")
    return {"path": f"{query['root']}/{p.relative_to(root).as_posix()}",
            "content": content}


# ---- HTTP/1.1, hand-rolled ---------------------------------------------------

STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8")}
TEXT = "text/plain; charset=utf-8"


async def _request(r: asyncio.StreamReader) -> tuple[str, str, bytes] | None:
    """One request -> (method, path, body). None at EOF or on a junk line.
    ponytail: no keep-alive, no chunked bodies — Content-Length or nothing."""
    line = await r.readline()
    parts = line.decode("latin-1").split()
    if len(parts) < 2:
        return None
    headers: dict[str, str] = {}
    while (h := await r.readline()) not in (b"\r\n", b"\n", b""):
        k, _, v = h.decode("latin-1").partition(":")
        headers[k.strip().lower()] = v.strip()
    return parts[0], parts[1], await r.readexactly(int(headers.get("content-length") or 0))


async def _send(w: asyncio.StreamWriter, status: str, ctype: str, body: bytes) -> None:
    w.write(f"HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: "
            f"{len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
    await w.drain()


async def _json(w: asyncio.StreamWriter, obj: Any) -> None:
    await _send(w, "200 OK", "application/json",
                json.dumps(obj, default=str).encode())


async def _stream_run(w: asyncio.StreamWriter, req: dict[str, Any]) -> None:
    sid = str(req.get("session") or "")
    s = _open(sid)                                   # KeyError -> 404
    if sid in RUNNING:                               # one writer, one run
        return await _send(w, "409 Conflict", TEXT,
                           b"a run is already streaming on this session\n")
    handle = AGENT.run(str(req.get("input") or ""), session=s)
    RUNNING.add(sid)
    try:
        w.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Cache-Control: no-cache\r\nConnection: close\r\n\r\n")
        await w.drain()
        live = True
        async for frame in sse(handle.events()):
            if not live:
                continue          # client left: drain the stream, finish the run
            try:
                w.write(frame.encode())
                await w.drain()
            except OSError:
                live = False
    finally:
        # the run outlives the connection: never unlock the session while the
        # loop may still be appending to its log
        with suppress(Exception):
            await handle
        RUNNING.discard(sid)


async def _route(w: asyncio.StreamWriter, method: str, path: str, body: bytes) -> None:
    path, _, qs = path.partition("?")
    if method == "GET" and path in STATIC:
        name, ctype = STATIC[path]                   # allowlist, no path arithmetic
        return await _send(w, "200 OK", ctype, (HERE / name).read_bytes())
    if path == "/api/sessions" and method in ("GET", "POST"):
        return await _json(w, _sessions() if method == "GET" else {"id": _create().id})
    if method == "GET" and path.startswith("/api/sessions/"):
        return await _json(w, _history(_open(path[len("/api/sessions/"):])))
    if method == "GET" and path == "/api/adapters":
        return await _json(w, _adapters())
    if method == "GET" and path == "/api/file":
        return await _json(w, _file(dict(parse_qsl(qs))))   # KeyError -> 404
    if method == "POST" and path == "/api/run":
        return await _stream_run(w, json.loads(body or b"{}"))
    await _send(w, "404 Not Found", TEXT, b"no such route\n")


async def _handle(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
    try:
        if req := await _request(r):
            await _route(w, *req)
    except Exception as ex:   # a demo server: one bad request never kills it
        status, msg = (("404 Not Found", f"not found: {ex}")
                       if isinstance(ex, KeyError) else
                       ("500 Internal Server Error", f"{type(ex).__name__}: {ex}"))
        with suppress(OSError):   # client gone, or a stream head already sent
            await _send(w, status, TEXT, (msg + "\n").encode())
    finally:
        w.close()


async def start(port: int = PORT) -> asyncio.Server:
    """Bound and serving; tests bind port 0 and read server.sockets[0]."""
    return await asyncio.start_server(_handle, "127.0.0.1", port)


async def main() -> None:
    try:
        server = await start()
    except OSError as ex:
        sys.exit(f"cannot bind 127.0.0.1:{PORT} ({ex.strerror}) — "
                 "set STELLAR_WEB_PORT to a free port")
    print(f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())          # SIGINT -> KeyboardInterrupt -> graceful
    except KeyboardInterrupt:
        pass
    finally:
        for session in SESSIONS.values():
            session.close()
