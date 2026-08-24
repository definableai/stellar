"""examples/web — the bare agent behind a browser.

    export OPENAI_API_KEY=...
    uv run python -m examples.web.server        # http://127.0.0.1:8765

One ``Agent(OpenAILLM)`` booted over ``external/``: a brain, the kernel,
and two tools of its own — ``workspace_write`` (drop a .py into the
workspace) and ``internal_load`` (mount a shipped ``internal/`` adapter).
That is enough to watch it grow: it writes an adapter, loads it, calls the
tool it just registered, and the right-hand panel shows the new part.
Steps stream over SSE; every message lands in ``.web-sessions/<id>.jsonl``,
so a reload resumes rather than restarts.

Framework-free on purpose. ``core/transport.py`` is a plain async-iterator
serializer, so the "web framework" here is ``asyncio.start_server`` plus
fifty lines of HTTP/1.1 — swap in FastAPI and only this file changes.

NOT a sandbox, exactly like examples/cc: adapter code runs in-process with
full privileges and nothing is gated. Bind it to localhost, and wire
``internal/hook_approval`` onto ``workspace_write`` and ``adapter_*``
before letting anyone near this port you would not hand a shell.
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

from core import Agent, Session, Tool, ToolSpec, boot, sse
from internal.llm_openai import OpenAIResponsesLLM

# ---- knobs (env-overridable at import so a test can point them at a tmpdir) --

WORKSPACE = os.environ.get("STELLAR_WEB_WORKSPACE", "external")
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
WS = Path(WORKSPACE).resolve()

SYSTEM = """You are a stellar agent — a small Python agent loop that can
rewrite itself while it runs. This conversation is a browser tab: every
step you take streams into it live.

Your parts are adapters: .py files defining setup(ctx), where ctx offers
ctx.tool(Tool(...)), ctx.hook(point, fn), ctx.llm(...) and
ctx.effect(undo). Every registration records its inverse, so anything you
mount can be dropped again.

To grow a tool, write it and load it:
  workspace_write("calc.py", <source>)   a .py in your workspace
  adapter_load("calc.py")                mounts it — callable next step
adapter_reload picks up an edit, adapter_unload drops it, and adapter_list
is the mirror: exactly what you are made of right now. internal_load(
"hook_compaction", {...}) mounts a shipped internal/ adapter instead of
writing your own.

Adapters are real Python: import Tool, ToolSpec and Hook from `core`.
Prefer writing the tool you are missing over apologising for not having
it. Tools are real function calls — make them; never print imitation
tool-call JSON as text. Keep answers short.""" + ("""

web_search is built in (runs on OpenAI's side): use it when you need
current facts or documentation.""" if SEARCH else "")


# ---- the agent's two extra tools (both trust boundaries: the model is input) -

def workspace_write(cctx: Any, path: str = "", content: str = "") -> str:
    """Write a UTF-8 .py file into the workspace, creating parent
    directories. Nothing is live until adapter_load mounts it."""
    p = (WS / path).resolve()   # the kernel's jail, same three checks
    if not (p.is_relative_to(WS) and p.suffix == ".py"):
        raise ValueError(f"the workspace holds .py files under {WS}: {path!r}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content.splitlines())} lines to {p.relative_to(WS).as_posix()}"


INTERNAL_NAME = re.compile(r"^(llm|tool|hook)_\w+$")


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
    Tool(spec=ToolSpec("workspace_write", workspace_write.__doc__ or "", {
        "type": "object", "required": ["path", "content"], "properties": {
            "path": {"type": "string", "description": "Path relative to the "
                     "workspace, ending in .py (subdirectories allowed)."},
            "content": {"type": "string", "description": "The whole file — an "
                        "adapter defines setup(ctx)."}}}),
         handler=workspace_write, parallel_safe=False),
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
boot(AGENT, WORKSPACE)   # the kernel + whatever it has already written


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


def _adapters() -> dict[str, list[Any]]:
    return {"mounted": [{"name": s.name, "source": s.source, "notes": s.notes}
                        for s in AGENT.adapters.values()],
            "internal": _internal(),
            "external": sorted(p.relative_to(WS).as_posix()
                               for p in WS.rglob("*.py"))}


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
    if method == "GET" and path in STATIC:
        name, ctype = STATIC[path]                   # allowlist, no path arithmetic
        return await _send(w, "200 OK", ctype, (HERE / name).read_bytes())
    if path == "/api/sessions" and method in ("GET", "POST"):
        return await _json(w, _sessions() if method == "GET" else {"id": _create().id})
    if method == "GET" and path.startswith("/api/sessions/"):
        return await _json(w, _history(_open(path[len("/api/sessions/"):])))
    if method == "GET" and path == "/api/adapters":
        return await _json(w, _adapters())
    if method == "POST" and path == "/api/run":
        return await _stream_run(w, json.loads(body or b"{}"))
    await _send(w, "404 Not Found", TEXT, b"no such route\n")


async def _handle(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
    try:
        if req := await _request(r):
            await _route(w, *req)
    except Exception as ex:   # a demo server: one bad request never kills it
        status, msg = (("404 Not Found", f"no such session: {ex}")
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
