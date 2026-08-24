"""examples/web over real localhost HTTP: the agent works in its scratch
workspace, grows a tool into external/, mounts a shipped adapter, and the
panels / viewer / session log agree — scripted LLM, no network.

Run: uv run python tests/test_web.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# the server assembles its agent at import: knobs first, into a tmpdir
_TMP = tempfile.TemporaryDirectory()
os.environ["OPENAI_API_KEY"] = "unused-the-llm-is-scripted"
os.environ["STELLAR_WEB_EXTERNAL"] = str(Path(_TMP.name) / "external")
os.environ["STELLAR_WEB_SCRATCH"] = str(Path(_TMP.name) / "workspace")
os.environ["STELLAR_WEB_SESSIONS"] = str(Path(_TMP.name) / "sessions")

import httpx  # noqa: E402

from core import LLMDelta, LLMReply, Message, ToolCall  # noqa: E402
from examples.web import server  # noqa: E402

# the shape the SYSTEM prompt documents: setup(ctx) composing, handler
# taking the call context first and the schema's properties as kwargs
CALC_V1 = '''
from core import Tool, ToolSpec

def setup(ctx):
    def calc(cctx, a=0, b=0):
        return a + b
    ctx.tool(Tool(spec=ToolSpec("calc", "Add a and b.", {
        "type": "object", "required": ["a", "b"],
        "properties": {"a": {"type": "number"},
                       "b": {"type": "number"}}}), handler=calc))
'''


class Scripted:
    """tests/test_kernel.py's stub, plus a gate so a run can be held open
    while a second request races it."""

    def __init__(self, replies, gate=None):
        self.replies, self.gate = list(replies), gate

    async def stream(self, messages, tools, **params):
        if self.gate is not None:
            await self.gate.wait()
        reply = self.replies.pop(0)
        if reply.message.content:
            yield LLMDelta(text=reply.message.content)
        yield reply


def call(cid, name, args):
    return LLMReply(message=Message(
        role="assistant", tool_calls=[ToolCall(cid, name, args)]),
        stop_reason="tool_use")


def text(t):
    return LLMReply(message=Message(role="assistant", content=t))


def frames(raw: str) -> list[tuple[str, dict]]:
    """The counterpart of core.transport.sse: (event name, data) per frame."""
    out = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        name, data = "message", ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        out.append((name, json.loads(data or "{}")))
    return out


def fails(fn, *a) -> str:
    try:
        fn(*a)
    except ValueError as ex:
        return str(ex)
    raise AssertionError(f"{getattr(fn, '__name__', fn)}{a} should have raised")


def events(fs, kind, phase):
    return [e["payload"] for name, e in fs
            if name != "done" and e["kind"] == kind and e["phase"] == phase]


async def run_sse(c, sid: str, prompt: str) -> list[tuple[str, dict]]:
    async with c.stream("POST", "/api/run",
                        json={"session": sid, "input": prompt}) as r:
        assert r.status_code == 200, (r.status_code, await r.aread())
        return frames((await r.aread()).decode())


# ---- scenarios ---------------------------------------------------------------

async def static_routes(c) -> None:
    r = await c.get("/")
    assert r.status_code == 200 and "<title>stellar</title>" in r.text
    assert r.headers["content-type"].startswith("text/html")
    assert (await c.get("/app.js")).status_code == 200
    assert (await c.get("/nope")).status_code == 404
    assert (await c.get("/api/sessions/../../etc/passwd")).status_code == 404


async def grow_a_tool(c) -> None:
    sid = (await c.post("/api/sessions")).json()["id"]
    server.AGENT.llm = Scripted([
        call("c1", "adapter_write", {"path": "tool_calc.py", "content": CALC_V1}),
        call("c2", "adapter_load", {"path": "tool_calc.py"}),
        call("c3", "calc", {"a": 2, "b": 3}),
        text("calc says 5."),
    ])
    fs = await run_sse(c, sid, "write yourself a calculator")

    assert fs[-1][0] == "done", fs[-1]                    # terminating frame
    deltas = events(fs, "text", "delta")
    assert "".join(d["text"] for d in deltas) == "calc says 5."
    assert [p["name"] for p in events(fs, "tool", "start")] == [
        "adapter_write", "adapter_load", "calc"]
    ends = events(fs, "tool", "end")
    assert not any(p["is_error"] for p in ends), ends
    assert ends[2]["result"] == 5                         # the new tool ran
    assert events(fs, "run", "end")[0]["status"] == "completed"

    a = (await c.get("/api/adapters")).json()
    grown = next(m for m in a["mounted"] if m["name"] == "tool_calc")  # the file
    assert grown["notes"] == ["tool:calc"]                             # what it added
    assert a["external"] == ["tool_calc.py"]
    assert "llm_anthropic" in a["internal"]               # has setup(ctx)
    assert "llm_common" not in a["internal"]              # a helper, not an adapter
    # the primitives are mounted like everything else, from internal/
    assert {"tool_fs", "tool_bash"} <= {m["name"] for m in a["mounted"]}
    assert {"bash", "read_file", "write_file", "edit_file", "list_files"} <= \
        set(server.AGENT.tools)

    # external/ is browsable through the same viewer the panel clicks
    f = (await c.get("/api/file?root=external&path=tool_calc.py")).json()
    assert f["path"] == "external/tool_calc.py" and f["content"] == CALC_V1

    h = (await c.get(f"/api/sessions/{sid}")).json()
    assert {"role": "user", "content": "write yourself a calculator"} in h
    assert {"role": "assistant", "content": "calc says 5."} in h
    ww = next(m for m in h if m["role"] == "tool" and m["name"] == "adapter_write")
    assert '"path"' in ww["args"] and "tool_calc.py" in ww["args"]   # input visible
    assert ww["result"] and not ww["error"]                         # output paired in
    calc = next(m for m in h if m["role"] == "tool" and m["name"] == "calc")
    assert calc["result"] == "5"                                # rendered, not wire

    rows = await c.get("/api/sessions")
    row = next(s for s in rows.json() if s["id"] == sid)
    assert row["messages"] == 8   # user + 3x(assistant+tool) + assistant


async def works_in_the_workspace(c) -> None:
    """The scratch space: a shell, files, and the viewer reading them back."""
    sid = (await c.post("/api/sessions")).json()["id"]
    server.AGENT.llm = Scripted([
        call("w1", "bash", {"command": "printf hi"}),
        call("w2", "write_file", {"path": "notes/hello.txt",
                                  "content": "hello workspace\n"}),
        call("w3", "read_file", {"path": "notes/hello.txt"}),
        text("scratch works."),
    ])
    fs = await run_sse(c, sid, "try your hands")

    ends = events(fs, "tool", "end")
    assert not any(p["is_error"] for p in ends), ends
    assert [p["result"] for p in ends] == [
        "hi", "Wrote 1 line to notes/hello.txt", "hello workspace"]

    scratch = Path(os.environ["STELLAR_WEB_SCRATCH"])
    assert (scratch / "notes" / "hello.txt").read_text() == "hello workspace\n"

    f = (await c.get("/api/file?root=workspace&path=notes/hello.txt")).json()
    assert f == {"path": "workspace/notes/hello.txt", "content": "hello workspace\n"}

    (scratch / "__pycache__").mkdir(exist_ok=True)
    (scratch / "__pycache__" / "junk.pyc").write_text("x")
    listing = (await c.get("/api/adapters")).json()["workspace"]
    assert "notes/hello.txt" in listing, listing
    assert not any("__pycache__" in n for n in listing), listing


def adapter_write_gate() -> None:
    """external/ is the manifest: .py, jailed, and named for what it adds."""
    write = server.AGENT.tools["adapter_write"].handler
    ws = Path(os.environ["STELLAR_WEB_EXTERNAL"])

    assert write(None, "tool_calc.py", CALC_V1).startswith("wrote")   # idempotent
    assert write(None, "tool/tool_x.py", "def setup(ctx): pass\n").startswith("wrote")

    for bad in ("calc.py", "skills_adapter.py", "tool/x.py", "TOOL_x.py"):
        msg = fails(write, None, bad, "# nope\n")
        assert "tool_*.py" in msg and "tool_calc.py" in msg, (bad, msg)

    for escape in ("../escaped.py", str(ws.parent / "escaped.py"), "notes.txt"):
        assert "holds .py files under" in fails(write, None, escape, "# nope\n"), escape
    assert not (ws.parent / "escaped.py").exists(), "a rejected write still wrote"


def the_prompt_does_not_lie() -> None:
    """The adapter example in SYSTEM is real code: lifted out of the prompt
    byte for byte it passes the naming gate, mounts, and the tool it claims
    to register answers a call. A prompt that documents a wrong handler
    signature is the bug this guards."""
    body, keep = [], False
    for ln in server.SYSTEM.splitlines():
        if ln.strip() == "# tool_calc.py":
            keep = True
        elif keep:
            if ln.strip() and not ln.startswith("    "):   # the block ends here
                break
            body.append(ln)
    src = textwrap.dedent("\n".join(body).strip("\n")) + "\n"
    assert "def setup(ctx):" in src and "cctx" in src, src

    server.AGENT.tools["adapter_write"].handler(None, "tool_example.py", src)
    assert "tool:add" in server.AGENT.tools["adapter_load"].handler(
        None, "tool_example.py")
    assert server.AGENT.tools["add"].handler(None, a=2, b=3) == 5
    server.AGENT.tools["adapter_unload"].handler(None, "tool_example")
    assert "add" not in server.AGENT.tools


async def file_endpoint(c) -> None:
    scratch = Path(os.environ["STELLAR_WEB_SCRATCH"])
    (scratch / "logo.png").write_bytes(b"\x89PNG\x00\xff\xfe")
    assert (await c.get("/api/file?root=workspace&path=logo.png")).json() == {
        "path": "workspace/logo.png", "content": "binary file (7 bytes)"}

    (scratch / "big.txt").write_text("x" * (server.MAX_FILE + 50))
    big = (await c.get("/api/file?root=workspace&path=big.txt")).json()["content"]
    assert big.endswith("\n… truncated (50 more chars)") and \
        len(big) == server.MAX_FILE + len("\n… truncated (50 more chars)")

    for bad in ("root=internal&path=llm_openai.py",        # not one of the two
                "root=&path=logo.png", "path=logo.png",    # no root at all
                "root=workspace&path=../../etc/passwd",    # out of the jail
                f"root=workspace&path={scratch.parent}/x", # absolute, likewise
                "root=workspace&path=notes",               # a directory
                "root=external&path=absent.py"):           # simply not there
        r = await c.get("/api/file?" + bad)
        assert r.status_code == 404, (bad, r.status_code)
        assert "not found" in r.text, (bad, r.text)

    # percent-encoded paths survive: the panel sends encodeURIComponent
    (scratch / "a b").mkdir(exist_ok=True)
    (scratch / "a b" / "c.txt").write_text("spaced\n")
    assert (await c.get("/api/file?root=workspace&path=a%20b%2Fc.txt")).json() == {
        "path": "workspace/a b/c.txt", "content": "spaced\n"}


async def mount_internal(c) -> None:
    sid = (await c.post("/api/sessions")).json()["id"]
    assert (await c.get("/api/sessions")).json()[0]["id"] == sid   # newest first
    server.AGENT.llm = Scripted([
        call("c4", "internal_load",
             {"name": "hook_compaction", "config": {"max_tokens": 100_000}}),
        call("c5", "internal_load", {"name": "llm_common"}),
        text("compaction is on."),
    ])
    fs = await run_sse(c, sid, "keep your context small")

    mounted, helper = events(fs, "tool", "end")
    assert not mounted["is_error"] and "hook:before_llm" in mounted["result"]
    assert helper["is_error"] and "no setup(ctx)" in helper["result"]
    assert "hook_compaction" in [
        m["name"] for m in (await c.get("/api/adapters")).json()["mounted"]]


async def one_run_per_session(c) -> None:
    sid = (await c.post("/api/sessions")).json()["id"]
    gate = asyncio.Event()
    server.AGENT.llm = Scripted([text("slow but done.")], gate=gate)
    async with c.stream("POST", "/api/run",
                        json={"session": sid, "input": "hold the line"}) as r:
        assert r.status_code == 200
        stream, raw = r.aiter_text(), ""
        async for chunk in stream:
            raw += chunk
            if "step_start" in raw:       # run/start: the session lock is held
                break
        second = await c.post("/api/run", json={"session": sid, "input": "race"})
        assert second.status_code == 409, second.status_code
        assert "already streaming" in second.text
        gate.set()
        async for chunk in stream:        # same iterator: httpx consumes once
            raw += chunk
    fs = frames(raw)
    assert fs[-1][0] == "done"
    assert events(fs, "run", "end")[0]["status"] == "completed"
    assert sid not in server.RUNNING      # released after the run, not the socket


async def survives_a_dropped_client(c) -> None:
    sid = (await c.post("/api/sessions")).json()["id"]
    gate = asyncio.Event()
    server.AGENT.llm = Scripted([text("nobody is listening.")], gate=gate)
    async with c.stream("POST", "/api/run",
                        json={"session": sid, "input": "walk away"}) as r:
        assert r.status_code == 200
        await r.aclose()                  # hang up mid-stream
    gate.set()
    while sid in server.RUNNING:          # the run finishes without a reader
        await asyncio.sleep(0.01)
    h = (await c.get(f"/api/sessions/{sid}")).json()
    assert h[-1] == {"role": "assistant", "content": "nobody is listening."}


# ---- driver ------------------------------------------------------------------

async def main() -> None:
    srv = await server.start(0)           # ephemeral port
    base = f"http://127.0.0.1:{srv.sockets[0].getsockname()[1]}"
    try:
        async with httpx.AsyncClient(base_url=base, timeout=10.0) as c:
            await static_routes(c)
            await grow_a_tool(c)          # asserts the exact external listing
            await works_in_the_workspace(c)
            adapter_write_gate()
            the_prompt_does_not_lie()
            await file_endpoint(c)
            await mount_internal(c)
            await one_run_per_session(c)
            await survives_a_dropped_client(c)
    finally:
        srv.close()
        await srv.wait_closed()
        for s in server.SESSIONS.values():
            s.close()
    print("test_web: all ok")


if __name__ == "__main__":
    asyncio.run(main())
