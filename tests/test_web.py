"""examples/web over real localhost HTTP: the agent grows a tool, mounts a
shipped adapter, and the panels/session log agree — scripted LLM, no network.

Run: uv run python tests/test_web.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# the server assembles its agent at import: knobs first, into a tmpdir
_TMP = tempfile.TemporaryDirectory()
os.environ["OPENAI_API_KEY"] = "unused-the-llm-is-scripted"
os.environ["STELLAR_WEB_WORKSPACE"] = str(Path(_TMP.name) / "workspace")
os.environ["STELLAR_WEB_SESSIONS"] = str(Path(_TMP.name) / "sessions")

import httpx  # noqa: E402

from core import LLMDelta, LLMReply, Message, ToolCall  # noqa: E402
from examples.web import server  # noqa: E402

CALC_V1 = '''
from core import Tool, ToolSpec

def setup(ctx):
    ctx.tool(Tool(ToolSpec("calc", "Add a and b.",
                           {"type": "object", "required": ["a", "b"],
                            "properties": {"a": {"type": "number"},
                                           "b": {"type": "number"}}}),
                  handler=lambda cctx, a, b: a + b))
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
        call("c1", "workspace_write", {"path": "calc.py", "content": CALC_V1}),
        call("c2", "adapter_load", {"path": "calc.py"}),
        call("c3", "calc", {"a": 2, "b": 3}),
        text("calc says 5."),
    ])
    fs = await run_sse(c, sid, "write yourself a calculator")

    assert fs[-1][0] == "done", fs[-1]                    # terminating frame
    deltas = events(fs, "text", "delta")
    assert "".join(d["text"] for d in deltas) == "calc says 5."
    assert [p["name"] for p in events(fs, "tool", "start")] == [
        "workspace_write", "adapter_load", "calc"]
    ends = events(fs, "tool", "end")
    assert not any(p["is_error"] for p in ends), ends
    assert ends[2]["result"] == 5                         # the new tool ran
    assert events(fs, "run", "end")[0]["status"] == "completed"

    a = (await c.get("/api/adapters")).json()
    assert "calc" in [m["name"] for m in a["mounted"]]
    assert a["external"] == ["calc.py"]
    assert "llm_anthropic" in a["internal"]               # has setup(ctx)
    assert "llm_common" not in a["internal"]              # a helper, not an adapter

    h = (await c.get(f"/api/sessions/{sid}")).json()
    assert {"role": "user", "content": "write yourself a calculator"} in h
    assert {"role": "assistant", "content": "calc says 5."} in h
    ww = next(m for m in h if m["role"] == "tool" and m["name"] == "workspace_write")
    assert '"path"' in ww["args"] and "calc.py" in ww["args"]   # input visible
    assert ww["result"] and not ww["error"]                     # output paired in
    calc = next(m for m in h if m["role"] == "tool" and m["name"] == "calc")
    assert calc["result"] == "5"                                # rendered, not wire

    rows = await c.get("/api/sessions")
    row = next(s for s in rows.json() if s["id"] == sid)
    assert row["messages"] == 8   # user + 3x(assistant+tool) + assistant


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
            await grow_a_tool(c)
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
