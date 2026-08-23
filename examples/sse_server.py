"""Stream a run over SSE with no web framework — raw asyncio.start_server.

    uv run python -m examples.sse_server          # offline self-demo
    uv run python -m examples.sse_server serve    # :8080, drive it with curl
        curl -ND- -X POST localhost:8080/run -d 'weather?'  # X-Run-Id header
        curl -N 'localhost:8080/events/<run_id>?after=4'    # resume the tail
        curl -X POST localhost:8080/stop/<run_id>

``sse()`` writes ``id: <seq>``, so a reconnect's Last-Event-ID goes straight into
``events(after_seq=...)``: replay then live, exactly what was missed, no dupes.
"""

from __future__ import annotations

import asyncio
import json
import sys
from urllib.parse import parse_qs

import httpx

from core import Agent, LLMDelta, LLMReply, Message, RunHandle, ToolCall, sse, tool

RUNS: dict[str, RunHandle] = {}


@tool(parameters={"type": "object", "properties": {"city": {"type": "string"}}})
async def weather(ctx, city: str = "here") -> str:
    """Canned forecast."""
    return f"{city}: sunny, 22C"


class ScriptedLLM:
    """Round 1: text deltas + a tool call. Round 2: a closing line. Input
    "loop" streams forever, so /stop has something to interrupt. The round
    is read off the transcript — one agent instance serves every run."""

    async def stream(self, messages, tools, **params):
        if messages[-1].content == "loop":
            while True:
                yield LLMDelta(text="tick ")
                await asyncio.sleep(0.02)
        if messages[-1].role == "tool":
            yield LLMDelta(text="looks sunny.")
            yield LLMReply(message=Message(role="assistant", content="looks sunny."))
            return
        yield LLMDelta(text="checking ")
        yield LLMDelta(text="the forecast... ")
        yield LLMReply(stop_reason="tool_use", message=Message(
            role="assistant", content="checking the forecast... ",
            tool_calls=[ToolCall("c1", "weather", {"city": "lisbon"})]))


AGENT = Agent(ScriptedLLM(), tools=[weather])


async def serve_http(reader, writer) -> None:
    """One request, one response, then close. ponytail: a demo, not a web
    server — no keep-alive, no chunked request bodies, no HTTP/1.0, no header
    folding, no body cap, no auth. What a real server owes you here, a real
    framework owes you there."""
    try:
        method, target, _ = (await reader.readline()).decode().split()
        headers: dict[str, str] = {}
        while (line := await reader.readline()) not in (b"\r\n", b"\n", b""):
            k, _, v = line.decode().partition(":")
            headers[k.strip().lower()] = v.strip()
        body = await reader.readexactly(int(headers.get("content-length", 0)))
        path, _, query = target.partition("?")
        route, _, run_id = path.strip("/").partition("/")

        if method == "POST" and route == "run":
            h = AGENT.run(body.decode() or "weather?")
            RUNS[h.run_id] = h
            await _stream(writer, h.events(), h.run_id)
        elif method == "GET" and route == "events" and run_id in RUNS:
            await _stream(writer, RUNS[run_id].events(
                after_seq=int(parse_qs(query).get("after", ["-1"])[0])))
        elif method == "POST" and route == "stop" and run_id in RUNS:
            RUNS[run_id].stop()
            _plain(writer, "200 OK", "stopped\n")
        else:
            _plain(writer, "404 Not Found", "no such route or run\n")
        await writer.drain()
    except (ConnectionResetError, asyncio.IncompleteReadError, ValueError):
        pass  # client vanished mid-stream, or sent us junk
    finally:
        writer.close()


async def _stream(writer, events, run_id: str = "") -> None:
    """The whole SSE endpoint: headers, then sse() frames flushed as they land."""
    writer.write(("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                  "Cache-Control: no-cache\r\nConnection: close\r\n"
                  f"{f'X-Run-Id: {run_id}' if run_id else ''}\r\n\r\n").encode())
    async for frame in sse(events):
        writer.write(frame.encode())
        await writer.drain()


def _plain(writer, status: str, body: str) -> None:
    writer.write(f"HTTP/1.1 {status}\r\nContent-Type: text/plain\r\nContent-Length: "
                 f"{len(body)}\r\nConnection: close\r\n\r\n{body}".encode())


async def read_sse(response) -> list[tuple[str, int, dict]]:
    """Frames -> (event, id, data), consumed as they arrive. ponytail: our sse()
    emits no comments, retry fields or multi-line data, so one split per line
    is the whole parser."""
    frames, f = [], {}
    async for line in response.aiter_lines():
        if line:
            f.update([line.split(": ", 1)])
        elif f:
            frames.append((f["event"], int(f.get("id", -1)), json.loads(f["data"])))
            f = {}
    return frames


async def demo() -> None:
    """Ephemeral port, same process, real sockets: stream, resume, stop."""
    server = await asyncio.start_server(serve_http, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    async with server, httpx.AsyncClient(base_url=url, timeout=10) as c:
        async with c.stream("POST", "/run", content=b"weather?") as r:
            assert r.headers["content-type"] == "text/event-stream"
            run_id, frames = r.headers["x-run-id"], await read_sse(r)
        names = [n for n, _, _ in frames]
        assert names[0] == "step_start" and names[-1] == "done", names
        assert {"step_start", "step_delta", "step_end"} == set(names[:-1]), names
        kinds = {(d["kind"], d["phase"]) for _, _, d in frames[:-1]}
        assert {("text", "delta"), ("tool", "end")} <= kinds, kinds
        assert frames[-2][2]["payload"]["status"] == "completed", frames[-2]
        seqs = [s for _, s, _ in frames[:-1]]   # contiguous: seq is the cursor
        assert seqs == list(range(1, len(seqs) + 1)), seqs

        # reconnect: Last-Event-ID -> events(after_seq=) replays only the tail
        async with c.stream("GET", f"/events/{run_id}?after={seqs[-3]}") as r:
            tail = await read_sse(r)
        assert [s for _, s, _ in tail[:-1]] == seqs[-2:], tail

        # stop: "loop" ticks forever until someone pulls the plug
        async with c.stream("POST", "/run", content=b"loop") as r:
            stop = await c.post(f"/stop/{r.headers['x-run-id']}")
            frames = await read_sse(r)
        assert stop.text == "stopped\n" and frames[-1][0] == "done", stop.text
        assert frames[-2][2]["payload"]["status"] == "stopped", frames[-2]
        assert (await c.post("/stop/nope")).status_code == 404
    print("sse server demo ok")


async def serve(port: int = 8080) -> None:
    """Serve mode: the curl lines are in the module docstring."""
    async with await asyncio.start_server(serve_http, "0.0.0.0", port) as s:
        print(f"listening on :{port}")
        await s.serve_forever()


if __name__ == "__main__":
    asyncio.run(serve() if "serve" in sys.argv[1:] else demo())
