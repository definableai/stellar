"""MCP client — plug any Model Context Protocol server's tools into an
Agent. Two transports, tools only (no resources/prompts/sampling).

    server = StdioMCP("npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp")
    await server.start()
    agent = Agent(llm, tools=await mcp_tools(server))
    ...
    await server.close()

    remote = HttpMCP("https://example.com/mcp", headers={"authorization": "..."})
    agent = Agent(llm, tools=await mcp_tools(remote))

Scope (ponytail, each a documented ceiling):
    * tools/list + tools/call only; server->client requests are answered
      "method not found", notifications are ignored.
    * HTTP reads each response fully — no incremental SSE progress; the
      final result is what matters to the loop.
    * No reconnect/resume; a dead server fails every pending call loudly.

Self-check (spawns a fake stdio server subprocess, mocks HTTP):

    uv run python -m internal.mcp
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

import httpx

from core import Tool, ToolSpec

PROTOCOL = "2025-06-18"
_CLIENT_INFO = {"name": "stellar", "version": "0.2"}


class MCPError(RuntimeError):
    def __init__(self, code: int, message: str):
        self.code, self.message = code, message
        super().__init__(f"MCP error {code}: {message}")


class StdioMCP:
    """One MCP server subprocess, JSON-RPC over stdin/stdout lines."""

    def __init__(self, command: str, *args: str,
                 env: dict[str, str] | None = None, timeout: float = 60.0):
        self.command, self.args = command, args
        self.env = {**os.environ, **(env or {})} if env else None
        self.timeout = timeout
        self._proc: asyncio.subprocess.Process | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._reader: asyncio.Task | None = None
        self._dead: MCPError | None = None
        self._id = 0

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            self.command, *self.args, env=self.env,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            limit=8 * 1024 * 1024)   # readline cap: default 64KiB bricks the
        # reader on the first big tool result (file reads, search dumps)
        self._reader = asyncio.create_task(self._read_loop())
        await self.call("initialize", {"protocolVersion": PROTOCOL,
                                       "capabilities": {},
                                       "clientInfo": _CLIENT_INFO})
        await self._send({"jsonrpc": "2.0",
                          "method": "notifications/initialized"})

    async def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if self._dead:
            raise self._dead
        self._id += 1
        rid = self._id           # local: self._id moves under concurrent calls
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await self._send({"jsonrpc": "2.0", "id": rid,
                          "method": method, "params": params or {}})
        try:
            msg = await asyncio.wait_for(fut, self.timeout)
        except asyncio.TimeoutError:
            raise MCPError(-1, f"timeout after {self.timeout}s: {method}") from None
        finally:
            self._pending.pop(rid, None)
        if "error" in msg:
            raise MCPError(msg["error"].get("code", -1),
                           msg["error"].get("message", "unknown"))
        return msg.get("result")

    async def close(self) -> None:
        if self._reader:
            self._reader.cancel()
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), 5)
            except asyncio.TimeoutError:
                self._proc.kill()

    async def _send(self, msg: dict[str, Any]) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(json.dumps(msg).encode() + b"\n")
        await self._proc.stdin.drain()

    async def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        exc = MCPError(-1, "server exited")
        try:
            while line := await self._proc.stdout.readline():
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue                  # stray non-protocol output
                if "id" in msg and ("result" in msg or "error" in msg):
                    fut = self._pending.get(msg["id"])
                    if fut and not fut.done():
                        fut.set_result(msg)
                elif "id" in msg and "method" in msg:  # server->client req
                    await self._send({"jsonrpc": "2.0", "id": msg["id"],
                                      "error": {"code": -32601,
                                                "message": "not supported"}})
                # notifications: ignored
        except Exception as ex:               # e.g. line over the limit
            exc = MCPError(-1, f"transport failed: {ex}")
        finally:
            # reader death (EOF, oversized line, decode explosion) must
            # fail everything loudly — current calls AND future ones —
            # never leave callers hanging to their timeouts
            self._dead = exc
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(exc)


class HttpMCP:
    """Streamable-HTTP MCP endpoint. Lazily initializes on first call."""

    def __init__(self, url: str, *, headers: dict[str, str] | None = None,
                 client: httpx.AsyncClient | None = None, timeout: float = 60.0):
        self.url = url
        self.client = client or httpx.AsyncClient(timeout=timeout)
        self.headers = {"accept": "application/json, text/event-stream",
                        "content-type": "application/json", **(headers or {})}
        self._session_id: str | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._id = 0

    async def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if not self._initialized and method != "initialize":
            async with self._init_lock:       # one initializer, others wait
                if not self._initialized:
                    await self.call("initialize",
                                    {"protocolVersion": PROTOCOL,
                                     "capabilities": {},
                                     "clientInfo": _CLIENT_INFO})
                    await self._post({"jsonrpc": "2.0",
                                      "method": "notifications/initialized"})
                    # only now: a failed initialize stays retryable, and the
                    # version header never rides on the negotiation itself
                    self._initialized = True
        self._id += 1
        rid = self._id
        r = await self._post({"jsonrpc": "2.0", "id": rid,
                              "method": method, "params": params or {}})
        if sid := r.headers.get("mcp-session-id"):
            self._session_id = sid
        msg = self._extract(r, rid)
        if "error" in msg:
            raise MCPError(msg["error"].get("code", -1),
                           msg["error"].get("message", "unknown"))
        return msg.get("result")

    async def close(self) -> None:
        if self._session_id:                  # best-effort session teardown
            try:
                await self.client.delete(self.url, headers=self._headers())
            except httpx.HTTPError:
                pass
        await self.client.aclose()

    def _headers(self) -> dict[str, str]:
        h = dict(self.headers)
        if self._session_id:
            h["mcp-session-id"] = self._session_id
        if self._initialized:
            h["mcp-protocol-version"] = PROTOCOL
        return h

    async def _post(self, msg: dict[str, Any]) -> httpx.Response:
        r = await self.client.post(self.url, json=msg, headers=self._headers())
        if r.status_code >= 400:
            raise MCPError(r.status_code, r.text[:500])
        return r

    @staticmethod
    def _extract(r: httpx.Response, rid: int) -> dict[str, Any]:
        """The response for ``rid``, from a JSON body or an SSE body.
        ponytail: whole body read at once; progress events are skipped."""
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        for line in r.text.splitlines():      # text/event-stream
            if line.startswith("data:"):
                try:
                    msg = json.loads(line[5:].strip())
                except ValueError:
                    continue
                if msg.get("id") == rid:
                    return msg
        raise MCPError(-1, f"no response for request {rid} in SSE body")


async def mcp_tools(server: Any, *, prefix: str = "",
                    parallel_safe: bool = False,
                    timeout: float | None = None) -> list[Tool]:
    """tools/list (paginated) -> core Tools. ``prefix`` namespaces tool
    names — without it, colliding names across servers silently resolve
    to the last one registered (Agent.tools is a dict). parallel_safe
    defaults False — MCP servers are commonly stateful."""
    tools: list[Tool] = []
    cursor: str | None = None
    while True:
        res = await server.call("tools/list",
                                {"cursor": cursor} if cursor else {})
        for t in res.get("tools", []):
            tools.append(_make_tool(server, t, prefix, parallel_safe, timeout))
        cursor = res.get("nextCursor")
        if not cursor:
            return tools


def _make_tool(server: Any, t: dict[str, Any], prefix: str,
               parallel_safe: bool, timeout: float | None) -> Tool:
    name = t["name"]

    async def handler(ctx: Any, **arguments: Any) -> Any:
        res = await server.call("tools/call",
                                {"name": name, "arguments": arguments})
        texts = [b.get("text", "") for b in res.get("content", [])
                 if b.get("type") == "text"]
        out = "\n".join(x for x in texts if x) \
            or json.dumps(res.get("content", []), default=str)
        if res.get("isError"):
            raise RuntimeError(out or "MCP tool error")
        return res.get("structuredContent") or out

    return Tool(spec=ToolSpec(name=prefix + name,
                              description=t.get("description", ""),
                              parameters=t.get("inputSchema")
                              or {"type": "object", "properties": {}}),
                handler=handler, parallel_safe=parallel_safe, timeout=timeout)


if __name__ == "__main__":
    _FAKE_SERVER = r'''
import json, sys
def send(m): sys.stdout.write(json.dumps(m)+"\n"); sys.stdout.flush()
print("stray log line that is not JSON")   # client must tolerate this
sys.stdout.flush()
for line in sys.stdin:
    m = json.loads(line)
    mid, meth = m.get("id"), m.get("method")
    if meth == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": m["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake", "version": "0"}}})
    elif meth == "tools/list":
        page = m.get("params", {}).get("cursor")
        if page is None:
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [
                {"name": "echo", "description": "Echo back",
                 "inputSchema": {"type": "object", "properties":
                                 {"text": {"type": "string"}},
                                 "required": ["text"]}}],
                "nextCursor": "p2"}})
        else:
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [
                {"name": "boom", "description": "Always errors",
                 "inputSchema": {"type": "object", "properties": {}}}]}})
    elif meth == "hang":
        pass                                   # never answered
    elif meth == "tools/call":
        p = m["params"]
        if p["name"] == "echo":
            send({"jsonrpc": "2.0", "id": mid, "result": {"content": [
                {"type": "text", "text": "echo: " + p["arguments"]["text"]}]}})
        elif p["name"] == "big":
            send({"jsonrpc": "2.0", "id": mid, "result": {"content": [
                {"type": "text", "text": "x" * 200_000}]}})
        else:
            send({"jsonrpc": "2.0", "id": mid, "result": {"content": [
                {"type": "text", "text": "kaboom"}], "isError": True}})
    elif mid is not None:
        send({"jsonrpc": "2.0", "id": mid,
              "error": {"code": -32601, "message": "nope"}})
'''

    async def _selfcheck() -> None:
        from core import Agent, LLMReply, Message, ToolCall

        # --- stdio, against a real subprocess -------------------------
        server = StdioMCP(sys.executable, "-c", _FAKE_SERVER, timeout=10)
        await server.start()
        tools = await mcp_tools(server)
        assert [t.spec.name for t in tools] == ["echo", "boom"]  # paginated
        assert tools[0].spec.parameters["required"] == ["text"]

        class Scripted:
            def __init__(self, replies): self.replies = list(replies)
            async def stream(self, messages, tools, **params):
                yield self.replies.pop(0)

        agent = Agent(Scripted([
            LLMReply(message=Message(role="assistant", tool_calls=[
                ToolCall("c1", "echo", {"text": "hi"}),
                ToolCall("c2", "boom", {})]), stop_reason="tool_use"),
            LLMReply(message=Message(role="assistant", content="done")),
        ]), tools=tools)
        result = await agent.run("go")
        assert result.status == "completed"
        tool_msgs = [m for m in result.messages if m.role == "tool"]
        assert tool_msgs[0].tool_result.content == "echo: hi"
        assert tool_msgs[1].tool_result.is_error
        assert "kaboom" in tool_msgs[1].tool_result.content

        # unknown method -> loud MCPError
        try:
            await server.call("resources/list")
            raise AssertionError("expected MCPError")
        except MCPError as e:
            assert e.code == -32601

        # concurrent calls: ids stay paired, nothing leaks in _pending
        results = await asyncio.gather(
            *(server.call("tools/call",
                          {"name": "echo", "arguments": {"text": str(i)}})
              for i in range(10)))
        assert [r["content"][0]["text"] for r in results] == \
            [f"echo: {i}" for i in range(10)]
        assert not server._pending

        # >64KiB result survives (limit raised past asyncio's default)
        big = await server.call("tools/call", {"name": "big", "arguments": {}})
        assert len(big["content"][0]["text"]) == 200_000

        # per-call timeout surfaces as MCPError, and cleans up
        server.timeout = 0.2
        try:
            await server.call("hang")
            raise AssertionError("expected timeout MCPError")
        except MCPError as e:
            assert "timeout" in e.message
        assert not server._pending
        server.timeout = 10
        await server.close()

        # --- HTTP, mocked: JSON response, SSE response, session id ----
        state = {"sid": None, "inits": 0}

        def handle(req: httpx.Request) -> httpx.Response:
            if req.method == "DELETE":       # session teardown, empty body
                return httpx.Response(200)
            msg = json.loads(req.content)
            if msg.get("method") == "initialize":
                state["inits"] += 1
                return httpx.Response(200, headers={"mcp-session-id": "s1"},
                    json={"jsonrpc": "2.0", "id": msg["id"], "result": {
                        "protocolVersion": PROTOCOL, "capabilities": {},
                        "serverInfo": {"name": "h", "version": "0"}}})
            if msg.get("method") == "notifications/initialized":
                return httpx.Response(202)
            state["sid"] = req.headers.get("mcp-session-id")
            if msg.get("method") == "tools/list":
                return httpx.Response(200, json={"jsonrpc": "2.0",
                    "id": msg["id"], "result": {"tools": [
                        {"name": "hello", "description": "",
                         "inputSchema": {"type": "object", "properties": {}}}]}})
            # tools/call answered over SSE, with noise events around it
            body = ("data: {\"method\": \"notifications/progress\"}\n\n"
                    f'data: {{"jsonrpc": "2.0", "id": {msg["id"]}, '
                    f'"result": {{"content": [{{"type": "text", '
                    f'"text": "hi from http"}}]}}}}\n\n')
            return httpx.Response(200, content=body,
                                  headers={"content-type": "text/event-stream"})

        http = HttpMCP("http://fake/mcp", client=httpx.AsyncClient(
            transport=httpx.MockTransport(handle)))
        htools = await mcp_tools(http, prefix="h_")
        assert [t.spec.name for t in htools] == ["h_hello"]
        assert state["inits"] == 1                       # lazy init, once

        class Ctx:  # ToolCallContext stand-in — handler only passes args
            pass
        out = await htools[0].handler(Ctx())
        assert out == "hi from http"                     # via SSE body
        assert state["sid"] == "s1"                      # session propagated
        await http.close()

        # failed initialize is retryable; version header never on the
        # negotiation request, always on post-init requests
        st = {"inits": 0}

        def flaky(req: httpx.Request) -> httpx.Response:
            if req.method == "DELETE":
                return httpx.Response(200)
            m = json.loads(req.content)
            if m.get("method") == "initialize":
                st["inits"] += 1
                assert "mcp-protocol-version" not in req.headers
                if st["inits"] == 1:
                    return httpx.Response(500, text="boom")
                return httpx.Response(200, json={
                    "jsonrpc": "2.0", "id": m["id"], "result": {
                        "protocolVersion": PROTOCOL, "capabilities": {},
                        "serverInfo": {"name": "f", "version": "0"}}})
            if m.get("method") == "notifications/initialized":
                return httpx.Response(202)
            assert req.headers.get("mcp-protocol-version") == PROTOCOL
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": m["id"],
                                             "result": {"tools": []}})

        fh = HttpMCP("http://fake/mcp", client=httpx.AsyncClient(
            transport=httpx.MockTransport(flaky)))
        try:
            await fh.call("tools/list")
            raise AssertionError("first call must fail on init 500")
        except MCPError:
            pass
        assert (await fh.call("tools/list"))["tools"] == []   # init retried
        assert st["inits"] == 2
        await fh.close()

        print("mcp client self-check ok")

    asyncio.run(_selfcheck())
