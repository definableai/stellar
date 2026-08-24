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

No ``setup(ctx)``: connecting and listing tools is async IO, and the core
adapter contract is sync composition (core/adapter.py).
"""

from __future__ import annotations

import asyncio
import json
import os
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
