"""An MCP server's tools, plugged straight into an Agent.

Shows: spawn an MCP stdio server, turn its ``tools/list`` into core Tools
with ``mcp_tools()``, hand them to an Agent, and let the loop drive
``tools/call`` — including a server-side tool error the model can route
around. State lives in the server process, so the notes written in step 1
are still there when step 2 reads them back.

Offline by design: the server here is a tiny "notes" server inlined as a
``python -c`` script, but it is a *real* subprocess speaking real JSON-RPC
over stdin/stdout, and the LLM is scripted. Nothing hits the network.
Swap either side independently — a real server:

    server = StdioMCP("npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp")

or a real LLM (``internal.llm.anthropic``), or both. The plumbing above is
identical either way; only the asserts below are pinned to this fake server.

    uv run python -m examples.mcp_agent
"""

from __future__ import annotations

import asyncio
import sys

from core import Agent, LLMReply, Message, ToolCall, new_id
from internal.mcp import StdioMCP, mcp_tools

# A minimal MCP server: two tools over one list that lives for the life of
# the process. ponytail: no framework, no SDK — the protocol is small enough
# to answer by hand, which is the point of showing it here.
_NOTES_SERVER = r'''
import json, sys
NOTES = []                                     # real, in-process state
TOOLS = [
    {"name": "add_note", "description": "Save a note.",
     "inputSchema": {"type": "object",
                     "properties": {"text": {"type": "string"}},
                     "required": ["text"]}},
    {"name": "list_notes", "description": "List every saved note.",
     "inputSchema": {"type": "object", "properties": {}}},
]
def send(m): sys.stdout.write(json.dumps(m) + "\n"); sys.stdout.flush()
def text(mid, s, err=False):
    send({"jsonrpc": "2.0", "id": mid, "result": {
        "content": [{"type": "text", "text": s}], "isError": err}})
for line in sys.stdin:
    m = json.loads(line)
    mid, meth = m.get("id"), m.get("method")
    if meth == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": m["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "notes", "version": "0"}}})
    elif meth == "tools/list":
        send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
    elif meth == "tools/call":
        p = m["params"]
        if p["name"] == "list_notes":
            text(mid, "\n".join("%d. %s" % (i, n)
                                for i, n in enumerate(NOTES, 1)) or "(empty)")
        elif p["arguments"].get("text", "").strip():
            NOTES.append(p["arguments"]["text"])
            text(mid, "saved note %d" % len(NOTES))
        else:
            text(mid, "note text is empty", err=True)
    elif mid is not None:
        send({"jsonrpc": "2.0", "id": mid,
              "error": {"code": -32601, "message": "not supported"}})
'''


class ScriptedLLM:
    """Stand-in LLM: one canned reply per step (see core/llm.py)."""

    def __init__(self, *replies: LLMReply):
        self.replies = list(replies)

    async def stream(self, messages, tools, **params):
        # the MCP schemas really do reach the model layer
        assert {t.name for t in tools} == {"add_note", "list_notes"}
        yield self.replies.pop(0)


def _calls(*calls: tuple[str, dict]) -> LLMReply:
    """An assistant turn that asks for tools, in order."""
    return LLMReply(message=Message(role="assistant", tool_calls=[
        ToolCall(new_id("call"), name, args) for name, args in calls]),
        stop_reason="tool_use")


async def main() -> None:
    server = StdioMCP(sys.executable, "-c", _NOTES_SERVER, timeout=10)
    try:
        # start() inside the try: a server that dies mid-handshake has
        # already forked, so only the finally below can reap it
        await server.start()
        tools = await mcp_tools(server)          # tools/list -> core Tools
        # parallel_safe=False by default (MCP servers are stateful), so the
        # calls in a step run in order — that ordering is what the asserts pin
        assert [t.spec.name for t in tools] == ["add_note", "list_notes"]
        assert tools[0].spec.parameters["required"] == ["text"]
        print("mcp tools:", [t.spec.name for t in tools])

        agent = Agent(ScriptedLLM(
            # step 1: one good note, one the server rejects
            _calls(("add_note", {"text": "buy milk"}),
                   ("add_note", {"text": "  "})),
            # step 2: recover, then read back what the server kept
            _calls(("add_note", {"text": "ship stellar v0.3"}),
                   ("list_notes", {})),
            LLMReply(message=Message(role="assistant", content="2 notes saved.")),
        ), tools=tools)
        result = await agent.run("jot down my todos, then read them back")

        out = [m.tool_result for m in result.messages if m.role == "tool"]
        for r in out:
            print(f"  {r.name:<10} -> {'[error] ' if r.is_error else ''}"
                  f"{r.content!r}")
        print("llm:", result.output)

        assert result.status == "completed"
        assert out[0].content == "saved note 1"
        assert out[1].is_error and "empty" in out[1].content   # server said no
        assert out[2].content == "saved note 2"
        # the payoff: state survived four separate JSON-RPC round trips
        # through a real subprocess, and the rejected note never landed
        assert out[3].content == "1. buy milk\n2. ship stellar v0.3"
    finally:
        await server.close()                     # even when an assert blows up

    print("mcp agent demo ok")


if __name__ == "__main__":
    asyncio.run(main())
