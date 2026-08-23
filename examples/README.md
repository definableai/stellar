# examples

Every example runs **offline by default** — scripted fake LLMs, real everything else (real loop, real sessions on disk, real subprocesses, real sockets) — and exits 0 printing `... demo ok`, so the gallery doubles as a smoke suite:

```
for e in quickstart durable_chat human_approval multi_agent \
         long_lived_worker mcp_agent custom_adapter sse_server; do
    uv run python -m examples.$e || break
done
```

Read them in this order:

| example | shows | extra modes |
|---|---|---|
| `quickstart.py` | the smallest real agent: one `@tool`, streamed events, `RunResult` | `live` (OPENAI_API_KEY) |
| `durable_chat.py` | the durability spine: crash mid-tool-call → `Session.load` repairs → resumes with full history | `live "msg"` appends to `./chat.jsonl` |
| `human_approval.py` | HITL gating: `approval_gate` rules + asker, `"always"` learns, audit trail in the event stream | `interactive` — you are the asker |
| `multi_agent.py` | `subagent()` delegation derived from the parent; the forwarded child events rendered as an indented tree | — |
| `long_lived_worker.py` | the hours-long agent: send / steer mid-turn / stop a runaway / close-drains / crash / resume — the session file it prints is the proof | — |
| `mcp_agent.py` | a real MCP stdio subprocess (inline fake server) → `mcp_tools()` → agent; server-side state round-trips | swap in any real MCP server |
| `custom_adapter.py` | write an LLM adapter in ~25 lines with `ReplyBuilder` — alien wire format in, core contract out, malformed-args policy included | — |
| `sse_server.py` | serving with zero web framework: SSE stream, `after=` reconnect replay, stop endpoint | `serve` binds :8080, curl lines in the docstring |

**`cc/`** is the flagship: Claude Code rebuilt on this core — same captured system prompt and tool schemas, durable `--session <id>`, kill/resume, REPL on a Worker. See `cc/cc.py`.
