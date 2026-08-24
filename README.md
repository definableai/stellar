# stellar core

A complete agent harness heart in **under 2,000 lines you can read in an afternoon**. Durable sessions, long-lived agents, subagents, MCP, prompt caching, streaming, human-in-the-loop — the capabilities of a 15,000-line platform, at a tenth of the code, with every line yours to change. The core is stdlib-only; there is no framework underneath, only typed interfaces and a loop.

```
core/                 the heart — under 2000 lines, stdlib only, py.typed
├── types.py          Message, ToolCall, ToolResult, Usage, content Blocks
├── events.py         StepEvent = kind × phase, the single event shape
├── llm.py            LLM protocol + ReplyBuilder (the adapter skeleton)
├── tools.py          Tool, @tool, ToolCallContext, validate_args guardrail
├── hooks.py          @hook decorator — before/after llm + tool
├── session.py        durable append-only JSONL log, crash repair, resume
├── run.py            RunHandle: stream, replay, stop, steer; bounded buffers
├── worker.py         Worker: one agent, one session, one turn at a time
├── tracer.py         Tracer protocol + Console/Jsonl sinks
├── transport.py      sse() / ws_frames() — pure serializers
└── agent.py          the loop (~360 lines) — everything else serves it

internal/             the adapters we ship (not in the budget)
├── llm_openai.py     chat + responses;  llm_anthropic.py — + prompt caching
├── llm_litellm.py    100+ providers through one optional dep
├── llm_deepseek.py   OpenAI-compatible reuse of llm_openai;  llm_moonshot.py too
├── llm_common.py     helper: shared block translation for llm adapters
├── hook_approval.py  approval_gate() — HITL gate;  hook_compaction.py — token-aware
├── tool_fs.py        fs_tools(root) — read / write / edit / list, path-jailed
├── tool_bash.py      bash_tool(root) — one shell command, rooted at a directory
├── tool_subagent.py  subagent() — spawn a child derived from the parent
├── tool_mcp.py       mcp_tools(server) — MCP (stdio + streamable HTTP) → Tools
└── tool_schema.py    helper: signature + docstring → JSON Schema

external/             the adapters you write, same three kinds
tests/                fake-LLM suites + hypothesis property tests, no pytest
examples/cc/          the flagship: Claude Code rebuilt on this core
```

## Architecture

```mermaid
flowchart LR
    U[your code] -->|"run(input, session=s)"| A

    subgraph CORE["core — stdlib only, provider-blind, under 2000 lines"]
        A["agent.py<br/>the loop"]
        WK["worker.py<br/>long-lived turns"] -.drives.-> A
        A <--> HK["hooks.py<br/>@hook points"]
        A <--> LP["llm.py<br/>LLM Protocol + ReplyBuilder"]
        A <--> TL["tools.py<br/>Tool + validate_args"]
        A <--> SS["session.py<br/>durable log"]
        A -->|StepEvents| RH["run.py<br/>RunHandle"]
        RH --> TR["tracer.py<br/>protocol + sinks"]
        RH --> TP["transport.py<br/>sse / ws_frames"]
    end

    subgraph ADAPTERS["internal + external — adapters, all replaceable"]
        OA["llm_* factories"] -.implement.-> LP
        AP["hook_* factories"] -.hook into.-> HK
        TB["tool_* factories"] -.produce.-> TL
    end

    RH -->|"events / RunResult"| U
```

Everything crossing a boundary speaks the `types.py` vocabulary — the narrow waist. Adapters translate provider wire formats to it; the loop never sees provider data.

## The loop

```
run/start                            (request derived from the session log)
repeat up to max_steps:
    drain steering inbox             handle.send() lands here
    before_llm hooks                 may rewrite messages/tools (compaction…)
    text/start → text/delta* → text/end
    after_llm hooks
    no tool calls? → break
    per tool call (parallel or sequential):
        tool/start
        validate_args guardrail      bad args → readable error, handler untouched
        before_tool hooks            may short-circuit (permission gate)
        execute handler              streams tool/delta* progress
        after_tool hooks             may replace/redact the result
        tool/end
run/end                              always emitted — completed | truncated |
                                     stopped | error
```

Every new message is appended through the session as it happens, so the durable transcript and the live one cannot diverge.

## Durable sessions — the reliability spine

The session is an append-only JSONL log and the **source of truth**: each run derives its request from it and appends through it. Stolen from the event-sourcing harnesses, at message granularity instead of token granularity — 95% of the value, 3% of the machinery.

```python
with Session("task.jsonl") as s:          # or Session.load() to continue
    await agent.run("do the thing", session=s)
```

* Kill the process at any point. `Session.load()` repairs the log — dangling tool calls get synthetic error results, torn tail writes are truncated — and the next run continues with full history.
* Compaction never rewrites the log: hooks shape the *request view*, the log keeps everything.
* Single-writer enforced by `flock`; corruption is loud, never silent.
* Property-tested: a transcript cut at **any byte** loads back to an exact prefix with no dangling calls (`tests/test_properties.py`).

Try it on the Claude Code replica:

```
uv run python -m examples.cc.cc --session 7b2f "refactor the parser"
^C  (or kill -9 it)
uv run python -m examples.cc.cc --session 7b2f "continue"   # picks up where it died
```

## The extension seams

**LLM adapter** — implement one method; `ReplyBuilder` is the skeleton that makes it hard to get wrong:

```python
class MyLLM:
    async def stream(self, messages, tools, **params):
        b = ReplyBuilder()
        async for ev in provider_stream(...):
            if ev.is_text:        yield b.text(ev.chunk)
            elif ev.is_thinking:  yield b.reasoning(ev.chunk)
            elif ev.opens_call:   b.tool_call(ev.index, id=ev.id, name=ev.name)
            elif ev.is_args:      yield b.tool_args(ev.index, ev.fragment)
        yield b.reply()
```

The builder owns assembly: text joins in order, tool-call fragments accumulate per index, truncated argument JSON becomes a flagged call that the loop turns into a readable "re-issue the call" error for the model — never a silent drop, never a handler crash. See `internal/llm_anthropic.py` (~215 lines, with prompt caching).

**Tools** — a JSON Schema and a function; the context streams progress and reaches the run:

```python
@tool(parameters={"type": "object",
                  "properties": {"city": {"type": "string"}},
                  "required": ["city"]})
async def weather(ctx, city: str):
    "Get weather for a city."
    await ctx.emit_delta({"progress": f"fetching {city}"})
    return {"city": city, "temp_c": 31}
```

Arguments are validated against the schema before dispatch (`Tool.validate`, on by default) — whatever the model or adapter produces, your handler only ever sees kwargs that fit its signature. Prefer inferred schemas? `internal/tool_schema.py` builds them from the signature + docstring.

**Hooks** — a decorated function taking its context; attach as a flat list:

```python
@hook("before_tool")
def gate(ctx):
    if ctx.call.name in BLOCKED:
        ctx.result = ToolResult(ctx.call.id, ctx.call.name, "denied", is_error=True)

agent = Agent(llm, tools=[...], hooks=[
    gate,
    compaction(max_tokens=100_000),        # hook_compaction — token-aware, exact
    approval_gate(rules, asker=my_asker),  # HITL permission gate, full audit trail
])
```

`before_tool` setting `ctx.result` short-circuits execution — that one rule is your permission gate, HITL approval, cache hit, and dry-run mode. Hook failures abort the run (fail closed); tracer failures are swallowed (fail open); tool failures become error results the model can read (fail forward).

**Tracer** — `async on_event(event)`, every event in order. `JsonlTracer` (core/tracer.py) is a replayable run record; point OTel, your DB, or a TUI at the same seam.

## Adapters — how the seams get filled

An adapter is one file with one plain factory. You call the factory. You pass what it hands back to `Agent(...)`. That is the whole contract.

There are three kinds, named for the seam they fill: `llm_*.py` gives an LLM, `tool_*.py` gives Tools, `hook_*.py` gives a Hook. The ones we ship live in `internal/`. The ones you write live in `external/`, same convention.

```python
from core import Agent
from internal.llm_anthropic import AnthropicLLM              # -> an LLM
from internal.tool_fs import fs_tools                        # -> a list of Tools
from internal.hook_approval import allow, approval_gate, console_asker   # -> a Hook

agent = Agent(
    AnthropicLLM(model="claude-opus-5"),
    tools=fs_tools("workdir"),
    hooks=[approval_gate([allow("read_file")], console_asker)],
)
```

Composition happens here and only here. An agent runs with the llm, tools and hooks it was built with — it cannot add a tool or swap its own model while it runs. Need a different agent? Build a different one. It is one call.

Nothing is auto-discovered and nothing is auto-mounted. If a file in `internal/` or `external/` is not imported and called, it does not exist to the agent.

## Long-lived agents

```python
worker = Worker(agent, Session.load("agent.jsonl"))   # crash? load + continue
task = asyncio.create_task(worker.serve())            # idles on the queue for hours
worker.send("new task")          # its own turn
worker.steer("also check X")     # joins the active turn — never silently lost
worker.stop_turn()               # graceful, partial reply persisted
await worker.close()             # drains unrun inputs into the log, unanswered
```

`Worker` is core (`core/worker.py`), in about 100 lines. `uv run python -m examples.cc.cc --session 7b2f` is a REPL built on it.

## Subagents and MCP

```python
lead = Agent(llm, tools=[
    *fs_tools("workdir"), bash_tool("workdir"),
    subagent(name="researcher", description="Delegate a research task",
             system="Research deeply; report findings only."),
    *await mcp_tools(server),     # any MCP server's tools, validated like local ones
], hooks=[approval_gate(rules, asker)])
```

A subagent is just a tool that spawns a child **derived from the running parent**: same LLM (override with `llm=` for a cheaper child), same tools minus itself (`tools=` to narrow or re-enable recursion), and — critically — the same hooks, so the parent's permission gate guards the child's `bash` too. The child's entire event stream forwards through the parent handle (`tool/delta` events, nesting for grandchildren), stop propagates down, depth is guarded. `tool_mcp.py` speaks stdio and streamable HTTP, survives 1MB tool results, dead servers fail loudly.

## Serving

`handle.events(after_seq=n)` replays the buffer then follows live — reconnects with `Last-Event-ID` just work:

```python
@app.post("/agent/run")
async def run(req: RunRequest):
    handle = agent.run(req.input, session=sessions[req.id])
    RUNS[handle.run_id] = handle
    return StreamingResponse(sse(handle.events()), media_type="text/event-stream")
```

Buffers are bounded (`RunHandle.max_buffer/max_queue`): a slow consumer sheds its oldest events, never `run/end` (holds for `max_queue >= 2`); hours-long runs don't grow memory without bound.

## Invariants (the reliability contract)

1. Every step emits `start` and `end` exactly once; `delta` zero or more in between.
2. Every run ends with exactly one `run/end` — completed, truncated, stopped, or error. Always.
3. `seq` is strictly increasing; it is the total order and the resume cursor.
4. `stop()` is graceful: partial text persisted (`meta.interrupted`), in-flight tools end `cancelled`, every event still reaches tracers and subscribers.
5. The request is derived from the session log; with a session attached, the durable and live transcripts cannot diverge.
6. Tool failures never crash the loop; hook failures abort it; tracer failures are invisible.
7. The loop only speaks `types.py`. Provider data inside `agent.py` is a bug.
8. Composition is fixed at construction: for the whole life of an agent, its llm, tools and hooks are the ones it was built with.

## Event grammar

Three phases (`step_start` / `step_delta` / `step_end` on the wire) × four kinds:

| kind | start | delta | end |
|---|---|---|---|
| `run` | `messages`, `run_id` | — | `status`, `steps`, `usage`, `stop_reason`, `error` |
| `text` | `step` | `text`, `channel` (`text`\|`reasoning`\|`tool_args`), `index` | `text`, `tool_calls[]`, `stop_reason`, `partial`, `usage` |
| `tool` | `call_id`, `name`, `arguments` | `call_id` + tool-defined (subagents: `child`) | `call_id`, `name`, `result`, `is_error`, `error?` |
| `hook` | `point`, `hook` | — | `point`, `hook`, `error?` |

## Testing

```
uv run python tests/test_agent.py        # loop, sessions, subagents — fake LLM
uv run python tests/test_session.py      # round-trip, repair, corruption
uv run python tests/test_primitives.py   # fs jail, line window, edit uniqueness, bash
uv run python tests/test_properties.py   # hypothesis: crash-cut recovery & more
uv run python tests/test_llm_adapters.py # anthropic + openai on fake transports
uv run ruff check .                      # lint gate — stays clean
```

## Deliberate non-features

No runtime self-modification — composition is fixed at construction, so there is no mounting, no unmounting and no lifecycle to get wrong. No DI container, no service registry, no dependency resolver, no plugin discovery, no phase machine, no delta-level persistence, no code-mode, no SDK codegen, no retry policy in the loop (wrap your LLM adapter if you need one). Multimodal input is typed blocks (`text_block` / `image_block` / `file_block`) — PDFs included, translated per provider. Everything else is an adapter, a hook, a tracer, or a tool you write on top: the seams are there, the opinions are not.
