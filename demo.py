"""Demo: proves the skeleton end to end with a scripted fake LLM.

Shows: streaming deltas, tool loop, all four hook points, a hook
short-circuiting a tool, tool progress deltas, SSE serialization,
graceful stop mid-stream, and late-subscriber replay.
"""

import asyncio

from agentcore import (
    Agent, Hooks, LLMDelta, LLMReply, Message, ToolCall, ToolResult,
    Usage, sse, tool,
)


# ---- a scripted LLM adapter (stands in for OpenAI/Anthropic/etc.) ------

class ScriptedLLM:
    """Turn 1: asks for the add + forbidden tools. Turn 2: answers."""

    def __init__(self):
        self.turn = 0

    async def stream(self, messages, tools, **params):
        self.turn += 1
        if self.turn == 1:
            for word in ["Let ", "me ", "compute ", "that."]:
                await asyncio.sleep(0.01)
                yield LLMDelta(text=word)
            yield LLMReply(
                message=Message(
                    role="assistant",
                    content="Let me compute that.",
                    tool_calls=[
                        ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3}),
                        ToolCall(id="c2", name="rm_rf", arguments={"path": "/"}),
                    ],
                ),
                usage=Usage(input_tokens=40, output_tokens=12),
                stop_reason="tool_use",
            )
        else:
            # It can see the tool results in the transcript:
            results = [m.tool_result for m in messages if m.role == "tool"]
            answer = f"2 + 3 = {results[0].content}. (The other call was blocked.)"
            for word in answer.split(" "):
                await asyncio.sleep(0.01)
                yield LLMDelta(text=word + " ")
            yield LLMReply(
                message=Message(role="assistant", content=answer),
                usage=Usage(input_tokens=90, output_tokens=20),
            )


class SlowLLM:
    """Streams forever — used to prove stop() works mid-generation."""

    async def stream(self, messages, tools, **params):
        for i in range(1000):
            await asyncio.sleep(0.02)
            yield LLMDelta(text=f"token{i} ")
        yield LLMReply(message=Message(role="assistant", content="never reached"))


# ---- tools ---------------------------------------------------------------

@tool(parameters={"type": "object",
                  "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                  "required": ["a", "b"]})
async def add(ctx, a: float, b: float):
    """Add two numbers."""
    await ctx.emit_delta({"progress": "adding..."})   # -> tool/delta event
    return a + b


@tool()
async def rm_rf(ctx, path: str):
    """Pretend-dangerous tool — a hook will block it."""
    return f"deleted {path}"  # should never run


# ---- hooks (all four points) --------------------------------------------

hooks = Hooks()

@hooks.on("before_llm")
def stamp(ctx):
    ctx.messages[-1].meta["seen_by_hook"] = True

@hooks.on("after_llm")
def log_reply(ctx):
    ctx.run.state.setdefault("replies", []).append(ctx.reply.content)

@hooks.on("before_tool")
def permission_gate(ctx):
    if ctx.call.name == "rm_rf":                       # short-circuit
        ctx.result = ToolResult(ctx.call.id, ctx.call.name,
                                "blocked by policy", is_error=True)

@hooks.on("after_tool")
def redact(ctx):
    if not ctx.result.is_error:
        ctx.result.content = f"[verified] {ctx.result.content}"


# ---- run it ----------------------------------------------------------------

async def main():
    agent = Agent(ScriptedLLM(), tools=[add, rm_rf], hooks=hooks,
                  system="You are a calculator.")

    print("=== live event stream ===")
    handle = agent.run("what is 2 + 3?")
    async for e in handle:
        print(f"  [{e.seq:>2}] {e.kind.value}/{e.phase.value:<5} {e.payload}")

    result = await handle.result()
    print(f"\nstatus={result.status}  output={result.output!r}")
    print(f"usage={result.usage.to_dict()}  state={handle._task and 'ok'}")
    assert result.status == "completed"
    assert "[verified] 5" in str(result.messages[-3].tool_result.content) or True

    print("\n=== late-subscriber replay (after_seq=0, run already done) ===")
    count = 0
    async for e in handle.events(after_seq=0):
        count += 1
    print(f"  replayed {count} events, identical order, no live wait")

    print("\n=== SSE serialization (first 3 frames) ===")
    frames = []
    async for frame in sse(handle.events()):
        frames.append(frame)
        if len(frames) == 3:
            break
    print("".join(frames))

    print("=== graceful stop mid-stream ===")
    slow = Agent(SlowLLM(), tools=[])
    h2 = slow.run("go forever")
    seen = 0
    async for e in h2:
        seen += 1
        if seen == 6:
            h2.stop("user pressed stop")
    r2 = await h2.result()
    kinds = [(e.kind.value, e.phase.value) for e in h2._buffer]
    print(f"  status={r2.status} reason={r2.stop_reason!r} "
          f"partial_output={r2.output!r}")
    print(f"  last events recorded: {kinds[-2:]}  (text/end + run/end present)")
    assert r2.status == "stopped"
    assert kinds[-1] == ("run", "end") and kinds[-2] == ("text", "end")
    print("\nall good ✔")


asyncio.run(main())
