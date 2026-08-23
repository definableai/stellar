"""Subagents: any Agent becomes a Tool of another Agent.

    researcher = Agent(llm, tools=[search], system="research deeply")
    lead = Agent(llm, tools=[
        as_tool(researcher, name="researcher",
                description="Delegate a research task to the researcher"),
    ])

Composition, not machinery:

    * The child's entire event stream forwards into the parent as
      ``tool/delta`` events (``payload["child"]`` = the child's StepEvent
      dict), so one subscriber on the parent handle sees the whole tree —
      grandchildren nest inside their parent's forwarded events.
    * Parent stop / tool timeout cancels the handler, which stops the
      child gracefully. ponytail: fire-and-forget stop — the child's
      last few events go unobserved while it winds down.
    * Depth is carried in run ``state["subagent_depth"]``; at
      ``max_depth`` the call fails as a normal tool error the model can
      read and route around.
    * The child returns its final text; a non-completed child prefixes
      its status, e.g. ``[truncated] ...`` — visible to the model.
      Child runs are ephemeral (no session); give the child agent its
      own durable session by wrapping run() yourself if you need it.
"""

from __future__ import annotations

import asyncio
from typing import Any

from core import Tool, ToolCallContext, ToolSpec

_PROMPT_SCHEMA = {
    "type": "object",
    "properties": {"prompt": {"type": "string",
                              "description": "The task for the subagent."}},
    "required": ["prompt"],
}


def as_tool(
    agent: Any,                    # Agent; Any avoids the circular import
    *,
    name: str,
    description: str,
    max_depth: int = 3,
    timeout: float | None = None,
    parallel_safe: bool = True,
) -> Tool:
    async def handler(ctx: ToolCallContext, prompt: str = "") -> Any:
        depth = ctx.run.state.get("subagent_depth", 0)
        if depth >= max_depth:
            raise RuntimeError(f"subagent depth limit ({max_depth}) reached")
        child = agent.run(prompt, state={"subagent_depth": depth + 1})
        try:
            async for e in child.events():
                await ctx.emit_delta({"child": e.to_dict()})
            result = await child
        except asyncio.CancelledError:   # parent stop / timeout: take it down
            child.stop("parent cancelled")
            raise
        if result.status == "error":
            err = result.error
            raise RuntimeError(f"subagent failed: {err.type}: {err.message}"
                               if err else "subagent failed")
        if result.status != "completed":
            return f"[{result.status}] {result.output or ''}"
        return result.output

    return Tool(
        spec=ToolSpec(name=name, description=description,
                      parameters=_PROMPT_SCHEMA),
        handler=handler, parallel_safe=parallel_safe, timeout=timeout,
    )
