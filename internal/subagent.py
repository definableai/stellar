"""Subagents: a tool that spawns a child agent DERIVED FROM ITS PARENT.

    lead = Agent(llm, tools=[read, bash, subagent(
        name="researcher",
        description="Delegate a deep research task to a focused child.",
        system="Research thoroughly; report findings only.",
    )], hooks=[approval_gate(rules, asker)])

Anything not overridden is inherited from the *running parent* at call
time (via ``ctx.run.agent``), on the same core:

    llm             parent's LLM (pass ``llm=`` for a cheaper child)
    tools           parent's tools minus this tool itself — a child can't
                    trivially self-spawn; pass ``tools=`` to narrow or to
                    re-enable recursion (the depth guard still backstops).
                    Exclusions accumulate down a chain: each child derives
                    from its immediate parent, so a grandchild's toolset
                    is strictly narrower than the root's
    hooks           parent's hooks — permission gates FOLLOW the
                    delegation; a child's bash faces the same approval
    system          parent's system prompt (usually override this one)
    max_steps, parallel_tools, params — parent's

Semantics:
    * The child's whole event stream forwards through the parent handle
      as ``tool/delta`` events (``payload["child"]``); grandchildren
      nest inside their parent's forwarded events.
    * Parent stop / tool timeout cancels the handler, which stops the
      child. ponytail: fire-and-forget stop — the child's last few
      events go unobserved while it winds down.
    * Depth rides run ``state["subagent_depth"]``; at ``max_depth`` the
      call fails as a readable tool error the model can route around.
    * Children are ephemeral (no session): the parent's log records the
      delegation and its result, not the child's internals.
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterable

from core import Agent, Hook, Tool, ToolCallContext, ToolSpec

_PROMPT_SCHEMA = {
    "type": "object",
    "properties": {"prompt": {"type": "string",
                              "description": "The task for the subagent."}},
    "required": ["prompt"],
}


def subagent(
    *,
    name: str,
    description: str,
    llm: Any = None,
    tools: Iterable[Tool] | None = None,
    hooks: Iterable[Hook] | Any = None,
    system: str | None = None,
    max_steps: int | None = None,
    max_depth: int = 3,
    timeout: float | None = None,
    parallel_safe: bool = True,
) -> Tool:
    async def handler(ctx: ToolCallContext, prompt: str = "") -> Any:
        depth = ctx.run.state.get("subagent_depth", 0)
        if depth >= max_depth:
            raise RuntimeError(f"subagent depth limit ({max_depth}) reached")
        parent = ctx.run.agent
        child_agent = Agent(
            llm=llm or parent.llm,
            tools=(list(tools) if tools is not None
                   else [t for t in parent.tools.values()
                         if t.spec.name != name]),
            hooks=hooks if hooks is not None else parent.hooks,
            system=system if system is not None else parent.system,
            max_steps=max_steps if max_steps is not None else parent.max_steps,
            parallel_tools=parent.parallel_tools,
            params=parent.params,
        )
        child = child_agent.run(prompt, state={"subagent_depth": depth + 1})
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
