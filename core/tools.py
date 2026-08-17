"""The tool layer.

A tool is a spec (name + description + JSON Schema) plus an async
handler. The handler receives a ``ToolCallContext`` first, then the
call's arguments as keyword args. Through the context a tool can
stream progress (``await ctx.emit_delta({...})`` -> tool/delta events)
and request a run stop (``ctx.run.stop(...)``).

Schemas are plain JSON Schema dicts on purpose: derive them however
you like (by hand, pydantic, TypeBox-style builders) — the core does
not care.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from .types import ToolCall

if TYPE_CHECKING:
    from .run import RunContext


@dataclass
class ToolSpec:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class ToolCallContext:
    """What a tool handler sees while it executes one call."""

    run: "RunContext"
    call: ToolCall
    emit_delta: Callable[[dict[str, Any]], Awaitable[Any]]  # -> tool/delta event


@dataclass
class Tool:
    spec: ToolSpec
    handler: Callable[..., Any]  # (ctx: ToolCallContext, **arguments) -> Any
    parallel_safe: bool = True   # False (writes/side effects) forces the batch sequential


def tool(
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    parallel_safe: bool = True,
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator: turn a function into a Tool.

        @tool(parameters={"type": "object",
                          "properties": {"a": {"type": "number"},
                                         "b": {"type": "number"}},
                          "required": ["a", "b"]})
        async def add(ctx, a: float, b: float):
            return a + b
    """

    def wrap(fn: Callable[..., Any]) -> Tool:
        spec = ToolSpec(
            name=name or fn.__name__,
            description=description or inspect.getdoc(fn) or "",
            parameters=parameters
            or {"type": "object", "properties": {}},
        )
        return Tool(spec=spec, handler=fn, parallel_safe=parallel_safe)

    return wrap


async def call_maybe_async(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Invoke a sync or async callable uniformly."""
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result
