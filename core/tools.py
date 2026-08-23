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
    timeout: float | None = None  # secs -> error result. Async handlers only:
    # a blocking sync handler holds the event loop and cannot be preempted.
    validate: bool = True        # check arguments against spec.parameters
    # before dispatch: mismatches become a readable error result for the
    # model, and the handler never sees kwargs its signature can't take.


def tool(
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    parallel_safe: bool = True,
    timeout: float | None = None,
    validate: bool = True,
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
            name=name or getattr(fn, "__name__", "tool"),
            description=description or inspect.getdoc(fn) or "",
            parameters=parameters
            or {"type": "object", "properties": {}},
        )
        return Tool(spec=spec, handler=fn, parallel_safe=parallel_safe,
                    timeout=timeout, validate=validate)

    return wrap


_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str, "integer": int, "number": (int, float),
    "boolean": bool, "array": list, "object": dict, "null": type(None),
}


def validate_args(schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
    """Model-facing problems with ``args`` against a JSON-Schema subset
    ([] = valid). Covers type / required / properties / enum / items —
    a guardrail between adapter output and handler signatures, not a
    full validator. ponytail: no anyOf/format/pattern; add if hit."""
    problems: list[str] = []
    props: dict[str, Any] = schema.get("properties") or {}
    for key in schema.get("required") or []:
        if key not in args:
            problems.append(f"missing required argument: {key!r}")
    if props and schema.get("additionalProperties") is not True:
        problems += [f"unknown argument: {k!r}" for k in args if k not in props]
    for key, val in args.items():
        if key in props:
            problems += [f"{key}: {p}" for p in _check(props[key], val)]
    return problems


def _check(schema: dict[str, Any], val: Any) -> list[str]:
    t = schema.get("type")
    if isinstance(t, str):
        want = _JSON_TYPES.get(t)
        if want and (not isinstance(val, want)
                     or (t in ("integer", "number") and isinstance(val, bool))):
            return [f"expected {t}, got {type(val).__name__}"]
    if "enum" in schema and val not in schema["enum"]:
        return [f"expected one of {schema['enum']!r}, got {val!r}"]
    if isinstance(val, list) and isinstance(schema.get("items"), dict):
        return [f"[{i}]: {p}" for i, v in enumerate(val)
                for p in _check(schema["items"], v)]
    if isinstance(val, dict) and isinstance(schema.get("properties"), dict):
        return validate_args(schema, val)
    return []


async def call_maybe_async(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Invoke a sync or async callable uniformly."""
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result
