"""One shortcut: a plain function becomes a Tool.

@tool takes one keyword, parallel=…, and nothing else on purpose. Want a
different name, description or schema? Write a Tool subclass — that is what
the class is for.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from functools import partial
from types import UnionType
from typing import (
    TYPE_CHECKING, Annotated, Any, Callable, Literal, Union, get_args,
    get_origin, get_type_hints, is_typeddict, overload,
)

from core.contracts import Tool, wrong

if TYPE_CHECKING:                       # names for the checker, no runtime edge
    from collections.abc import AsyncIterator

    from core.agent import Run
    from core.types import Part

__all__ = ["tool"]

TYPES = {                               # a hint the schema knows, or nothing
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    tuple: "array",
    set: "array",
    dict: "object",
}


@overload
def tool(fn: Callable[..., Any], /) -> Tool: ...
@overload
def tool(*, parallel: bool = ...) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    fn: Annotated[
        Callable[..., Any] | None,
        "called as it is, never bound — the wrapper's self goes unused"] = None,
    /,
    *,
    parallel: Annotated[
        bool, "True lets it run beside other parallel tools; serial is a barrier",
    ] = False,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """A plain function becomes a Tool: its name, its docstring, its signature.

    A sync function runs in a thread, an async one is awaited, and an async
    generator streams — every Part it yields rings tool.delta. Spell the
    first parameter run and the loop hands the run over; the schema never
    sees that one.

    Gives back an instance, ready for Agent(model, [that]). Raises
    ContractError on *args/**kwargs — an unreadable signature is no schema —
    and on anything without a __name__: the function's name is the tool's.
    """
    if fn is None:                      # @tool(parallel=True): come back with the fn
        return partial(tool, parallel=parallel)
    name = getattr(fn, "__name__", "")
    if not name:
        raise wrong("tool", f"@tool needs a named function, not {type(fn).__name__}")
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
           for p in inspect.signature(fn).parameters.values()):
        raise wrong("tool", f"@tool cannot read *args/**kwargs on {name}")
    wants = takes_run(fn)

    def call(run: Run, args: dict[str, Any]) -> Any:
        return fn(run, **args) if wants else fn(**args)

    execute: Callable[..., Any]         # one of three, picked by fn's kind
    if inspect.isasyncgenfunction(fn):
        # run is positional-only (/) so an arg may be named run too
        async def streams(self: Tool, run: Run, /,
                          **args: Any) -> AsyncIterator[Part]:
            """Pass the wrapped generator's Parts through, one at a time."""
            async for part in call(run, args):
                yield part
        execute = streams
    elif inspect.iscoroutinefunction(fn):
        async def awaits(self: Tool, run: Run, /, **args: Any) -> Any:
            """Await the wrapped coroutine."""
            return await call(run, args)
        execute = awaits
    else:
        async def bridges(self: Tool, run: Run, /, **args: Any) -> Any:
            """Run the wrapped sync function in a thread, so the loop breathes."""
            return await asyncio.to_thread(call, run, args)
        execute = bridges

    return type(name, (Tool,), {
        "name": name,
        "description": inspect.getdoc(fn) or "",
        "parameters": schema(fn),
        "execute": execute,
        "parallel": parallel,
    })()


def takes_run(
    fn: Annotated[
        Callable[..., Any],
        "only its parameter names are read; the annotations stay untouched"],
) -> bool:
    """True when the first parameter is spelled run: the loop fills that one in."""
    return next(iter(inspect.signature(fn).parameters), "") == "run"


def schema(
    fn: Annotated[
        Callable[..., Any],
        "read with eval_str, so a module's future-import strings resolve too"],
) -> dict[str, Any]:
    """The signature as a schema. No hint means anything; no default means required.

    An Annotated note becomes the property's description — the model reads it.
    """
    properties, required = {}, []
    skip = "run" if takes_run(fn) else None
    for name, p in inspect.signature(fn, eval_str=True).parameters.items():
        if name == skip:
            continue
        properties[name] = kind(p.annotation)     # no annotation reads as anything
        if p.default is p.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def kind(
    hint: Annotated[Any, "one evaluated hint, at any depth — never a string"],
) -> dict[str, Any]:
    """One hint as a schema, all the way down. What it cannot read is anything: {}."""
    origin, args = get_origin(hint), get_args(hint)
    if origin is Annotated:                   # a note this deep reaches the model too
        note = next((n for n in args[1:] if isinstance(n, str)), None)
        return kind(args[0]) | ({"description": note} if note else {})
    if origin in (Union, UnionType):
        rest = [a for a in args if a is not type(None)]
        one = kind(rest[0]) if len(rest) == 1 else {"anyOf": [kind(a) for a in rest]}
        if len(rest) == len(args):            # no None in it: the union stands as is
            return one
        t = one.get("type")
        return ({**one, "type": [t, "null"]} if isinstance(t, str)
                else {"anyOf": [*one.get("anyOf", [one]), {"type": "null"}]})
    if origin is Literal:
        shared = {TYPES.get(type(a)) for a in args}
        share = len(shared) == 1 and None not in shared
        return {"enum": list(args)} | ({"type": shared.pop()} if share else {})
    if origin in (list, set) or (origin is tuple and args[-1:] == (...,)):
        return {"type": "array", "items": kind(args[0])}
    if isinstance(hint, type) and issubclass(hint, Enum):
        return {"enum": [m.value for m in hint]}
    if is_dataclass(hint) or is_typeddict(hint):
        hints = get_type_hints(hint, include_extras=True)
        need = (hint.__required_keys__ if is_typeddict(hint) else
                {f.name for f in fields(hint)
                 if f.default is f.default_factory is MISSING})
        return {"type": "object",
                "properties": {n: kind(h) for n, h in hints.items()},
                "required": [n for n in hints if n in need]}
    # ponytail: dict[str, X] is a bare object — the value hint goes unsaid
    return {"type": t} if (t := TYPES.get(origin or hint)) else {}
