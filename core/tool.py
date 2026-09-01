"""One shortcut: a plain function becomes a Tool.

@tool takes no arguments, on purpose. Want a different name, description or
schema? Write a Tool subclass — that is what the class is for.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Annotated, Any, Callable, get_args, get_origin

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
    dict: "object",
}


def tool(
    fn: Annotated[
        Callable[..., Any],
        "called as it is, never bound — the wrapper's self goes unused"],
) -> Tool:
    """A plain function becomes a Tool: its name, its docstring, its signature.

    A sync function runs in a thread, an async one is awaited, and an async
    generator streams — every Part it yields rings tool.delta. Spell the
    first parameter run and the loop hands the run over; the schema never
    sees that one.

    Gives back an instance, ready for Agent(model, [that]). Raises
    ContractError on *args/**kwargs — an unreadable signature is no schema —
    and on anything without a __name__: the function's name is the tool's.
    """
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
        ann = p.annotation
        hint, *notes = get_args(ann) if get_origin(ann) is Annotated else (ann,)
        kind = TYPES.get(get_origin(hint) or hint)
        properties[name] = {"type": kind} if kind else {}
        if notes and isinstance(notes[0], str):   # the note the model reads
            properties[name]["description"] = notes[0]
        if p.default is p.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}
