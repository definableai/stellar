"""One shortcut: a plain function becomes a Tool.

@tool takes no arguments, on purpose. Want a different name, description or
schema? Write a Tool subclass — that is what the class is for.
"""

import asyncio
import inspect
from typing import get_origin

from core.contracts import Tool, wrong

TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def takes_run(fn) -> bool:
    """True when the first parameter is spelled run: the loop fills that one in."""
    return next(iter(inspect.signature(fn).parameters), "") == "run"


def schema(fn) -> dict:
    """The signature as a schema. No hint means anything; no default means required."""
    properties, required = {}, []
    skip = "run" if takes_run(fn) else None
    for name, p in inspect.signature(fn).parameters.items():
        if name == skip:
            continue
        kind = TYPES.get(get_origin(p.annotation) or p.annotation)
        properties[name] = {"type": kind} if kind else {}
        if p.default is p.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def tool(fn) -> Tool:
    """A plain function becomes a Tool: its name, its docstring, its signature.

    A sync function runs in a thread, an async one is awaited, and an async
    generator streams — every Part it yields rings tool.delta. Spell the
    first parameter run and the loop hands the run over; the schema never
    sees that one.
    """
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
           for p in inspect.signature(fn).parameters.values()):
        raise wrong("tool", f"@tool cannot read *args/**kwargs on {fn.__name__}")
    wants = takes_run(fn)

    def call(run, args):
        return fn(run, **args) if wants else fn(**args)

    if inspect.isasyncgenfunction(fn):
        async def execute(self, run, /, **args):   # / so an arg may be named run too
            async for part in call(run, args):
                yield part
    elif inspect.iscoroutinefunction(fn):
        async def execute(self, run, /, **args):
            return await call(run, args)
    else:
        async def execute(self, run, /, **args):
            return await asyncio.to_thread(call, run, args)

    return type(fn.__name__, (Tool,), {
        "name": fn.__name__,
        "description": inspect.getdoc(fn) or "",
        "parameters": schema(fn),
        "execute": execute,
    })()
