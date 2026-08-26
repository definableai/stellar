"""Two shortcuts: a function becomes a Tool, a function becomes a Hook.

@tool takes no arguments, on purpose. Want a different name, description or
schema? Write a Tool subclass — that is what the class is for.
"""

import inspect
from typing import get_origin

from core.contracts import EVENTS, Hook, Tool, wrong

TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def takes_agent(fn) -> bool:
    """True when the first parameter is spelled agent: the loop fills that one in."""
    return next(iter(inspect.signature(fn).parameters), "") == "agent"


def schema(fn) -> dict:
    """The signature as a schema. No hint means anything; no default means required."""
    properties, required = {}, []
    skip = "agent" if takes_agent(fn) else None
    for name, p in inspect.signature(fn).parameters.items():
        if name == skip:
            continue
        kind = TYPES.get(get_origin(p.annotation) or p.annotation)
        properties[name] = {"type": kind} if kind else {}
        if p.default is p.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def tool(fn) -> Tool:
    """A plain function becomes a Tool: its name, its docstring, its signature."""
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
           for p in inspect.signature(fn).parameters.values()):
        raise wrong("tool", f"@tool cannot read *args/**kwargs on {fn.__name__}")
    wants = takes_agent(fn)

    def execute(self, agent, **args):
        return fn(agent, **args) if wants else fn(**args)

    async def aexecute(self, agent, **args):
        return await execute(self, agent, **args)   # fn handed back a coroutine

    body = {
        "name": fn.__name__,
        "description": inspect.getdoc(fn) or "",
        "parameters": schema(fn),
    }
    if inspect.iscoroutinefunction(fn):
        body["aexecute"] = aexecute
    else:
        body["execute"] = execute
    return type(fn.__name__, (Tool,), body)()


def on(event: str, fn) -> Hook:
    """A plain function becomes a Hook that listens for one event."""
    if event not in EVENTS:
        raise wrong(
            "hook", f"{event!r} is not an event; pick one of: " + ", ".join(EVENTS)
        )
    return type(f"On_{event}", (Hook,), {event: lambda self, agent: fn(agent)})()
