"""The loop: ask the model, run what it asked for, ask again.

Nothing here imports the Agent class. These functions take any object with
the right attributes, so agent.py can import this file and this file never
has to import back.
"""

import inspect
import json

from core.contracts import EVENTS, Hook, Model, Stop, Tool, fire, fold, wrong
from core.types import Message, ToolCall


def _own(obj, base, name):
    """The method obj's own class writes itself, or None.

    Subclasses pass. So do look-alikes that never heard of the base class.
    A plain function has none of the names, so it fails.
    """
    fn = getattr(type(obj), name, None)
    return None if fn is None or fn is getattr(base, name, None) else fn


def _sync(fn) -> bool:
    """True when the method was not written async — the loop cannot run it."""
    return not (inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn))


def check(agent) -> None:
    """Read the wiring and say what a dev got wrong, before it runs."""
    invoke = _own(agent.model, Model, "invoke")
    if invoke is None:
        raise wrong("model", f"{type(agent.model).__name__} is not a Model: "
                    "subclass Model and write an async invoke, or "
                    "ProviderModel and write encode + send")
    if _sync(invoke):
        raise wrong("model", f"{type(agent.model).__name__}.invoke must be "
                    "async — sync code is one asyncio.to_thread line inside it")
    seen = set()
    for tool in agent.tools:
        name = getattr(tool, "name", None)
        if not name:
            raise wrong("tool", f"{type(tool).__name__} needs a name")
        if not isinstance(getattr(tool, "parameters", None), dict):
            raise wrong("tool", f"tool {name!r} needs parameters to be a dict")
        execute = _own(tool, Tool, "execute")
        if execute is None:
            raise wrong("tool", f"tool {name!r} defines no execute")
        if _sync(execute):
            raise wrong("tool", f"tool {name!r}.execute must be async — sync "
                        "code is one asyncio.to_thread line inside it")
        if name in seen:
            raise wrong("tool", f"two tools answer to {name!r}")
        seen.add(name)
    for hook in agent.hooks:
        if not any(_own(hook, Hook, event) for event in EVENTS):
            raise wrong(
                "hook",
                f"{type(hook).__name__} listens for nothing; write one of: "
                + ", ".join(EVENTS),
            )


def coerce(result, call: ToolCall) -> Message:
    """Whatever the tool handed back, make it one tool message."""
    if isinstance(result, Message):
        result.role = "tool"
        result.tool_call_id = result.tool_call_id or call.id
        return result
    if not isinstance(result, str):
        result = json.dumps(result, default=str)
    return Message("tool", result, tool_call_id=call.id)


async def _use(agent, tool, call) -> None:
    """Run one tool. An async generator streams: each Part rings tool_delta."""
    out = tool.execute(agent, **call.args)
    if not inspect.isasyncgen(out):
        agent.result = await out
        return
    agent.result = Message("tool", tool_call_id=call.id)
    try:
        async for part in out:
            fold(agent.result, part)
            agent.delta = part
            await fire(agent, "tool_delta")
    finally:
        agent.delta = None


async def run(agent):
    """Ask, act, repeat. Ends when the model stops asking for tools."""
    try:
        await fire(agent, event="run_pre")
        while True:
            check(agent)                     # the wiring may have changed
            agent.step += 1
            await fire(agent, event="model_pre")
            answer = await agent.model.invoke(agent)
            if not isinstance(answer, Message):
                raise wrong(
                    kind="model",
                    problem=f"{type(agent.model).__name__}.invoke returned "
                    f"{type(answer).__name__}, expected Message",
                )
            agent.response = answer
            await fire(agent, event="model_post")
            agent.messages.append(agent.response)   # a hook may have swapped it
            calls = agent.response.tool_calls
            if not calls:
                break
            for call in calls:
                agent.call, agent.result = call, None
                await fire(agent, event="tool_pre")       # a filled result means denied
                if agent.result is None:
                    tool = next((t for t in agent.tools if t.name == call.name), None)
                    if tool is None:
                        agent.result = f"error: unknown tool: {call.name}"
                    else:
                        try:
                            await _use(agent, tool, call)
                        except Stop:
                            raise                   # Stop always means stop
                        except Exception as ex:
                            agent.result = f"error: {type(ex).__name__}: {ex}"
                agent.result = coerce(agent.result, call)
                await fire(agent, event="tool_post")      # this always sees a message
                agent.messages.append(agent.result)
                agent.call = agent.result = None
    except Stop:
        pass
    finally:
        try:
            await fire(agent, event="run_post")           # teardown, no matter what
        except Stop:
            pass                                    # too late to stop; ignore
    return agent
