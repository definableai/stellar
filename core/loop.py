"""The loop: ask the model, run what it asked for, ask again.

Nothing here imports the Agent class. These functions take any object with
the right attributes, so agent.py can import this file and this file never
has to import back.
"""

import inspect
import json

from core.contracts import EVENTS, Hook, Model, ProviderModel, Stop, Tool, wrong
from core.types import Message, ToolCall


def _overrides(obj, base, *names) -> bool:
    """True if obj's own class writes any of these methods itself.

    Subclasses pass. So do look-alikes that never heard of the base class.
    A plain function has none of the names, so it fails.
    """
    for name in names:
        own = getattr(type(obj), name, None)
        if own is not None and own is not getattr(base, name, None):
            return True
    return False


def check(agent) -> None:
    """Read the wiring and say what a dev got wrong, before it runs."""
    if not _overrides(agent.model, Model, "invoke", "ainvoke"):
        raise wrong(
            "model",
            f"{type(agent.model).__name__} defines neither invoke nor ainvoke"
            " — wrap it in a Model subclass",
        )
    for key, tool in agent.tools.items():
        name = getattr(tool, "name", None)
        if not name or name != key:
            raise wrong(
                "tool", f"tools[{key!r}] calls itself {name!r}; the two must match"
            )
        if not isinstance(getattr(tool, "parameters", None), dict):
            raise wrong("tool", f"tool {key!r} needs parameters to be a dict")
        if not _overrides(tool, Tool, "execute", "aexecute"):
            raise wrong("tool", f"tool {key!r} defines neither execute nor aexecute")
    for hook in agent.hooks:
        if not _overrides(hook, Hook, *EVENTS):
            raise wrong(
                "hook",
                f"{type(hook).__name__} listens for nothing; write one of: "
                + ", ".join(EVENTS),
            )


async def fire(agent, event: str) -> None:
    """Ring one bell. Every hook hears it, in list order."""
    for hook in list(agent.hooks):           # a hook may add hooks mid-ring
        listen = getattr(hook, event, None)
        answer = listen(agent) if listen else None
        if inspect.isawaitable(answer):
            await answer


def add(agent, *tools) -> None:
    """Put tools in the toolbox. Two tools with one name is a mistake."""
    for tool in tools:
        if tool.name in agent.tools:
            raise wrong("tool", f"two tools answer to {tool.name!r}")
        agent.tools[tool.name] = tool


def coerce(result, call: ToolCall) -> Message:
    """Whatever the tool handed back, make it one tool message."""
    if isinstance(result, Message):
        result.role = "tool"
        result.tool_call_id = result.tool_call_id or call.id
        return result
    if not isinstance(result, str):
        result = json.dumps(result, default=str)
    return Message("tool", result, tool_call_id=call.id)


async def run(agent):
    """Ask, act, repeat. Ends when the model stops asking for tools."""
    try:
        await fire(agent, "run_pre")
        while True:
            check(agent)                     # the wiring may have changed
            agent.step += 1
            await fire(agent, "model_pre")
            answer = await agent.model.ainvoke(agent)
            if not isinstance(answer, Message):
                template = isinstance(agent.model, ProviderModel)
                raise wrong(
                    "provider" if template else "model",
                    f"{type(agent.model).__name__}.ainvoke returned "
                    f"{type(answer).__name__}, expected Message"
                    + (" — to_core must build one" if template else ""),
                )
            agent.response = answer
            await fire(agent, "model_post")
            agent.messages.append(agent.response)   # a hook may have swapped it
            calls = agent.response.tool_calls
            if not calls:
                break
            for call in calls:
                agent.call, agent.result = call, None
                await fire(agent, "tool_pre")       # a filled result means denied
                if agent.result is None:
                    tool = agent.tools.get(call.name)
                    if tool is None:
                        agent.result = f"error: unknown tool: {call.name}"
                    else:
                        try:
                            agent.result = await tool.aexecute(agent, **call.args)
                        except Stop:
                            raise                   # Stop always means stop
                        except Exception as ex:
                            agent.result = f"error: {type(ex).__name__}: {ex}"
                agent.result = coerce(agent.result, call)
                await fire(agent, "tool_post")      # this always sees a message
                agent.messages.append(agent.result)
                agent.call = agent.result = None
    except Stop:
        pass
    finally:
        try:
            await fire(agent, "run_post")           # teardown, no matter what
        except Stop:
            pass                                    # too late to stop; ignore
    return agent
