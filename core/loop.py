"""The loop: ask the model, run what it asked for, ask again.

Nothing here imports the Agent class at runtime — only under TYPE_CHECKING,
for the checker. These functions take any object with the right attributes,
so agent.py can import this file and this file never has to import back.

Agent.run() calls run(); Agent.__post_init__ and every step call check().
"""

import inspect
import json
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

from core.contracts import ContractError, Model, Stop, Tool, fire, fold, wrong
from core.types import Message, ToolCall

if TYPE_CHECKING:                       # names for the checker, no runtime edge
    from core.agent import Agent, Run


def _own(obj: Any, base: type, name: str) -> Callable[..., Any] | None:
    """The method obj's own class writes itself, or None.

    Subclasses pass. So do look-alikes that never heard of the base class.
    A plain function has none of the names, so it fails.
    """
    fn = getattr(type(obj), name, None)
    return None if fn is None or fn is getattr(base, name, None) else fn


def _sync(fn: Callable[..., Any]) -> bool:
    """True when the method was not written async — the loop cannot run it."""
    return not (inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn))


def check(agent: "Agent") -> None:
    """Read the wiring and say what a dev got wrong, before it runs.

    Raises ContractError, with the code to type instead. Runs at construction
    and again at the top of every step, so a hot swap is checked too.
    """
    invoke = _own(agent.model, Model, "invoke")
    if invoke is None:
        raise wrong("model", f"{type(agent.model).__name__} is not a Model: "
                    "subclass Model and write an async invoke, or "
                    "ProviderModel and write encode + send")
    if _sync(invoke):
        raise wrong("model", f"{type(agent.model).__name__}.invoke must be "
                    "async — sync code is one asyncio.to_thread line inside it")
    if not isinstance(agent.tools, Mapping):
        raise wrong("tool", "agent.tools is a mapping of name to tool; the "
                    "constructor keys an iterable for you, so hand it one")
    for filed, tool in agent.tools.items():
        name = getattr(tool, "name", None)
        if not name:
            raise wrong("tool", f"{type(tool).__name__} needs a name")
        if filed != name:
            raise wrong("tool", f"tool {name!r} is filed under {filed!r}; "
                        "the key is the name")
        if not isinstance(getattr(tool, "parameters", None), dict):
            raise wrong("tool", f"tool {name!r} needs parameters to be a dict")
        execute = _own(tool, Tool, "execute")
        if execute is None:
            raise wrong("tool", f"tool {name!r} defines no execute")
        if _sync(execute):
            raise wrong("tool", f"tool {name!r}.execute must be async — sync "
                        "code is one asyncio.to_thread line inside it")


def coerce(result: Any, call: ToolCall) -> Message:
    """Whatever the tool handed back, make it one tool message.

    A Message is re-roled and stamped with the call id; a str is the content;
    anything else is json.dumps'd, str() for whatever will not serialise.
    """
    if isinstance(result, Message):
        result.role = "tool"
        result.tool_call_id = result.tool_call_id or call.id
        return result
    if not isinstance(result, str):
        result = json.dumps(result, default=str)
    return Message("tool", result, tool_call_id=call.id)


async def use(run: "Run", tool: Tool, call: ToolCall) -> Message | Any:
    """Run one tool. An async generator streams: each Part rings tool.delta.

    A streaming tool gives back the folded Message; any other tool gives back
    whatever it returned, and coerce() shapes that downstream.
    """
    # Any on purpose: execute is a coroutine OR an async generator, and no
    # single declared type can say so — the isasyncgen sniff is the truth.
    out: Any = tool.execute(run, **call.args)
    if not inspect.isasyncgen(out):
        return await out
    result = Message("tool", tool_call_id=call.id)
    async for part in out:
        [part] = await fire(run, "tool.delta", part)      # hook before fold
        fold(result, part)
        run.emit("tool.delta", part, source=call.name)
    return result


async def run(run: "Run") -> "Run":
    """Ask, act, repeat. Ends when the model stops asking for tools.

    Stop — from any hook or tool — ends it cleanly; run.post fires either way.
    Hands back the same Run, its notebook filled in.
    """
    try:
        [run.messages[-1]] = await fire(run, "run.pre", run.messages[-1])
        run.emit("run.pre", run.messages[-1], source="loop")
        while True:
            check(run.agent)                     # the wiring may have changed
            run.step += 1
            [said] = await fire(run, "model.pre", run.messages)
            run.messages = list(said)            # a replacement is permanent
            run.emit("model.pre", list(said), source="loop")  # a copy: the log
                                                 # must not grow with the run
            answer = await run.agent.model.invoke(run)
            if not isinstance(answer, Message):
                raise wrong(
                    kind="model",
                    problem=f"{type(run.agent.model).__name__}.invoke returned "
                    f"{type(answer).__name__}, expected Message",
                )
            [answer] = await fire(run, "model.post", answer)
            run.emit("model.post", answer, source="loop")
            run.messages.append(answer)
            if not answer.tool_calls:
                break
            for call in answer.tool_calls:
                out = await fire(run, "tool.pre", call)
                if isinstance(out, tuple):
                    call, result = out[0], None
                else:
                    result = out                 # a str or Message: denied
                run.emit("tool.pre", call, source="loop")
                if result is None:
                    toolbox = cast(Mapping[str, Tool], run.agent.tools)
                    tool = toolbox.get(call.name)   # check() proved the mapping
                    if tool is None:
                        result = f"error: unknown tool: {call.name}"
                    else:
                        try:
                            result = await use(run, tool, call)
                        except (Stop, ContractError):
                            raise                # stop means stop; a broken
                        except Exception as ex:  # contract stays loud
                            result = f"error: {type(ex).__name__}: {ex}"
                reply = coerce(result, call)
                [call, reply] = await fire(run, "tool.post", call, reply)
                run.emit("tool.post", reply, source="loop")
                run.messages.append(reply)
    except Stop:
        pass
    finally:
        last = run.messages[-1] if run.messages else Message("assistant")
        try:
            await fire(run, "run.post", last)    # teardown, no matter what
        except Stop:
            pass                                 # too late to stop; ignore
        finally:
            run.emit("run.post", last, source="loop")   # streams end on this
    return run
