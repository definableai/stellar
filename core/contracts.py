"""What you implement: Model, Tool, hooks — plus the two control signals.

Everything is async. A Model or a Tool takes one argument, the run; a hook
takes the payload of its stage, and the run behind it if it asks. There are
no sync twins and no bridges: sync code is one asyncio.to_thread line the
implementer writes inside the async method.

The eight stage names live here, and so does fold() — the Part protocol
every streaming adapter obeys. core/loop.py rings the stages; nothing in
this file knows what an Agent is.
"""

from __future__ import annotations

import inspect
from typing import (
    TYPE_CHECKING, Annotated, Any, AsyncIterator, Callable, Literal, Sequence, cast,
    overload,
)

from core.types import Message, Part, ToolCall

if TYPE_CHECKING:                  # the checker's eyes only: agent.py imports
    from core.agent import Run     # this file, so the runtime graph stays one-way

__all__ = [
    "PAYLOAD", "SKELETONS", "STAGES", "ContractError", "Hooks", "Model",
    "ProviderModel", "Stop", "Tool", "fold", "hook",
]

# The eight bells, in ring order. A hook name and an event name are one word.
STAGES = ("run.pre", "model.pre", "model.delta", "model.post",
          "tool.pre", "tool.delta", "tool.post", "run.post")

PAYLOAD: dict[str, tuple[type, ...]] = {   # what each stage hands its hooks
    "run.pre": (Message,), "model.pre": (list,), "model.delta": (Part,),
    "model.post": (Message,), "tool.pre": (ToolCall,), "tool.delta": (Part,),
    "tool.post": (ToolCall, Message), "run.post": (Message,),
}


class ContractError(Exception):
    """You implemented it wrong. The message says what to type instead."""


class Stop(Exception):
    """End the run cleanly. Raise it from any hook or tool."""


class Model:
    """The brain. Override invoke — async — and hand back a Message."""

    async def invoke(
        self,
        run: Annotated[
            Run, "run.messages is the notebook; run.agent holds tools and hooks"],
    ) -> Message:
        """One turn: read run.messages, answer with one assistant Message."""
        raise NotImplementedError("invoke")


class ProviderModel(Model):
    """A Model in two steps: build the request, send it as a stream of Parts.

    encode is pure translation. send owns the network — retries live there —
    and yields Parts; a provider that does not stream yields them off one
    response. invoke rings model.delta per Part and folds what comes back,
    so streaming UIs are written once and work with every adapter.
    """

    def encode(
        self,
        run: Annotated[Run, "every message, plus run.agent.tools — all a body needs"],
    ) -> Any:
        """The request body, built from the run. Pure: send() takes it from here."""
        raise _todo("encode")

    def send(
        self,
        run: Annotated[
            Run, "whatever the body left out — run.id, run.extra, the backpack"],
        body: Annotated[Any, "exactly what encode() built; send() never rebuilds it"],
    ) -> AsyncIterator[Part]:
        """Yield the reply as Parts. An async generator; the network lives here."""
        raise _todo("send")

    async def invoke(
        self,
        run: Annotated[
            Run, "passed straight to encode and send; its bus carries the deltas"],
    ) -> Message:
        """encode, send, fold — rings model.delta per Part, hands back the Message."""
        answer = Message("assistant")
        async for part in self.send(run, self.encode(run)):
            if not isinstance(part, Part):
                raise wrong("provider", f"{type(self).__name__}.send "
                            f"yielded {type(part).__name__}, expected Part")
            [part] = await run.hooks.fire("model.delta", run, part)  # then fold
            fold(answer, part)
            run.emit("model.delta", part, source="loop")
        return answer


class Tool:
    """One thing the agent can do. Override execute — async.

    An async generator works too: each Part it yields rings tool.delta and
    folds into the tool message, same protocol as a streaming model. Set
    parallel = True on a tool that may run beside other parallel tools; the
    default is a barrier.
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}  # read-only
    parallel: bool = False   # beside other parallel tools; serial is a barrier

    # the real shape is (self, run, **args); typed loose so an override
    # may name the args its schema promises without an override complaint
    async def execute(
        self,
        *args: Annotated[
            Any, "the first argument, always — the loop fills that one in"],
        **kwargs: Annotated[
            Any, "the parameters your schema promised, by name, off the call"],
    ) -> Any:
        """Do the thing. execute(run, **args) in, anything out — coerce shapes it."""
        raise NotImplementedError("execute")


class Hooks:
    """The rule cards, filed by stage. Attach order is call order.

    A registry may chain to a parent whose cards ring first — a Run's hooks
    parent onto its Agent's, so one bell reaches both. The parent binds when
    the Run is born: swap cards with attach and detach, not by reassigning
    agent.hooks mid-flight.
    """

    def __init__(
        self,
        parent: Annotated[Hooks | None, "rings first; a tool.pre skip from it "
                          "means this registry's cards never hear the bell"] = None,
    ) -> None:
        self.parent = parent
        # a filed card is (fn, wants the run after its payload) — sniffed at attach
        self.fns: dict[str, list[tuple[Any, bool]]] = {s: [] for s in STAGES}

    def attach(
        self,
        fn: Annotated[
            Callable[..., Any],
            "filed as-is and left a function; one may sit on several stages"],
        *stages: Annotated[
            str, "the bells to file it under — one or more of the eight names"],
    ) -> Hooks:
        """File one function: the stages named here, else the ones @hook tagged.

        Raises ContractError now — not mid-run — if fn is not async, names no
        stage, or does not fit one. Hands back self, so attaches chain.
        """
        wanted = _named(stages or getattr(fn, "stages", ()), _who(fn))
        if not inspect.iscoroutinefunction(fn):
            raise wrong("hook", f"{_who(fn)} must be async — write async def; "
                        "sync work is one asyncio.to_thread line inside it")
        for stage, with_run in [(s, _wants_run(fn, s)) for s in wanted]:
            self.fns[stage].append((fn, with_run))   # sniffed all, then filed all
        return self

    def detach(
        self,
        fn: Annotated[
            Callable[..., Any],
            "matched by identity — the same object that was attached"],
    ) -> Hooks:
        """Unfile it, from every stage it listened to."""
        for filed in self.fns.values():
            filed[:] = [card for card in filed if card[0] is not fn]
        return self

    @overload
    async def fire(self, stage: Literal["tool.pre"], run: Run,
                   *payload: Any) -> tuple[Any, ...] | str | Message: ...
    @overload
    async def fire(self, stage: str, run: Run,
                   *payload: Any) -> tuple[Any, ...]: ...

    async def fire(
        self,
        stage: Annotated[
            str, "one of the eight names; its cards are the ones that ring"],
        run: Annotated[
            Run, "handed over only to a hook that left room for it after payload"],
        *payload: Annotated[
            Any, "the stage's arguments — PAYLOAD says the shape it wants"],
    ) -> tuple[Any, ...] | str | Message:
        """Ring one stage: the parent's cards first, then these, in attach order.

        What a hook returns is what the next one hears; None leaves the
        payload alone. At tool.pre a str or Message is the tool's result,
        and the hooks behind it never hear the bell.

        So: the payload back as a tuple, or — tool.pre only — that str or
        Message. Raises ContractError if a hook returns the wrong shape.
        """
        if self.parent is not None:
            up: tuple[Any, ...] | str | Message = (
                await self.parent.fire(stage, run, *payload))
            if not isinstance(up, tuple):
                return up             # the parent said skip; our cards miss it
            payload = up
        want = PAYLOAD[stage][-1]
        for fn, with_run in list(self.fns[stage]):   # a card added mid-ring waits
            out = await (fn(*payload, run) if with_run else fn(*payload))
            if out is None:
                continue
            if stage == "tool.pre" and isinstance(out, (str, Message)):
                return out                  # the tool is skipped; this is its result
            if not isinstance(out, want):
                takes = ("ToolCall, str or Message" if stage == "tool.pre"
                         else want.__name__)
                raise wrong("hook", f"{_who(fn)} at {stage} returned "
                            f"{type(out).__name__}; return {takes}, or None to "
                            "leave it alone")
            payload = payload[:-1] + (out,)
        return payload


def fold(
    message: Annotated[Message, "folded into in place — nothing is handed back"],
    part: Annotated[Part, "one piece off the stream; its type picks the rule above"],
) -> None:
    """One streamed Part into the growing Message. The Part protocol:

    "text" is a delta — consecutive text parts concatenate into one.
    "tool_call" arrives whole — data is a complete ToolCall; partial JSON
    is buffered wherever it came from, never here. "meta" is a dict merged
    into message.meta, later keys winning. Every other type lands in
    content untouched.
    """
    held = cast(list[Part], message.content)   # always parts after construction
    if part.type == "text":
        if held and held[-1].type == "text":
            held[-1].data += part.data
        else:
            held.append(Part("text", part.data))
    elif part.type == "tool_call":
        if not isinstance(part.data, ToolCall):
            raise wrong("provider", "a tool_call part must carry a whole "
                        f"ToolCall, got {type(part.data).__name__}")
        message.tool_calls.append(part.data)
    elif part.type == "meta":
        message.meta.update(part.data)
    else:
        held.append(part)


def wrong(
    kind: Annotated[
        str, "which skeleton to print — 'model', 'provider', 'tool' or 'hook'"],
    problem: Annotated[
        str, "the sentence naming what is wrong, in the caller's own words"],
) -> ContractError:
    """The complaint, then the code to type instead."""
    return ContractError(f"{problem}\n\n{SKELETONS[kind]}")


def hook(
    *stages: Annotated[str, "the bells to listen for — one or more of the eight names"],
) -> Callable[[Any], Any]:
    """Tag a function with the stages it listens for. It stays a function.

    Raises ContractError right here if a stage is misspelled. attach() reads
    the tag, so agent.hooks.attach(fn) then needs no stage names.
    """
    _named(stages, "@hook")

    def tag(fn: Any) -> Any:   # Any: the tag writes .stages onto the function
        fn.stages = stages
        return fn

    return tag


def _todo(name: str) -> NotImplementedError:
    """A missing template method, with the whole template to copy."""
    return NotImplementedError(
        f"{name} — a ProviderModel implements both:\n\n{SKELETONS['provider']}"
    )


def _who(fn: Callable[..., Any]) -> str:
    """What to call a hook in a complaint."""
    return getattr(fn, "__name__", type(fn).__name__)


def _named(
    stages: Sequence[str],
    who: Annotated[str, "the name to blame in the complaint — '@hook', or the fn's"],
) -> tuple[str, ...]:
    """The stage names, checked. Nobody gets to listen for nothing."""
    if not stages:
        raise wrong("hook", f"{who} listens for nothing; name a stage: "
                    + ", ".join(STAGES))
    for stage in stages:
        if stage not in STAGES:
            raise wrong("hook", f"{stage!r} is not a stage; pick one of: "
                        + ", ".join(STAGES))
    return tuple(stages)


def _wants_run(fn: Callable[..., Any], stage: str) -> bool:
    """True when there is room for run after the payload. Sniffed once, at attach."""
    takes = [p for p in inspect.signature(fn).parameters.values()
             if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    hands = len(PAYLOAD[stage])
    if len(takes) not in (hands, hands + 1):
        raise wrong("hook", f"{_who(fn)} does not fit {stage}, which hands over "
                    + ", ".join(t.__name__ for t in PAYLOAD[stage])
                    + " — take those, and run after them if you want it")
    return len(takes) == hands + 1


# data, not logic — parked last so the code reads first
SKELETONS: dict[str, str] = {
    "model": """class MyModel(Model):
    async def invoke(self, run) -> Message:
        said = await asyncio.to_thread(my_sdk.complete, run.messages)
        return Message("assistant", said)
""",
    "provider": """class MyProvider(ProviderModel):
    def encode(self, run) -> dict:             # pure: every message — including
        ...                                    # tool_calls and tool_call_id — plus
                                               # run.agent.tools, as one request body

    async def send(self, run, body):           # the network call; retries live
        ...                                    # here. Yield Parts: "text" deltas,
        yield Part("text", "…")                # whole "tool_call" ToolCalls,
                                               # one "meta" dict
""",
    "tool": """class MyTool(Tool):
    name = "shout"
    description = "Shout a word."
    parameters = {"type": "object", "properties": {"word": {"type": "string"}}}

    async def execute(self, run, word):
        return word.upper()

# or decorate a plain function with @tool — sync, async, or async generator
""",
    "hook": """@hook("tool.pre")
async def no_rm(call: ToolCall):        # add a trailing `run` arg if you need it
    if call.name == "rm":
        return "denied"                 # the tool is skipped; this is its result

agent.hooks.attach(no_rm)               # untagged? attach(no_rm, "tool.pre")
""",
}
