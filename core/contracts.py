"""What you implement: Model, Tool, Hook — plus the two control signals.

Everything is async and every method takes one argument, the agent. There
are no sync twins and no bridges: sync code is one asyncio.to_thread line
the implementer writes inside the async method.
"""

import inspect
from typing import Any, AsyncIterator, Awaitable, cast

from core.types import Message, Part, ToolCall

EVENTS = ("run_pre", "model_pre", "model_delta", "model_post",
          "tool_pre", "tool_delta", "tool_post", "run_post")

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


async def fire(agent, event: str) -> None:
    """Ring one bell. Every hook hears it, in list order."""
    for hook in list(agent.hooks):           # a hook may add hooks mid-ring
        listen = getattr(hook, event, None)
        answer = listen(agent) if listen else None
        if inspect.isawaitable(answer):
            await answer


def fold(message: Message, part: Part) -> None:
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


class Model:
    """The brain. Override invoke — async — and hand back a Message."""

    async def invoke(self, agent) -> Message:
        raise NotImplementedError("invoke")


def _todo(name: str) -> NotImplementedError:
    """A missing template method, with the whole template to copy."""
    return NotImplementedError(
        f"{name} — a ProviderModel implements both:\n\n{SKELETONS['provider']}"
    )


class ProviderModel(Model):
    """A Model in two steps: build the request, send it as a stream of Parts.

    encode is pure translation. send owns the network — retries live there —
    and yields Parts; a provider that does not stream yields them off one
    response. invoke folds them into the reply and rings model_delta per
    Part, so streaming UIs are written once and work with every adapter.
    """

    def encode(self, agent) -> Any:
        raise _todo("encode")

    def send(self, agent, body) -> AsyncIterator[Part]:
        raise _todo("send")

    async def invoke(self, agent) -> Message:
        answer = agent.response = Message("assistant")
        try:
            async for part in self.send(agent, self.encode(agent)):
                if not isinstance(part, Part):
                    raise wrong("provider", f"{type(self).__name__}.send "
                                f"yielded {type(part).__name__}, expected Part")
                fold(answer, part)
                agent.delta = part
                await fire(agent, "model_delta")
        finally:
            agent.delta = None
        return answer


class Tool:
    """One thing the agent can do. Override execute — async.

    An async generator works too: each Part it yields rings tool_delta and
    folds into the tool message, same protocol as a streaming model.
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}  # read-only

    # the real shape is (self, agent, **args); typed loose so an override
    # may name the args its schema promises without an override complaint
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("execute")


def wrong(kind: str, problem: str) -> ContractError:
    """The complaint, then the code to type instead."""
    return ContractError(f"{problem}\n\n{SKELETONS[kind]}")


class Hook:
    """Rule cards. Override the moments you care about; the rest do nothing.

    The eight method names are the eight event names. Each may be sync or
    async. At a delta the Part is agent.delta, already folded into
    agent.response (model) or agent.result (tool).
    """

    def run_pre(self, agent) -> Awaitable[None] | None:
        """The run is about to start."""

    def model_pre(self, agent) -> Awaitable[None] | None:
        """The model is about to be asked."""

    def model_delta(self, agent) -> Awaitable[None] | None:
        """The model streamed one Part — agent.delta."""

    def model_post(self, agent) -> Awaitable[None] | None:
        """The model just answered."""

    def tool_pre(self, agent) -> Awaitable[None] | None:
        """A tool is about to run."""

    def tool_delta(self, agent) -> Awaitable[None] | None:
        """A tool yielded one Part — agent.delta."""

    def tool_post(self, agent) -> Awaitable[None] | None:
        """A tool just finished."""

    def run_post(self, agent) -> Awaitable[None] | None:
        """The run is over. This always happens."""


def _who(fn) -> str:
    """What to call a hook in a complaint."""
    return getattr(fn, "__name__", type(fn).__name__)


def _named(stages, who: str) -> tuple[str, ...]:
    """The stage names, checked. Nobody gets to listen for nothing."""
    if not stages:
        raise wrong("stage", f"{who} listens for nothing; name a stage: "
                    + ", ".join(STAGES))
    for stage in stages:
        if stage not in STAGES:
            raise wrong("stage", f"{stage!r} is not a stage; pick one of: "
                        + ", ".join(STAGES))
    return tuple(stages)


def _wants_run(fn, stage: str) -> bool:
    """True when there is room for run after the payload. Sniffed once, at attach."""
    takes = [p for p in inspect.signature(fn).parameters.values()
             if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    hands = len(PAYLOAD[stage])
    if len(takes) not in (hands, hands + 1):
        raise wrong("stage", f"{_who(fn)} does not fit {stage}, which hands over "
                    + ", ".join(t.__name__ for t in PAYLOAD[stage])
                    + " — take those, and run after them if you want it")
    return len(takes) == hands + 1


def hook(*stages: str):
    """Tag a function with the stages it listens for. It stays a function."""
    _named(stages, "@hook")

    def tag(fn):
        fn.stages = stages
        return fn

    return tag


class Hooks:
    """The rule cards, filed by stage. Attach order is call order."""

    def __init__(self) -> None:
        self.fns: dict[str, list[tuple[Any, bool]]] = {s: [] for s in STAGES}

    def attach(self, fn, *stages: str) -> "Hooks":
        """File one function: the stages named here, else the ones @hook tagged."""
        wanted = _named(stages or getattr(fn, "stages", ()), _who(fn))
        if not inspect.iscoroutinefunction(fn):
            raise wrong("stage", f"{_who(fn)} must be async — write async def; "
                        "sync work is one asyncio.to_thread line inside it")
        for stage, with_run in [(s, _wants_run(fn, s)) for s in wanted]:
            self.fns[stage].append((fn, with_run))   # sniffed all, then filed all
        return self

    def detach(self, fn) -> "Hooks":
        """Unfile it, from every stage it listened to."""
        for filed in self.fns.values():
            filed[:] = [card for card in filed if card[0] is not fn]
        return self

    async def fire(self, stage: str, run, *payload) -> Any:
        """Ring one stage here. Every hook hears it, in attach order.

        What a hook returns is what the next one hears; None leaves the
        payload alone. At tool.pre a str or Message is the tool's result,
        and the hooks behind it never hear the bell.
        """
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
                raise wrong("stage", f"{_who(fn)} at {stage} returned "
                            f"{type(out).__name__}; return {takes}, or None to "
                            "leave it alone")
            payload = payload[:-1] + (out,)
        return payload


async def ring(run, stage: str, *payload) -> Any:
    """Both registries: the agent's cards first, then this run's own.

    Becomes fire when the old bells retire (task 03).
    """
    out = await run.agent.hooks.fire(stage, run, *payload)
    if not isinstance(out, tuple):
        return out                    # tool.pre said skip; the run's cards miss it
    return await run.hooks.fire(stage, run, *out)


SKELETONS: dict[str, str] = {
    "model": """class MyModel(Model):
    async def invoke(self, agent) -> Message:
        said = await asyncio.to_thread(my_sdk.complete, agent.messages)
        return Message("assistant", said)
""",
    "provider": """class MyProvider(ProviderModel):
    def encode(self, agent) -> dict:           # pure: every message — including
        ...                                    # tool_calls and tool_call_id — plus
                                               # agent.tools, as one request body

    async def send(self, agent, body):         # the network call; retries live
        ...                                    # here. Yield Parts: "text" deltas,
        yield Part("text", "…")                # whole "tool_call" ToolCalls,
                                               # one "meta" dict
""",
    "tool": """class MyTool(Tool):
    name = "shout"
    description = "Shout a word."
    parameters = {"type": "object", "properties": {"word": {"type": "string"}}}

    async def execute(self, agent, word):
        return word.upper()

# or decorate a plain function with @tool — sync, async, or async generator
""",
    "hook": """class MyHook(Hook):
    def run_pre(self, agent) -> None:
        print("starting with", len(agent.messages), "messages")
""",
    "stage": """@hook("tool.pre")
async def no_rm(call: ToolCall):        # add a trailing `run` arg if you need it
    if call.name == "rm":
        return "denied"                 # the tool is skipped; this is its result

agent.hooks.attach(no_rm)               # untagged? attach(no_rm, "tool.pre")
""",
}
