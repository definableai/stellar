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
}
