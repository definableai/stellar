"""What you implement: Model, Tool, Hook — plus the two control signals.

Every method takes one argument, the agent. Model and Tool come in pairs:
write the sync one or the async one, and the base class bridges the other
with a thread.
"""

import asyncio
from functools import partial
from typing import Any

from core.types import Message

EVENTS = ("run_pre", "model_pre", "model_post", "tool_pre", "tool_post", "run_post")


class ContractError(Exception):
    """You implemented it wrong. The message says what to type instead."""


class Stop(Exception):
    """End the run cleanly. Raise it from any hook or tool."""


class Model:
    """The brain. Override invoke or ainvoke — one of the two."""

    def invoke(self, agent) -> Message:
        raise NotImplementedError("invoke")

    async def ainvoke(self, agent) -> Message:
        return await asyncio.to_thread(self.invoke, agent)


def _todo(name: str) -> NotImplementedError:
    """A missing template method, with the whole template to copy."""
    return NotImplementedError(
        f"{name} — a ProviderModel implements all three:\n\n{SKELETONS['provider']}"
    )


class ProviderModel(Model):
    """A Model in three steps: build the request, send it, read the reply.

    to_provider and to_core are pure translation. Retries belong in send.
    """

    def to_provider(self, agent) -> Any:
        raise _todo("to_provider")

    async def send(self, body) -> Any:
        raise _todo("send")

    def to_core(self, raw) -> Message:
        raise _todo("to_core")

    async def ainvoke(self, agent) -> Message:
        return self.to_core(await self.send(self.to_provider(agent)))


class Tool:
    """One thing the agent can do. Override execute or aexecute."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}  # read-only

    def execute(self, agent, **args) -> Any:
        raise NotImplementedError("execute")

    async def aexecute(self, agent, **args) -> Any:
        return await asyncio.to_thread(partial(self.execute, agent, **args))


def wrong(kind: str, problem: str) -> ContractError:
    """The complaint, then the code to type instead."""
    return ContractError(f"{problem}\n\n{SKELETONS[kind]}")


class Hook:
    """Rule cards. Override the moments you care about; the rest do nothing.

    The six method names are the six event names. Each may be sync or async.
    """

    def run_pre(self, agent) -> None:
        """The run is about to start."""

    def model_pre(self, agent) -> None:
        """The model is about to be asked."""

    def model_post(self, agent) -> None:
        """The model just answered."""

    def tool_pre(self, agent) -> None:
        """A tool is about to run."""

    def tool_post(self, agent) -> None:
        """A tool just finished."""

    def run_post(self, agent) -> None:
        """The run is over. This always happens."""


SKELETONS: dict[str, str] = {
    "model": """class MyModel(Model):
    def invoke(self, agent) -> Message:
        return Message(role="assistant", content="hello")
""",
    "provider": """class MyProvider(ProviderModel):
    def to_provider(self, agent) -> dict:      # pure: messages+tools -> request body
        ...

    async def send(self, body) -> dict:        # the network call; retries live here
        ...

    def to_core(self, raw) -> Message:         # pure: response -> assistant Message
        ...
""",
    "tool": """class MyTool(Tool):
    name = "shout"
    description = "Shout a word."
    parameters = {"type": "object", "properties": {"word": {"type": "string"}}}

    def execute(self, agent, word):
        return word.upper()

# or decorate a plain function with @tool
""",
    "hook": """class MyHook(Hook):
    def run_pre(self, agent) -> None:
        print("starting with", len(agent.messages), "messages")
""",
}
