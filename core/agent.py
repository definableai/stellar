"""Two nouns: the Agent you build once, and the Run it starts each time.

The front door: agent.run(prompt) is the whole API. This file wires the
backpack together and hands it to core/loop.py, which never imports back.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any
from uuid import uuid4

from core.contracts import ContractError, Hooks, Model, Tool, wrong
from core.events import Event, Events
from core.loop import check, run as loop
from core.types import Message

__all__ = ["Agent", "Run"]


@dataclass
class Agent:
    """A robot with one backpack, shared by every run it starts.

    Swap any part by assignment: check() reads the whole backpack again at
    the top of every step. agent.hooks.attach mid-flight reaches every run
    already in the air, at its next stage; run.hooks is the isolated lane.
    A hundred users is asyncio.gather(*(agent.run(p) for p in prompts)).
    """

    model: Annotated[Model, "the brain; every run of this agent asks this one"]
    tools: Annotated[
        Iterable[Tool] | Mapping[str, Tool],
        "hand it either; after __post_init__ it is always the mapping"] = ()
    hooks: Annotated[Hooks, "control plane"] = field(default_factory=Hooks)
    events: Annotated[Events, "data plane"] = field(default_factory=Events)
    extra: Annotated[
        dict[str, Any],
        "the spare pocket. Namespaced keys: extra['budget.tokens']",
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Key the toolbox by tool name, then check the wiring — build fails loud."""
        if not isinstance(self.tools, Mapping):
            filed: dict = {}                 # a nameless one files under None
            for tool in self.tools:          # and check() says so in a moment
                name = getattr(tool, "name", None)
                if name in filed:
                    raise wrong("tool", f"two tools answer to {name!r}")
                filed[name] = tool
            self.tools = filed
        check(self)

    async def run(
        self,
        prompt: Annotated[
            str | Message | None,
            "appended after messages=; None only if messages= says enough"] = None,
        *,
        messages: Annotated[
            list[Message] | None,
            "the notebook to open on, copied — your list is left alone"] = None,
        run_id: Annotated[
            str | None,
            "reuse one to resume that run; None mints a fresh hex id"] = None,
    ) -> Run:
        """Ask, act, repeat. The only method; the work is in core/loop.py.

        A str prompt becomes a user message, a Message is taken as it is,
        and messages= opens the notebook — with run_id, that is a resume.
        Gives back the Run: its notebook, its step count, its id.
        """
        said = list(messages or [])
        if prompt is not None:
            said.append(prompt if isinstance(prompt, Message)
                        else Message("user", prompt))
        if not said:
            raise ContractError("run needs a prompt or messages")
        return await loop(Run(self, run_id or uuid4().hex, said))  # the loop's run


@dataclass
class Run:
    """One conversation: its own notebook, its own hooks, its own id.

    What pickles is the checkpoint — id, messages, step, extra — so a Run
    survives a restart and agent.run(messages=…, run_id=…) picks it back up.
    """

    agent: Annotated[Agent, "the backpack behind this run; dropped from the checkpoint"]
    id: Annotated[str, "hex, minted per run — or the one you handed in to resume"]
    messages: Annotated[list[Message], "the notebook, in order; the loop appends to it"]
    step: Annotated[
        int, "model turns taken — the loop counts up before each invoke"] = 0
    hooks: Annotated[Hooks, "this run's own lane"] = field(default_factory=Hooks)
    extra: Annotated[
        dict[str, Any],
        "the spare pocket for this run alone"] = field(default_factory=dict)

    def emit(
        self,
        name: Annotated[
            str, "dotted event name — the loop uses the eight, you use anything"],
        data: Annotated[Any, "whatever should ride along; stored as-is"] = None,
        source: Annotated[
            str | None, "who is speaking — 'loop', a tool's name, or yours"] = None,
    ) -> Event:
        """Say one thing on the agent's bus, stamped with this run's id.

        Never blocks and never raises; the Event it hands back is already logged.
        """
        return self.agent.events.emit(name, data, source, run_id=self.id)

    def __getstate__(self) -> dict[str, Any]:
        """The checkpoint: everything except the wiring.

        The agent, its hooks and its bus are dropped — pickle keeps the
        notebook, and agent.run(messages=…, run_id=…) rebuilds the rest.
        """
        return {"id": self.id, "messages": self.messages, "step": self.step,
                "extra": self.extra}
