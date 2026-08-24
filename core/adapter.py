"""Adapters: the unit of self-composition.

An adapter is a ``setup(ctx)`` function. Everything registered through
the ``Ctx`` records its inverse in a ``Scope``; ``agent.drop(name)``
unwinds them newest-first:

    def metrics(ctx):
        ctx.tool(report)                    # a Tool
        ctx.hook("after_tool", record)      # or ctx.hook(some_Hook)
        ctx.agent.tracers.append(sink)      # any open mutation +
        ctx.effect(lambda: ctx.agent.tracers.remove(sink))  # its inverse

    agent.use(metrics)
    agent.drop("metrics")          # all three gone, LIFO

Rules:
    * A raising ``setup()`` unwinds its partial work; nothing mounts.
    * The loop resolves llm/tools/hooks at each use: the LLM's toolset
      updates next step; within one tool batch, a tool mounted by
      call #1 is already callable by call #2.
    * ``setup`` is sync, idempotent composition, not IO — ``drop()``
      re-runs the setups of adapters mounted above the dropped one.
    * ``ctx.agent`` is the real Agent, no jail: registrars are the
      recorded paths, ``ctx.effect`` covers any other mutation.
      Loading adapter *files* is ``core/kernel.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .hooks import Hook, HookFn, HookPoint
from .tools import Tool
from .types import ErrorInfo

if TYPE_CHECKING:
    from .agent import Agent

_MISSING = object()


class Scope:
    """One mounted adapter: name, source file (if any), what it
    contributed (``notes``), and the undo stack."""

    def __init__(self, name: str, source: str | None = None) -> None:
        self.name, self.source = name, source
        self.notes: list[str] = []
        self._undo: list[Callable[[], Any]] = []
        self._remount: tuple[Callable, dict] = (lambda ctx: None, {})  # set by use()

    def dispose(self) -> list[ErrorInfo]:
        """Run inverses LIFO. Never raises: one failing inverse must
        not strand the ones registered before it — errors collect."""
        errors: list[ErrorInfo] = []
        while self._undo:
            fn = self._undo.pop()
            try:
                fn()
            except Exception as ex:
                errors.append(ErrorInfo.from_exc(ex, "adapter"))
        return errors


class Ctx:
    """What ``setup()`` receives. Every registration records its inverse."""

    def __init__(self, agent: "Agent", scope: Scope, config: dict[str, Any]) -> None:
        self.agent, self.scope, self.config = agent, scope, config

    def effect(self, dispose: Callable[[], Any]) -> None:
        """Record a custom inverse (close a file, cancel a task, ...)."""
        self.scope._undo.append(dispose)

    def tool(self, t: Tool) -> Tool:
        d, name = self.agent.tools, t.spec.name
        prior = d.get(name, _MISSING)   # shadowing restores, not deletes
        d[name] = t

        def restore() -> None:
            if prior is _MISSING:
                d.pop(name, None)
            else:
                d[name] = prior
        self.effect(restore)
        self.scope.notes.append(f"tool:{name}")
        return t

    def hook(self, point: HookPoint | Hook, fn: HookFn | None = None) -> HookFn:
        """``ctx.hook("before_tool", fn)`` or ``ctx.hook(some_Hook)``."""
        if isinstance(point, Hook):
            point, fn = point.point, point.fn
        if fn is None:
            raise TypeError("ctx.hook needs a Hook or (point, fn)")
        self.agent.hooks.add(point, fn)
        self.effect(lambda: self.agent.hooks.remove(point, fn))
        self.scope.notes.append(f"hook:{point}:{getattr(fn, '__qualname__', '?')}")
        return fn

    def llm(self, llm: Any) -> None:
        """Swap the agent's LLM; the inverse restores the prior one."""
        prior = self.agent.llm
        self.agent.llm = llm
        self.effect(lambda: setattr(self.agent, "llm", prior))
        self.scope.notes.append(f"llm:{type(llm).__name__}")
