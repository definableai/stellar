"""HITL approval gate for tools — policy allowlist first, then a human.

    from internal.hook_approval import approval_gate, allow, deny, console_asker

    agent = Agent(llm, tools=[...], hooks=[approval_gate(
        rules=[allow("read_file"),
               allow("bash", arg="command", prefix="git "),
               deny("bash", arg="command", prefix="rm ")],
        asker=console_asker,        # or your web/slack asker
        timeout=120.0,
    )])

Decision order: deny rules > allow rules > default "ask". A prefix rule
must name its argument — it matches only that argument's value, so other
model-controlled fields can't smuggle a match. On "ask" the gate emits
``tool/delta {"approval": "pending"}`` into the event stream, then awaits
the asker raced against run-stop and timeout — both fail closed (deny).
The decision is emitted as a second tool/delta: full audit trail in the
run record.

An asker is ``async (ToolCall) -> ("allow"|"deny"|"always", reason|None)``
(bare string accepted; anything malformed or unrecognized = deny).
"always" allows and learns an allow rule for that EXACT call (tool +
identical arguments) — different arguments re-ask. Learned rules live in
the gate instance: build one approval_gate per user session, never share
across users. Asker exceptions abort the run — fail closed, like any hook.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

from core import Hook, ToolCall, ToolResult

Asker = Callable[[ToolCall], "Awaitable[tuple[str, str | None] | str]"]


@dataclass
class Rule:
    action: str                     # "allow" | "deny"
    tool: str
    arg: str | None = None          # argument the prefix applies to
    prefix: str | None = None       # matches ONLY that argument's value
    exact: dict | None = None       # matches identical arguments ("always")

    def __post_init__(self) -> None:
        if self.prefix is not None and self.arg is None:
            raise ValueError("prefix rule must name its argument: "
                             "allow(tool, arg=..., prefix=...)")

    def matches(self, call: ToolCall) -> bool:
        if self.tool != call.name:
            return False
        if self.exact is not None:
            return call.arguments == self.exact
        if self.prefix is None:
            return True
        v = call.arguments.get(self.arg)
        return isinstance(v, str) and v.startswith(self.prefix)


def allow(tool: str, arg: str | None = None, prefix: str | None = None) -> Rule:
    return Rule("allow", tool, arg, prefix)


def deny(tool: str, arg: str | None = None, prefix: str | None = None) -> Rule:
    return Rule("deny", tool, arg, prefix)


def decide(rules: Iterable[Rule], call: ToolCall) -> str:
    if any(r.action == "deny" and r.matches(call) for r in rules):
        return "deny"
    if any(r.action == "allow" and r.matches(call) for r in rules):
        return "allow"
    return "ask"


async def console_asker(call: ToolCall) -> tuple[str, str | None]:
    """Terminal asker. y = allow once, a = always (this exact call),
    else deny. Non-blocking stdin reader — cancellation leaves no stuck
    thread. ponytail: unix only (loop.add_reader on stdin)."""
    import sys
    loop = asyncio.get_running_loop()
    print(f"allow {call.name}({json.dumps(call.arguments)})? "
          f"[y=once / a=always this exact call / N=deny] ", end="", flush=True)
    fut: asyncio.Future[str] = loop.create_future()
    loop.add_reader(sys.stdin, lambda: not fut.done()
                    and fut.set_result(sys.stdin.readline()))
    try:
        ans = (await fut).strip().lower()
    finally:
        loop.remove_reader(sys.stdin)
    return {"y": ("allow", None), "a": ("always", None)}.get(
        ans, ("deny", "rejected at console"))


def _deny(ctx: Any, reason: str) -> None:
    """Short-circuit the call with a model-readable denial."""
    ctx.result = ToolResult(ctx.call.id, ctx.call.name,
                            f"Denied: {reason}", is_error=True)


def approval_gate(rules: Iterable[Rule], asker: Asker, timeout: float = 120.0):
    """-> before_tool Hook enforcing rules, asking the human on gaps.
    Attach directly: ``Agent(llm, hooks=[approval_gate(rules, asker)])``."""
    rules = list(rules)   # session-local; "always" answers append here

    async def gate(ctx: Any) -> None:
        d = decide(rules, ctx.call)
        if d == "allow":
            return
        if d == "deny":
            return _deny(ctx, "blocked by policy")

        if ctx.emit_delta:
            await ctx.emit_delta({"approval": "pending",
                                  "arguments": ctx.call.arguments})
        ask = asyncio.ensure_future(asker(ctx.call))
        stop = asyncio.ensure_future(ctx.run.handle.wait_stop())
        done, _ = await asyncio.wait({ask, stop}, timeout=timeout,
                                     return_when=asyncio.FIRST_COMPLETED)
        if ask in done:
            stop.cancel()
            resp = ask.result()
            try:
                decision, reason = (resp, None) if isinstance(resp, str) else resp
            except (TypeError, ValueError):
                decision, reason = "deny", f"invalid asker response: {resp!r}"
            if decision not in ("allow", "deny", "always"):
                decision, reason = "deny", f"unrecognized answer: {decision!r}"
        else:
            ask.cancel()
            stop.cancel()
            # let the asker's cleanup finish (a console asker tears down
            # its stdin reader in a finally) before the next gate installs
            # its own — otherwise one timeout kills every later prompt
            await asyncio.gather(ask, stop, return_exceptions=True)
            decision, reason = "deny", ("run stopped" if stop in done else "timeout")

        if ctx.emit_delta:   # audit the REAL answer — "always" has policy
            await ctx.emit_delta({"approval": decision,   # consequence and
                                  **({"reason": reason} if reason else {})})
        if decision == "always":                          # must be visible
            # learns this exact call only — different arguments re-ask
            rules.append(Rule("allow", ctx.call.name,
                              exact=dict(ctx.call.arguments)))
            decision = "allow"
        if decision != "allow":
            _deny(ctx, reason or "rejected by user")

    return Hook("before_tool", gate)


def setup(ctx: Any) -> None:
    """Adapter shape (core/adapter.py); config = approval_gate kwargs
    (``rules=``, ``asker=``, ``timeout=``)."""
    ctx.hook(approval_gate(**ctx.config))
