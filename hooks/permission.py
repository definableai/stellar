"""Permission: the guard at the tool gate, answering before a tool runs."""

from core import hook


def matches(rule, run, call) -> bool:
    """A rule is a set of names, or a predicate (run, call) -> bool."""
    if rule is None:
        return False
    return bool(rule(run, call)) if callable(rule) else call.name in rule


async def asked(run, call) -> str | None:
    """Put the question to the human. None means go ahead."""
    ask = run.extra.get("ui.ask") or run.agent.extra.get("ui.ask")
    if not callable(ask):
        return "denied: no ui.ask channel"      # a guard with no voice says no
    said = await ask(f"Allow {call.name}({call.args})?", ["allow", "deny"])
    return None if said == "allow" else f"denied: {call.name}"


def Permission(deny=None, allow=None, ask: bool = False):
    """deny wins; an allow list then settles it alone; only then is a human asked."""

    @hook("tool.pre")
    async def guard(call, run) -> str | None:
        """A returned str is the refusal: the tool never runs, that is its result."""
        if matches(deny, run, call):
            return f"denied: {call.name}"
        if allow is not None:
            return None if matches(allow, run, call) else f"denied: {call.name}"
        return await asked(run, call) if ask else None

    return guard
