"""Permission: the guard at the tool gate, answering before a tool runs."""

# ponytail: 06 rewrites these as @hook functions


def matches(rule, agent, call) -> bool:
    """A rule is a set of names, or a predicate (agent, call) -> bool."""
    if rule is None:
        return False
    return bool(rule(agent, call)) if callable(rule) else call.name in rule


async def asked(agent, call) -> str | None:
    """Put the question to the human. None means go ahead."""
    ask = agent.extra.get("ui.ask")
    if not callable(ask):
        return "denied: no ui.ask channel"      # a guard with no voice says no
    said = await ask(f"Allow {call.name}({call.args})?", ["allow", "deny"])
    return None if said == "allow" else f"denied: {call.name}"


class Permission:
    """deny wins; an allow list then settles it alone; only then is a human asked."""

    def __init__(self, deny=None, allow=None, ask: bool = False) -> None:
        self.deny, self.allow, self.ask = deny, allow, ask

    async def tool_pre(self, agent) -> None:
        """A filled-in result is the refusal: the tool never runs."""
        call = agent.call
        if matches(self.deny, agent, call):
            agent.result = f"denied: {call.name}"
        elif self.allow is not None:
            if not matches(self.allow, agent, call):
                agent.result = f"denied: {call.name}"
        elif self.ask:
            agent.result = await asked(agent, call)
