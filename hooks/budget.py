"""Budget: how many tokens the run may spend."""

from core import Hook, Stop

KINDS = ("input_tokens", "output_tokens")


class Budget(Hook):
    """Adds up what every reply cost; ends the run once the total goes past max."""

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens

    def model_post(self, agent) -> None:
        """A reply with no usage meta costs nothing — not every adapter counts."""
        usage = agent.response.meta.get("usage") or {}
        spent = sum(usage.get(kind) or 0 for kind in KINDS)
        total = agent.extra.get("budget.tokens", 0) + spent
        agent.extra["budget.tokens"] = total
        if total > self.max_tokens:
            agent.messages.append(agent.response)   # paid for — keep it; the
            raise Stop                              # loop's own append is skipped
