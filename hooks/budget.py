"""Budget: how many tokens the run may spend."""

from core import Stop, hook

KINDS = ("input_tokens", "output_tokens")


def Budget(max_tokens: int):
    """Adds up what every reply cost; ends the run once the total goes past max.

    The total lives in run.extra, so it is per conversation and it rides
    along in the checkpoint — one card on the agent, one purse per run.
    """

    @hook("model.post")
    async def spend(answer, run) -> None:
        """A reply with no usage meta costs nothing — not every adapter counts."""
        usage = answer.meta.get("usage") or {}
        total = run.extra.get("budget.tokens", 0) + sum(
            usage.get(kind) or 0 for kind in KINDS)
        run.extra["budget.tokens"] = total
        if total > max_tokens:
            run.messages.append(answer)     # paid for — keep it; the loop's
            raise Stop                      # own append is skipped

    return spend
