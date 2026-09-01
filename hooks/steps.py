"""Steps: how many turns the agent gets with the model."""

from core import Stop, hook


def Steps(n: int):
    """A ration of model calls. Steps(3) buys exactly three replies.

    The run counts for us, so the card keeps nothing of its own: attach one
    to the agent and every run gets its own ration.
    """

    @hook("model.pre")
    async def ration(messages, run) -> None:
        """The step number went up before this bell, so it counts this turn."""
        if run.step > n:
            raise Stop

    return ration
