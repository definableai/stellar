"""Steps: how many turns the agent gets with the model."""

from core import Stop  # ponytail: 06 rewrites these as @hook functions


class Steps:
    """A ration of model calls. Steps(3) buys exactly three replies."""

    def __init__(self, n: int) -> None:
        self.n = n

    def model_pre(self, agent) -> None:
        """The step number went up before this bell, so it counts this turn."""
        if agent.step > self.n:
            raise Stop
