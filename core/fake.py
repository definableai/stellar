"""FakeModel: a model that reads its answers off a list you wrote.

The one Model core ships: no network, no key, so tests and examples run
anywhere. Every adapter in models/ is graded against the same loop it rides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, cast

from core.contracts import ContractError, ProviderModel
from core.types import Message, Part

if TYPE_CHECKING:                  # the checker's eyes only: no runtime edge
    from core.agent import Run

__all__ = ["FakeModel"]


class FakeModel(ProviderModel):
    """Hands back the next entry each time. A str becomes an assistant line.

    A ProviderModel on purpose: each scripted reply is taken apart and
    streamed as Parts, so tests ride the same fold and model.delta path a
    real adapter uses — with no network.
    """

    def __init__(self, script: list[Message | str]) -> None:
        self.script = list(script)     # copied: your list is left alone

    def encode(self, run: Run) -> None:
        """Nothing to build: the answers are already written.

        run: unread — the script owes nothing to the conversation.
        """
        return None

    async def send(self, run: Run, body: Any) -> AsyncIterator[Part]:
        """The next scripted reply, taken apart into Parts.

        Raises ContractError when the script runs out — a run that asked for
        one more turn than you wrote.

        run: only run.step is read, and only to name the reply that ran out.
        body: always None; encode() builds nothing to send.
        """
        if not self.script:
            raise ContractError(
                f"FakeModel script exhausted after reply {run.step - 1}"
            )
        answer = self.script.pop(0)
        if isinstance(answer, str):
            answer = Message("assistant", answer)
        for part in cast(list[Part], answer.content):
            yield part
        for call in answer.tool_calls:
            yield Part("tool_call", call)
        if answer.meta:
            yield Part("meta", answer.meta)
