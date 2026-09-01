"""FakeModel: a model that reads its answers off a list you wrote."""

from typing import cast

from core.contracts import ContractError, ProviderModel
from core.types import Message, Part


class FakeModel(ProviderModel):
    """Hands back the next entry each time. A str becomes an assistant line.

    A ProviderModel on purpose: each scripted reply is taken apart and
    streamed as Parts, so tests ride the same fold and model.delta path a
    real adapter uses — with no network.
    """

    def __init__(self, script: list[Message | str]) -> None:
        self.script = list(script)

    def encode(self, run) -> None:
        return None

    async def send(self, run, body):
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
