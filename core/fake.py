"""FakeModel: a model that reads its answers off a list you wrote."""

from core.contracts import ContractError, Model
from core.types import Message


class FakeModel(Model):
    """Hands back the next entry each time. A str becomes an assistant line."""

    def __init__(self, script: list[Message | str]) -> None:
        self.script = list(script)

    async def ainvoke(self, agent) -> Message:
        if not self.script:
            raise ContractError("FakeModel script exhausted")
        answer = self.script.pop(0)
        return Message("assistant", answer) if isinstance(answer, str) else answer
