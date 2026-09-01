"""stellar core — the whole vocabulary, in one import.

An agent is a robot with one backpack: a brain (Model), a toolbox (Tools),
rule cards (Hooks), a notebook (messages) and one spare pocket (extra).
"""

from core.agent import Agent
from core.conformance import check_model
from core.contracts import (
    ContractError, Hook, Model, ProviderModel, Stop, Tool, fire,
)
from core.fake import FakeModel
from core.loop import check, run
from core.tool import on, tool
from core.types import Message, Part, ToolCall
