"""stellar core — the whole vocabulary, in one import.

An agent is a robot with one backpack: a brain (Model), a toolbox (Tools),
rule cards (Hooks) and one spare pocket (extra). Every conversation it holds
is a Run — its own notebook and its own id — and everything that happens
lands on the bus (Events) for anyone watching.
"""

from core.agent import Agent, Run
from core.conformance import check_model
from core.contracts import (
    ContractError, Hooks, Model, ProviderModel, Stop, Tool, fire, hook,
)
from core.events import Event, Events
from core.fake import FakeModel
from core.loop import check, run
from core.tool import tool
from core.types import Message, Part, ToolCall
