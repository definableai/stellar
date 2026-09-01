"""stellar core — the whole vocabulary, in one import.

An agent is a robot with one backpack: a brain (Model), a toolbox (Tools),
rule cards (Hooks) and one spare pocket (extra). One errand is one Run — its
own notebook, its own id — and everything that happens goes out over the
radio (Events) for whoever is listening.
"""

from core.agent import Agent, Run
from core.contracts import (
    ContractError, Hooks, Model, ProviderModel, STAGES, Stop, Tool, hook,
)
from core.events import Event, Events
from core.fake import FakeModel
from core.loop import check, run
from core.tool import tool
from core.types import Message, Part, ToolCall
