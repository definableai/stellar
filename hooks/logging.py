"""Log: every bell the agent rings, handed to one emit callable.

emit(event_name, payload) — the same signature drivers.events.EventStream
has, so `Log(stream.emit)` is the whole wiring. Any callable of that shape
does; it may be sync or async. Payloads are JSON-safe.
"""

import inspect

from core import Hook


class Log(Hook):
    """Six methods, six event names, one listener."""

    def __init__(self, emit) -> None:
        self.emit = emit

    async def say(self, name: str, payload: dict) -> None:
        answer = self.emit(name, payload)
        if inspect.isawaitable(answer):
            await answer

    async def run_pre(self, agent) -> None:
        await self.say("run_pre", {"step": agent.step,
                                   "messages": len(agent.messages)})

    async def model_pre(self, agent) -> None:
        await self.say("model_pre", {"step": agent.step})

    async def model_post(self, agent) -> None:
        said = agent.response.content
        await self.say("model_post", {
            "content": said if isinstance(said, str) else None,   # parts stay home
            "tool_calls": [{"id": c.id, "name": c.name, "args": c.args}
                           for c in agent.response.tool_calls],
            "usage": agent.response.meta.get("usage"),
        })

    async def tool_pre(self, agent) -> None:
        await self.say("tool_pre", {"id": agent.call.id, "name": agent.call.name,
                                    "args": agent.call.args})

    async def tool_post(self, agent) -> None:
        await self.say("tool_post", {"id": agent.result.tool_call_id,
                                     "content": agent.result.content})

    async def run_post(self, agent) -> None:
        await self.say("run_post", {"step": agent.step,
                                    "messages": len(agent.messages)})
