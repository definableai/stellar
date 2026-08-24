"""Transport: pure serializers over the event stream.

Deliberately framework-free. These are async-iterator -> async-iterator
functions; plug them into FastAPI's StreamingResponse, Starlette
WebSockets, aiohttp — whatever you run.

    # SSE (FastAPI)
    @app.post("/agent/run")
    async def run(req: Req):
        handle = agent.run(req.input, history=load(req.session))
        return StreamingResponse(sse(handle.events()),
                                 media_type="text/event-stream")

    # WebSocket
    async for frame in ws_frames(handle.events(after_seq=cursor)):
        await websocket.send_text(frame)

The SSE ``event:`` field is one of exactly three values — step_start,
step_delta, step_end — matching the three event categories. The step
kind travels in the JSON data. ``id:`` carries ``seq`` so a client can
reconnect with Last-Event-ID and resume via ``events(after_seq=...)``.
"""

from __future__ import annotations

from typing import AsyncIterator

from core import StepEvent


async def sse(events: AsyncIterator[StepEvent]) -> AsyncIterator[str]:
    """Serialize events as Server-Sent Events frames."""
    async for event in events:
        yield (f"id: {event.seq}\nevent: {event.name}\n"
               f"data: {event.to_json()}\n\n")
    yield "event: done\ndata: {}\n\n"


async def ws_frames(events: AsyncIterator[StepEvent]) -> AsyncIterator[str]:
    """Serialize events as one JSON text frame each."""
    async for event in events:
        yield event.to_json()
