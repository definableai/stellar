"""Two serializers over a stream of StepEvents. Framework-free on purpose.

Async iterator in, strings out — hand them to FastAPI's StreamingResponse,
a Starlette websocket, aiohttp, whatever you run:

    return StreamingResponse(sse(stream.subscribe()),
                             media_type="text/event-stream")
"""

import json
from collections.abc import AsyncIterator
from dataclasses import asdict


def dump(event) -> str:
    """One event as one line of JSON."""
    return json.dumps(asdict(event), default=str)


async def sse(events) -> AsyncIterator[str]:
    """Server-Sent Events. `id:` carries seq, `event:` names the bell."""
    async for event in events:
        yield f"id: {event.seq}\nevent: {event.event}\ndata: {dump(event)}\n\n"
    yield "event: done\ndata: {}\n\n"


async def ws_frames(events) -> AsyncIterator[str]:
    """Websocket text frames: one whole event per frame."""
    async for event in events:
        yield dump(event)
