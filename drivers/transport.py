"""Two serializers over a stream of Events. Framework-free on purpose.

Async iterator in, strings out — hand them to FastAPI's StreamingResponse,
a Starlette websocket, aiohttp, whatever you run:

    return StreamingResponse(sse(agent.events.stream(run_id=rid)),
                             media_type="text/event-stream")

The bus does the hard half: one run's events, replayed from `since` and
then live, ending on that run's run.post. These two only frame them.
"""

import json
from collections.abc import AsyncIterator
from dataclasses import asdict


def dump(event) -> str:
    """One event as one line of JSON — the Messages and Parts on it too."""
    # ponytail: asdict deep-copies the payload — emit records, not live handles
    return json.dumps(asdict(event), default=str)


async def sse(events) -> AsyncIterator[str]:
    """Server-Sent Events. `id:` carries the seq, and the seq is the cursor.

    A browser that drops hands its last id back as Last-Event-ID; reopen the
    stream with since=int(last) + 1 and the client misses nothing. The event
    name rides in the JSON, so every frame is one `message`.
    """
    async for event in events:
        yield f"id: {event.seq}\ndata: {dump(event)}\n\n"


async def ws_frames(events) -> AsyncIterator[str]:
    """Websocket text frames: one whole event per frame."""
    async for event in events:
        yield dump(event)
