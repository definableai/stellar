"""One run, one stream, two serializers — end to end.

Run: uv run python tests/test_drivers.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Agent, FakeModel, Message, Tool, ToolCall  # noqa: E402
from drivers.events import EventStream, StepEvent, offer  # noqa: E402
from drivers.transport import sse, ws_frames  # noqa: E402
from hooks.logging import Log  # noqa: E402

BELLS = ["run_pre", "model_pre", "model_delta", "model_delta", "model_post",
         "tool_pre", "tool_post",
         "model_pre", "model_delta", "model_post", "run_post"]
DONE = "event: done\ndata: {}\n\n"


class Shout(Tool):
    """Uppercases a word."""

    name = "shout"
    description = "Shout a word."
    parameters = {"type": "object", "properties": {"word": {"type": "string"}}}

    async def execute(self, agent, word) -> str:
        return word.upper()


def wired(stream: EventStream) -> Agent:
    """One tool call, then a goodbye — with every bell going to the stream."""
    return Agent(
        FakeModel([
            Message("assistant", "", [ToolCall("c1", "shout", {"word": "hi"})],
                    meta={"usage": {"input_tokens": 10, "output_tokens": 2}}),
            Message("assistant", "done"),
        ]),
        [Shout()],
        [Log(stream.emit)],
    )


def fields(frame: str) -> dict:
    """One SSE frame back into the fields it carries."""
    return dict(line.split(": ", 1) for line in frame.strip().splitlines())


async def live() -> list[str]:
    """Subscribe first, then run: the frames arrive as they happen."""
    stream = EventStream()
    frames = asyncio.create_task(collect(sse(stream.subscribe())))
    await asyncio.sleep(0)                    # let the subscriber take its place
    await wired(stream).run()
    return await frames


async def replay(serialize) -> list[str]:
    """Run to the very end, then read the whole stream back afterwards."""
    stream = EventStream()
    await wired(stream).run()
    assert stream.closed                      # the run_post event shut it
    return await collect(serialize(stream.subscribe()))


async def collect(frames) -> list[str]:
    return [frame async for frame in frames]


def test_a_run_streams_every_bell_in_order() -> None:
    frames = asyncio.run(live())
    assert frames[-1] == DONE
    sent = [fields(f) for f in frames[:-1]]
    assert [f["event"] for f in sent] == BELLS
    assert [int(f["id"]) for f in sent] == list(range(1, len(sent) + 1))


def test_every_data_line_is_the_whole_event() -> None:
    sent = [json.loads(fields(f)["data"]) for f in asyncio.run(live())[:-1]]
    assert [e["event"] for e in sent] == BELLS
    assert [e["seq"] for e in sent] == list(range(1, len(sent) + 1))
    assert all(isinstance(e["ts"], float) for e in sent)
    assert sent[0]["payload"] == {"step": 0, "messages": 0}
    assert sent[2]["payload"] == {"type": "tool_call", "text": None}
    assert sent[3]["payload"] == {"type": "meta", "text": None}
    assert sent[4]["payload"] == {
        "content": "",
        "tool_calls": [{"id": "c1", "name": "shout", "args": {"word": "hi"}}],
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }
    assert sent[5]["payload"] == {"id": "c1", "name": "shout", "args": {"word": "hi"}}
    assert sent[6]["payload"] == {"id": "c1", "content": "HI"}
    assert sent[8]["payload"] == {"type": "text", "text": "done"}
    assert sent[9]["payload"]["usage"] is None      # that reply carried no meta
    assert sent[10]["payload"] == {"step": 2, "messages": 3}


def test_a_late_subscriber_gets_the_replay_and_the_done_frame() -> None:
    frames = asyncio.run(replay(sse))
    assert [fields(f)["event"] for f in frames[:-1]] == BELLS
    assert frames[-1] == DONE


def test_websocket_frames_are_one_json_object_each() -> None:
    frames = [json.loads(f) for f in asyncio.run(replay(ws_frames))]
    assert [e["event"] for e in frames] == BELLS   # no done frame: the socket ends
    assert frames[5]["payload"]["name"] == "shout"


def test_an_event_stamps_itself() -> None:
    event = StepEvent("run_pre", 1)
    assert (event.event, event.seq, event.payload) == ("run_pre", 1, {})
    assert isinstance(event.ts, float)


def test_a_full_queue_sheds_its_oldest_event() -> None:
    queue = asyncio.Queue(maxsize=2)
    for n in (1, 2, 3):
        offer(queue, n)
    assert [queue.get_nowait(), queue.get_nowait()] == [2, 3]


if __name__ == "__main__":
    for test in (
        test_a_run_streams_every_bell_in_order,
        test_every_data_line_is_the_whole_event,
        test_a_late_subscriber_gets_the_replay_and_the_done_frame,
        test_websocket_frames_are_one_json_object_each,
        test_an_event_stamps_itself,
        test_a_full_queue_sheds_its_oldest_event,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_drivers: all ok")
