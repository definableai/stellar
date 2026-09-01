"""One run off the bus, framed for the wire: SSE lines and websocket frames.

Run: uv run python tests/test_drivers.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Agent, FakeModel, Message, Tool, ToolCall  # noqa: E402
from drivers.transport import sse, ws_frames  # noqa: E402

RUN = "r1"
BELLS = ["run.pre", "model.pre", "model.delta", "model.delta", "model.post",
         "tool.pre", "tool.post",
         "model.pre", "model.delta", "model.post", "run.post"]


class Shout(Tool):
    """Uppercases a word."""

    name = "shout"
    description = "Shout a word."
    parameters = {"type": "object", "properties": {"word": {"type": "string"}}}

    async def execute(self, run, word) -> str:
        return word.upper()


def wired() -> Agent:
    """One tool call, then a goodbye — every bell of it lands on the bus."""
    return Agent(
        FakeModel([
            Message("assistant", "", [ToolCall("c1", "shout", {"word": "hi"})],
                    meta={"usage": {"input_tokens": 10, "output_tokens": 2}}),
            Message("assistant", "done"),
        ]),
        [Shout()],
    )


def fields(frame: str) -> dict:
    """One SSE frame back into the fields it carries."""
    return dict(line.split(": ", 1) for line in frame.strip().splitlines())


async def collect(frames) -> list[str]:
    return [frame async for frame in frames]


async def live() -> list[str]:
    """Open the stream first, then run: the frames arrive as they happen."""
    agent = wired()
    frames = asyncio.create_task(collect(sse(agent.events.stream(run_id=RUN))))
    await asyncio.sleep(0)                    # let the stream take its place
    await agent.run("go", run_id=RUN)
    return await frames


async def replay(serialize) -> list[str]:
    """Run to the very end, then read the whole run back afterwards."""
    agent = wired()
    await agent.run("go", run_id=RUN)
    return await collect(serialize(agent.events.stream(run_id=RUN)))


def test_a_live_stream_frames_every_event_in_order() -> None:
    sent = [fields(f) for f in asyncio.run(live())]
    assert [json.loads(f["data"])["name"] for f in sent] == BELLS
    assert [f["id"] for f in sent] == [str(n) for n in range(len(BELLS))]
    assert list(sent[0]) == ["id", "data"]    # two fields, and the frame ends
    assert asyncio.run(live())[0].endswith("\n\n")


def test_every_data_line_is_the_whole_event() -> None:
    sent = [json.loads(fields(f)["data"]) for f in asyncio.run(live())]
    assert [e["seq"] for e in sent] == list(range(len(BELLS)))
    assert {e["run_id"] for e in sent} == {RUN}
    assert {e["source"] for e in sent} == {"loop"}
    assert all(isinstance(e["ts"], float) for e in sent)
    assert sent[0]["data"]["content"] == [{"type": "text", "data": "go"}]
    assert sent[2]["data"] == {"type": "tool_call", "data": {
        "id": "c1", "name": "shout", "args": {"word": "hi"}}}
    assert sent[3]["data"] == {"type": "meta", "data": {
        "usage": {"input_tokens": 10, "output_tokens": 2}}}
    assert sent[4]["data"]["tool_calls"] == [
        {"id": "c1", "name": "shout", "args": {"word": "hi"}}]
    assert sent[5]["data"] == {"id": "c1", "name": "shout", "args": {"word": "hi"}}
    assert sent[6]["data"] == {"role": "tool", "tool_call_id": "c1", "meta": {},
                               "content": [{"type": "text", "data": "HI"}],
                               "tool_calls": []}
    assert sent[-1]["data"]["content"] == [{"type": "text", "data": "done"}]


def test_a_late_reader_gets_the_whole_run_replayed() -> None:
    frames = asyncio.run(replay(sse))
    assert [json.loads(fields(f)["data"])["name"] for f in frames] == BELLS
    # the run.post in the log ended it: a finished run never hangs a reader


def test_a_reconnect_asks_for_what_it_missed() -> None:
    async def dropped() -> None:
        agent = wired()
        await agent.run("go", run_id=RUN)
        whole = await collect(sse(agent.events.stream(run_id=RUN)))
        last = int(fields(whole[4])["id"])          # Last-Event-ID, as sent back
        again = await collect(
            sse(agent.events.stream(run_id=RUN, since=last + 1)))
        assert again == whole[5:]                   # what the drop cost, exactly
        assert len(again) == len(BELLS) - 5

    asyncio.run(dropped())


def test_websocket_frames_are_one_json_object_each() -> None:
    frames = [json.loads(f) for f in asyncio.run(replay(ws_frames))]
    assert [e["name"] for e in frames] == BELLS     # the same records, unframed
    assert frames[5]["data"]["name"] == "shout"


def test_one_stream_hears_one_run() -> None:
    async def two() -> None:
        agent = Agent(FakeModel(["done", "done"]))
        await agent.run("go", run_id="r1")
        await agent.run("go", run_id="r2")
        frames = await collect(sse(agent.events.stream(run_id="r1")))
        sent = [json.loads(fields(f)["data"]) for f in frames]
        assert {e["run_id"] for e in sent} == {"r1"}
        assert [e["name"] for e in sent] == [
            "run.pre", "model.pre", "model.delta", "model.post", "run.post"]
        assert [e["seq"] for e in sent] == [0, 1, 2, 3, 4]   # the bus numbers all

    asyncio.run(two())


if __name__ == "__main__":
    for test in (
        test_a_live_stream_frames_every_event_in_order,
        test_every_data_line_is_the_whole_event,
        test_a_late_reader_gets_the_whole_run_replayed,
        test_a_reconnect_asks_for_what_it_missed,
        test_websocket_frames_are_one_json_object_each,
        test_one_stream_hears_one_run,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_drivers: all ok")
