"""The bus's promises: numbering, filtering, forgiveness, replay into live.

Run: uv run python tests/test_events.py
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.contracts import ContractError  # noqa: E402
from core.events import Events  # noqa: E402


class Catcher(logging.Handler):
    """Keeps the log records instead of printing them."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_events_are_numbered_and_stamped() -> None:
    bus = Events()
    before = time.time()
    first = bus.emit("run.pre", "hello", source="loop", run_id="r1")
    second = bus.emit("run.post")
    assert (first.seq, second.seq) == (0, 1)
    assert (first.name, first.data) == ("run.pre", "hello")
    assert (first.source, first.run_id) == ("loop", "r1")
    assert before <= first.ts <= second.ts
    assert (second.data, second.source, second.run_id) == (None, None, None)
    assert list(bus.log) == [first, second]
    assert Events().seq == 0                 # the numbering is per bus


def test_listen_hands_the_function_back() -> None:
    bus, heard = Events(), []

    @bus.listen
    def watch(event) -> None:
        heard.append(event.name)

    bus.emit("hi")
    assert heard == ["hi"] and callable(watch)


def test_a_prefix_stops_at_a_dot() -> None:
    bus, heard = Events(), []
    bus.listen(lambda e: heard.append(("tool", e.name)), "tool")
    bus.listen(lambda e: heard.append(("all", e.name)))
    for name in ("tool", "tool.pre", "toolbox.x"):
        bus.emit(name)
    assert [n for tag, n in heard if tag == "tool"] == ["tool", "tool.pre"]
    assert [n for tag, n in heard if tag == "all"] == [
        "tool", "tool.pre", "toolbox.x"]


def test_a_listener_can_pick_one_run() -> None:
    bus, heard = Events(), []
    bus.listen(lambda e: heard.append(e.run_id), run_id="r1")
    bus.listen(lambda e: heard.append(("tool", e.run_id)), "tool", "r1")
    bus.emit("tool.pre", run_id="r1")
    bus.emit("tool.pre", run_id="r2")
    bus.emit("tool.pre")                     # no run at all
    assert heard == ["r1", ("tool", "r1")]


def test_a_broken_listener_does_not_stop_the_rest() -> None:
    bus, heard = Events(), []
    caught, logger = Catcher(), logging.getLogger("core.events")
    logger.addHandler(caught)
    bus.listen(lambda e: heard.append("first"))
    bus.listen(lambda e: 1 / 0)
    bus.listen(lambda e: heard.append("third"))
    try:
        event = bus.emit("boom")             # the emitter hears nothing of it
    finally:
        logger.removeHandler(caught)
    assert heard == ["first", "third"] and event.name == "boom"
    assert len(caught.records) == 1
    assert caught.records[0].exc_info[0] is ZeroDivisionError


def test_a_listener_can_detach_itself_mid_emit() -> None:
    bus, heard = Events(), []

    def once(event) -> None:
        heard.append(event.name)
        bus.detach(once)

    bus.listen(once)
    tail = bus.listen(lambda e: heard.append("tail"))
    bus.emit("one")                          # the copy is already iterating
    bus.emit("two")
    assert heard == ["one", "tail", "tail"]
    bus.detach(tail)
    bus.detach(tail)                         # again, and a stranger: no-ops
    bus.detach(len)
    bus.emit("three")
    assert heard == ["one", "tail", "tail"] and bus.listeners == []


def test_stream_replays_the_log_then_goes_live() -> None:
    async def watch() -> None:
        bus, seen = Events(), []
        bus.emit("run.pre", run_id="r1")
        bus.emit("tool.pre", run_id="r2")    # another run, never ours

        async def read() -> None:
            async for event in bus.stream(run_id="r1"):
                seen.append(event.name)

        task = asyncio.create_task(read())
        await asyncio.sleep(0)               # replayed; now parked on the queue
        bus.emit("model.post", run_id="r1")
        bus.emit("run.post", run_id="r2")    # not the run being watched
        bus.emit("run.post", run_id="r1")
        bus.emit("late", run_id="r1")        # after the ending: too late
        await asyncio.wait_for(task, 1)
        assert seen == ["run.pre", "model.post", "run.post"]
        assert bus.listeners == []           # the queue listener let go

    asyncio.run(watch())


def test_an_event_born_mid_replay_arrives_once() -> None:
    async def watch() -> None:
        bus, seen = Events(), []
        for n in range(3):
            bus.emit(f"old.{n}", run_id="r1")

        async def read() -> None:
            async for event in bus.stream(run_id="r1"):
                seen.append(event.seq)
                if event.seq == 0:           # emitted while replay is paused
                    bus.emit("mid.flight", run_id="r1")
                    bus.emit("run.post", run_id="r1")

        await asyncio.wait_for(read(), 1)
        assert seen == [0, 1, 2, 3, 4]       # in order, each of them once

    asyncio.run(watch())


def test_since_trims_the_replay_and_not_the_live_tail() -> None:
    async def watch() -> None:
        bus, seen = Events(), []
        for n in range(4):
            bus.emit(f"step.{n}", run_id="r1")
        bus.emit("run.post", run_id="r1")
        assert [e.seq async for e in bus.stream(run_id="r1", since=3)] == [3, 4]
        assert [e.seq async for e in bus.stream(run_id="r1")] == [0, 1, 2, 3, 4]

        fresh = Events()
        fresh.emit("run.pre", run_id="r1")

        async def read() -> None:
            async for event in fresh.stream(run_id="r1", since=99):
                seen.append(event.name)              # a cursor past the end

        task = asyncio.create_task(read())
        await asyncio.sleep(0)
        fresh.emit("model.post", run_id="r1")
        fresh.emit("run.post", run_id="r1")
        await asyncio.wait_for(task, 1)
        assert seen == ["model.post", "run.post"]    # nothing replayed, still ends

    asyncio.run(watch())


def test_a_prefixed_stream_still_ends_with_its_run() -> None:
    async def watch() -> None:
        bus = Events()
        bus.emit("tool.pre", run_id="r1")
        bus.emit("toolbox.x", run_id="r1")
        bus.emit("tool.post", run_id="r2")   # another run's tool
        bus.emit("run.post", run_id="r1")    # filtered out, but it still ends
        seen = [e.name async for e in bus.stream(run_id="r1", prefix="tool")]
        assert seen == ["tool.pre"]

    asyncio.run(watch())


def test_an_async_listener_is_refused_up_front() -> None:
    bus = Events()

    async def eager(event) -> None: ...

    try:
        bus.listen(eager)
    except ContractError as e:
        assert "stream()" in str(e)
    else:
        raise AssertionError("an async listener would never be awaited")


def test_a_stream_lets_go_when_it_is_closed() -> None:
    async def watch() -> None:
        bus = Events()
        bus.emit("one")
        stream = bus.stream()                # no run_id: it would never end
        assert (await anext(stream)).name == "one"
        assert len(bus.listeners) == 1
        await stream.aclose()
        assert bus.listeners == []

    asyncio.run(watch())


if __name__ == "__main__":
    for test in (
        test_events_are_numbered_and_stamped,
        test_listen_hands_the_function_back,
        test_a_prefix_stops_at_a_dot,
        test_a_listener_can_pick_one_run,
        test_a_broken_listener_does_not_stop_the_rest,
        test_a_listener_can_detach_itself_mid_emit,
        test_stream_replays_the_log_then_goes_live,
        test_an_event_born_mid_replay_arrives_once,
        test_since_trims_the_replay_and_not_the_live_tail,
        test_a_prefixed_stream_still_ends_with_its_run,
        test_an_async_listener_is_refused_up_front,
        test_a_stream_lets_go_when_it_is_closed,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_events: all ok")
