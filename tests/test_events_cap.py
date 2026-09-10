"""The log is bounded: the oldest events drop, and a replay starts where it now begins.

Run: uv run python tests/test_events_cap.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.events import Events  # noqa: E402


def test_keep_drops_the_oldest_and_the_seq_keeps_counting() -> None:
    bus = Events(keep=10)
    for n in range(15):
        bus.emit(f"step.{n}", run_id="r1")
    assert len(bus.log) == 10                # the cap holds
    assert bus.log[0].seq == 5               # the first five are gone
    assert (bus.log[-1].seq, bus.seq) == (14, 15)


def test_a_replay_from_zero_gives_back_what_the_log_kept() -> None:
    async def watch() -> None:
        bus, seen = Events(keep=10), []
        for n in range(15):
            bus.emit(f"step.{n}", run_id="r1")

        async def read() -> None:
            async for event in bus.stream(run_id="r1"):
                seen.append(event.seq)

        task = asyncio.create_task(read())
        await asyncio.sleep(0)               # replayed; now parked on the queue
        bus.emit("run.post", run_id="r1")
        await asyncio.wait_for(task, 1)
        assert seen == list(range(5, 16))    # the ten kept, in order, then live

    asyncio.run(watch())


if __name__ == "__main__":
    for test in (
        test_keep_drops_the_oldest_and_the_seq_keeps_counting,
        test_a_replay_from_zero_gives_back_what_the_log_kept,
    ):
        test()
        print(f"  ok {test.__name__}")
    print("test_events_cap: all ok")
