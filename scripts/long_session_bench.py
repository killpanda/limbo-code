"""Benchmark: does the UI get slower as a session grows?

Drives N turns through the real MainScreen with a scripted LLM (thinking,
a Markdown reply with code, and a ``read`` tool call per iteration, four
iterations per turn) and reports, every ``--every`` turns:

- turn wall time and the worst event-loop stall during the turn (a 10 ms
  ticker measures how late it wakes up — what the user feels as lag),
- a forced full reflow + render of the screen,
- widgets in the DOM, live/frozen GC objects, RSS,
- the longest full (gen2) garbage collection.

Numbers that keep rising with the turn count are the long-session lag.

    uv run python scripts/long_session_bench.py --turns 40
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import tempfile
import time
from pathlib import Path

from limbo.config import Config
from limbo.models import CompletionMeta, TextChunk, ThinkingChunk, ToolCallEvent
from limbo.ui.app import LimboApp
from limbo.ui.screens.main import MainScreen

REPLY = (
    "Here is some **analysis** of the code.\n\n- point one about `foo`\n"
    "- point two\n\n```python\ndef f(x):\n    return x * 2\n```\n\n"
    "More prose follows here. "
) * 6


class ScriptedLLM:
    def __init__(self, chunk_delay: float) -> None:
        self.calls = 0
        self.chunk_delay = chunk_delay

    async def chat(self, messages, tools=None, on_request=None):
        if on_request:
            # Same shape as the real clients: the whole history every call.
            on_request({"messages": [m.model_dump() for m in messages], "tools": tools})
        self.calls += 1
        for _ in range(40):
            yield ThinkingChunk(text="thinking token " * 3)
            await asyncio.sleep(0)
        for i in range(0, len(REPLY), 20):
            yield TextChunk(text=REPLY[i : i + 20])
            await asyncio.sleep(self.chunk_delay)
        if messages[-1].role != "tool" or self.calls % 4 != 0:
            yield ToolCallEvent(id=f"c{self.calls}", name="read", arguments={"path": "big.py"})
        yield CompletionMeta(usage={"prompt_tokens": 1000, "completion_tokens": 100})


async def run(turns: int, every: int, chunk_delay: float) -> None:
    workdir = Path(tempfile.mkdtemp())
    (workdir / "big.py").write_text(
        "\n".join(f"line {i}: some content here {i}" for i in range(400))
    )
    config = Config()
    config.llm.api_key = "bench"
    app = LimboApp(
        workdir=workdir,
        config=config,
        llm_client=ScriptedLLM(chunk_delay),
        session_dir=workdir / "sessions",
    )
    gc_pauses: list[float] = []
    started: list[float] = []

    def on_gc(phase: str, info: dict) -> None:
        if info["generation"] != 2:
            return
        if phase == "start":
            started.append(time.perf_counter())
        elif started:
            gc_pauses.append(time.perf_counter() - started.pop())

    gc.callbacks.append(on_gc)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        screen = app.screen_stack[-1]
        assert isinstance(screen, MainScreen)
        worst = [0.0]
        done = False

        async def ticker() -> None:
            last = time.perf_counter()
            while not done:
                await asyncio.sleep(0.01)
                now = time.perf_counter()
                worst[0] = max(worst[0], now - last - 0.01)
                last = now

        tick_task = asyncio.create_task(ticker())
        for turn in range(1, turns + 1):
            worst[0] = 0.0
            gc_pauses.clear()
            t0 = time.perf_counter()
            await screen._handle_turn(f"turn {turn}: read big.py and explain")
            await pilot.pause()
            elapsed = time.perf_counter() - t0
            if turn % every and turn != turns:
                continue
            reflow = []
            for _ in range(9):
                t1 = time.perf_counter()
                screen._refresh_layout()
                screen._compositor.render_update(full=True).render_segments(app.console)
                reflow.append(time.perf_counter() - t1)
            reflow.sort()
            rss = int(Path("/proc/self/statm").read_text().split()[1]) * 4096 / 1e6
            print(
                f"turn {turn:4d}  {elapsed * 1000:6.0f} ms  "
                f"worst stall {worst[0] * 1000:5.0f} ms  "
                f"gen2 max {max(gc_pauses, default=0) * 1000:4.0f} ms  "
                f"reflow {reflow[4] * 1000:5.1f} ms  "
                f"widgets {len(screen.query('*')):4d}  "
                f"objects {len(gc.get_objects()):7d}+{gc.get_freeze_count():7d} frozen  "
                f"rss {rss:4.0f} MB",
                flush=True,
            )
        done = True
        await tick_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument(
        "--chunk-delay",
        type=float,
        default=0.001,
        help="seconds between 20-char reply chunks (0.03 ≈ a real stream)",
    )
    args = parser.parse_args()
    asyncio.run(run(args.turns, args.every, args.chunk_delay))


if __name__ == "__main__":
    main()
