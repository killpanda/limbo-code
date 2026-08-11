"""Benchmark: full cost of one scroll step in the real app layout.

Measures scroll_to(immediate) + the scroll layout pass (reflow_visible) +
compositor render + terminal-output build, for both a light chat and a
"busy" chat (expanded tool bodies with long outputs).
"""

from __future__ import annotations

import time

from textual.app import App, ComposeResult

from limbo.ui.widgets.chat import ChatWidget


class BenchApp(App[None]):
    CSS = """
    Screen { layout: vertical; }
    ChatWidget { width: 1fr; height: 1fr; }
    """

    def __init__(self, n_msgs: int = 120, heavy: bool = False) -> None:
        super().__init__()
        self.n_msgs = n_msgs
        self.heavy = heavy
        self.chat: ChatWidget | None = None
        self._profiled = False

    def compose(self) -> ComposeResult:
        yield ChatWidget(id="chat")

    async def on_mount(self) -> None:
        self.chat = self.query_one("#chat", ChatWidget)
        for i in range(self.n_msgs):
            if i % 3 == 0:
                self.chat.add_user_message(f"用户消息 #{i} " + "x" * 40)
            elif i % 3 == 1:
                self.chat.add_assistant_message(
                    f"assistant #{i}\n\n- item one\n- item two\n\n```python\nprint({i})\n```"
                )
            else:
                card = self.chat.add_tool_card(
                    f"tool-{i}", "read", {"path": f"/tmp/file_{i}.py"}
                )
                card.set_success(
                    "\n".join(f"line {j}: content {i}" for j in range(80))
                )
        await self.chat.mount_all(self.chat.children)
        if self.heavy:
            for card in self.chat.tool_cards.values():
                card.toggle()
        await self.pause(0.1)
        self._bench()

    async def pause(self, secs: float) -> None:
        import asyncio

        await asyncio.sleep(secs)

    def _bench(self) -> None:
        import gc

        gc.disable()
        chat = self.chat
        assert chat is not None
        max_y = chat.max_scroll_y
        print(f"heavy={self.heavy} widgets={len(chat.children)} max_scroll_y={max_y:.0f}")

        # Full scroll-step: immediate scroll (fires watch_scroll_y) then let
        # the screen do its scroll-layout + compositor pass, then build the
        # terminal output like _compositor_refresh does.
        step = max(1.0, max_y / 200)
        pos = max_y * 0.5
        times: list[float] = []
        seg_times: list[float] = []
        phase_times: dict[str, list[float]] = {
            "scroll_to": [],
            "refresh_layout": [],
            "render_update": [],
            "render_segments": [],
        }
        for _ in range(30):
            t0 = time.perf_counter()
            chat.scroll_to(y=pos, animate=False, immediate=True)
            phase_times["scroll_to"].append(time.perf_counter() - t0)
            # what the screen timer does on _scroll_required:
            t1 = time.perf_counter()
            self.screen._refresh_layout(scroll=True)
            layout_cost = time.perf_counter() - t1
            phase_times["refresh_layout"].append(layout_cost)
            if layout_cost > 0.05 and self._profiled is False:
                self._profiled = True
                import cProfile, pstats, io

                pr = cProfile.Profile()
                pr.enable()
                self.screen._refresh_layout(scroll=True)
                pr.disable()
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
                ps.print_stats(18)
                print("=== SLOW LAYOUT PROFILE ===")
                print(s.getvalue())
            t2 = time.perf_counter()
            update = self.screen._compositor.render_update()
            phase_times["render_update"].append(time.perf_counter() - t2)
            # build the terminal escape-sequence output (the part that shows
            # up even in fast terminals; the write itself is excluded here)
            seg_start = time.perf_counter()
            update.render_segments(self.app.console)
            phase_times["render_segments"].append(time.perf_counter() - seg_start)
            times.append(time.perf_counter() - t0)
            seg_times.append(time.perf_counter() - seg_start)
            pos += step
            if pos > max_y:
                pos = 0.0
        times.sort()
        med = times[len(times) // 2]
        seg_times.sort()
        seg_med = seg_times[len(seg_times) // 2]
        print(
            f"  full scroll step: median={med * 1000:.2f} ms  "
            f"p95={times[int(len(times) * 0.95)] * 1000:.2f} ms  "
            f"-> ~{1.0 / med:.0f} fps"
        )
        print(f"  output-build (render_segments): median={seg_med * 1000:.2f} ms")
        for name, vals in phase_times.items():
            vals.sort()
            print(
                f"    {name}: median={vals[len(vals) // 2] * 1000:.2f} ms  "
                f"max={vals[-1] * 1000:.2f} ms"
            )
        self.exit()


if __name__ == "__main__":
    import asyncio

    for heavy in (False, True):
        app = BenchApp(n_msgs=500, heavy=heavy)
        asyncio.run(app.run_async(headless=True))
