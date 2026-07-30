#!/usr/bin/env python3
"""状态走查脚本（RFC LIM-16 v1.1 Test Plan §7.4）。

用 pilot 一次性渲染五个关键状态并导出 SVG，供肉眼比对 §6.3 用色规范表：

1. idle            — 新会话首屏（紧凑 banner + 状态栏）
2. thinking        — thinking + 运行中的工具卡片（primary 蓝 spinner）
3. tool_success    — 工具成功（success 绿 + 耗时）
4. tool_error      — 工具失败（error 符号 + error-bright 文字 + error-bg 底）
5. llm_error       — 结构化错误消息块（✗ + 缩进详情）

输出到 ``docs/assets/walkthrough/``（dark 与 light 两套）。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from limbo.config import Config
from limbo.ui.app import LimboApp
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.status_bar import StatusBar

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets" / "walkthrough"


class FakeLLMClient:
    async def chat(self, messages, tools, on_request=None):
        return
        yield  # pragma: no cover


def make_app(workdir: Path, theme: str) -> LimboApp:
    cfg = Config()
    cfg.llm.api_key = "walkthrough"
    cfg.ui.theme = theme
    return LimboApp(
        workdir=workdir,
        config=cfg,
        llm_client=FakeLLMClient(),
        session_dir=workdir / "sessions",
    )


async def shoot(app: LimboApp, name: str, setup, out_dir: Path) -> None:
    async with app.run_test(size=(90, 26)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        chat = screen.query_one("#chat", ChatWidget)
        statusbar = screen.query_one("#statusbar", StatusBar)
        await setup(chat, statusbar, pilot)
        await pilot.pause(0.3)
        pilot.app.save_screenshot(str(out_dir / f"{name}.svg"))
        print(f"  {name}.svg")


async def _idle(chat, statusbar, pilot):
    statusbar.set_tokens(0)


async def _thinking(chat, statusbar, pilot):
    statusbar.set_state("thinking…", "thinking")
    chat.add_user_message("帮我看一下 agent.py 里的工具调用是怎么实现的？")
    chat.add_tool_card("c1", "read", {"path": "src/limbo/agent.py"})


async def _tool_success(chat, statusbar, pilot):
    statusbar.set_state("thinking…", "thinking")
    statusbar.set_tokens(12300)
    chat.add_user_message("运行一下测试")
    card = chat.add_tool_card("c1", "bash", {"command": "pytest tests/ -q"})
    card.set_success("418 passed in 19.5s")


async def _tool_error(chat, statusbar, pilot):
    statusbar.set_state("idle")
    chat.add_user_message("删掉 build 目录")
    card = chat.add_tool_card("c1", "bash", {"command": "rm -rf build/"})
    card.set_error("permission denied")


async def _llm_error(chat, statusbar, pilot):
    statusbar.set_state("idle")
    chat.add_error("LLM 请求失败：HTTP 429 rate limited\n请稍后重试，或检查 API 额度")


async def _edit_diff(chat, statusbar, pilot):
    statusbar.set_state("idle")
    statusbar.set_tokens(24100)
    chat.add_user_message("把超时时间改成 30 秒")
    card = chat.add_tool_card(
        "c1", "edit", {"path": "src/limbo/config.py", "old_text": "timeout = 10"}
    )
    card.set_success(
        "--- a/src/limbo/config.py\n"
        "+++ b/src/limbo/config.py\n"
        "@@ -1,3 +1,3 @@\n"
        " class LLMConfig:\n"
        "-    timeout = 10\n"
        "+    timeout = 30\n"
    )
    await pilot.pause()  # 等 card body 组合完成
    card.toggle()  # 展开显示 diff 高亮


STATES = [
    ("1_idle", _idle),
    ("2_thinking_tool_running", _thinking),
    ("3_tool_success", _tool_success),
    ("4_tool_error", _tool_error),
    ("5_llm_error", _llm_error),
    ("6_edit_diff", _edit_diff),
]


async def main() -> None:
    workdir = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/limbo-walkthrough")
    workdir.mkdir(parents=True, exist_ok=True)
    for theme in ("limbo-dark", "limbo-light"):
        out_dir = OUT_DIR / theme
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{theme}] -> {out_dir}")
        for name, setup in STATES:
            await shoot(make_app(workdir, theme), name, setup, out_dir)


if __name__ == "__main__":
    asyncio.run(main())
