"""``/tracing-viewer`` command logic, shared by the TUI and headless frontends.

The frontends are thin adapters: they own one :class:`TracingViewerCommand`,
pass it the command argument, print the returned message, and call
:meth:`TracingViewerCommand.close` on exit.

Usage: ``/tracing-viewer`` opens the current session, ``/tracing-viewer
<id prefix>`` a past one, ``/tracing-viewer off`` stops the server.
"""

from __future__ import annotations

import os
import sys
import webbrowser
from collections.abc import Callable
from typing import TYPE_CHECKING

from limbo.sessions import AmbiguousSessionError, SessionNotFoundError, find_session
from limbo.tracing.server import TraceViewer

if TYPE_CHECKING:
    from limbo.agent import Agent

USAGE = "/tracing-viewer [会话 id 前缀|off]"


def can_open_browser() -> bool:
    """Whether launching a browser can work (and won't hijack the terminal).

    Over SSH, or on Linux without a display, ``webbrowser`` would fall back
    to a text browser (lynx/w3m) that takes over the TUI's terminal; the
    URL is printed instead.
    """
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return False
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


class TracingViewerCommand:
    def __init__(
        self,
        agent: Callable[[], Agent],
        open_browser: Callable[[str], bool] = webbrowser.open,
        browser_available: Callable[[], bool] = can_open_browser,
    ) -> None:
        self._agent = agent
        self._open_browser = open_browser
        self._browser_available = browser_available
        self.viewer: TraceViewer | None = None

    def run(self, arg: str) -> str:
        """Execute the command; returns the line to show the user."""
        arg = arg.strip()
        if arg == "off":
            if self.viewer is None or not self.viewer.running:
                return "Tracing viewer 未在运行"
            self.close()
            return "已关闭 tracing viewer"
        agent = self._agent()
        session_dir = agent.session_path.parent
        target = agent.session_id
        if arg:
            try:
                target = find_session(session_dir, arg).stem
            except SessionNotFoundError:
                return f"找不到会话：{arg}（用法：{USAGE}）"
            except AmbiguousSessionError as e:
                return f"会话前缀 {arg} 不唯一：{e}"
        if self.viewer is None:
            self.viewer = TraceViewer(
                session_dir=session_dir,
                workdir=agent.workdir,
                current_session=lambda: self._agent().session_id,
            )
        try:
            self.viewer.start()
        except OSError as e:
            return f"Tracing viewer 启动失败：{e}"
        url = self.viewer.url(target)
        if self._browser_available():
            try:
                opened = self._open_browser(url)
            except Exception:  # noqa: BLE001 - a broken browser setup must not crash the UI
                opened = False
            if opened:
                return f"Tracing viewer 已在浏览器打开：{url}"
        port = self.viewer.port
        return (
            f"Tracing viewer：{url}\n"
            f"（未能自动打开浏览器，请复制链接；远程机器可先 "
            f"ssh -L {port}:127.0.0.1:{port} 转发端口）"
        )

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.stop()
            self.viewer = None
