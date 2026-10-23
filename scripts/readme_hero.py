#!/usr/bin/env python3
"""Render the README hero screenshot (a realistic mid-task session).

    uv run python scripts/readme_hero.py        # -> docs/assets/limbo-hero.svg
                                                #    docs/assets/limbo-models.svg

The SVG comes straight from a pilot run of the real app (same trick as
``scripts/ui_walkthrough.py``), so it cannot drift from the actual UI.
Content is deliberately ASCII: the SVG positions each text run by cell
metrics, and non-ASCII fallback fonts render with shifted glyph widths.

Convert to PNG for the README with:

    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless \\
        --disable-gpu --hide-scrollbars --window-size=1116,610 \\
        --screenshot=docs/assets/limbo-hero.png \\
        file://$PWD/docs/assets/limbo-hero.svg

(same for limbo-models.svg -> limbo-models.png).
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from limbo.config import Config
from limbo.ui.app import LimboApp
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.status_bar import StatusBar

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"
OUT = ASSETS / "limbo-hero.svg"
MODELS_OUT = ASSETS / "limbo-models.svg"

# Screenshots catch animations mid-frame; render the settled UI.
os.environ.setdefault("LIMBO_MOTION", "off")


class FakeLLMClient:
    async def chat(self, messages, tools, on_request=None):
        return
        yield  # pragma: no cover


def _config() -> Config:
    cfg = Config()
    cfg.llm.api_key = "hero"
    cfg.llm.model = "deepseek-v4-flash"
    cfg.ui.theme = "limbo-dusk"
    return cfg


async def render_hero(workdir: Path) -> None:
    app = LimboApp(
        workdir=workdir,
        config=_config(),
        llm_client=FakeLLMClient(),
        session_dir=Path(tempfile.mkdtemp()),  # never write into the repo
    )
    async with app.run_test(size=(100, 25)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        chat = screen.query_one("#chat", ChatWidget)
        statusbar = screen.query_one("#statusbar", StatusBar)
        chat.clear()  # drop the fresh-session landing card
        statusbar.set_model("deepseek-v4-flash")
        statusbar.set_tokens(18400)

        chat.add_user_message("why is the LLM request timing out? fix it and run tests")
        chat.add_thinking_message(
            "The timeout lives in LLMConfig. Let me read it, then bump it to 30s."
        )
        chat.add_tool_card("c1", "read", {"path": "src/limbo/config.py"}).set_success(
            " 1 | class LLMConfig(BaseModel):\n"
            " 2 |     timeout: float = 60.0\n"
            " 3 |     max_retries: int = 3"
        )
        chat.add_assistant_message(
            "The default is 60s but `bash` calls pass a shorter budget; bumping "
            "`LLMConfig.timeout` to 30s and re-running the suite."
        )
        card = chat.add_tool_card(
            "c2",
            "edit",
            {
                "path": "src/limbo/config.py",
                "old_text": "    timeout: float = 60.0",
                "new_text": "    timeout: float = 30.0",
            },
        )
        card.set_success(
            "--- a/src/limbo/config.py\n"
            "+++ b/src/limbo/config.py\n"
            "@@ -2,3 +2,3 @@\n"
            " class LLMConfig(BaseModel):\n"
            "-    timeout: float = 60.0\n"
            "+    timeout: float = 30.0\n"
            "     max_retries: int = 3"
        )
        bash_card = chat.add_tool_card("c3", "bash", {"command": "pytest tests/ -q"})
        bash_card.set_success("418 passed in 19.5s")
        await pilot.pause(0.3)
        chat.tool_cards["c2"].toggle()  # expand the edit card to show the diff
        bash_card.toggle()  # ... and the bash result
        await pilot.pause(0.4)
        pilot.app.save_screenshot(str(OUT))
    print(f"wrote {OUT}")


async def render_models(workdir: Path) -> None:
    """The ctrl+l model picker: providers, recents, and inline login."""
    cfg = _config()
    cfg.ui.show_banner = False  # keep the landing card out from behind the dialog
    app = LimboApp(
        workdir=workdir,
        config=cfg,
        llm_client=FakeLLMClient(),
        session_dir=Path(tempfile.mkdtemp()),
    )
    async with app.run_test(size=(100, 25)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+l")
        await pilot.pause(0.2)
        # A query hides the (CJK) placeholder, which the SVG renders poorly.
        await pilot.press(*"glm")
        await pilot.pause(0.4)
        pilot.app.save_screenshot(str(MODELS_OUT))
    print(f"wrote {MODELS_OUT}")


async def main() -> None:
    # Default to the repo itself: the status bar shows the workdir, so the shot
    # reads like a real session (no throwaway /tmp path).
    repo = Path(__file__).resolve().parent.parent
    workdir = Path(sys.argv[1]) if len(sys.argv) > 1 else repo
    workdir.mkdir(parents=True, exist_ok=True)
    await render_hero(workdir)
    await render_models(workdir)


if __name__ == "__main__":
    asyncio.run(main())
