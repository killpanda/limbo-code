import pytest

from limbo.ui.app import LimboApp


@pytest.mark.asyncio
async def test_app_mounts(tmp_path):
    app = LimboApp(workdir=".", session_dir=tmp_path / "sessions")
    async with app.run_test() as pilot:
        assert pilot.app.is_running


