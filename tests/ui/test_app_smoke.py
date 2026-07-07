import pytest

from limbo.ui.app import LimboApp


@pytest.mark.asyncio
async def test_app_mounts():
    app = LimboApp(workdir=".")
    async with app.run_test() as pilot:
        assert pilot.app.is_running
