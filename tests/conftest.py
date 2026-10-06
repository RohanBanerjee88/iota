import pytest


@pytest.fixture(autouse=True)
def _main_configs(monkeypatch):
    """Tests always see the main configs, whatever IOTA_CONFIG_DIR the notebook set
    for an ablation run (r07 sets configs/fixinit before running the test suite)."""
    monkeypatch.delenv("IOTA_CONFIG_DIR", raising=False)
