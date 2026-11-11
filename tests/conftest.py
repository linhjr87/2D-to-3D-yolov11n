from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"


@pytest.fixture
def spaced_tmp(tmp_path):
    d = tmp_path / "dir with space" / "non-ascii-é-路径"
    d.mkdir(parents=True)
    return d
