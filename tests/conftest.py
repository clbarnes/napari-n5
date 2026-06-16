from pathlib import Path

import pytest


@pytest.fixture
def data_dir():
    test_dir = Path(__file__).resolve().parent
    project_dir = test_dir.parent
    return project_dir / "data"
