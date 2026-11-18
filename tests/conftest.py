from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.config import load_config  # noqa: E402
from retrieval.service import build_state  # noqa: E402

T007, T008, T009 = "det01:v001:t007", "det01:v001:t008", "det01:v001:t009"
V2T1, V2T2 = "det01:v002:t001", "det01:v002:t002"


@pytest.fixture(scope="session")
def config():
    return load_config(ROOT / "configs/retrieval.yaml")


@pytest.fixture(scope="session")
def state(config):
    s = build_state(config)
    assert s.health()["status"] == "ok", s.health()
    return s


@pytest.fixture(scope="session")
def engine(state):
    return state.galleries["gallery_dev"].engine


@pytest.fixture()
def fixture_copy(tmp_path):
    """수정해도 되는 예제 사본과 그 사본을 가리키는 설정 파일."""
    dst = tmp_path / "repo"
    shutil.copytree(ROOT / "fixtures", dst / "fixtures")
    (dst / "configs").mkdir()
    shutil.copy(ROOT / "configs/retrieval.yaml", dst / "configs/retrieval.yaml")
    return dst
