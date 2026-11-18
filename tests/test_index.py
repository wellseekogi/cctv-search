"""인덱스 묶음 검사: 해시·행 대응·차원·정규화·인코더 일치."""
from __future__ import annotations

import json

import numpy as np
import pytest

from retrieval.config import ConfigError, load_config
from retrieval.index import IndexNotReady, load_gallery
from retrieval.service import build_state


def _gallery_cfg(repo):
    return load_config(repo / "configs/retrieval.yaml").galleries["gallery_dev"]


def test_fixture_index_loads(config):
    g = load_gallery(config.galleries["gallery_dev"])
    assert len(g.items) == 10 and g.vectors.shape == (10, 4) and g.vectors.dtype == np.float32
    assert [it["vector_row"] for it in g.items] == list(range(10))
    assert g.index_version == "idx_d_small_v1"


def test_tampered_vectors_detected_by_hash(fixture_copy):
    path = fixture_copy / "fixtures/d_small/run/vectors.npy"
    v = np.load(path)
    v[0] = [0, 0, 0, 1]
    np.save(path, v)
    with pytest.raises(IndexNotReady) as e:
        load_gallery(_gallery_cfg(fixture_copy))
    assert any("해시" in p for p in e.value.problems)


def _rewrite_manifest_hashes(repo):
    from retrieval.index import sha256_file
    run = repo / "fixtures/d_small/run"
    m = json.loads((run / "index_manifest.json").read_text(encoding="utf-8"))
    for name in ("items.jsonl", "vectors.npy"):
        m["files"][name]["sha256"] = sha256_file(run / name)
    (run / "index_manifest.json").write_text(json.dumps(m), encoding="utf-8")


def test_unnormalized_vectors_rejected(fixture_copy):
    path = fixture_copy / "fixtures/d_small/run/vectors.npy"
    v = np.load(path)
    v[3] *= 1.01
    np.save(path, v)
    _rewrite_manifest_hashes(fixture_copy)
    with pytest.raises(IndexNotReady) as e:
        load_gallery(_gallery_cfg(fixture_copy))
    assert any("정규화" in p for p in e.value.problems)


def test_row_count_mismatch_rejected(fixture_copy):
    path = fixture_copy / "fixtures/d_small/run/vectors.npy"
    np.save(path, np.load(path)[:9])
    _rewrite_manifest_hashes(fixture_copy)
    with pytest.raises(IndexNotReady) as e:
        load_gallery(_gallery_cfg(fixture_copy))
    assert any("행" in p for p in e.value.problems)


def test_missing_file_rejected(fixture_copy):
    (fixture_copy / "fixtures/d_small/run/items.jsonl").unlink()
    with pytest.raises(IndexNotReady):
        load_gallery(_gallery_cfg(fixture_copy))


def test_encoder_model_mismatch_marks_not_ready(fixture_copy):
    run = fixture_copy / "fixtures/d_small/run"
    m = json.loads((run / "index_manifest.json").read_text(encoding="utf-8"))
    m["model_id"] = "other-model"
    (run / "index_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    state = build_state(load_config(fixture_copy / "configs/retrieval.yaml"))
    assert state.model_ready and not state.index_ready
    assert any("model_id" in p for p in state.galleries["gallery_dev"].problems)


def test_empty_index_is_valid(fixture_copy):
    """빈 결과(items 0행, vectors [0,d])는 오류가 아니라 정상 상태입니다."""
    run = fixture_copy / "fixtures/d_small/run"
    (run / "items.jsonl").write_text("", encoding="utf-8")
    np.save(run / "vectors.npy", np.zeros((0, 4), dtype=np.float32))
    m = json.loads((run / "index_manifest.json").read_text(encoding="utf-8"))
    for name in ("items.jsonl", "vectors.npy"):
        m["files"][name]["rows"] = 0
    (run / "index_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    _rewrite_manifest_hashes(fixture_copy)
    state = build_state(load_config(fixture_copy / "configs/retrieval.yaml"))
    assert state.index_ready
    out = state.galleries["gallery_dev"].engine.search(np.array([1, 0, 0, 0], dtype=np.float32))
    assert out["results"] == [] and out["total_candidates"] == 0


def test_config_rejects_multiple_workers(fixture_copy):
    cfg = fixture_copy / "configs/retrieval.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8").replace("workers: 1", "workers: 2"), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(cfg)
