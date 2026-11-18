"""predictions.jsonl: 질의별 전체 트랙 순위, 재현성, exit code."""
from __future__ import annotations

import json

from retrieval.__main__ import main

from .conftest import ROOT, T007

ARGS = ["predict", "--config", str(ROOT / "configs/retrieval.yaml"), "--gallery", "gallery_dev",
        "--queries", str(ROOT / "fixtures/d_small/queries.jsonl")]


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_predictions_full_ranking(tmp_path):
    out = tmp_path / "predictions.jsonl"
    assert main([*ARGS, "--out", str(out)]) == 0
    rows = _rows(out)
    # 질의 4개(텍스트 3 + 이미지 1) x 후보 트랙 5개 = 20행. 상위 K 로 자르지 않은 전체 순위
    assert len(rows) == 20
    q1 = [r for r in rows if r["query_id"] == "q001"]
    assert [r["rank"] for r in q1] == [1, 2, 3, 4, 5] and q1[0]["track_uid"] == T007
    assert all(r["schema_version"] == "cctv-3.0" and r["score_type"] == "rrf" for r in rows)
    # 이미지 질의(빨간 패치)는 q001 과 같은 벡터 -> 같은 순위
    q4 = [r for r in rows if r["query_id"] == "q004"]
    assert [r["track_uid"] for r in q4] == [r["track_uid"] for r in q1]
    summary = json.loads((tmp_path / "predictions_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "success" and summary["queries"] == 4 and summary["rows"] == 20


def test_predictions_reproducible(tmp_path):
    a, b = tmp_path / "a/predictions.jsonl", tmp_path / "b/predictions.jsonl"
    assert main([*ARGS, "--out", str(a)]) == 0 and main([*ARGS, "--out", str(b)]) == 0
    assert a.read_bytes() == b.read_bytes()


def test_predictions_all_representations(tmp_path):
    out = tmp_path / "predictions.jsonl"
    assert main([*ARGS, "--out", str(out), "--representation", "all"]) == 0
    rows = _rows(out)
    counts = {rep: sum(r["query_id"] == "q001" and r["representation"] == rep for r in rows)
              for rep in ("scene", "object", "combined")}
    assert counts == {"scene": 5, "object": 4, "combined": 5}


def test_exit_codes(tmp_path):
    out = str(tmp_path / "p.jsonl")
    assert main([*ARGS[:-1], str(tmp_path / "missing.jsonl"), "--out", out]) == 2
    bad = [a if a != "gallery_dev" else "nope" for a in ARGS]
    assert main([*bad, "--out", out]) == 2
    assert main(["check", "--config", str(ROOT / "configs/retrieval.yaml")]) == 0
