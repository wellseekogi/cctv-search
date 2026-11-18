"""관측 파일(tracks.jsonl) 검사: 회의에서 정한 B 형식, track_uid 구성, 품질식, 중복·정렬."""
from __future__ import annotations

import copy
import json

import pytest

from retrieval.config import load_config
from retrieval.service import build_state
from retrieval.tracks import compute_quality, load_tracks

from .conftest import ROOT, T007, T009

SAMPLE = ROOT / "fixtures/b_sample/tracks_sample.jsonl"
CITYFLOW = {"cityflow_S02_c007": {"video_id": "cityflow_S02_c007", "width": 1920, "height": 1080}}
BASE_ROW = json.loads(SAMPLE.read_text(encoding="utf-8").splitlines()[0])


def _write(tmp_path, rows, name="tracks.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _row(**patch):
    r = copy.deepcopy(BASE_ROW)
    for k, v in patch.items():
        if k.startswith("quality_"):
            r["quality"][k[len("quality_"):]] = v
        else:
            r[k] = v
    return r


# ---- 팀이 보낸 실제 형식 -----------------------------------------------------

def test_team_sample_row_is_accepted():
    idx = load_tracks(SAMPLE, CITYFLOW, expected_source="pred")
    assert idx.problems == [] and idx.warnings == []
    key = ("cityflow_S02_c007", "det01_b_colab_001_000:cityflow_S02_c007:t000001")
    (obs,) = idx.observations[key]
    assert obs.timestamp_ms == 0 and obs.frame_index == 0 and obs.class_name == "car"
    assert obs.confidence == pytest.approx(0.8827624320983887)
    assert obs.quality_score == pytest.approx(0.7109654215527837)
    assert obs.bbox == pytest.approx([1325.21435546875, 370.3994140625, 1668.09130859375, 504.34686279296875])
    assert idx.summary()["track_source_ids"] == ["det01_b_colab_001_000"]


def test_team_sample_score_matches_quality_formula():
    q = BASE_ROW["quality"]
    assert compute_quality(q["area_ratio"], q["blur_var"], q["border_touch"]) == pytest.approx(q["score"], abs=1e-12)


@pytest.mark.parametrize("area,blur,border,expected", [
    (0.0154321, 100.0, False, 0.404321),   # 안내서 A 4 예제
    (0.0154321, 900.0, False, 0.604321),   # 같은 박스, blur_var 900
    (0.0154321, 100.0, True, 0.3234568),   # 경계 접촉 x0.8
    (0.2, 0.0, False, 0.5),                # 면적 점수 상한 1
])
def test_quality_formula(area, blur, border, expected):
    assert compute_quality(area, blur, border) == pytest.approx(expected, abs=1e-6)


# ---- 오류 (problems: 인덱스 미준비) --------------------------------------------

@pytest.mark.parametrize("patch,message", [
    ({"track_uid": "det01_b_colab_001_000:other_video:t000001"}, "track_uid 가"),
    ({"bbox": [1668.0, 370.0, 1325.0, 504.0]}, "x1<x2"),
    ({"bbox": [1, 2, 3]}, "숫자 4개"),
    ({"timestamp_ms": 12.5}, "timestamp_ms"),
    ({"source": "detector"}, "source"),
    ({"schema_version": "cctv-2.0"}, "schema_version"),
    ({"confidence": None}, "confidence"),
    ({"quality_score": 1.7}, "quality.score"),
])
def test_invalid_rows_are_problems(tmp_path, patch, message):
    idx = load_tracks(_write(tmp_path, [_row(**patch)]), CITYFLOW)
    assert any(message in p for p in idx.problems), idx.problems


def test_duplicate_observation_is_problem(tmp_path):
    idx = load_tracks(_write(tmp_path, [_row(), _row()]), CITYFLOW)
    assert any("두 행 이상" in p and "2행" in p for p in idx.problems), idx.problems


def test_missing_file_is_problem(tmp_path):
    idx = load_tracks(tmp_path / "none.jsonl", CITYFLOW)
    assert idx.problems and "관측 파일이 없습니다" in idx.problems[0]


def test_source_must_match_index(tmp_path):
    idx = load_tracks(_write(tmp_path, [_row()]), CITYFLOW, expected_source="gt")
    assert any("track_source=gt" in p for p in idx.problems)


def test_mixed_pred_and_gt_is_problem(tmp_path):
    gt = _row(source="gt", confidence=None, timestamp_ms=200)
    idx = load_tracks(_write(tmp_path, [_row(), gt]), CITYFLOW)
    assert any("섞여" in p for p in idx.problems)


# ---- 경고 (warnings: 검색은 가능, B·F 에게 알림) ---------------------------------

def test_score_mismatch_is_warning(tmp_path):
    idx = load_tracks(_write(tmp_path, [_row(quality_score=0.5)]), CITYFLOW)
    assert idx.problems == [] and any("품질식" in w for w in idx.warnings)
    assert next(iter(idx.observations.values()))[0].quality_score == 0.5   # 전달값을 그대로 사용


def test_missing_score_is_recomputed(tmp_path):
    r = _row()
    del r["quality"]["score"]
    idx = load_tracks(_write(tmp_path, [r]), CITYFLOW)
    assert idx.problems == [] and idx.warnings == [] and idx.score_recomputed == 1
    assert next(iter(idx.observations.values()))[0].quality_score == pytest.approx(0.7109654215527837)


def test_area_ratio_mismatch_and_out_of_frame_are_warnings(tmp_path):
    rows = [_row(quality_area_ratio=0.03, quality_score=compute_quality(0.03, 4652.25025543198, False)),
            _row(timestamp_ms=200, bbox=[1800.0, 370.0, 1930.0, 504.0])]
    idx = load_tracks(_write(tmp_path, rows), CITYFLOW)
    assert idx.problems == []
    assert any("area_ratio" in w for w in idx.warnings)
    assert any("범위를 벗어남" in w for w in idx.warnings)


def test_unsorted_rows_are_sorted_with_warning(tmp_path):
    rows = [_row(timestamp_ms=400), _row(timestamp_ms=0), _row(timestamp_ms=200)]
    idx = load_tracks(_write(tmp_path, rows), CITYFLOW)
    obs = next(iter(idx.observations.values()))
    assert [o.timestamp_ms for o in obs] == [0, 200, 400]
    assert any("오름차순" in w for w in idx.warnings)


def test_gt_rows_allow_null_confidence(tmp_path):
    idx = load_tracks(_write(tmp_path, [_row(source="gt", confidence=None)]), CITYFLOW, expected_source="gt")
    assert idx.problems == [] and next(iter(idx.observations.values()))[0].confidence is None


def test_other_video_rows_are_skipped(tmp_path):
    idx = load_tracks(_write(tmp_path, [_row(), _row(video_id="cityflow_S02_c008")]), CITYFLOW)
    assert idx.rows == 1 and idx.problems == []
    assert any("gallery 에 없는 영상" in w for w in idx.warnings)


# ---- 인덱스와의 연결 -----------------------------------------------------------

def _state(repo):
    return build_state(load_config(repo / "configs/retrieval.yaml"))


def test_items_track_missing_from_tracks_marks_not_ready(fixture_copy):
    path = fixture_copy / "fixtures/d_small/run/tracks.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows if r["track_uid"] != T007), encoding="utf-8")
    state = _state(fixture_copy)
    assert not state.index_ready
    assert any("관측 파일에 없습니다" in p for p in state.galleries["gallery_dev"].problems)


def test_missing_tracks_file_marks_not_ready(fixture_copy):
    (fixture_copy / "fixtures/d_small/run/tracks.jsonl").unlink()
    state = _state(fixture_copy)
    assert not state.index_ready
    assert any("관측 파일이 없습니다" in p for p in state.galleries["gallery_dev"].problems)


def test_results_carry_observation_fields(engine):
    import numpy as np
    res = {r["track_uid"]: r for r in engine.search(np.array([1, 0, 0, 0], dtype=np.float32))["results"]}
    # object 대표(obj001 @1200ms): 200x160 박스 -> 안내서 A 4 예제와 같은 품질 0.404321
    assert (res[T007]["class_name"], res[T007]["confidence"], res[T007]["quality_score"]) == ("car", 0.9, 0.404321)
    # scene 대표(scn001 @1200ms): 사람 100x220 박스
    assert res[T009]["class_name"] == "person"
    assert res[T009]["quality_score"] == pytest.approx(0.5 * (22000 / 2073600) / 0.05 + 0.25, abs=1e-6)


def test_health_shows_observation_summary(state):
    g = state.health()["galleries"][0]
    assert g["observations"]["rows"] == 51 and g["observations"]["sources"] == ["pred"]
    assert g["observations"]["class_names"] == {"car": 31, "person": 4, "truck": 16}
    assert g["warnings"] == []
