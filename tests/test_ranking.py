"""D 완료 확인: 벡터 10개 순위의 손계산 대조, 동점, 중복 트랙, 결합, 재생 구간."""
from __future__ import annotations

import numpy as np
import pytest

from retrieval.ranking import (
    dot_scores,
    link_scene_to_tracks,
    nearest_observation,
    playback_window,
    rank_tracks,
)

from .conftest import T007, T008, T009, V2T1, V2T2

Q_WHITE_CAR = np.array([1, 0, 0, 0], dtype=np.float32)  # "a white car"


def _by_uid(out):
    return {r["track_uid"]: r for r in out["results"]}


def test_dot_scores_hand_calculated(engine):
    """질의 [1,0,0,0] 의 내적은 각 벡터의 첫 성분과 같습니다."""
    scores = dot_scores(Q_WHITE_CAR, engine.gallery.vectors)
    got = {it["item_id"]: round(float(s), 6) for it, s in zip(engine.gallery.items, scores)}
    assert got == {
        "obj001": 0.8, "obj002": 0.6, "obj003": 0.8, "obj004": 0.96, "obj005": 0.28,
        "scn001": 0.6, "scn002": 1.0, "scn003": 0.0, "scn004": 0.0, "scn005": 0.6,
    }


def test_object_ranking_duplicate_track_and_tie(engine):
    out = engine.search(Q_WHITE_CAR, "object")
    assert out["score_type"] == "cosine"
    # t007 은 obj001(0.8)·obj002(0.6) 두 항목 -> 최댓값 0.8 한 건으로 모음
    # t007 과 t008 은 0.8 동점 -> track_uid 사전순
    assert [(r["track_uid"], r["score"]) for r in out["results"]] == [
        (V2T1, 0.96), (T007, 0.8), (T008, 0.8), (V2T2, 0.28),
    ]
    assert out["results"][1]["item_id"] == "obj001"
    assert out["scene_results"] == []


def test_scene_ranking_links_and_tie(engine):
    out = engine.search(Q_WHITE_CAR, "scene")
    # scn002(1.0) -> t007·t008, scn001(0.6) -> t007·t008·t009, scn005(0.6) -> v002:t002, scn003(0) -> v002:t001
    assert [(r["track_uid"], r["score"]) for r in out["results"]] == [
        (T007, 1.0), (T008, 1.0), (T009, 0.6), (V2T2, 0.6), (V2T1, 0.0),
    ]
    # 트랙과 연결되지 않은 scn004 도 장면 결과 목록에는 포함
    assert [r["item_id"] for r in out["scene_results"]] == ["scn002", "scn001", "scn005", "scn003", "scn004"]
    assert out["scene_results"][-1]["linked_track_uids"] == []
    assert all(r["track_uid"] is None and r["result_type"] == "scene" for r in out["scene_results"])


def test_combined_rrf_hand_calculated(engine):
    out = engine.search(Q_WHITE_CAR, "combined")
    assert out["score_type"] == "rrf"
    assert out["total_candidates"] == 5
    expected = [
        # track, scene 순위, object 순위
        (T007, 1, 2),
        (T008, 2, 3),
        (V2T1, 5, 1),
        (V2T2, 4, 4),
        (T009, 3, None),   # object 항이 없으면 0
    ]
    assert [r["track_uid"] for r in out["results"]] == [e[0] for e in expected]
    for r, (uid, s_rank, o_rank) in zip(out["results"], expected):
        hand = 1 / (60 + s_rank) + (1 / (60 + o_rank) if o_rank else 0.0)
        assert r["score"] == pytest.approx(hand, abs=1e-6), uid
        assert (r["scene_rank"], r["object_rank"]) == (s_rank, o_rank)
    assert out["results"][0]["score"] == pytest.approx(0.032522, abs=1e-6)  # 안내서 예제의 0.03252


def test_representative_item_and_playback_window(engine):
    res = _by_uid(engine.search(Q_WHITE_CAR, "combined"))
    top = res[T007]   # 최고 object(obj001, 1200ms) 가 대표 -> 안내서 D 3 예제와 같은 값
    assert (top["item_id"], top["representative_ms"], top["start_ms"], top["end_ms"]) == ("obj001", 1200, 200, 2200)
    assert top["bbox"] == [172, 200, 372, 360] and (top["width"], top["height"]) == (1920, 1080)
    # object 가 없는 트랙은 최고 scene 이 대표, 박스는 그 시각의 관측에서 가져옴
    t9 = res[T009]
    assert (t9["item_id"], t9["item_representation"], t9["bbox"]) == ("scn001", "scene", [1500, 300, 1600, 520])
    # 영상 시작·끝으로 구간을 맞춤
    assert (res[V2T1]["start_ms"], res[V2T1]["end_ms"]) == (0, 1400)
    assert (res[V2T2]["start_ms"], res[V2T2]["end_ms"]) == (3000, 4800)
    for r in res.values():
        assert r["start_ms"] <= r["representative_ms"] < r["end_ms"]


def test_video_filter_applied_before_ranking(engine):
    out = engine.search(Q_WHITE_CAR, "combined", ["v002"])
    # v002 만 남기면 순위를 다시 매김: object t001=1,t002=2 / scene t002=1,t001=2 -> 동점 -> 사전순
    assert [r["track_uid"] for r in out["results"]] == [V2T1, V2T2]
    assert out["results"][0]["score"] == out["results"][1]["score"] == pytest.approx(1 / 61 + 1 / 62, abs=1e-6)
    assert all(r["video_id"] == "v002" for r in out["scene_results"])


def test_empty_filter_returns_no_candidates(engine):
    out = engine.search(Q_WHITE_CAR, "combined", [])
    assert out["results"] == [] and out["scene_results"] == [] and out["total_candidates"] == 0


def test_same_input_same_output(engine):
    assert engine.search(Q_WHITE_CAR, "combined") == engine.search(Q_WHITE_CAR, "combined")


def test_scene_link_tolerance_boundary():
    obs = {("v", "a"): [(1000, [0, 0, 1, 1])], ("v", "b"): [(1300, [0, 0, 1, 1])], ("w", "c"): [(1100, None)]}
    assert link_scene_to_tracks("v", 1100, obs, 100) == ["a"]          # 차이 100ms: 연결
    assert link_scene_to_tracks("v", 1101, obs, 100) == []             # a 101ms, b 199ms: 미연결
    assert link_scene_to_tracks("v", 1200, obs, 100) == ["b"]
    assert nearest_observation([(1000, "x"), (1200, "y")], 1100, 100) == (1000, "x")  # 같은 거리면 앞 관측


def test_rank_tracks_missing_terms_and_tie_order():
    items = [
        {"item_id": "o1", "representation": "object", "track_uid": "b", "video_id": "v"},
        {"item_id": "o2", "representation": "object", "track_uid": "a", "video_id": "v"},
        {"item_id": "s1", "representation": "scene", "track_uid": None, "video_id": "v"},
    ]
    scores = np.array([0.5, 0.5, 0.9])
    ranked = rank_tracks(items, scores, [0, 0, 0], {2: ["c"]}, range(3), "combined", rrf_k=60)
    # a(object 1위)=1/61, c(scene 1위)=1/61, b(object 2위)=1/62 -> a·c 동률은 사전순
    assert [(t.track_uid, round(t.score, 6)) for t in ranked] == [("a", 0.016393), ("c", 0.016393), ("b", 0.016129)]


@pytest.mark.parametrize(
    "rep_ms,start,end,expected",
    [(1200, 0, 3200, (200, 2200)), (400, 0, 3200, (0, 1400)), (3000, 0, 3200, (2000, 3200)),
     (1500, 1000, 2000, (1000, 2000)), (500, 1000, 2000, (1000, 1500)), (100, 2000, 3000, (2000, 2001))],
)
def test_playback_window(rep_ms, start, end, expected):
    assert playback_window(rep_ms, 1000, start, end) == expected
