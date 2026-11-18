"""순위 계산 (안내서 D 1). 파일·네트워크를 쓰지 않는 순수 함수만 둡니다.

- score = q · v (정규화 벡터의 내적)
- 같은 트랙의 object 점수·scene 점수는 각각 최댓값으로 모음
- scene 은 같은 영상·같은 시각 ±tolerance 이내에 관측된 트랙에 연결
- 결합 = 1/(k+scene 순위) + 1/(k+object 순위), 없는 항은 0, 동률은 track_uid 사전순
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Hit:
    """한 트랙에서 가장 점수가 높은 항목."""

    score: float
    item_index: int   # items 목록의 위치(= vector_row)
    timestamp_ms: int
    item_id: str


@dataclass(frozen=True)
class TrackRank:
    rank: int
    track_uid: str
    video_id: str
    score: float                 # 표현에 맞는 최종 점수 (rrf 또는 cosine)
    object_rank: int | None
    scene_rank: int | None
    object_hit: Hit | None
    scene_hit: Hit | None


def dot_scores(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """정규화한 질의 q 와 후보 행렬 V 의 내적. float64 [N] 을 돌려줍니다."""
    q = np.asarray(query, dtype=np.float64).reshape(-1)
    if vectors.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    if vectors.shape[1] != q.shape[0]:
        raise ValueError(f"질의 차원 {q.shape[0]} ≠ 인덱스 차원 {vectors.shape[1]}")
    return vectors.astype(np.float64) @ q


def _better(a: Hit | None, b: Hit) -> Hit:
    """같은 트랙 안에서 대표 항목 고르기: 점수 높은 순 → 이른 시각 → item_id 사전순."""
    if a is None:
        return b
    return min(a, b, key=lambda h: (-h.score, h.timestamp_ms, h.item_id))


def nearest_observation(
    observations: Sequence[tuple[int, Any]], timestamp_ms: int, tolerance_ms: int
) -> tuple[int, Any] | None:
    """시각 오름차순 관측에서 timestamp_ms 와 가장 가까운 관측. 허용차 밖이면 None.

    같은 거리면 앞 관측을 사용합니다(B 1의 프레임 선택 규칙과 같은 방향).
    """
    if not observations:
        return None
    i = bisect_left(observations, timestamp_ms, key=lambda o: o[0])  # 목록 복사 없이 이진 탐색
    best: tuple[int, Any] | None = None
    for j in (i - 1, i):
        if 0 <= j < len(observations):
            if best is None or abs(observations[j][0] - timestamp_ms) < abs(best[0] - timestamp_ms):
                best = observations[j]
    if best is None or abs(best[0] - timestamp_ms) > tolerance_ms:
        return None
    return best


def link_scene_to_tracks(
    video_id: str,
    timestamp_ms: int,
    observations: dict[tuple[str, str], Sequence[tuple[int, Any]]],
    tolerance_ms: int,
) -> list[str]:
    """scene 시각에서 tolerance 이내에 관측이 있는 같은 영상의 track_uid 목록(사전순)."""
    linked = [
        uid
        for (vid, uid), obs in observations.items()
        if vid == video_id and nearest_observation(obs, timestamp_ms, tolerance_ms) is not None
    ]
    return sorted(linked)


def _ranks(best: dict[str, Hit]) -> dict[str, int]:
    """점수 내림차순, 동률은 track_uid 사전순으로 1부터 순위를 매깁니다."""
    order = sorted(best, key=lambda uid: (-best[uid].score, uid))
    return {uid: i + 1 for i, uid in enumerate(order)}


def rank_tracks(
    items: Sequence[dict[str, Any]],
    scores: np.ndarray,
    item_times: Sequence[int],
    scene_links: dict[int, list[str]],
    candidate_rows: Iterable[int],
    representation: str,
    rrf_k: int = 60,
) -> list[TrackRank]:
    """후보 행(candidate_rows)만 사용해 트랙 전체 순위를 만듭니다.

    scene_links: scene 항목의 행 번호 -> 연결된 track_uid 목록
    representation: scene | object | combined
    """
    if representation not in ("scene", "object", "combined"):
        raise ValueError(f"representation={representation!r}")

    object_best: dict[str, Hit] = {}
    scene_best: dict[str, Hit] = {}
    video_of: dict[str, str] = {}

    for row in candidate_rows:
        it = items[row]
        hit = Hit(float(scores[row]), row, int(item_times[row]), str(it["item_id"]))
        if it["representation"] == "object":
            uid = it["track_uid"]
            object_best[uid] = _better(object_best.get(uid), hit)
            video_of[uid] = it["video_id"]
        else:
            for uid in scene_links.get(row, ()):
                scene_best[uid] = _better(scene_best.get(uid), hit)
                video_of.setdefault(uid, it["video_id"])

    object_rank = _ranks(object_best)
    scene_rank = _ranks(scene_best)

    if representation == "object":
        uids = list(object_best)
        final = {u: object_best[u].score for u in uids}
    elif representation == "scene":
        uids = list(scene_best)
        final = {u: scene_best[u].score for u in uids}
    else:
        uids = sorted(set(object_best) | set(scene_best))
        final = {
            u: (1.0 / (rrf_k + scene_rank[u]) if u in scene_rank else 0.0)
            + (1.0 / (rrf_k + object_rank[u]) if u in object_rank else 0.0)
            for u in uids
        }

    ordered = sorted(uids, key=lambda u: (-final[u], u))
    use_obj = representation in ("object", "combined")
    use_scn = representation in ("scene", "combined")
    return [
        TrackRank(
            rank=i + 1,
            track_uid=u,
            video_id=video_of[u],
            score=final[u],
            object_rank=object_rank.get(u) if use_obj else None,
            scene_rank=scene_rank.get(u) if use_scn else None,
            object_hit=object_best.get(u) if use_obj else None,
            scene_hit=scene_best.get(u) if use_scn else None,
        )
        for i, u in enumerate(ordered)
    ]


def playback_window(
    representative_ms: int, context_ms: int, available_start_ms: int, available_end_ms: int
) -> tuple[int, int]:
    """대표 시각 전후 context_ms 를 유효 원본 구간 [start, end) 으로 맞춥니다."""
    start = max(int(available_start_ms), int(representative_ms) - context_ms)
    end = min(int(available_end_ms), int(representative_ms) + context_ms)
    if end <= start:  # 대표 시각이 유효 구간 밖인 비정상 입력: 최소 1ms 구간 보장
        start = min(max(int(representative_ms), int(available_start_ms)), int(available_end_ms) - 1)
        end = start + 1
    return start, end
