"""관측 파일(tracks.jsonl) 읽기와 검사.

B(pred) 또는 F(GT)가 만든 관측 행을 읽습니다. 한 행은 "한 영상의 한 시각에 검출된 객체 하나"입니다.

  {"schema_version": "cctv-3.0", "source": "pred", "track_source_id": "det01_b_colab_001_000",
   "video_id": "cityflow_S02_c007", "frame_index": 0, "timestamp_ms": 0,
   "local_track_id": "t000001",
   "track_uid": "det01_b_colab_001_000:cityflow_S02_c007:t000001",
   "class_name": "car", "bbox": [1325.21, 370.40, 1668.09, 504.35], "confidence": 0.8828,
   "quality": {"area_ratio": 0.02215, "blur_var": 4652.25, "border_touch": false, "score": 0.71097}}

D가 이 파일을 쓰는 곳
  1. scene 항목을 같은 시각(±100ms)에 관측된 트랙에 연결 (D 1)
  2. scene 이 대표인 결과의 박스 (D 3)
  3. 결과에 class_name · confidence · quality_score 표시

검사 결과는 두 종류로 나눕니다.
  problems : 검색 결과가 틀어질 수 있는 오류. 하나라도 있으면 인덱스 미준비(503)
  warnings : 검색은 가능하지만 B·F 에게 알려야 하는 사항. /health 에 표시
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, NamedTuple

from .config import SCHEMA_VERSION

# A 4 품질식 (안내서)
AREA_FULL = 0.05          # area_score = min(area_ratio / 0.05, 1)
BLUR_HALF = 100.0         # sharp_score = blur_var / (blur_var + 100)
BORDER_PENALTY = 0.8      # 경계 접촉 시 quality * 0.8
SCORE_TOLERANCE = 1e-6    # 전달된 score 와 다시 계산한 값의 허용 차이
AREA_TOLERANCE = 1e-4     # area_ratio 와 bbox 면적/영상 면적의 허용 상대 차이
BBOX_EDGE_TOLERANCE = 1.0 # 원본 범위 밖 박스 허용 픽셀

MAX_LISTED = 5            # 같은 종류의 문제는 이 개수까지만 행 번호를 보여 줌

SOURCES = ("pred", "gt")


class Observation(NamedTuple):
    """트랙의 한 시각 관측. [0]=시각, [1]=박스 순서는 ranking 모듈이 그대로 사용합니다."""

    timestamp_ms: int
    bbox: list[float]
    frame_index: int | None
    class_name: str | None
    confidence: float | None
    quality_score: float | None


def compute_quality(area_ratio: float, blur_var: float, border_touch: bool) -> float:
    """A 4 품질식. B 가 기록한 quality.score 를 다시 계산해 대조할 때 사용합니다."""
    area_score = min(float(area_ratio) / AREA_FULL, 1.0)
    sharp_score = float(blur_var) / (float(blur_var) + BLUR_HALF)
    q = 0.5 * area_score + 0.5 * sharp_score
    return q * BORDER_PENALTY if border_touch else q


@dataclass
class TrackIndex:
    """영상별·트랙별로 시각 오름차순 정렬한 관측."""

    # (video_id, track_uid) -> [Observation, ...]
    observations: dict[tuple[str, str], list[Observation]] = field(default_factory=dict)
    # video_id -> {(video_id, track_uid): [Observation, ...]}  (scene 연결 속도용)
    by_video: dict[str, dict[tuple[str, str], list[Observation]]] = field(default_factory=dict)
    rows: int = 0
    sources: set[str] = field(default_factory=set)
    track_source_ids: set[str] = field(default_factory=set)
    class_names: dict[str, int] = field(default_factory=dict)
    score_recomputed: int = 0     # score 가 없어 D 가 계산한 행 수
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "tracks": len(self.observations),
            "videos": len(self.by_video),
            "sources": sorted(self.sources),
            "track_source_ids": sorted(self.track_source_ids),
            "class_names": dict(sorted(self.class_names.items())),
            "score_recomputed_rows": self.score_recomputed,
        }


class _Issues:
    """같은 종류의 문제를 모아 '종류 (n행, 예: 3·8·12행)' 한 줄로 만듭니다."""

    def __init__(self) -> None:
        self._rows: dict[str, list[int]] = {}

    def add(self, kind: str, line_no: int) -> None:
        self._rows.setdefault(kind, []).append(line_no)

    def lines(self, file_name: str) -> list[str]:
        out = []
        for kind, rows in self._rows.items():
            shown = "·".join(str(r) for r in rows[:MAX_LISTED])
            more = f" 외 {len(rows) - MAX_LISTED}행" if len(rows) > MAX_LISTED else ""
            out.append(f"{file_name}: {kind} ({len(rows)}행, 예: {shown}행{more})")
        return out


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def load_tracks(
    path: Path,
    videos: dict[str, dict[str, Any]],
    expected_source: str | None = None,
) -> TrackIndex:
    """tracks.jsonl 을 읽어 검사합니다.

    videos: manifest 의 video_id -> 행 (gallery 에 포함된 영상만). 그 밖의 영상 관측은 건너뜁니다.
    expected_source: index_manifest 의 track_source (pred·gt). 지정되면 관측 출처와 대조합니다.
    """
    idx = TrackIndex()
    name = path.name
    if not path.is_file():
        idx.problems.append(f"관측 파일이 없습니다: {name} (scene 을 트랙에 연결할 수 없음)")
        return idx

    bad = _Issues()      # problems
    warn = _Issues()     # warnings
    seen: set[tuple[str, str, int]] = set()
    last_time: dict[tuple[str, str], int] = {}
    skipped_videos: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad.add("JSON 형식 오류", line_no)
                continue
            if not isinstance(r, dict):
                bad.add("JSON 객체가 아님", line_no)
                continue

            vid = r.get("video_id")
            if vid not in videos:
                skipped_videos.add(str(vid))
                continue
            idx.rows += 1

            # --- 필수 필드와 형식 -------------------------------------------------
            if r.get("schema_version") != SCHEMA_VERSION:
                bad.add(f"schema_version 이 {SCHEMA_VERSION} 이 아님", line_no)
            source = r.get("source")
            if source not in SOURCES:
                bad.add("source 가 pred·gt 가 아님", line_no)
            else:
                idx.sources.add(source)

            tsid, local, uid = r.get("track_source_id"), r.get("local_track_id"), r.get("track_uid")
            if not all(isinstance(x, str) and x for x in (tsid, local, uid)):
                bad.add("track_source_id·local_track_id·track_uid 누락", line_no)
                continue
            if uid != f"{tsid}:{vid}:{local}":
                # 공통 2: track_uid = track_source_id + video_id + local_track_id
                bad.add("track_uid 가 track_source_id:video_id:local_track_id 와 다름", line_no)
            idx.track_source_ids.add(tsid)

            ts = r.get("timestamp_ms")
            if not isinstance(ts, int) or isinstance(ts, bool) or ts < 0:
                bad.add("timestamp_ms 가 0 이상의 정수가 아님", line_no)
                continue
            fi = r.get("frame_index")
            if fi is not None and (not isinstance(fi, int) or isinstance(fi, bool) or fi < 0):
                bad.add("frame_index 가 0 이상의 정수가 아님", line_no)
                fi = None

            bbox = r.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4 and all(_is_number(v) for v in bbox)):
                bad.add("bbox 가 숫자 4개가 아님", line_no)
                continue
            x1, y1, x2, y2 = (float(v) for v in bbox)
            if not (x1 < x2 and y1 < y2):
                bad.add("bbox 가 x1<x2, y1<y2 를 만족하지 않음", line_no)
                continue
            v = videos[vid]
            W, H = float(v.get("width") or 0), float(v.get("height") or 0)
            if W and H and (x1 < -BBOX_EDGE_TOLERANCE or y1 < -BBOX_EDGE_TOLERANCE
                            or x2 > W + BBOX_EDGE_TOLERANCE or y2 > H + BBOX_EDGE_TOLERANCE):
                warn.add("bbox 가 원본 영상 범위를 벗어남", line_no)

            class_name = r.get("class_name")
            if not isinstance(class_name, str) or not class_name:
                warn.add("class_name 누락", line_no)
                class_name = None
            else:
                idx.class_names[class_name] = idx.class_names.get(class_name, 0) + 1

            conf = r.get("confidence")
            if source == "gt":
                if conf is not None:
                    warn.add("GT 의 confidence 가 null 이 아님", line_no)
                conf = None
            elif not (_is_number(conf) and 0.0 <= conf <= 1.0):
                bad.add("pred 의 confidence 가 0~1 숫자가 아님", line_no)
                conf = None

            # --- 품질값: score 가 있으면 A 4 식과 대조, 없으면 D 가 계산 ------------
            score = None
            q = r.get("quality")
            if q is None:
                warn.add("quality 가 null (quality_score=null 로 처리)", line_no)
            elif not isinstance(q, dict):
                bad.add("quality 가 객체가 아님", line_no)
            else:
                ar, bv, bt = q.get("area_ratio"), q.get("blur_var"), q.get("border_touch")
                parts_ok = _is_number(ar) and _is_number(bv) and bv >= 0 and isinstance(bt, bool)
                if not parts_ok:
                    warn.add("quality 의 area_ratio·blur_var·border_touch 형식 오류", line_no)
                else:
                    if W and H:
                        area = (x2 - x1) * (y2 - y1) / (W * H)
                        if abs(area - ar) > AREA_TOLERANCE * max(area, 1e-12) + 1e-9:
                            warn.add("area_ratio 가 bbox 면적/영상 면적과 다름", line_no)
                    expected = compute_quality(ar, bv, bt)
                given = q.get("score")
                if given is None:
                    if parts_ok:
                        score = expected
                        idx.score_recomputed += 1
                elif not (_is_number(given) and 0.0 <= given <= 1.0):
                    bad.add("quality.score 가 0~1 숫자가 아님", line_no)
                else:
                    score = float(given)
                    if parts_ok and abs(score - expected) > SCORE_TOLERANCE:
                        warn.add("quality.score 가 A 4 품질식 계산값과 다름", line_no)

            # --- 중복과 정렬 (B 3: 같은 video·track·시각은 한 행, 트랙별 시각 오름차순) --
            key = (vid, uid)
            if (vid, uid, ts) in seen:
                bad.add("같은 video_id·track_uid·timestamp_ms 가 두 행 이상", line_no)
                continue
            seen.add((vid, uid, ts))
            if key in last_time and ts < last_time[key]:
                warn.add("트랙 안에서 시각이 오름차순이 아님 (D 가 다시 정렬)", line_no)
            last_time[key] = ts

            idx.observations.setdefault(key, []).append(
                Observation(ts, [x1, y1, x2, y2], fi, class_name, conf, score)
            )

    for obs in idx.observations.values():
        obs.sort(key=lambda o: o.timestamp_ms)
    for key, obs in idx.observations.items():
        idx.by_video.setdefault(key[0], {})[key] = obs

    if expected_source and idx.sources and idx.sources != {expected_source}:
        idx.problems.append(
            f"{name}: 관측 출처 {sorted(idx.sources)} 가 인덱스의 track_source={expected_source} 와 다릅니다 "
            "(GT 인덱스에는 GT 관측, pred 인덱스에는 pred 관측을 연결)"
        )
    if len(idx.sources) > 1:
        idx.problems.append(f"{name}: pred 와 gt 관측이 한 파일에 섞여 있습니다")
    idx.problems.extend(bad.lines(name))
    idx.warnings.extend(warn.lines(name))
    if skipped_videos - {"None"}:
        n = len(skipped_videos)
        idx.warnings.append(f"{name}: gallery 에 없는 영상의 관측은 사용하지 않음 (영상 {n}개)")
    return idx


def check_items_against_tracks(items: Iterable[dict[str, Any]], tracks: TrackIndex) -> list[str]:
    """object 항목의 track_uid 가 관측 파일에 있는지 확인합니다.

    C 의 인덱스와 다른 실행의 관측 파일을 연결하면 scene 연결이 조용히 틀어지므로 오류로 처리합니다.
    """
    missing: list[str] = []
    item_sources: set[str] = set()
    for it in items:
        if it.get("track_source_id"):
            item_sources.add(str(it["track_source_id"]))
        if it.get("representation") != "object":
            continue
        if (it.get("video_id"), it.get("track_uid")) not in tracks.observations:
            missing.append(str(it.get("item_id")))
    problems: list[str] = []
    if missing:
        shown = ", ".join(missing[:MAX_LISTED]) + (f" 외 {len(missing) - MAX_LISTED}개" if len(missing) > MAX_LISTED else "")
        problems.append(
            f"object 항목 {len(missing)}개의 track_uid 가 관측 파일에 없습니다 ({shown}). "
            "items 와 같은 실행의 tracks.jsonl 인지 확인하세요"
        )
    unknown_sources = item_sources - tracks.track_source_ids
    if unknown_sources and tracks.track_source_ids:
        problems.append(
            f"items 의 track_source_id {sorted(unknown_sources)} 가 관측 파일에 없습니다"
        )
    return problems
