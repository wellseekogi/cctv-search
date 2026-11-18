"""검색 실행: 영상 필터 → 내적 → 트랙 순위 → 대표 항목·재생 구간·미디어 연결."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import RetrievalConfig
from .index import Gallery, _item_time
from .ranking import (
    Hit,
    TrackRank,
    dot_scores,
    link_scene_to_tracks,
    nearest_observation,
    playback_window,
    rank_tracks,
)

SCORE_TYPE = {"combined": "rrf", "object": "cosine", "scene": "cosine"}
SCORE_DECIMALS = 6


@dataclass(frozen=True)
class Media:
    media_id: str
    path: Path
    content_type: str


_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def _media_id(prefix: str, gallery_id: str, rel: str) -> str:
    return prefix + hashlib.sha1(f"{gallery_id}|{rel}".encode("utf-8")).hexdigest()[:12]


def _resolve_inside(base: Path, rel: str) -> Path | None:
    """base 폴더 밖을 가리키는 경로는 등록하지 않습니다."""
    p = (base / rel).resolve()
    try:
        p.relative_to(base.resolve())
    except ValueError:
        return None
    return p


class SearchEngine:
    """한 gallery 에 대한 검색기. 만들 때 scene↔track 연결과 미디어 등록을 미리 계산합니다."""

    def __init__(self, gallery: Gallery, cfg: RetrievalConfig):
        self.gallery = gallery
        self.cfg = cfg
        self.item_times = [_item_time(it) for it in gallery.items]
        self.media: dict[str, Media] = {}
        self._frame_media: dict[int, str | None] = {}   # 행 -> 전체 프레임 media_id
        self._crop_media: dict[int, str | None] = {}    # 행 -> crop media_id (object 만)
        self._video_media: dict[str, dict[str, Any]] = {}
        self.rows_by_video: dict[str, list[int]] = {v: [] for v in gallery.videos}
        self.scene_links: dict[int, list[str]] = {}

        for row, it in enumerate(gallery.items):
            self.rows_by_video[it["video_id"]].append(row)
            if it["representation"] == "scene":
                self.scene_links[row] = link_scene_to_tracks(
                    it["video_id"], self.item_times[row],
                    gallery.tracks.by_video.get(it["video_id"], {}),   # 같은 영상의 트랙만 비교
                    cfg.scene_link_tolerance_ms,
                )
                self._frame_media[row] = self._register("f_", gallery.config.run_dir, it.get("media_path"))
            else:
                full = it.get("full_frame_path") or it.get("media_path")
                self._frame_media[row] = self._register("f_", gallery.config.run_dir, full)
                self._crop_media[row] = self._register("c_", gallery.config.run_dir, it.get("media_path"))

        for vid, row in gallery.videos.items():
            ov = gallery.config.playback_overrides.get(vid)
            rel = (ov or {}).get("path") or row["relative_path"]
            media_id = self._register("v_", gallery.config.data_root, rel)
            self._video_media[vid] = {
                "media_id": media_id,
                "source_offset_ms": int((ov or {}).get("source_offset_ms", row.get("source_offset_ms") or 0)),
                "reencoded": bool(ov),
                "codec": (ov or {}).get("codec"),
                "file_sha256": (ov or {}).get("sha256") or row.get("file_sha256"),
            }

    # ---- 미디어 -------------------------------------------------------------
    def _register(self, prefix: str, base: Path, rel: str | None) -> str | None:
        if not rel:
            return None
        rel = str(rel).replace("\\", "/")
        path = _resolve_inside(base, rel)
        if path is None:
            return None
        media_id = _media_id(prefix, self.gallery.gallery_id, rel)
        ctype = _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
        self.media[media_id] = Media(media_id, path, ctype)
        return media_id

    @staticmethod
    def _url(media_id: str | None) -> str | None:
        return f"/media/{media_id}" if media_id else None

    def video_info(self, video_id: str) -> dict[str, Any]:
        row = self.gallery.videos[video_id]
        vm = self._video_media[video_id]
        duration = int(row["duration_ms"])
        return {
            "video_id": video_id,
            "gallery_id": self.gallery.gallery_id,
            "width": int(row["width"]),
            "height": int(row["height"]),
            "duration_ms": duration,
            "fps": row.get("fps"),
            "source_offset_ms": vm["source_offset_ms"],
            "available_start_ms": int(row.get("available_start_ms") or 0),
            "available_end_ms": int(row.get("available_end_ms") or duration),
            "media_id": vm["media_id"],
            "video_url": self._url(vm["media_id"]),
            "reencoded": vm["reencoded"],
            "codec": vm["codec"],
            "file_sha256": vm["file_sha256"],
        }

    # ---- 검색 ---------------------------------------------------------------
    def candidate_rows(self, video_ids: Sequence[str] | None) -> list[int]:
        """영상 필터를 먼저 적용합니다. None=gallery 전체, []=후보 0개."""
        if video_ids is None:
            return list(range(len(self.gallery.items)))
        rows: list[int] = []
        for vid in dict.fromkeys(video_ids):  # 중복 제거·순서 유지
            rows.extend(self.rows_by_video[vid])
        return sorted(rows)

    def search(
        self,
        query_vector: np.ndarray,
        representation: str = "combined",
        video_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """전체 트랙 순위와 scene 순위를 돌려줍니다. 상위 K 자르기는 호출자가 합니다."""
        g = self.gallery
        rows = self.candidate_rows(video_ids)
        scores = dot_scores(query_vector, g.vectors)
        ranked = rank_tracks(
            g.items, scores, self.item_times, self.scene_links, rows, representation, self.cfg.rrf_k
        )
        results = [self._track_result(tr, representation) for tr in ranked]

        scene_results: list[dict[str, Any]] = []
        if representation in ("scene", "combined"):
            scene_rows = [r for r in rows if g.items[r]["representation"] == "scene"]
            scene_rows.sort(key=lambda r: (-float(scores[r]), g.items[r]["item_id"]))
            scene_results = [
                self._scene_result(i + 1, r, float(scores[r])) for i, r in enumerate(scene_rows)
            ]
        return {
            "representation": representation,
            "score_type": SCORE_TYPE[representation],
            "total_candidates": len(results),
            "results": results,
            "scene_results": scene_results,
        }

    def _window(self, video_id: str, representative_ms: int) -> tuple[int, int]:
        v = self.gallery.videos[video_id]
        duration = int(v["duration_ms"])
        return playback_window(
            representative_ms,
            self.cfg.playback_context_ms,
            int(v.get("available_start_ms") or 0),
            int(v.get("available_end_ms") or duration),
        )

    def _track_result(self, tr: TrackRank, representation: str) -> dict[str, Any]:
        g = self.gallery
        # 대표 항목: 최고 object 우선, 없으면 최고 scene
        hit: Hit = tr.object_hit or tr.scene_hit  # type: ignore[assignment]
        item = g.items[hit.item_index]
        rep_ms = hit.timestamp_ms
        # 대표 시각에 가장 가까운 해당 트랙의 관측 (class_name·confidence·quality_score 출처)
        obs = nearest_observation(
            g.observations.get((tr.video_id, tr.track_uid), ()), rep_ms,
            self.cfg.scene_link_tolerance_ms,
        )
        if item["representation"] == "object":
            bbox = item.get("bbox")
        else:  # scene 대표: 관측의 박스
            bbox = list(obs.bbox) if obs else None
        start_ms, end_ms = self._window(tr.video_id, rep_ms)
        v = g.videos[tr.video_id]
        vm = self._video_media[tr.video_id]
        return {
            "rank": tr.rank,
            "score": round(tr.score, SCORE_DECIMALS),
            "result_type": "track",
            "item_id": item["item_id"],
            "item_representation": item["representation"],
            "video_id": tr.video_id,
            "track_uid": tr.track_uid,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "representative_ms": rep_ms,
            "bbox": bbox,
            "width": int(v["width"]),
            "height": int(v["height"]),
            "class_name": obs.class_name if obs else None,
            "confidence": obs.confidence if obs else None,
            "quality_score": round(obs.quality_score, SCORE_DECIMALS) if obs and obs.quality_score is not None else None,
            "thumbnail_url": self._url(self._frame_media.get(hit.item_index)),
            "crop_url": self._url(self._crop_media.get(hit.item_index)),
            "video_url": self._url(vm["media_id"]),
            "source_offset_ms": vm["source_offset_ms"],
            "object_rank": tr.object_rank,
            "scene_rank": tr.scene_rank,
            "object_score": round(tr.object_hit.score, SCORE_DECIMALS) if tr.object_hit else None,
            "scene_score": round(tr.scene_hit.score, SCORE_DECIMALS) if tr.scene_hit else None,
        }

    def _scene_result(self, rank: int, row: int, score: float) -> dict[str, Any]:
        g = self.gallery
        item = g.items[row]
        vid = item["video_id"]
        rep_ms = self.item_times[row]
        start_ms, end_ms = self._window(vid, rep_ms)
        v = g.videos[vid]
        vm = self._video_media[vid]
        return {
            "rank": rank,
            "score": round(score, SCORE_DECIMALS),
            "result_type": "scene",
            "item_id": item["item_id"],
            "video_id": vid,
            "track_uid": None,
            "linked_track_uids": self.scene_links.get(row, []),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "representative_ms": rep_ms,
            "bbox": None,
            "width": int(v["width"]),
            "height": int(v["height"]),
            "thumbnail_url": self._url(self._frame_media.get(row)),
            "video_url": self._url(vm["media_id"]),
            "source_offset_ms": vm["source_offset_ms"],
        }
