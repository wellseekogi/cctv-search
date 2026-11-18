"""인덱스 묶음 읽기와 검사.

C가 전달한 items.jsonl · vectors.npy · index_manifest.json 과
F의 manifest.jsonl, B 또는 F의 tracks.jsonl 을 읽습니다.
D는 해시·행 수·차원·정규화·모델·전처리를 확인한 뒤에만 검색을 시작합니다(공통 4).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import SCHEMA_VERSION, GalleryConfig
from .tracks import Observation, TrackIndex, check_items_against_tracks, load_tracks

NORM_TOLERANCE = 1e-4  # C 2 검사 기준과 같은 값


class IndexNotReady(RuntimeError):
    """인덱스 검사에 실패해 검색할 수 없는 상태."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name} {line_no}행: JSON 오류 ({e.msg})") from e
    return rows


@dataclass
class Gallery:
    """검색에 필요한 한 gallery 의 모든 자료."""

    gallery_id: str
    config: GalleryConfig
    index_version: str
    manifest: dict[str, Any]                      # index_manifest.json
    items: list[dict[str, Any]]                   # vector_row 순서
    vectors: np.ndarray                           # float32 [N, d]
    videos: dict[str, dict[str, Any]]             # video_id -> manifest 행
    tracks: TrackIndex = field(default_factory=TrackIndex)
    warnings: list[str] = field(default_factory=list)

    @property
    def observations(self) -> dict[tuple[str, str], list[Observation]]:
        """(video_id, track_uid) -> 시각 오름차순 관측."""
        return self.tracks.observations

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])


def _item_time(item: dict[str, Any]) -> int:
    t = item.get("actual_timestamp_ms")
    return int(item["timestamp_ms"] if t is None else t)


def load_gallery(cfg: GalleryConfig) -> Gallery:
    """gallery 를 읽고 검사합니다. 문제가 하나라도 있으면 IndexNotReady 를 냅니다."""
    problems: list[str] = []
    items_path = cfg.run_dir / "items.jsonl"
    vectors_path = cfg.run_dir / "vectors.npy"
    manifest_path = cfg.run_dir / "index_manifest.json"
    for p in (items_path, vectors_path, manifest_path, cfg.manifest):
        if not p.is_file():
            problems.append(f"파일 없음: {p.name}")
    if problems:
        raise IndexNotReady(problems)

    try:
        index_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        items = read_jsonl(items_path)
        vectors = np.load(vectors_path, allow_pickle=False)
        video_rows = read_jsonl(cfg.manifest)
    except Exception as e:  # 읽기 단계 오류는 모두 '미준비'로 처리
        raise IndexNotReady([f"읽기 실패: {e}"]) from e

    # 1) 파일 해시·행 수
    files = index_manifest.get("files") or {}
    for name, path in (("items.jsonl", items_path), ("vectors.npy", vectors_path)):
        meta = files.get(name) or {}
        expected = meta.get("sha256")
        if not expected:
            problems.append(f"index_manifest 에 {name} 해시가 없습니다")
        elif sha256_file(path) != expected:
            problems.append(f"{name} 해시가 index_manifest 와 다릅니다")
        if meta.get("rows") is not None and int(meta["rows"]) != len(items):
            problems.append(f"{name} 행 수가 index_manifest({meta['rows']})와 다릅니다({len(items)})")

    # 2) 벡터 형태·dtype·값
    if vectors.ndim != 2:
        problems.append(f"vectors.npy 는 2차원이어야 합니다 (현재 {vectors.ndim}차원)")
    else:
        n, d = vectors.shape
        if n != len(items):
            problems.append(f"items {len(items)}행과 vectors {n}행이 다릅니다")
        if index_manifest.get("embedding_dim") is not None and int(index_manifest["embedding_dim"]) != d:
            problems.append(f"벡터 차원 {d} 이 index_manifest({index_manifest['embedding_dim']})와 다릅니다")
        if vectors.dtype != np.float32:
            problems.append(f"vectors dtype 은 float32 여야 합니다 (현재 {vectors.dtype})")
        if n and not np.isfinite(vectors).all():
            problems.append("vectors 에 유한하지 않은 값이 있습니다")
        elif n:
            err = float(np.abs(np.linalg.norm(vectors.astype(np.float64), axis=1) - 1.0).max())
            if err > NORM_TOLERANCE:
                problems.append(f"정규화 오차 {err:.6f} 이 허용값 {NORM_TOLERANCE} 을 넘습니다")
    if index_manifest.get("normalized") is False:
        problems.append("index_manifest.normalized 가 false 입니다")
    if index_manifest.get("schema_version") not in (None, SCHEMA_VERSION):
        problems.append(f"schema_version 이 {SCHEMA_VERSION} 이 아닙니다")

    # 3) 영상 목록
    videos: dict[str, dict[str, Any]] = {}
    for row in video_rows:
        if cfg.split and row.get("split") != cfg.split:
            continue
        vid = row.get("video_id")
        if vid in videos:
            problems.append(f"manifest 에 video_id 중복: {vid}")
        videos[vid] = row

    # 4) items 행 대응과 필수 필드
    seen_rows: set[int] = set()
    seen_ids: set[str] = set()
    for i, it in enumerate(items):
        row = it.get("vector_row")
        if row != i:
            problems.append(f"items {i}행의 vector_row={row} (0부터 연속이어야 함)")
            break
        seen_rows.add(row)
        if it.get("item_id") in seen_ids:
            problems.append(f"item_id 중복: {it.get('item_id')}")
        seen_ids.add(it.get("item_id"))
        rep = it.get("representation")
        if rep not in ("scene", "object"):
            problems.append(f"{it.get('item_id')}: representation={rep!r}")
        elif rep == "object" and (not it.get("track_uid") or it.get("bbox") is None):
            problems.append(f"{it.get('item_id')}: object 항목에 track_uid·bbox 가 필요합니다")
        if it.get("video_id") not in videos:
            problems.append(f"{it.get('item_id')}: gallery 에 없는 video_id={it.get('video_id')}")
        if it.get("timestamp_ms") is None and it.get("actual_timestamp_ms") is None:
            problems.append(f"{it.get('item_id')}: 시각 필드가 없습니다")

    if problems:
        raise IndexNotReady(problems)

    # 5) 관측 파일 (B 의 pred 또는 F 의 GT). 형식·track_uid 구성·품질값·중복을 검사합니다.
    try:
        tracks = load_tracks(cfg.tracks, videos, index_manifest.get("track_source"))
    except Exception as e:
        raise IndexNotReady([f"관측 파일 읽기 실패: {e}"]) from e
    problems.extend(tracks.problems)
    problems.extend(check_items_against_tracks(items, tracks))
    if problems:
        raise IndexNotReady(problems)

    index_version = str(
        index_manifest.get("index_version") or "idx_" + sha256_file(manifest_path)[:12]
    )
    return Gallery(
        gallery_id=cfg.gallery_id,
        config=cfg,
        index_version=index_version,
        manifest=index_manifest,
        items=items,
        vectors=vectors,
        videos=videos,
        tracks=tracks,
        warnings=list(tracks.warnings),
    )


def check_encoder_matches(gallery: Gallery, encoder: Any) -> list[str]:
    """질의 인코더와 인덱스의 모델·전처리·차원이 같은지 확인합니다(C 1의 5번)."""
    problems: list[str] = []
    m = gallery.manifest
    if int(getattr(encoder, "embedding_dim", gallery.dim)) != gallery.dim:
        problems.append(f"인코더 차원 {encoder.embedding_dim} ≠ 인덱스 차원 {gallery.dim}")
    for key in ("model_id", "model_revision", "preprocess_hash"):
        enc_v, idx_v = getattr(encoder, key, None), m.get(key)
        if enc_v is not None and idx_v is not None and str(enc_v) != str(idx_v):
            problems.append(f"인코더 {key}={enc_v} ≠ 인덱스 {key}={idx_v}")
    return problems


__all__ = [
    "Gallery",
    "IndexNotReady",
    "check_encoder_matches",
    "load_gallery",
    "read_jsonl",
    "sha256_file",
    "_item_time",
]
