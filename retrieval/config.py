"""설정 읽기 (안내서 D 4).

configs/retrieval.yaml 을 읽어 검색·API·gallery 설정 객체로 바꿉니다.
상대 경로는 모두 저장소 루트(설정 파일의 상위 폴더) 기준입니다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "cctv-3.0"
REPRESENTATIONS = ("scene", "object", "combined")


class ConfigError(ValueError):
    """설정 파일 내용이 잘못됐을 때 발생합니다."""


@dataclass(frozen=True)
class RetrievalConfig:
    fusion: str = "rrf"
    rrf_k: int = 60
    scene_link_tolerance_ms: int = 100
    playback_context_ms: int = 1000


@dataclass(frozen=True)
class ApiConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    top_k_default: int = 10
    top_k_max: int = 50
    text_max_chars: int = 500
    image_max_bytes: int = 5_000_000
    image_max_pixels: int = 16_000_000
    image_min_side_px: int = 32
    log_path: str = "logs/api.jsonl"


@dataclass(frozen=True)
class GalleryConfig:
    gallery_id: str
    run_dir: Path
    data_root: Path
    manifest: Path
    tracks: Path
    split: str | None = None
    playback_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class AppConfig:
    root: Path
    retrieval: RetrievalConfig
    api: ApiConfig
    encoder_factory: str
    encoder_options: dict[str, Any]
    galleries: dict[str, GalleryConfig]


def _only_known(section: str, raw: dict[str, Any], cls: type) -> dict[str, Any]:
    known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"{section}: 알 수 없는 설정 키 {unknown}")
    return raw


def load_config(path: str | Path, root: str | Path | None = None) -> AppConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"설정 파일이 없습니다: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = Path(root).resolve() if root else path.resolve().parent.parent

    retrieval = RetrievalConfig(**_only_known("retrieval", raw.get("retrieval") or {}, RetrievalConfig))
    api = ApiConfig(**_only_known("api", raw.get("api") or {}, ApiConfig))
    if retrieval.fusion != "rrf":
        raise ConfigError("retrieval.fusion 은 rrf 만 지원합니다")
    if retrieval.rrf_k <= 0 or retrieval.scene_link_tolerance_ms < 0 or retrieval.playback_context_ms <= 0:
        raise ConfigError("retrieval 설정값의 범위가 잘못됐습니다")
    if not (1 <= api.top_k_default <= api.top_k_max):
        raise ConfigError("api.top_k_default 는 1 이상 top_k_max 이하여야 합니다")
    if api.workers != 1:
        raise ConfigError("api.workers 는 1 로 고정합니다 (안내서 D 3)")

    enc = raw.get("encoder") or {}
    if "factory" not in enc:
        raise ConfigError("encoder.factory 가 필요합니다")

    galleries: dict[str, GalleryConfig] = {}
    for gid, g in (raw.get("galleries") or {}).items():
        for key in ("run_dir", "data_root", "manifest"):
            if key not in g:
                raise ConfigError(f"galleries.{gid}.{key} 가 필요합니다")
        run_dir = (base / g["run_dir"]).resolve()
        galleries[str(gid)] = GalleryConfig(
            gallery_id=str(gid),
            run_dir=run_dir,
            data_root=(base / g["data_root"]).resolve(),
            manifest=(base / g["manifest"]).resolve(),
            tracks=(base / g["tracks"]).resolve() if g.get("tracks") else run_dir / "tracks.jsonl",
            split=g.get("split"),
            playback_overrides=dict(g.get("playback_overrides") or {}),
        )
    if not galleries:
        raise ConfigError("galleries 에 최소 한 개의 gallery 를 등록해야 합니다")

    return AppConfig(
        root=base,
        retrieval=retrieval,
        api=api,
        encoder_factory=str(enc["factory"]),
        encoder_options=dict(enc.get("options") or {}),
        galleries=galleries,
    )
