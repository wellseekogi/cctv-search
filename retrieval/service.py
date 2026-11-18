"""인코더와 gallery 를 한 번에 준비하는 공용 상태. API 와 CLI 가 함께 사용합니다."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import AppConfig
from .encoder import load_encoder
from .index import IndexNotReady, check_encoder_matches, load_gallery
from .search import SearchEngine


@dataclass
class GalleryState:
    gallery_id: str
    engine: SearchEngine | None = None
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.engine is not None and not self.problems


@dataclass
class ServiceState:
    config: AppConfig
    encoder: Any = None
    encoder_problem: str | None = None
    galleries: dict[str, GalleryState] = field(default_factory=dict)

    @property
    def model_ready(self) -> bool:
        return self.encoder is not None

    @property
    def index_ready(self) -> bool:
        return bool(self.galleries) and all(g.ready for g in self.galleries.values())

    def health(self) -> dict[str, Any]:
        enc = self.encoder
        galleries = []
        for gid, gs in sorted(self.galleries.items()):
            g = gs.engine.gallery if gs.engine else None
            galleries.append({
                "gallery_id": gid,
                "index_ready": gs.ready,
                "index_version": g.index_version if g else None,
                "items": len(g.items) if g else 0,
                "embedding_dim": g.dim if g else None,
                "videos": len(g.videos) if g else 0,
                "tracks": len(g.observations) if g else 0,
                "observations": g.tracks.summary() if g else None,
                "problems": gs.problems,
                "warnings": gs.warnings,
            })
        return {
            "status": "ok" if (self.model_ready and self.index_ready) else "not_ready",
            "model_ready": self.model_ready,
            "index_ready": self.index_ready,
            "model": {
                "model_id": getattr(enc, "model_id", None),
                "model_revision": getattr(enc, "model_revision", None),
                "preprocess_hash": getattr(enc, "preprocess_hash", None),
                "embedding_dim": getattr(enc, "embedding_dim", None),
                "problem": self.encoder_problem,
            },
            "galleries": galleries,
        }


def build_state(config: AppConfig) -> ServiceState:
    """오류가 나도 예외를 던지지 않고 '미준비' 상태로 기록합니다(/health·503 응답에 사용)."""
    state = ServiceState(config=config)
    try:
        state.encoder = load_encoder(config.encoder_factory, config.encoder_options, config.root)
    except Exception as e:
        state.encoder_problem = f"{type(e).__name__}: {e}"

    for gid, gcfg in config.galleries.items():
        gs = GalleryState(gallery_id=gid)
        try:
            gallery = load_gallery(gcfg)
            gs.engine = SearchEngine(gallery, config.retrieval)
            gs.warnings.extend(gallery.warnings)
            if state.encoder is not None:
                gs.problems.extend(check_encoder_matches(gallery, state.encoder))
        except IndexNotReady as e:
            gs.problems.extend(e.problems)
        except Exception as e:
            gs.problems.append(f"{type(e).__name__}: {e}")
        state.galleries[gid] = gs
    return state
