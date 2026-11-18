"""D 검색 모듈 명령.

  python -m retrieval check   --config configs/retrieval.yaml
  python -m retrieval search  --config configs/retrieval.yaml --gallery gallery_dev --text "a white car"
  python -m retrieval predict --config configs/retrieval.yaml --gallery gallery_dev \
        --queries fixtures/d_small/queries.jsonl --out runs/d_small/predictions.jsonl

exit code: 0 정상, 2 입력·검사 오류, 1 내부 처리 오류 (G 5 규칙)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .config import REPRESENTATIONS, SCHEMA_VERSION, ConfigError, load_config
from .index import read_jsonl
from .service import ServiceState, build_state


class InputError(Exception):
    pass


def _ready_engine(state: ServiceState, gallery_id: str):
    if not state.model_ready:
        raise InputError(f"질의 인코더 미준비: {state.encoder_problem}")
    gs = state.galleries.get(gallery_id)
    if gs is None:
        raise InputError(f"등록되지 않은 gallery: {gallery_id}")
    if not gs.ready:
        raise InputError("인덱스 미준비: " + "; ".join(gs.problems))
    return gs.engine


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def encode_query(state: ServiceState, query: dict[str, Any], base: Path) -> np.ndarray:
    if query.get("modality") == "text":
        text = " ".join(str(query.get("text") or "").split())
        if not text:
            raise InputError(f"{query.get('query_id')}: 빈 텍스트 질의")
        return state.encoder.encode_text([text])[0]
    if query.get("modality") == "image":
        path = base / str(query.get("image_path"))
        if not path.is_file():
            raise InputError(f"{query.get('query_id')}: 질의 이미지 없음 {path}")
        return state.encoder.encode_image([_load_rgb(path)])[0]
    raise InputError(f"{query.get('query_id')}: modality={query.get('modality')!r}")


def _atomic_write(path: Path, text: str) -> None:
    """임시 파일에 쓰고 검사 뒤 최종 이름으로 확정합니다(G 4)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def cmd_check(state: ServiceState, args: argparse.Namespace) -> int:
    health = state.health()
    print(json.dumps(health, ensure_ascii=False, indent=2))
    return 0 if health["status"] == "ok" else 2


def cmd_search(state: ServiceState, args: argparse.Namespace) -> int:
    engine = _ready_engine(state, args.gallery)
    q = state.encoder.encode_text([" ".join(args.text.split())])[0]
    out = engine.search(q, args.representation, args.video_ids)
    out["results"] = out["results"][: args.top_k]
    out["scene_results"] = out["scene_results"][: args.top_k]
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_predict(state: ServiceState, args: argparse.Namespace) -> int:
    engine = _ready_engine(state, args.gallery)
    queries_path = Path(args.queries)
    if not queries_path.is_file():
        raise InputError(f"질의 파일 없음: {queries_path}")
    base = Path(args.image_root) if args.image_root else queries_path.parent
    reps = list(REPRESENTATIONS) if args.representation == "all" else [args.representation]

    seen: set[str] = set()
    lines: list[str] = []
    per_query: list[dict[str, Any]] = []
    skipped = 0
    for q in read_jsonl(queries_path):
        qid = q.get("query_id")
        if q.get("gallery_id") not in (None, args.gallery) or (args.split and q.get("split") != args.split):
            skipped += 1
            continue
        if qid in seen:
            raise InputError(f"query_id 중복: {qid}")
        seen.add(qid)
        vec = encode_query(state, q, base)
        for rep in reps:
            out = engine.search(vec, rep, None)
            for r in out["results"]:
                lines.append(json.dumps({
                    "schema_version": SCHEMA_VERSION,
                    "query_id": qid,
                    "gallery_id": args.gallery,
                    "index_version": engine.gallery.index_version,
                    "representation": rep,
                    "score_type": out["score_type"],
                    "rank": r["rank"],
                    "track_uid": r["track_uid"],
                    "video_id": r["video_id"],
                    "score": r["score"],
                    "item_id": r["item_id"],
                    "representative_ms": r["representative_ms"],
                    "start_ms": r["start_ms"],
                    "end_ms": r["end_ms"],
                    "object_rank": r["object_rank"],
                    "scene_rank": r["scene_rank"],
                }, ensure_ascii=False))
            per_query.append({"query_id": qid, "representation": rep, "tracks": len(out["results"])})

    out_path = Path(args.out)
    _atomic_write(out_path, "".join(l + "\n" for l in lines))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "gallery_id": args.gallery,
        "index_version": engine.gallery.index_version,
        "queries_file": queries_path.as_posix(),
        "representations": reps,
        "queries": len(seen),
        "queries_skipped": skipped,
        "rows": len(lines),
        "retrieval": {
            "fusion": state.config.retrieval.fusion,
            "rrf_k": state.config.retrieval.rrf_k,
            "scene_link_tolerance_ms": state.config.retrieval.scene_link_tolerance_ms,
            "playback_context_ms": state.config.retrieval.playback_context_ms,
        },
        "per_query": per_query,
    }
    _atomic_write(out_path.with_name("predictions_summary.json"),
                  json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"predictions: 질의 {len(seen)}개, {len(lines)}행 -> {out_path.as_posix()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m retrieval", description="D 검색 모듈")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", default="configs/retrieval.yaml")

    c = sub.add_parser("check", help="인코더·인덱스 준비 상태 검사")
    common(c)

    s = sub.add_parser("search", help="텍스트 한 건 검색")
    common(s)
    s.add_argument("--gallery", required=True)
    s.add_argument("--text", required=True)
    s.add_argument("--representation", choices=REPRESENTATIONS, default="combined")
    s.add_argument("--top-k", type=int, default=10)
    s.add_argument("--video-ids", nargs="*", default=None)

    pr = sub.add_parser("predict", help="질의 목록 전체의 트랙 순위를 predictions.jsonl 로 저장")
    common(pr)
    pr.add_argument("--gallery", required=True)
    pr.add_argument("--queries", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--representation", choices=[*REPRESENTATIONS, "all"], default="combined")
    pr.add_argument("--split", default=None, help="지정하면 해당 분할 질의만 실행")
    pr.add_argument("--image-root", default=None, help="질의 image_path 의 기준 폴더 (기본: 질의 파일 폴더)")

    args = p.parse_args(argv)
    try:
        state = build_state(load_config(args.config))
        return {"check": cmd_check, "search": cmd_search, "predict": cmd_predict}[args.cmd](state, args)
    except (ConfigError, InputError, KeyError) as e:
        print(f"[입력 오류] {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"[내부 오류] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
