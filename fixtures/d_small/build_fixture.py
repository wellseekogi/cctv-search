"""D 작은 검색 예제(벡터 10개) 생성 스크립트.

  python fixtures/d_small/build_fixture.py            # jsonl·npy·이미지 생성
  python fixtures/d_small/build_fixture.py --videos   # ffmpeg 로 재생 확인용 MP4 도 생성

C·F·B의 실제 자료를 받기 전까지 D의 순위·API 를 검사하기 위한 가짜 입력입니다.
벡터는 4차원이라 순위를 손으로 계산할 수 있습니다(README 의 손계산 표 참고).
같은 스크립트를 다시 실행하면 같은 파일이 만들어집니다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
DATA = HERE / "data"
SCHEMA = "cctv-3.0"
SRC = "det01"

VIDEOS = {
    "v001": {"width": 1920, "height": 1080, "duration_ms": 3200},
    "v002": {"width": 1280, "height": 720, "duration_ms": 4800},
}

# 단위 벡터 (제곱합이 1)
A = [1.0, 0.0, 0.0, 0.0]
B = [0.8, 0.6, 0.0, 0.0]
C = [0.6, 0.8, 0.0, 0.0]
D = [0.0, 1.0, 0.0, 0.0]
E = [0.6, 0.0, 0.8, 0.0]
F = [0.8, 0.0, 0.6, 0.0]
H = [0.0, 0.6, 0.8, 0.0]
I = [0.28, 0.96, 0.0, 0.0]
J = [0.96, 0.28, 0.0, 0.0]


def uid(video: str, local: str) -> str:
    return f"{SRC}:{video}:{local}"


def t007_box(ts: int) -> list[int]:
    cx = {0: 200, 200: 210, 400: 260, 600: 270}.get(ts, 272 if ts <= 1200 else 280)
    return [cx - 100, 200, cx + 100, 360]


def build_tracks() -> list[dict]:
    rows = []

    def add(video, local, cls, ts, box):
        # 회의에서 정한 B 관측 형식: 원본 좌표 실수 bbox, quality.score(A 4 품질식) 포함
        v = VIDEOS[video]
        box = [float(c) for c in box]
        area = (box[2] - box[0]) * (box[3] - box[1]) / (v["width"] * v["height"])
        blur = 100.0
        border = box[0] <= 0 or box[1] <= 0 or box[2] >= v["width"] or box[3] >= v["height"]
        score = 0.5 * min(area / 0.05, 1.0) + 0.5 * blur / (blur + 100.0)
        if border:
            score *= 0.8
        rows.append({
            "schema_version": SCHEMA, "source": "pred", "track_source_id": SRC,
            "video_id": video, "frame_index": ts * 30 // 1000, "timestamp_ms": ts,
            "local_track_id": local, "track_uid": uid(video, local), "class_name": cls,
            "bbox": box, "confidence": 0.9,
            "quality": {"area_ratio": area, "blur_var": blur, "border_touch": border, "score": score},
        })

    for ts in [*range(0, 1201, 200), *range(2400, 3001, 200)]:
        add("v001", "t007", "car", ts, t007_box(ts))
    for ts in range(0, 3001, 200):
        add("v001", "t008", "truck", ts, [900, 500, 1200, 700])
    for ts in range(1000, 1601, 200):
        add("v001", "t009", "person", ts, [1500, 300, 1600, 520])
    for ts in range(0, 2001, 200):
        add("v002", "t001", "car", ts, [100, 100, 300, 260])
    for ts in range(3000, 4601, 200):
        add("v002", "t002", "car", ts, [600, 300, 860, 480])
    rows.sort(key=lambda r: (r["video_id"], r["track_uid"], r["timestamp_ms"]))
    return rows


# (item_id, representation, video, local_track, timestamp_ms, vector)
ITEMS = [
    ("obj001", "object", "v001", "t007", 1200, B),
    ("obj002", "object", "v001", "t007", 0, C),
    ("obj003", "object", "v001", "t008", 600, F),
    ("obj004", "object", "v002", "t001", 400, J),
    ("obj005", "object", "v002", "t002", 4000, I),
    ("scn001", "scene", "v001", None, 1200, E),
    ("scn002", "scene", "v001", None, 2600, A),
    ("scn003", "scene", "v002", None, 400, D),
    ("scn004", "scene", "v002", None, 2500, H),   # 이 시각에는 관측된 트랙이 없음
    ("scn005", "scene", "v002", None, 4000, C),
]

TEXT_QUERIES = {
    "a white car": A,
    "a red truck": D,
    "a person walking": [0.0, 0.0, 1.0, 0.0],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def draw_frame(video: str, ts: int, tracks: list[dict]) -> Image.Image:
    v = VIDEOS[video]
    im = Image.new("RGB", (v["width"], v["height"]), (58, 62, 70))
    d = ImageDraw.Draw(im)
    colors = {"car": (235, 235, 235), "truck": (200, 60, 50), "person": (70, 140, 220)}
    for r in tracks:
        if r["video_id"] == video and r["timestamp_ms"] == ts:
            d.rectangle(r["bbox"], fill=colors[r["class_name"]])
    d.text((20, 20), f"{video} {ts}ms", fill=(255, 255, 255))
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", action="store_true", help="ffmpeg 로 MP4 생성")
    args = ap.parse_args()

    tracks = build_tracks()
    box_at = {(r["video_id"], r["track_uid"], r["timestamp_ms"]): r["bbox"] for r in tracks}
    if RUN.exists():
        shutil.rmtree(RUN)
    (RUN / "media/frames").mkdir(parents=True)
    (RUN / "media/crops").mkdir(parents=True)
    write_jsonl(RUN / "tracks.jsonl", tracks)

    manifest = []
    for vid, v in VIDEOS.items():
        mp4 = DATA / f"videos/{vid}.mp4"
        if args.videos:
            mp4.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                "-i", f"testsrc2=size={v['width']}x{v['height']}:rate=5:duration={v['duration_ms'] / 1000}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "38", "-g", "5",
                "-movflags", "+faststart", "-map_metadata", "-1", "-fflags", "+bitexact",
                str(mp4),
            ], check=True)
        manifest.append({
            "video_id": vid, "dataset": "fixture", "camera_id": None,
            "relative_path": f"videos/{vid}.mp4", "width": v["width"], "height": v["height"],
            "duration_ms": v["duration_ms"], "fps": 5, "time_base": None, "source_offset_ms": 0,
            "available_start_ms": 0, "available_end_ms": v["duration_ms"], "split": "dev",
            "file_sha256": sha256(mp4) if mp4.is_file() else None,
        })
    write_jsonl(HERE / "manifest.jsonl", manifest)

    ordered = sorted(ITEMS, key=lambda x: (x[2], x[4], x[1], x[3] or ""))
    items, vectors = [], []
    for row, (item_id, rep, vid, local, ts, vec) in enumerate(ordered):
        assert math.isclose(sum(c * c for c in vec), 1.0, abs_tol=1e-9), item_id
        v = VIDEOS[vid]
        frame_rel = f"media/frames/{vid}_{ts:06d}.jpg"
        frame_path = RUN / frame_rel
        if not frame_path.exists():
            draw_frame(vid, ts, tracks).save(frame_path, quality=80)
        track_uid = uid(vid, local) if local else None
        bbox = crop_bbox = None
        media_rel = frame_rel
        if rep == "object":
            bbox = box_at[(vid, track_uid, ts)]
            mx, my = (bbox[2] - bbox[0]) * 0.10, (bbox[3] - bbox[1]) * 0.10
            crop_bbox = [max(0, math.floor(bbox[0] - mx)), max(0, math.floor(bbox[1] - my)),
                         min(v["width"], math.ceil(bbox[2] + mx)), min(v["height"], math.ceil(bbox[3] + my))]
            media_rel = f"media/crops/{item_id}.jpg"
            Image.open(frame_path).crop(crop_bbox).save(RUN / media_rel, quality=85)
        items.append({
            "schema_version": SCHEMA, "item_id": item_id, "video_id": vid,
            "frame_index": ts * 30 // 1000, "timestamp_ms": ts, "representation": rep,
            "track_uid": track_uid, "bbox": bbox, "event_ids": [], "reasons": ["fixture"],
            "policy": "fixture", "source_run_id": "d_small", "track_source_id": SRC,
            "vector_row": row, "model_id": "fixture-encoder", "model_revision": "fixture",
            "embedding_dim": 4, "preprocess_hash": "fixture", "media_path": media_rel,
            "full_frame_path": frame_rel, "actual_timestamp_ms": ts, "crop_bbox": crop_bbox,
        })
        vectors.append(vec)
    write_jsonl(RUN / "items.jsonl", items)
    np.save(RUN / "vectors.npy", np.asarray(vectors, dtype=np.float32))

    (RUN / "index_manifest.json").write_text(json.dumps({
        "schema_version": SCHEMA, "index_version": "idx_d_small_v1",
        "files": {
            "items.jsonl": {"sha256": sha256(RUN / "items.jsonl"), "rows": len(items)},
            "vectors.npy": {"sha256": sha256(RUN / "vectors.npy"), "rows": len(items)},
        },
        "embedding_dim": 4, "dtype": "float32", "normalized": True,
        "model_id": "fixture-encoder", "model_revision": "fixture", "preprocess_hash": "fixture",
        "split": "dev", "track_source": "pred", "track_source_id": SRC,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (HERE / "query_vectors.json").write_text(json.dumps({
        "model_id": "fixture-encoder", "preprocess_hash": "fixture", "embedding_dim": 4,
        "texts": TEXT_QUERIES,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (HERE / "queries").mkdir(exist_ok=True)
    Image.new("RGB", (64, 64), (255, 0, 0)).save(HERE / "queries/red_patch.png")
    queries = [
        {"query_id": f"q{i + 1:03d}", "modality": "text", "split": "dev", "language": "en",
         "text": t, "image_path": None, "gallery_id": "gallery_dev"}
        for i, t in enumerate(TEXT_QUERIES)
    ]
    queries.append({"query_id": "q004", "modality": "image", "split": "dev", "language": None,
                    "text": None, "image_path": "queries/red_patch.png", "gallery_id": "gallery_dev"})
    write_jsonl(HERE / "queries.jsonl", queries)
    print(f"생성 완료: items {len(items)}행, tracks {len(tracks)}행, videos {len(manifest)}개")


if __name__ == "__main__":
    main()
