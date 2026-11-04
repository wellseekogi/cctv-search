"""테스트용 RGB 이미지 프레임. B의 영상 디코더를 대신하지 않습니다."""

from pathlib import Path
import json

import numpy as np
from PIL import Image

from embedding.contracts import file_sha256, within

ROOT = Path(__file__).parent


def read_frame(video_id, requested_timestamp_ms):
    if video_id != "fixture01" or requested_timestamp_ms not in (0, 200):
        raise ValueError("Unknown fixture frame")
    source = json.loads((ROOT / "frames_source.json").read_text(encoding="utf-8"))
    row = next(r for r in source["frames"] if r["timestamp_ms"] == requested_timestamp_ms)
    path = within(ROOT, row["path"])
    if file_sha256(path) != row["sha256"]:
        raise ValueError("Fixture source hash mismatch")
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    return {"image_rgb": rgb, "actual_timestamp_ms": requested_timestamp_ms,
            "frame_index": requested_timestamp_ms // 200, "width": rgb.shape[1],
            "height": rgb.shape[0], "time_source": "fixture", "source_offset_ms": 0}
