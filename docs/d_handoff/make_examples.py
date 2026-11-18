"""E·F·G 에게 전달할 응답 예제를 다시 만드는 스크립트.

  python docs/d_handoff/make_examples.py

서버를 띄우지 않고 같은 앱 코드를 직접 호출해 docs/d_handoff/examples/ 에 저장합니다.
request_id·latency_ms 는 실행할 때마다 달라집니다.
"""
from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from api.app import create_app  # noqa: E402
from retrieval.config import load_config  # noqa: E402

OUT = ROOT / "docs/d_handoff/examples"
BASE = {"modality": "text", "text": "a white car", "top_k": 10,
        "gallery_id": "gallery_dev", "representation": "combined"}


def png(size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def save(name: str, request: dict, response) -> None:
    body = response.json()
    (OUT / f"{name}.json").write_text(json.dumps(
        {"request": request, "status": response.status_code, "response": body},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    code = body.get("error", {}).get("code", "-")
    print(f"{response.status_code} {code:16s} {name}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c = TestClient(create_app(load_config(ROOT / "configs/retrieval.yaml")))

    def text(name, **patch):
        body = {**BASE, **patch}
        save(name, {"method": "POST", "path": "/search", "json": body}, c.post("/search", json=body))

    def image(name, data, **fields):
        form = {"gallery_id": "gallery_dev", **fields}
        save(name, {"method": "POST", "path": "/search/image", "form": form, "image_bytes": len(data)},
             c.post("/search/image", files={"image": ("q.png", data, "image/png")}, data=form))

    text("200_text_combined")
    text("200_text_object", representation="object")
    text("200_text_scene", representation="scene")
    text("200_empty_filter", video_ids=[])
    image("200_image_combined", png(), top_k="5", representation="combined")
    text("422_invalid_input_top_k", top_k=0)
    text("422_text_too_long", text="a" * 501)
    text("404_unknown_gallery", gallery_id="nope")
    text("404_unknown_video", video_ids=["v999"])
    image("413_image_too_large", png((4001, 4000)))
    image("415_invalid_image", b"not an image")
    for name, path in (("200_health", "/health"), ("200_video_v001", "/videos/v001"),
                       ("404_videos_unknown", "/videos/v999")):
        save(name, {"method": "GET", "path": path}, c.get(path))

    # 미준비 상태(503): 사본의 벡터 파일을 망가뜨려 재현
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        shutil.copytree(ROOT / "fixtures", repo / "fixtures")
        (repo / "configs").mkdir()
        shutil.copy(ROOT / "configs/retrieval.yaml", repo / "configs/retrieval.yaml")
        (repo / "fixtures/d_small/run/vectors.npy").write_bytes(b"broken")
        broken = TestClient(create_app(load_config(repo / "configs/retrieval.yaml")))
        save("503_index_not_ready", {"method": "POST", "path": "/search", "json": BASE},
             broken.post("/search", json=BASE))
        save("200_health_not_ready", {"method": "GET", "path": "/health"}, broken.get("/health"))


if __name__ == "__main__":
    main()
