"""API 검사: 정상·빈 결과·입력 오류·이미지 조건·404·503·미디어 Range·재현성."""
from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.app import create_app
from retrieval.config import load_config

from .conftest import ROOT, T007, V2T1, V2T2

BASE = {"modality": "text", "text": "a white car", "top_k": 10,
        "gallery_id": "gallery_dev", "representation": "combined"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import dataclasses
    cfg = load_config(ROOT / "configs/retrieval.yaml")
    log = tmp_path_factory.mktemp("logs") / "api.jsonl"
    cfg = dataclasses.replace(cfg, api=dataclasses.replace(cfg.api, log_path=str(log)))
    c = TestClient(create_app(cfg))
    c.log_path = log
    return c


def _png(size=(64, 64), color=(255, 0, 0), fmt="PNG", **kw) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt, **kw)
    return buf.getvalue()


def _post_image(client, data: bytes, name="q.png", ctype="image/png", **fields):
    form = {"gallery_id": "gallery_dev", **{k: str(v) for k, v in fields.items()}}
    return client.post("/search/image", files={"image": (name, data, ctype)}, data=form)


def _assert_error(r, status, code):
    assert r.status_code == status, r.text
    body = r.json()
    assert body["error"]["code"] == code and body["error"]["message"]
    assert body["request_id"] == r.headers["X-Request-ID"]


# ---- 정상 응답 ----------------------------------------------------------------

def test_health_ready(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_ready"] and body["index_ready"]
    assert body["galleries"][0]["items"] == 10


def test_text_search_response_fields(client):
    r = client.post("/search", json=BASE)
    assert r.status_code == 200
    body = r.json()
    assert body["schema_version"] == "cctv-3.0" and body["score_type"] == "rrf"
    assert body["index_version"] == "idx_d_small_v1" and body["total_candidates"] == 5
    top = body["results"][0]
    for key in ("rank", "score", "result_type", "item_id", "video_id", "track_uid", "start_ms", "end_ms",
                "representative_ms", "bbox", "width", "height", "thumbnail_url", "video_url", "source_offset_ms"):
        assert key in top
    assert (top["rank"], top["track_uid"], top["representative_ms"]) == (1, T007, 1200)
    assert client.get(top["thumbnail_url"]).headers["content-type"] == "image/jpeg"
    assert client.get(top["crop_url"]).status_code == 200


def test_top_k_limits_results_but_not_total(client):
    body = client.post("/search", json={**BASE, "top_k": 2}).json()
    assert len(body["results"]) == 2 and body["total_candidates"] == 5


def test_defaults_and_whitespace(client):
    body = client.post("/search", json={"text": "  a   white car \n"}).json()
    assert body["representation"] == "combined" and body["results"][0]["track_uid"] == T007


def test_empty_video_filter_returns_200_empty(client):
    r = client.post("/search", json={**BASE, "video_ids": []})
    assert r.status_code == 200
    assert r.json()["results"] == [] and r.json()["total_candidates"] == 0


def test_video_filter(client):
    body = client.post("/search", json={**BASE, "video_ids": ["v002"]}).json()
    assert [x["track_uid"] for x in body["results"]] == [V2T1, V2T2]


def test_same_request_twice_is_identical(client):
    a = client.post("/search", json=BASE)
    b = client.post("/search", json=BASE)
    strip = lambda d: {k: v for k, v in d.items() if k not in ("request_id", "latency_ms")}  # noqa: E731
    assert a.status_code == b.status_code == 200
    assert strip(a.json()) == strip(b.json())
    assert a.json()["request_id"] != b.json()["request_id"]


# ---- 입력 오류 422 -------------------------------------------------------------

@pytest.mark.parametrize("patch", [
    {"text": ""}, {"text": "   "}, {"text": 123}, {"text": None},
    {"top_k": 0}, {"top_k": 51}, {"top_k": "10"}, {"top_k": 1.5}, {"top_k": True},
    {"representation": "both"}, {"video_ids": "v001"}, {"video_ids": [1]}, {"modality": "image"},
])
def test_invalid_input_422(client, patch):
    _assert_error(client.post("/search", json={**BASE, **patch}), 422, "INVALID_INPUT")


def test_invalid_json_and_content_type(client):
    _assert_error(client.post("/search", content=b"{not json", headers={"content-type": "application/json"}),
                  422, "INVALID_INPUT")
    _assert_error(client.post("/search", content=b"text=a", headers={"content-type": "text/plain"}),
                  422, "INVALID_INPUT")
    _assert_error(client.post("/search", json=[1, 2]), 422, "INVALID_INPUT")


def test_text_length_boundary(client):
    assert client.post("/search", json={**BASE, "text": "a" * 500}).status_code == 200
    _assert_error(client.post("/search", json={**BASE, "text": "a" * 501}), 422, "TEXT_TOO_LONG")


def test_token_length_checked(client):
    # 글자 수는 500 이하지만 예제 인코더 기준 토큰 수(단어+2)가 77 을 넘는 입력
    assert client.post("/search", json={**BASE, "text": " ".join(["a"] * 75)}).status_code == 200
    _assert_error(client.post("/search", json={**BASE, "text": " ".join(["a"] * 76)}), 422, "TEXT_TOO_LONG")


# ---- 404 ----------------------------------------------------------------------

def test_unknown_gallery_and_video(client):
    _assert_error(client.post("/search", json={**BASE, "gallery_id": "nope"}), 404, "UNKNOWN_GALLERY")
    _assert_error(client.post("/search", json={**BASE, "video_ids": ["v001", "v999"]}), 404, "UNKNOWN_VIDEO")
    _assert_error(client.get("/videos/v999"), 404, "UNKNOWN_VIDEO")
    _assert_error(client.get("/videos/v001?gallery_id=nope"), 404, "UNKNOWN_GALLERY")
    _assert_error(client.get("/media/v_doesnotexist"), 404, "UNKNOWN_MEDIA")
    assert client.get("/no/such/path").json()["error"]["code"] == "NOT_FOUND"


# ---- 이미지 검색 ----------------------------------------------------------------

def test_image_search_png_and_jpeg(client):
    r = _post_image(client, _png(), top_k=3, representation="object")
    assert r.status_code == 200
    body = r.json()
    # 빨간 이미지 -> 예제 인코더에서 [1,0,0,0] -> "a white car" 의 object 순위와 같음
    assert [x["track_uid"] for x in body["results"]] == [V2T1, T007, "det01:v001:t008"]
    assert body["score_type"] == "cosine"
    assert _post_image(client, _png(fmt="JPEG"), name="q.jpg", ctype="image/jpeg").status_code == 200


def test_image_video_ids_json_string(client):
    r = _post_image(client, _png(), video_ids='["v002"]')
    assert {x["video_id"] for x in r.json()["results"]} == {"v002"}
    assert _post_image(client, _png(), video_ids="[]").json()["results"] == []
    _assert_error(_post_image(client, _png(), video_ids="v002"), 422, "INVALID_INPUT")
    _assert_error(_post_image(client, _png(), video_ids='["v999"]'), 404, "UNKNOWN_VIDEO")


def test_image_form_errors(client):
    _assert_error(client.post("/search/image", data={"gallery_id": "gallery_dev"}, files={"other": ("a", b"x")}),
                  422, "INVALID_INPUT")
    _assert_error(_post_image(client, _png(), top_k="abc"), 422, "INVALID_INPUT")
    _assert_error(_post_image(client, _png(), top_k=51), 422, "INVALID_INPUT")
    _assert_error(client.post("/search/image", json={"a": 1}), 422, "INVALID_INPUT")


def test_image_too_many_bytes_413(client):
    import os
    big = _png() + os.urandom(5_000_001)
    _assert_error(_post_image(client, big), 413, "IMAGE_TOO_LARGE")


def test_image_too_many_pixels_413(client):
    assert len(_png((4000, 4000))) < 5_000_000
    assert _post_image(client, _png((4000, 4000))).status_code == 200       # 정확히 1600만 픽셀
    _assert_error(_post_image(client, _png((4001, 4000))), 413, "IMAGE_TOO_LARGE")


def test_image_min_side_415(client):
    assert _post_image(client, _png((32, 32))).status_code == 200
    _assert_error(_post_image(client, _png((31, 64))), 415, "INVALID_IMAGE")


def test_image_bad_format_415(client):
    _assert_error(_post_image(client, b"this is not an image"), 415, "INVALID_IMAGE")
    _assert_error(_post_image(client, _png(fmt="GIF"), name="q.gif", ctype="image/gif"), 415, "INVALID_IMAGE")
    _assert_error(_post_image(client, _png()[:60]), 415, "INVALID_IMAGE")   # 잘린(손상) 파일
    _assert_error(_post_image(client, b""), 415, "INVALID_IMAGE")


def test_exif_rotation_applied():
    """EXIF 회전 6(시계 방향 90도) 이미지는 회전 후 크기로 인코더에 전달됩니다."""
    from api.image_input import decode_query_image
    from retrieval.config import ApiConfig
    im = Image.new("RGB", (80, 40), (0, 255, 0))
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif)
    rgb = decode_query_image(buf.getvalue(), ApiConfig())
    assert rgb.shape == (80, 40, 3)   # (높이, 너비) = 회전 후 40x80 이미지


# ---- 영상 정보와 미디어 ----------------------------------------------------------

def test_video_info(client):
    body = client.get("/videos/v001").json()
    assert (body["width"], body["height"], body["duration_ms"], body["source_offset_ms"]) == (1920, 1080, 3200, 0)
    assert body["video_url"].startswith("/media/v_")
    search_url = client.post("/search", json=BASE).json()["results"][0]["video_url"]
    assert body["video_url"] == search_url


def test_media_full_and_range(client):
    url = client.get("/videos/v001").json()["video_url"]
    data = (ROOT / "fixtures/d_small/data/videos/v001.mp4").read_bytes()
    size = len(data)

    full = client.get(url)
    assert full.status_code == 200 and full.content == data
    assert full.headers["accept-ranges"] == "bytes" and full.headers["content-type"] == "video/mp4"

    part = client.get(url, headers={"Range": "bytes=100-199"})
    assert part.status_code == 206 and part.content == data[100:200]
    assert part.headers["content-range"] == f"bytes 100-199/{size}" and part.headers["content-length"] == "100"

    tail = client.get(url, headers={"Range": "bytes=-50"})
    assert tail.status_code == 206 and tail.content == data[-50:]

    open_end = client.get(url, headers={"Range": f"bytes={size - 10}-"})
    assert open_end.status_code == 206 and open_end.content == data[-10:]

    over = client.get(url, headers={"Range": f"bytes=0-{size + 999}"})
    assert over.status_code == 206 and len(over.content) == size

    bad = client.get(url, headers={"Range": f"bytes={size}-"})
    assert bad.status_code == 416 and bad.headers["content-range"] == f"bytes */{size}"

    head = client.head(url)
    assert head.status_code == 200 and head.headers["content-length"] == str(size) and head.content == b""


# ---- 미준비 상태 503 -------------------------------------------------------------

def test_index_not_ready_503(fixture_copy):
    (fixture_copy / "fixtures/d_small/run/vectors.npy").write_bytes(b"broken")
    c = TestClient(create_app(load_config(fixture_copy / "configs/retrieval.yaml")))
    health = c.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "not_ready" and health.json()["index_ready"] is False
    assert health.json()["galleries"][0]["problems"]
    _assert_error(c.post("/search", json=BASE), 503, "INDEX_NOT_READY")
    _assert_error(_post_image(c, _png()), 503, "INDEX_NOT_READY")
    _assert_error(c.get("/videos/v001"), 503, "INDEX_NOT_READY")
    # 미준비 상태에서도 입력 오류는 422, 없는 gallery 는 404 로 같은 코드가 나옵니다
    _assert_error(c.post("/search", json={**BASE, "top_k": 0}), 422, "INVALID_INPUT")
    _assert_error(c.post("/search", json={**BASE, "gallery_id": "nope"}), 404, "UNKNOWN_GALLERY")


def test_internal_error_500(fixture_copy, monkeypatch):
    c = TestClient(create_app(load_config(fixture_copy / "configs/retrieval.yaml")), raise_server_exceptions=False)
    state = c.app.state.service

    def boom(texts):
        raise RuntimeError("encoder exploded")

    monkeypatch.setattr(state.encoder, "encode_text", boom)
    r = c.post("/search", json=BASE)
    _assert_error(r, 500, "INTERNAL_ERROR")
    assert "exploded" not in r.text          # 내부 사정은 응답에 싣지 않고
    log = (fixture_copy / "logs/api.jsonl").read_text(encoding="utf-8").splitlines()
    rec = json.loads(log[-1])                # request_id 로 로그에서 찾습니다
    assert rec["request_id"] == r.json()["request_id"] and "exploded" in rec["exception"]


def test_request_log_links_request_id(client):
    r = client.post("/search", json=BASE)
    records = [json.loads(line) for line in client.log_path.read_text(encoding="utf-8").splitlines()]
    rec = next(x for x in records if x["request_id"] == r.json()["request_id"])
    assert rec["status"] == 200 and rec["path"] == "/search" and rec["total_candidates"] == 5
