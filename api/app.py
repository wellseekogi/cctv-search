"""검색 API (안내서 D 2 · D 3).

  POST /search            텍스트 검색 (application/json)
  POST /search/image      이미지 검색 (multipart/form-data)
  GET  /health            모델·인덱스 준비 상태
  GET  /videos/{video_id} 영상 크기·길이·offset·재생 URL
  GET  /media/{media_id}  등록한 이미지·영상 (HTTP Range 지원)

모든 응답과 로그는 request_id 로 연결합니다.
"""
from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from retrieval.config import REPRESENTATIONS, SCHEMA_VERSION, AppConfig, load_config
from retrieval.search import SearchEngine
from retrieval.service import ServiceState, build_state

from . import errors
from .errors import ApiError
from .image_input import decode_query_image
from .media import file_response

Handler = Callable[[Request, str, dict[str, Any]], Awaitable[Response]]


def _new_request_id() -> str:
    return "req_" + uuid.uuid4().hex[:16]


class RequestLog:
    """request_id 로 연결되는 JSONL 요청 로그."""

    def __init__(self, path: Path | None):
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("cctv.api")

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        self._logger.info(line)


def _error_body(request_id: str, code: str, message: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "error": {"code": code, "message": message},
    }


# ---- 입력 검사 ---------------------------------------------------------------

def _parse_top_k(value: Any, cfg) -> int:
    if value is None:
        return cfg.top_k_default
    if isinstance(value, bool) or not isinstance(value, int):
        raise errors.invalid_input("top_k 는 정수여야 합니다")
    if not (1 <= value <= cfg.top_k_max):
        raise errors.invalid_input(f"top_k 는 1~{cfg.top_k_max} 범위여야 합니다")
    return value


def _parse_representation(value: Any) -> str:
    if value is None:
        return "combined"
    if value not in REPRESENTATIONS:
        raise errors.invalid_input("representation 은 scene·object·combined 중 하나여야 합니다")
    return value


def _parse_video_ids(value: Any) -> list[str] | None:
    """생략(None)=gallery 전체, []=후보 0개."""
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise errors.invalid_input("video_ids 는 문자열 배열이어야 합니다")
    return value


def _resolve_gallery(state: ServiceState, gallery_id: Any) -> str:
    if gallery_id is None:
        if len(state.galleries) == 1:
            return next(iter(state.galleries))
        raise errors.invalid_input("gallery_id 가 필요합니다")
    if not isinstance(gallery_id, str) or not gallery_id:
        raise errors.invalid_input("gallery_id 는 문자열이어야 합니다")
    if gallery_id not in state.galleries:
        raise errors.unknown_gallery(gallery_id)
    return gallery_id


def _ready_engine(state: ServiceState, gallery_id: str) -> SearchEngine:
    if not state.model_ready:
        raise errors.index_not_ready("질의 모델이 준비되지 않았습니다. GET /health 를 확인하세요")
    gs = state.galleries[gallery_id]
    if not gs.ready:
        raise errors.index_not_ready("인덱스가 준비되지 않았습니다. GET /health 를 확인하세요")
    return gs.engine  # type: ignore[return-value]


def _check_videos(engine: SearchEngine, video_ids: list[str] | None) -> None:
    for vid in video_ids or []:
        if vid not in engine.gallery.videos:
            raise errors.unknown_video(vid)


def _search_response(
    engine: SearchEngine, request_id: str, vector, representation: str,
    video_ids: list[str] | None, top_k: int, started: float,
) -> dict[str, Any]:
    out = engine.search(vector, representation, video_ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "index_version": engine.gallery.index_version,
        "gallery_id": engine.gallery.gallery_id,
        "representation": representation,
        "score_type": out["score_type"],
        "total_candidates": out["total_candidates"],
        "latency_ms": int(round((time.perf_counter() - started) * 1000)),
        "results": out["results"][:top_k],
        "scene_results": out["scene_results"][:top_k],
    }


# ---- 앱 ---------------------------------------------------------------------

def create_app(config: AppConfig | str | Path = "configs/retrieval.yaml",
               state: ServiceState | None = None) -> FastAPI:
    if not isinstance(config, AppConfig):
        config = load_config(config)
    state = state or build_state(config)
    log = RequestLog(config.root / config.api.log_path if config.api.log_path else None)

    app = FastAPI(title="CCTV 검색 API", version=SCHEMA_VERSION, docs_url="/docs", redoc_url=None)
    app.state.service = state

    async def run(request: Request, handler: Handler) -> Response:
        """request_id 발급 → 처리 → 오류를 공통 본문으로 변환 → 로그 기록."""
        request_id = _new_request_id()
        started = time.perf_counter()
        info: dict[str, Any] = {}
        code: str | None = None
        try:
            response = await handler(request, request_id, info)
        except ApiError as e:
            code = e.code
            response = JSONResponse(_error_body(request_id, e.code, e.message), status_code=e.status)
        except Exception as e:  # noqa: BLE001
            code = "INTERNAL_ERROR"
            info["exception"] = f"{type(e).__name__}: {e}"
            info["traceback"] = traceback.format_exc()
            response = JSONResponse(
                _error_body(request_id, code, "내부 오류가 발생했습니다. request_id 로 로그를 확인하세요"),
                status_code=500,
            )
        response.headers["X-Request-ID"] = request_id
        log.write({
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "error_code": code,
            "latency_ms": int(round((time.perf_counter() - started) * 1000)),
            **info,
        })
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """등록하지 않은 경로·메서드도 같은 오류 본문으로 응답합니다."""
        request_id = _new_request_id()
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
        return JSONResponse(
            _error_body(request_id, code, str(exc.detail)),
            status_code=exc.status_code, headers={"X-Request-ID": request_id},
        )

    # -- POST /search ---------------------------------------------------------
    async def _search_text(request: Request, request_id: str, info: dict[str, Any]) -> Response:
        started = time.perf_counter()
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise errors.invalid_input("Content-Type 은 application/json 이어야 합니다")
        try:
            body = json.loads(await request.body())
        except (ValueError, UnicodeDecodeError):
            raise errors.invalid_input("본문이 올바른 JSON 이 아닙니다") from None
        if not isinstance(body, dict):
            raise errors.invalid_input("본문은 JSON 객체여야 합니다")
        if body.get("modality", "text") != "text":
            raise errors.invalid_input("POST /search 의 modality 는 text 입니다. 이미지는 POST /search/image 를 사용하세요")

        text = body.get("text")
        if not isinstance(text, str):
            raise errors.invalid_input("text 는 문자열이어야 합니다")
        text = " ".join(text.split())  # 앞뒤 공백 제거, 연속 공백은 한 칸으로
        if not text:
            raise errors.invalid_input("text 는 공백을 제외하고 1자 이상이어야 합니다")
        if len(text) > config.api.text_max_chars:
            raise errors.text_too_long(f"text 는 {config.api.text_max_chars}자 이하여야 합니다 (현재 {len(text)}자)")
        top_k = _parse_top_k(body.get("top_k"), config.api)
        representation = _parse_representation(body.get("representation"))
        video_ids = _parse_video_ids(body.get("video_ids"))
        gallery_id = _resolve_gallery(state, body.get("gallery_id"))
        info.update(modality="text", gallery_id=gallery_id, representation=representation,
                    top_k=top_k, text_chars=len(text),
                    video_filter=None if video_ids is None else len(video_ids))

        engine = _ready_engine(state, gallery_id)
        _check_videos(engine, video_ids)
        tokens = state.encoder.count_tokens(text) if hasattr(state.encoder, "count_tokens") else None
        max_tokens = getattr(state.encoder, "max_tokens", None)
        if tokens is not None and max_tokens is not None and tokens > max_tokens:
            raise errors.text_too_long(f"text 가 모델 토큰 길이 {max_tokens} 을 넘습니다 (현재 {tokens})")

        vector = state.encoder.encode_text([text])[0]
        payload = _search_response(engine, request_id, vector, representation, video_ids, top_k, started)
        info.update(total_candidates=payload["total_candidates"], returned=len(payload["results"]))
        return JSONResponse(payload)

    @app.post("/search", openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["text"],
                        "properties": {
                            "modality": {"type": "string", "enum": ["text"], "default": "text"},
                            "text": {"type": "string", "minLength": 1, "maxLength": 500},
                            "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                            "gallery_id": {"type": "string", "default": "gallery_dev"},
                            "representation": {"type": "string", "enum": ["object", "scene", "combined"], "default": "combined"},
                            "video_ids": {"type": "array", "items": {"type": "string"}},
                        },
                        "example": {
                            "modality": "text", "text": "a white car", "top_k": 10,
                            "gallery_id": "gallery_dev", "representation": "combined",
                        },
                    },
                },
            },
        },
    })
    async def search_text(request: Request) -> Response:
        return await run(request, _search_text)

    # -- POST /search/image ---------------------------------------------------
    async def _search_image(request: Request, request_id: str, info: dict[str, Any]) -> Response:
        started = time.perf_counter()
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype != "multipart/form-data":
            raise errors.invalid_input("Content-Type 은 multipart/form-data 여야 합니다")
        # 본문 전체 크기를 먼저 확인해 과도한 업로드를 읽지 않습니다(폼 필드 여유 64KB).
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > config.api.image_max_bytes + 65536:
            raise errors.image_too_large(f"이미지는 {config.api.image_max_bytes:,} bytes 이하여야 합니다")
        try:
            form = await request.form()
        except Exception:
            raise errors.invalid_input("multipart 본문을 읽을 수 없습니다") from None
        try:
            uploads = [v for v in form.getlist("image") if hasattr(v, "read")]
            if len(uploads) != 1:
                raise errors.invalid_input("image 파일 한 개가 필요합니다")

            raw_top_k = form.get("top_k")
            if raw_top_k in (None, ""):
                top_k = config.api.top_k_default
            else:
                try:
                    top_k = _parse_top_k(int(str(raw_top_k)), config.api)
                except ValueError:
                    raise errors.invalid_input("top_k 는 정수여야 합니다") from None
            representation = _parse_representation(form.get("representation") or None)
            raw_ids = form.get("video_ids")
            if raw_ids in (None, ""):
                video_ids = None
            else:
                try:
                    video_ids = _parse_video_ids(json.loads(str(raw_ids)))
                except ValueError:
                    raise errors.invalid_input('video_ids 는 JSON 배열 문자열이어야 합니다 (예: ["v001"])') from None
            gallery_id = _resolve_gallery(state, form.get("gallery_id") or None)

            data = await uploads[0].read(config.api.image_max_bytes + 1)
            rgb = decode_query_image(data, config.api)
            info.update(modality="image", gallery_id=gallery_id, representation=representation,
                        top_k=top_k, image_bytes=len(data), image_size=[rgb.shape[1], rgb.shape[0]],
                        video_filter=None if video_ids is None else len(video_ids))

            engine = _ready_engine(state, gallery_id)
            _check_videos(engine, video_ids)
            vector = state.encoder.encode_image([rgb])[0]
            payload = _search_response(engine, request_id, vector, representation, video_ids, top_k, started)
            info.update(total_candidates=payload["total_candidates"], returned=len(payload["results"]))
            return JSONResponse(payload)
        finally:
            await form.close()  # 임시 업로드 파일 정리

    @app.post("/search/image")
    async def search_image(request: Request) -> Response:
        return await run(request, _search_image)

    # -- GET /health ----------------------------------------------------------
    async def _health(request: Request, request_id: str, info: dict[str, Any]) -> Response:
        return JSONResponse({"schema_version": SCHEMA_VERSION, "request_id": request_id, **state.health()})

    @app.get("/health")
    async def health(request: Request) -> Response:
        return await run(request, _health)

    # -- GET /videos/{video_id} -----------------------------------------------
    async def _video(request: Request, request_id: str, info: dict[str, Any]) -> Response:
        video_id = request.path_params["video_id"]
        gallery_id = request.query_params.get("gallery_id")
        if gallery_id is not None and gallery_id not in state.galleries:
            raise errors.unknown_gallery(gallery_id)
        targets = [gallery_id] if gallery_id else sorted(state.galleries)
        any_ready = False
        for gid in targets:
            gs = state.galleries[gid]
            if not gs.ready:
                continue
            any_ready = True
            if video_id in gs.engine.gallery.videos:  # type: ignore[union-attr]
                return JSONResponse({
                    "schema_version": SCHEMA_VERSION, "request_id": request_id,
                    **gs.engine.video_info(video_id),  # type: ignore[union-attr]
                })
        if not any_ready:
            raise errors.index_not_ready("인덱스가 준비되지 않았습니다. GET /health 를 확인하세요")
        raise errors.unknown_video(video_id)

    @app.get("/videos/{video_id}")
    async def video(request: Request) -> Response:
        return await run(request, _video)

    # -- GET /media/{media_id} ------------------------------------------------
    async def _media(request: Request, request_id: str, info: dict[str, Any]) -> Response:
        media_id = request.path_params["media_id"]
        for gs in state.galleries.values():
            m = gs.engine.media.get(media_id) if gs.engine else None
            if m is not None:
                if not m.path.is_file():
                    info["missing_file"] = m.path.as_posix()
                    break
                info["media_id"] = media_id
                return file_response(m.path, m.content_type, request.headers.get("range"),
                                     head=request.method == "HEAD")
        raise errors.unknown_media(media_id)

    @app.api_route("/media/{media_id}", methods=["GET", "HEAD"])
    async def media(request: Request) -> Response:
        return await run(request, _media)

    return app
