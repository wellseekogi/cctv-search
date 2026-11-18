"""API 오류 코드 (안내서 D 3 상태와 오류 응답 표)."""
from __future__ import annotations


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def invalid_input(message: str) -> ApiError:
    return ApiError(422, "INVALID_INPUT", message)


def text_too_long(message: str) -> ApiError:
    return ApiError(422, "TEXT_TOO_LONG", message)


def image_too_large(message: str) -> ApiError:
    return ApiError(413, "IMAGE_TOO_LARGE", message)


def invalid_image(message: str) -> ApiError:
    return ApiError(415, "INVALID_IMAGE", message)


def unknown_video(video_id: str) -> ApiError:
    return ApiError(404, "UNKNOWN_VIDEO", f"등록되지 않은 video_id 입니다: {video_id}")


def unknown_gallery(gallery_id: str) -> ApiError:
    return ApiError(404, "UNKNOWN_GALLERY", f"등록되지 않은 gallery_id 입니다: {gallery_id}")


def unknown_media(media_id: str) -> ApiError:
    # 안내서 표에는 없는 코드. 미디어 404 를 영상·gallery 404 와 구분하려고 추가했습니다.
    return ApiError(404, "UNKNOWN_MEDIA", f"등록되지 않은 media_id 입니다: {media_id}")


def index_not_ready(message: str) -> ApiError:
    return ApiError(503, "INDEX_NOT_READY", message)
