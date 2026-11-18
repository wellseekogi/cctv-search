"""질의 이미지 검사와 변환 (안내서 D 2 이미지 검색).

JPEG·PNG, 5,000,000 bytes 이하, 1600만 픽셀 이하, 가로·세로 각 32픽셀 이상.
EXIF 회전을 적용한 뒤 RGB 배열로 바꿔 질의 인코더에 넘깁니다.
"""
from __future__ import annotations

import io
import warnings

import numpy as np
from PIL import Image, ImageOps

from retrieval.config import ApiConfig

from .errors import image_too_large, invalid_image

_ALLOWED_FORMATS = {"JPEG", "PNG"}


def decode_query_image(data: bytes, cfg: ApiConfig) -> np.ndarray:
    if len(data) > cfg.image_max_bytes:
        raise image_too_large(f"이미지는 {cfg.image_max_bytes:,} bytes 이하여야 합니다 (현재 {len(data):,})")
    if not data:
        raise invalid_image("빈 이미지 파일입니다")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            im = Image.open(io.BytesIO(data))
            fmt = im.format
            width, height = im.size
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise image_too_large(f"이미지는 {cfg.image_max_pixels:,} 픽셀 이하여야 합니다") from None
    except Exception:
        raise invalid_image("이미지를 읽을 수 없습니다. JPEG 또는 PNG 파일을 사용하세요") from None

    if fmt not in _ALLOWED_FORMATS:
        raise invalid_image(f"JPEG·PNG 만 지원합니다 (현재 {fmt})")
    if width * height > cfg.image_max_pixels:
        raise image_too_large(
            f"이미지는 {cfg.image_max_pixels:,} 픽셀 이하여야 합니다 (현재 {width}x{height})"
        )
    if min(width, height) < cfg.image_min_side_px:
        raise invalid_image(
            f"가로·세로가 각각 {cfg.image_min_side_px}픽셀 이상이어야 합니다 (현재 {width}x{height})"
        )
    try:
        im.load()  # 실제 디코딩. 손상 파일은 여기서 걸립니다.
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        raise invalid_image("손상된 이미지입니다") from None
    finally:
        im.close()
    return np.asarray(rgb)
