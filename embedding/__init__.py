"""C 담당: RGB 이미지, CLIP 벡터, 검색 인코더와 인덱스 묶음."""

from .encoder import ClipEncoder
from .pipeline import build_index, validate_index

__all__ = ["ClipEncoder", "build_index", "validate_index"]
