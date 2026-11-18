"""질의 인코더 연결.

실제 인코더는 C가 제공합니다: encode_text(texts), encode_image(images_rgb).
D는 아래 인터페이스만 사용하므로, 설정의 encoder.factory 만 바꾸면 교체됩니다.

  model_id · model_revision · preprocess_hash · embedding_dim · max_tokens
  encode_text(list[str])            -> float32 [N, d] (L2 정규화)
  encode_image(list[np.ndarray RGB]) -> float32 [N, d] (L2 정규화)
  count_tokens(str) -> int | None   (없으면 토큰 길이 검사를 건너뜀)
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


class FixtureEncoder:
    """작은 예제·검사 전용 인코더. 실제 CLIP 이 아닙니다.

    - 텍스트: table(JSON) 에 등록된 문장은 그 벡터를, 그 외에는 문장 해시로 만든 고정 벡터를 반환
    - 이미지: 평균 [R, G, B] 를 앞 세 차원에 넣고 정규화
    같은 입력에는 항상 같은 벡터가 나오므로 순위를 손계산과 대조할 수 있습니다.
    """

    model_revision = "fixture"
    max_tokens = 77

    def __init__(self, table: str | None = None, root: str | Path | None = None, **_: Any):
        self._table: dict[str, list[float]] = {}
        self.embedding_dim = 4
        self.model_id = "fixture-encoder"
        self.preprocess_hash = "fixture"
        if table:
            path = Path(table)
            if not path.is_absolute() and root:
                path = Path(root) / path
            data = json.loads(path.read_text(encoding="utf-8"))
            self.embedding_dim = int(data["embedding_dim"])
            self.model_id = data.get("model_id", self.model_id)
            self.preprocess_hash = data.get("preprocess_hash", self.preprocess_hash)
            self._table = {k: v for k, v in data["texts"].items()}

    def count_tokens(self, text: str) -> int:
        return len(text.split()) + 2  # 시작·끝 토큰 포함한 근사값

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for i, t in enumerate(texts):
            if t in self._table:
                out[i] = self._table[t]
            else:
                digest = hashlib.sha256(t.encode("utf-8")).digest()
                out[i] = [b / 255.0 + 0.01 for b in digest[: self.embedding_dim]]
        return _l2(out)

    def encode_image(self, images_rgb: Sequence[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(images_rgb), self.embedding_dim), dtype=np.float32)
        for i, img in enumerate(images_rgb):
            mean = np.asarray(img, dtype=np.float64).reshape(-1, 3).mean(axis=0) / 255.0
            out[i, :3] = mean
            if not mean.any():
                out[i, 0] = 1.0
        return _l2(out)


class FunctionEncoder:
    """C의 모듈 함수(encode_text·encode_image)를 감싸는 연결부.

    options:
      module: C 모듈 경로 (예: embedding.encoder)
    모델 정보는 모듈의 MODEL_ID · MODEL_REVISION · PREPROCESS_HASH · EMBEDDING_DIM · MAX_TOKENS
    상수와 count_tokens 함수를 있으면 사용합니다(이름은 C와 확정 필요).
    """

    def __init__(self, module: str, **_: Any):
        mod = importlib.import_module(module)
        self._encode_text = mod.encode_text
        self._encode_image = mod.encode_image
        self._count_tokens = getattr(mod, "count_tokens", None)
        self.model_id = getattr(mod, "MODEL_ID", None)
        self.model_revision = getattr(mod, "MODEL_REVISION", None)
        self.preprocess_hash = getattr(mod, "PREPROCESS_HASH", None)
        self.max_tokens = int(getattr(mod, "MAX_TOKENS", 77))
        dim = getattr(mod, "EMBEDDING_DIM", None)
        self.embedding_dim = int(dim) if dim else int(self.encode_text(["a"]).shape[1])

    def count_tokens(self, text: str) -> int | None:
        return int(self._count_tokens(text)) if self._count_tokens else None

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        return _l2(np.asarray(self._encode_text(list(texts))))

    def encode_image(self, images_rgb: Sequence[np.ndarray]) -> np.ndarray:
        return _l2(np.asarray(self._encode_image(list(images_rgb))))


def load_encoder(factory: str, options: dict[str, Any], root: str | Path | None = None) -> Any:
    """'패키지.모듈:클래스' 형식의 factory 로 인코더를 만듭니다."""
    module_name, _, attr = factory.partition(":")
    if not attr:
        raise ValueError("encoder.factory 는 '모듈:클래스' 형식이어야 합니다")
    cls = getattr(importlib.import_module(module_name), attr)
    try:
        return cls(root=root, **options)
    except TypeError:
        return cls(**options)
