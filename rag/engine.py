import heapq
import time
from pathlib import Path

import numpy as np

from embedding.contracts import check_vectors, file_sha256, read_jsonl
from embedding.pipeline import validate_index

from .errors import RagError
from .prompts import SAFE_SUMMARIES


class FrameRecommender:
    def __init__(self, index_dir, encoder, llm=None):
        self.root = Path(index_dir).resolve()
        self.encoder = encoder
        self.llm = llm
        self.index = validate_index(self.root, encoder.metadata)
        self.items = read_jsonl(self.root / "items.jsonl")
        self.vectors = np.load(self.root / "vectors.npy", mmap_mode="r", allow_pickle=False)
        self.index_hash = file_sha256(self.root / "index_manifest.json")
        self.video_ids = set(self.index["sources"]) | {r["video_id"] for r in self.items}
        for row in self.items:
            stamp = row.get("actual_timestamp_ms", row["timestamp_ms"])
            offset = row.get("source_offset_ms", 0)
            if (type(row["frame_index"]) is not int or row["frame_index"] < 0 or
                    type(stamp) is not int or type(offset) is not int or not 0 <= offset <= stamp):
                raise ValueError("Index frame/time metadata is invalid")

    def health(self):
        return {"status": "ready", "model_ready": True, "index_ready": True,
                "llm_ready": bool(self.llm is not None and getattr(self.llm, "ready", True)),
                "row_count": len(self.items),
                "embedding_model": self.encoder.metadata["model_id"],
                "embedding_dim": self.encoder.dim, "index_sha256": self.index_hash}

    def recommend(self, question, top_k=5, video_ids=None, representation="combined", use_llm=True):
        started = time.perf_counter()
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1000:
            raise RagError("INVALID_QUESTION", "question must contain 1 to 1000 characters")
        question = question.strip()
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise RagError("INVALID_TOP_K", "top_k must be an integer from 1 to 20")
        if representation not in ("scene", "object", "combined"):
            raise RagError("INVALID_REPRESENTATION", "Use scene, object, or combined")
        if type(use_llm) is not bool:
            raise RagError("INVALID_INPUT", "use_llm must be boolean")
        if video_ids is not None:
            if (not isinstance(video_ids, list) or
                    any(not isinstance(v, str) or not v for v in video_ids)):
                raise RagError("INVALID_VIDEO_IDS", "video_ids must be a string list or null")
            if set(video_ids) - self.video_ids:
                raise RagError("UNKNOWN_VIDEO", "A requested video is not registered in this index")
        allowed = None if video_ids is None else set(video_ids)
        candidates = [r for r in self.items if (allowed is None or r["video_id"] in allowed)
                      and (representation == "combined" or r["representation"] == representation)]
        recommendations = []
        if candidates:
            if hasattr(self.encoder, "text_token_lengths"):
                if self.encoder.text_token_lengths([question])[0] > self.encoder.text_max_length:
                    raise RagError("TEXT_TOO_LONG", f"Embedding input exceeds {self.encoder.text_max_length} tokens")
            try:
                query = self.encoder.encode_text([question])
            except (ValueError, RuntimeError) as exc:
                if "too long" in str(exc).lower() or "TEXT_TOO_LONG" in str(exc):
                    raise RagError("TEXT_TOO_LONG", "Question exceeds the embedding encoder token limit") from exc
                raise
            check_vectors(query, 1, self.encoder.dim)
            scores = self.vectors @ query[0]
            best = {}
            for row in candidates:
                stamp = row.get("actual_timestamp_ms", row["timestamp_ms"])
                score = float(np.clip(scores[row["vector_row"]], -1, 1))
                order = (-score, row["video_id"], row["frame_index"], row["item_id"])
                key = (row["video_id"], row["frame_index"], stamp)
                if key not in best or order < best[key][0]:
                    best[key] = (order, row, stamp, score)
            for _, row, stamp, score in heapq.nsmallest(top_k, best.values(), key=lambda v: v[0]):
                offset = row.get("source_offset_ms", 0)
                recommendations.append({"rank": len(recommendations) + 1, "item_id": row["item_id"],
                                        "video_id": row["video_id"], "frame_index": row["frame_index"],
                                        "timestamp_ms": stamp, "seconds": stamp / 1000,
                                        "source_offset_ms": offset, "playback_time_ms": stamp - offset,
                                        "score": score, "representation": row["representation"],
                                        "bbox": row["bbox"]})
        if not recommendations:
            summary, llm_status = "검색 가능한 프레임 후보가 없습니다.", "skipped_empty"
        else:
            summary, llm_status = SAFE_SUMMARIES[0], "disabled" if not use_llm else "fallback"
            if use_llm and self.llm is not None:
                try:
                    generated = self.llm.summarize(question, len(recommendations))
                    if generated in SAFE_SUMMARIES:
                        summary, llm_status = generated, "generated"
                except Exception:
                    # Frame recommendations remain usable when the optional narrator fails.
                    pass
        lines = [summary]
        for row in recommendations:
            lines.append(f"{row['rank']}. 영상 {row['video_id']}, 프레임 {row['frame_index']} "
                         f"(0부터 시작), {row['seconds']:.3f}초, 유사도 {row['score']:.4f}")
        return {"status": "success", "question": question, "summary": summary,
                "answer": "\n".join(lines), "recommendations": recommendations, "llm_status": llm_status,
                "retrieval": "frame_cosine_max_per_frame", "score_type": "cosine",
                "index_sha256": self.index_hash, "latency_ms": (time.perf_counter() - started) * 1000}
