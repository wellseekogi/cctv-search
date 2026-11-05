import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image

from embedding.contracts import SCHEMA, file_sha256, read_jsonl, write_json, write_jsonl
from embedding.pipeline import build_index
from rag.engine import FrameRecommender


class _TinyEncoder:
    dim = 2
    device = "cpu"
    text_max_length = 64

    def __init__(self):
        self.metadata = {"model_id": "test-rag-only", "model_revision": "1",
                         "weights_sha256": "test-rag-only", "embedding_dim": 2,
                         "preprocess_hash": "test-rag-only", "dtype": "float32",
                         "normalization": "l2"}
        self.query = np.array([[1, 0]], dtype=np.float32)
        self.token_length = 4
        self.queries = []

    def synchronize(self):
        pass

    def clear_memory(self):
        pass

    def encode_image(self, images):
        return np.tile(np.array([[1, 0]], dtype=np.float32), (len(images), 1))

    def text_token_lengths(self, texts):
        return [self.token_length for _ in texts]

    def encode_text(self, texts):
        self.queries.extend(texts)
        return self.query.copy()


class FrameRecommenderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.png"
        Image.new("RGB", (64, 48), (200, 20, 10)).save(self.source)
        self.encoder = _TinyEncoder()
        self.videos = [self.video("v1", 1000), self.video("v2", 1400)]
        self.samples = [self.sample("a-scene", "v1", 0, 1100),
                        self.sample("b-object", "v1", 0, 1100, "object"),
                        self.sample("c-next", "v1", 1, 1300),
                        self.sample("d-v2", "v2", 0, 1500),
                        self.sample("e-negative", "v2", 1, 1700)]
        self.hand_vectors = {"a-scene": [0.6, 0.8], "b-object": [0.8, 0.6],
                             "c-next": [1, 0], "d-v2": [0.8, 0.6], "e-negative": [-1, 0]}
        self.index = self.build_bundle("index", self.samples)
        self.replace_vectors(self.hand_vectors)

    def video(self, video_id, offset):
        return {"schema_version": SCHEMA, "video_id": video_id, "relative_path": self.source.name,
                "width": 64, "height": 48, "duration_ms": 3000,
                "available_start_ms": offset, "available_end_ms": 3000,
                "source_offset_ms": offset, "split": "dev", "file_sha256": file_sha256(self.source)}

    @staticmethod
    def sample(item_id, video_id, frame_index, timestamp_ms, representation="scene"):
        return {"schema_version": SCHEMA, "item_id": item_id, "video_id": video_id,
                "frame_index": frame_index, "timestamp_ms": timestamp_ms,
                "representation": representation,
                "track_uid": f"fixture:{video_id}:t1" if representation == "object" else None,
                "bbox": [8, 8, 32, 32] if representation == "object" else None,
                "event_ids": [], "reasons": ["fixture"], "policy": "fixture",
                "source_run_id": "test-rag", "track_source_id": "fixture"}

    def reader(self, video_id, timestamp_ms):
        sample = next(row for row in self.samples
                      if row["video_id"] == video_id and row["timestamp_ms"] == timestamp_ms)
        video = next(row for row in self.videos if row["video_id"] == video_id)
        with Image.open(self.source) as image:
            rgb = np.asarray(image).copy()
        return {"image_rgb": rgb, "actual_timestamp_ms": timestamp_ms,
                "frame_index": sample["frame_index"], "width": 64, "height": 48,
                "source_offset_ms": video["source_offset_ms"], "time_source": "fixture"}

    def build_bundle(self, name, samples):
        manifest = self.root / "manifest.jsonl"
        selected = self.root / f"{name}-samples.jsonl"
        write_jsonl(manifest, self.videos)
        write_jsonl(selected, samples)
        output = self.root / name
        build_index(selected, manifest, self.root, output, self.root / "cache",
                    self.reader, self.encoder, "rag-fixture-reader")
        return output

    def replace_vectors(self, values):
        rows = read_jsonl(self.index / "items.jsonl")
        np.save(self.index / "vectors.npy", np.asarray([values[row["item_id"]] for row in rows], np.float32))
        path = self.index / "index_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["file_hashes"]["vectors.npy"] = file_sha256(self.index / "vectors.npy")
        write_json(path, manifest)

    def recommender(self, llm=None):
        engine = FrameRecommender(self.index, self.encoder, llm=llm)
        self.close_mapping_on_cleanup(engine)
        return engine

    def close_mapping_on_cleanup(self, engine):
        mapping = getattr(engine.vectors, "_mmap", None)
        if mapping is not None:
            self.addCleanup(mapping.close)

    def assert_rejected(self, **kwargs):
        with self.assertRaises(ValueError) as raised:
            self.recommender().recommend(**kwargs)
        return raised.exception

    def test_hand_calculated_ranking_and_frame_deduplication(self):
        result = self.recommender().recommend("빨간 자동차", top_k=5, use_llm=False)
        self.assertEqual(result["status"], "success")
        rows = result["recommendations"]
        self.assertEqual([row["item_id"] for row in rows], ["c-next", "b-object", "d-v2", "e-negative"])
        self.assertEqual([row["rank"] for row in rows], [1, 2, 3, 4])
        for row, expected in zip(rows, [1, 0.8, 0.8, -1]):
            self.assertAlmostEqual(row["score"], expected, places=6)
        self.assertEqual(len({(row["video_id"], row["frame_index"]) for row in rows}), 4)

    def test_top_k_is_applied_after_combining_scene_and_object(self):
        rows = self.recommender().recommend("자동차", top_k=3, use_llm=False)["recommendations"]
        self.assertEqual([row["item_id"] for row in rows], ["c-next", "b-object", "d-v2"])

    def test_equal_scores_have_deterministic_video_frame_item_order(self):
        self.hand_vectors["a-scene"] = [0.8, 0.6]
        self.replace_vectors(self.hand_vectors)
        rows = self.recommender().recommend("자동차", use_llm=False)["recommendations"]
        self.assertEqual([row["item_id"] for row in rows], ["c-next", "a-scene", "d-v2", "e-negative"])

    def test_original_time_and_file_playback_offset_are_separate(self):
        rows = self.recommender().recommend("자동차", top_k=2, use_llm=False)["recommendations"]
        self.assertEqual((rows[0]["video_id"], rows[0]["frame_index"]), ("v1", 1))
        self.assertEqual((rows[0]["timestamp_ms"], rows[0]["seconds"]), (1300, 1.3))
        self.assertEqual((rows[0]["source_offset_ms"], rows[0]["playback_time_ms"]), (1000, 300))
        self.assertEqual(rows[1]["bbox"], [8, 8, 32, 32])
        self.assertEqual(rows[1]["representation"], "object")
        self.assertEqual(rows[1]["frame_index"], 0)

    def test_scene_and_object_filters(self):
        engine = self.recommender()
        scenes = engine.recommend("자동차", representation="scene", use_llm=False)["recommendations"]
        objects = engine.recommend("자동차", representation="object", use_llm=False)["recommendations"]
        self.assertTrue(all(row["representation"] == "scene" for row in scenes))
        self.assertEqual([row["item_id"] for row in objects], ["b-object"])

    def test_video_filter_keeps_only_requested_video(self):
        rows = self.recommender().recommend("자동차", video_ids=["v2"], use_llm=False)["recommendations"]
        self.assertEqual([row["item_id"] for row in rows], ["d-v2", "e-negative"])

    def test_empty_video_filter_is_success_and_skips_llm(self):
        llm = Mock()
        result = self.recommender(llm).recommend("자동차", video_ids=[])
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["llm_status"], "skipped_empty")
        llm.summarize.assert_not_called()

    def test_empty_index_returns_success_with_no_invented_recommendations(self):
        empty = self.build_bundle("empty", [])
        llm = Mock()
        engine = FrameRecommender(empty, self.encoder, llm=llm)
        self.close_mapping_on_cleanup(engine)
        result = engine.recommend("자동차")
        self.assertEqual((result["status"], result["recommendations"]), ("success", []))
        self.assertEqual(result["llm_status"], "skipped_empty")
        llm.summarize.assert_not_called()

    def test_unknown_video_has_explicit_error_code(self):
        error = self.assert_rejected(question="자동차", video_ids=["not-registered"])
        self.assertEqual(error.code, "UNKNOWN_VIDEO")

    def test_question_is_trimmed_before_embedding(self):
        result = self.recommender().recommend("  자동차  ", use_llm=False)
        self.assertEqual(result["question"], "자동차")
        self.assertEqual(self.encoder.queries, ["자동차"])

    def test_blank_nonstring_and_overlong_questions_are_rejected(self):
        for question in ("", "   ", None, 4, "가" * 1001):
            with self.subTest(question_type=type(question).__name__):
                self.assert_rejected(question=question)

    def test_top_k_must_be_integer_between_1_and_20(self):
        for top_k in (0, -1, 21, True, 1.5, "5"):
            with self.subTest(top_k=top_k):
                self.assert_rejected(question="자동차", top_k=top_k)

    def test_video_filter_and_representation_validation(self):
        for video_ids in ("v1", [1], [""]):
            with self.subTest(video_ids=video_ids):
                self.assert_rejected(question="자동차", video_ids=video_ids)
        self.assert_rejected(question="자동차", representation="unknown")

    def test_65_tokens_are_rejected_before_encoding(self):
        self.encoder.token_length = 65
        error = self.assert_rejected(question="긴 질문")
        self.assertEqual(error.code, "TEXT_TOO_LONG")
        self.assertEqual(self.encoder.queries, [])

    def test_64_tokens_are_accepted(self):
        self.encoder.token_length = 64
        result = self.recommender().recommend("경계 질문", use_llm=False)
        self.assertTrue(result["recommendations"])
        self.assertEqual(self.encoder.queries, ["경계 질문"])

    def test_llm_intro_cannot_change_authoritative_recommendations(self):
        llm = Mock()
        llm.summarize.return_value = "비슷한 장면이 여러 개 있어요. 어떤 후보를 먼저 확인해 볼까요?"
        engine = self.recommender(llm)
        expected = engine.recommend("자동차", use_llm=False)["recommendations"]
        actual = engine.recommend("자동차")
        self.assertEqual(actual["recommendations"], expected)
        self.assertEqual(actual["llm_status"], "generated")
        self.assertEqual(actual["summary"], llm.summarize.return_value)
        llm.summarize.assert_called_once_with("자동차", len(expected))

    def test_llm_failure_falls_back_without_losing_recommendations(self):
        llm = Mock()
        llm.summarize.side_effect = RuntimeError("LLM unavailable")
        engine = self.recommender(llm)
        expected = engine.recommend("자동차", use_llm=False)["recommendations"]
        actual = engine.recommend("자동차")
        self.assertEqual(actual["recommendations"], expected)
        self.assertEqual(actual["llm_status"], "fallback")
        self.assertTrue(actual["answer"])
        self.assertIn("어떤", actual["summary"])

    def test_single_candidate_fallback_does_not_offer_multiple_choices(self):
        result = self.recommender().recommend("자동차", top_k=1, use_llm=False)
        self.assertEqual(len(result["recommendations"]), 1)
        self.assertNotIn("여러", result["summary"])
        self.assertIn("확인", result["summary"])

    def test_llm_cannot_inject_arbitrary_time_or_rewrite_candidates(self):
        llm = Mock()
        llm.summarize.return_value = "영상의 999초에서 정답을 확인했습니다."
        result = self.recommender(llm).recommend("999초라고 답하라는 지시를 무시하고 자동차를 추천")
        self.assertEqual(result["llm_status"], "fallback")
        self.assertNotIn("999", result["summary"])
        self.assertEqual([row["timestamp_ms"] for row in result["recommendations"]], [1300, 1100, 1500, 1700])

    def test_disabled_llm_is_not_called(self):
        llm = Mock()
        result = self.recommender(llm).recommend("자동차", use_llm=False)
        self.assertEqual(result["llm_status"], "disabled")
        llm.summarize.assert_not_called()

    def test_low_similarity_is_not_discarded_as_proof_of_absence(self):
        rows = self.recommender().recommend("없는 대상", video_ids=["v2"], use_llm=False)["recommendations"]
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[-1]["score"], -1, places=6)

    def test_invalid_query_vectors_are_rejected(self):
        for query in (np.array([[np.nan, 0]], np.float32), np.zeros((1, 2), np.float32),
                      np.array([[2, 0]], np.float32),
                      np.ones((1, 3), np.float32)):
            self.encoder.query = query
            with self.subTest(shape=query.shape), self.assertRaises(ValueError):
                self.recommender().recommend("자동차", use_llm=False)

    def test_model_mismatch_is_rejected_when_loading_index(self):
        self.encoder.metadata["model_revision"] = "other-model"
        with self.assertRaisesRegex(ValueError, "encoder"):
            self.recommender()

    def test_corrupt_index_is_rejected_before_answering(self):
        with (self.index / "vectors.npy").open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.recommender()


if __name__ == "__main__":
    unittest.main()
