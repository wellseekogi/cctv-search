import io
import json
import unittest
from unittest.mock import Mock

from rag.prompts import SAFE_SUMMARIES
from rag.runtime import LocalLLM


class LocalLLMTests(unittest.TestCase):
    def setUp(self):
        self.model = LocalLLM("fixture-llama-server.exe", "fixture-model.gguf")
        self.model.process = Mock()
        self.model.process.poll.return_value = None
        self.model.opener = Mock()

    def response(self, summary=SAFE_SUMMARIES[0], reason="stop"):
        body = {"choices": [{"finish_reason": reason,
                              "message": {"content": json.dumps({"summary": summary}, ensure_ascii=False)}}]}
        self.model.opener.open.return_value = io.BytesIO(json.dumps(body, ensure_ascii=False).encode("utf-8"))

    def test_schema_wrapper_and_auth_are_sent_with_non_thinking(self):
        self.response()
        self.assertEqual(self.model.summarize("빨간차", 3), SAFE_SUMMARIES[0])
        request = self.model.opener.open.call_args.args[0]
        body = json.loads(request.data)
        schema = body["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["summary"]["enum"], list(SAFE_SUMMARIES))
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(request.get_header("Authorization"), "Bearer " + self.model.api_key)
        self.assertEqual(body["max_tokens"], 128)

    def test_incomplete_or_unsupported_generation_is_rejected(self):
        for summary, reason in (("영상 999초에 정답이 있습니다", "stop"), (SAFE_SUMMARIES[0], "length")):
            self.response(summary, reason)
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                self.model.summarize("자동차", 2)

    def test_stopped_model_is_not_reported_ready_or_called(self):
        self.model.process.poll.return_value = 0
        self.assertFalse(self.model.ready)
        with self.assertRaises(RuntimeError):
            self.model.summarize("자동차", 2)
        self.model.opener.open.assert_not_called()

    def test_close_stops_only_its_owned_process_and_closes_log(self):
        process = self.model.process
        log = self.model.log_file = Mock()
        self.model.close()
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=10)
        log.close.assert_called_once()
        self.assertFalse(self.model.ready)
        self.model.close()

    def test_resource_limits_are_checked_before_starting(self):
        for kwargs in ({"port": 0}, {"context": 100}, {"output_tokens": 1000}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LocalLLM("server", "model", **kwargs)


if __name__ == "__main__":
    unittest.main()
