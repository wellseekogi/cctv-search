import http.client
import importlib.util
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from rag.client import recommend_frames
from rag.errors import RagError
from rag.http_server import MAX_BODY_BYTES, make_server


class FakeRecommender:
    def __init__(self):
        self.calls = []

    def health(self):
        return {"ready": True, "index_id": "test-index"}

    def recommend(self, question, top_k, video_ids, representation, use_llm):
        self.calls.append((question, top_k, video_ids, representation, use_llm))
        return {"question": question, "recommendations": [{"frame_index": 6, "timestamp_ms": 1200}],
                "summary": "추천 프레임을 확인하세요."}


class RagHttpTests(unittest.TestCase):
    def start_server(self, engine=None, key=None):
        engine = engine or FakeRecommender()
        server = make_server(engine, port=0, api_key=key)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def close():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.addCleanup(close)
        return server, engine, f"http://127.0.0.1:{server.server_port}"

    def request(self, server, method="POST", path="/recommend", body=b'{"question":"car"}', headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        try:
            actual_headers = {"Content-Type": "application/json", **(headers or {})}
            connection.request(method, path, body=body, headers=actual_headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_client_roundtrip_and_defaults(self):
        server, engine, url = self.start_server()
        result = recommend_frames("빨간 자동차", base_url=url)
        self.assertEqual(result["recommendations"][0]["timestamp_ms"], 1200)
        self.assertEqual(engine.calls, [("빨간 자동차", 5, None, "combined", True)])

    def test_client_preserves_empty_filter_and_options(self):
        server, engine, url = self.start_server()
        recommend_frames("car", 3, [], "object", False, base_url=url + "/")
        self.assertEqual(engine.calls, [("car", 3, [], "object", False)])

    def test_authentication_for_health_and_recommend(self):
        key = "test-api-key"
        server, engine, url = self.start_server(key=key)
        for method, path in (("GET", "/health"), ("POST", "/recommend")):
            status, body = self.request(server, method, path)
            self.assertEqual(status, 401)
            self.assertEqual(body["error"]["code"], "UNAUTHORIZED")
        self.assertEqual(engine.calls, [])
        status, body = self.request(server, "GET", "/health", headers={"Authorization": "Bearer " + key})
        self.assertEqual((status, body["ready"]), (200, True))
        with patch.dict(os.environ, {"CCTV_RAG_API_KEY": key}):
            self.assertIn("recommendations", recommend_frames("car", base_url=url))

    def test_nonloopback_bind_requires_long_key_before_binding(self):
        for host, key in (("0.0.0.0", None), ("192.0.2.1", "short")):
            with self.subTest(host=host), self.assertRaisesRegex(ValueError, "32 characters"):
                make_server(FakeRecommender(), host=host, port=0, api_key=key)

    def test_json_errors_unknown_fields_and_missing_question(self):
        server, engine, _ = self.start_server()
        for body, expected_status in ((b"{broken", 400), (b'[]', 422),
                                      (b'{"question":"car","path":"secret"}', 422),
                                      (b'{"top_k":2}', 422),
                                      (b'{"question":NaN}', 400), (b'\xff', 400)):
            with self.subTest(body=body):
                status, result = self.request(server, body=body)
                self.assertEqual(status, expected_status)
                self.assertEqual(result["error"]["code"], "INVALID_INPUT")
        self.assertEqual(engine.calls, [])

    def test_oversized_body_and_wrong_content_type(self):
        server, engine, _ = self.start_server()
        status, result = self.request(server, body=b"x" * (MAX_BODY_BYTES + 1))
        self.assertEqual((status, result["error"]["code"]), (413, "REQUEST_TOO_LARGE"))
        status, _ = self.request(server, headers={"Content-Type": "text/plain"})
        self.assertEqual(status, 415)
        self.assertEqual(engine.calls, [])

    def test_error_code_and_status_propagate_to_client(self):
        class Failing(FakeRecommender):
            def recommend(self, *args):
                raise RagError("TEXT_TOO_LONG", "maximum 64 tokens", 422)

        server, _, url = self.start_server(Failing())
        status, body = self.request(server)
        self.assertEqual((status, body["error"]["code"]), (422, "TEXT_TOO_LONG"))
        with self.assertRaisesRegex(RuntimeError, "TEXT_TOO_LONG: maximum 64 tokens"):
            recommend_frames("car", base_url=url)

    def test_internal_exceptions_are_not_exposed(self):
        class Failing(FakeRecommender):
            def recommend(self, *args):
                raise RuntimeError("private-file-path and secret")

        server, _, url = self.start_server(Failing())
        status, body = self.request(server)
        self.assertEqual((status, body["error"]["code"]), (500, "INTERNAL_ERROR"))
        self.assertNotIn("secret", json.dumps(body))
        with self.assertRaisesRegex(RuntimeError, "INTERNAL_ERROR: Internal server error"):
            recommend_frames("car", base_url=url)

    def test_busy_is_rejected_while_health_remains_available(self):
        entered, release = threading.Event(), threading.Event()

        class Blocking(FakeRecommender):
            def recommend(self, *args):
                entered.set()
                release.wait(timeout=4)
                return super().recommend(*args)

        server, _, url = self.start_server(Blocking())
        results = []
        worker = threading.Thread(target=lambda: results.append(recommend_frames("first", base_url=url)), daemon=True)
        worker.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            status, body = self.request(server)
            self.assertEqual((status, body["error"]["code"]), (503, "BUSY"))
            status, _ = self.request(server, "GET", "/health")
            self.assertEqual(status, 200)
        finally:
            release.set()
            worker.join(timeout=3)
        self.assertEqual(len(results), 1)

    def test_unknown_endpoint(self):
        server, _, _ = self.start_server()
        status, body = self.request(server, "GET", "/private")
        self.assertEqual((status, body["error"]["code"]), (404, "NOT_FOUND"))

    def test_invalid_base_urls_and_timeout(self):
        for url in ("file:///secret", "http://user:pass@localhost", "http://localhost/path",
                    "http://localhost?token=secret", "http://localhost/#x", "http://localhost:99999"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                recommend_frames("car", base_url=url)
        for timeout in (0, -1, True, float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                recommend_frames("car", timeout=timeout)

    def test_client_file_can_be_imported_without_rag_package(self):
        path = Path(__file__).resolve().parents[1] / "rag/client.py"
        spec = importlib.util.spec_from_file_location("d_standalone_client", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        server, _, url = self.start_server()
        self.assertIn("recommendations", module.recommend_frames("car", base_url=url))

    def test_incomplete_body_times_out_before_engine_call(self):
        server, engine, _ = self.start_server()
        with patch("rag.http_server.REQUEST_TIMEOUT_SECONDS", 0.1):
            status, body = self.request(server, body=b"{", headers={"Content-Length": "100"})
        self.assertEqual((status, body["error"]["code"]), (408, "REQUEST_TIMEOUT"))
        self.assertEqual(engine.calls, [])

    def test_client_timeout_and_proxy_environment(self):
        server, _, url = self.start_server()
        with patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1", "NO_PROXY": ""}):
            self.assertIn("recommendations", recommend_frames("car", base_url=url))
        release = threading.Event()

        class Slow(FakeRecommender):
            def recommend(self, *args):
                release.wait(timeout=3)
                return super().recommend(*args)

        _, _, slow_url = self.start_server(Slow())
        try:
            with self.assertRaisesRegex(RuntimeError, "SERVER_UNAVAILABLE"):
                recommend_frames("car", base_url=slow_url, timeout=0.05)
        finally:
            release.set()

    def test_client_does_not_forward_api_key_to_redirect(self):
        redirected = threading.Event()

        class Redirecting(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.send_response(302)
                self.send_header("Location", "/somewhere-else")
                self.end_headers()

            def do_GET(self):
                redirected.set()
                self.send_response(200)
                self.end_headers()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Redirecting)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "HTTP_ERROR: RAG server returned HTTP 302"):
                recommend_frames("car", base_url=f"http://127.0.0.1:{server.server_port}", api_key="secret")
            self.assertFalse(redirected.is_set())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
