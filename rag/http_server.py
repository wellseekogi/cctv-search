import hmac
import ipaddress
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .errors import RagError

MAX_BODY_BYTES = 16 * 1024
REQUEST_TIMEOUT_SECONDS = 10
REQUEST_FIELDS = {"question", "top_k", "video_ids", "representation", "use_llm"}


def reject_constant(value):
    raise ValueError("Nonfinite JSON number")


def make_server(recommender, host="127.0.0.1", port=8765, api_key=None):
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.lower() == "localhost"
    if api_key is not None and (not isinstance(api_key, str) or not api_key):
        raise ValueError("api_key must be a nonempty string")
    if not loopback and (api_key is None or len(api_key) < 32):
        raise ValueError("A non-loopback server requires an API key of at least 32 characters")
    busy = threading.BoundedSemaphore(1)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(REQUEST_TIMEOUT_SECONDS)

        def log_message(self, format, *args):
            pass

        def send_json(self, status, value):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (ConnectionError, TimeoutError):
                pass

        def send_error_json(self, code, message, status):
            self.send_json(status, {"error": {"code": code, "message": message}})

        def authorized(self):
            if api_key is None:
                return True
            supplied = self.headers.get("Authorization", "").encode("utf-8")
            if hmac.compare_digest(supplied, ("Bearer " + api_key).encode("utf-8")):
                return True
            self.send_error_json("UNAUTHORIZED", "A valid Bearer API key is required", 401)
            return False

        def do_GET(self):
            if not self.authorized():
                return
            if self.path != "/health":
                self.send_error_json("NOT_FOUND", "Unknown endpoint", 404)
                return
            try:
                self.send_json(200, recommender.health())
            except RagError as exc:
                self.send_error_json(exc.code, str(exc), exc.status)
            except Exception:
                self.send_error_json("INTERNAL_ERROR", "Internal server error", 500)

        def read_payload(self):
            if self.headers.get("Transfer-Encoding") is not None:
                raise RagError("INVALID_INPUT", "Transfer-Encoding is unsupported", 400)
            lengths = self.headers.get_all("Content-Length", [])
            if not lengths:
                raise RagError("LENGTH_REQUIRED", "Content-Length is required", 411)
            if len(lengths) != 1:
                raise RagError("INVALID_INPUT", "Supply exactly one Content-Length", 400)
            try:
                length = int(lengths[0])
            except ValueError as exc:
                raise RagError("INVALID_INPUT", "Invalid Content-Length", 400) from exc
            if length < 0:
                raise RagError("INVALID_INPUT", "Invalid Content-Length", 400)
            if length > MAX_BODY_BYTES:
                raise RagError("REQUEST_TOO_LARGE", "Request body exceeds 16 KiB", 413)
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                raise RagError("INVALID_INPUT", "Content-Type must be application/json", 415)
            try:
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("Incomplete body")
                payload = json.loads(body.decode("utf-8"), parse_constant=reject_constant)
            except (ValueError, UnicodeError) as exc:
                raise RagError("INVALID_INPUT", "Body must be valid UTF-8 JSON", 400) from exc
            if not isinstance(payload, dict):
                raise RagError("INVALID_INPUT", "Body must be a JSON object")
            if "question" not in payload or set(payload) - REQUEST_FIELDS:
                raise RagError("INVALID_INPUT", "Supply question and only supported request fields")
            return payload

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != "/recommend":
                self.send_error_json("NOT_FOUND", "Unknown endpoint", 404)
                return
            try:
                payload = self.read_payload()
            except RagError as exc:
                self.send_error_json(exc.code, str(exc), exc.status)
                return
            except (TimeoutError, OSError):
                self.send_error_json("REQUEST_TIMEOUT", "Request body was not received in time", 408)
                return
            if not busy.acquire(blocking=False):
                self.send_error_json("BUSY", "Another recommendation is in progress; retry shortly", 503)
                return
            try:
                result = recommender.recommend(payload["question"], payload.get("top_k", 5),
                                               payload.get("video_ids"), payload.get("representation", "combined"),
                                               payload.get("use_llm", True))
                self.send_json(200, result)
            except RagError as exc:
                self.send_error_json(exc.code, str(exc), exc.status)
            except Exception:
                self.send_error_json("INTERNAL_ERROR", "Internal server error", 500)
            finally:
                busy.release()

    class Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

    return Server((host, port), Handler)
