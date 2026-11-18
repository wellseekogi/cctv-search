"""미디어 파일 제공과 HTTP Range (안내서 D 3). 브라우저의 MP4 탐색(seek)에 필요합니다."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from fastapi import Response
from fastapi.responses import StreamingResponse

CHUNK = 256 * 1024


def parse_range(header: str | None, size: int) -> tuple[int, int] | None | str:
    """Range 헤더를 (start, end) 로 바꿉니다(end 포함).

    None  -> Range 없음 또는 무시할 형식(전체 200 응답)
    "bad" -> 만족할 수 없는 범위(416 응답)
    """
    if not header or not header.startswith("bytes="):
        return None
    spec = header[len("bytes="):].strip()
    if "," in spec:  # 다중 범위는 지원하지 않고 전체를 보냅니다
        return None
    first, sep, last = spec.partition("-")
    if not sep:
        return None
    try:
        if first == "":           # bytes=-N : 끝에서 N바이트
            n = int(last)
            if n <= 0:
                return "bad"
            start, end = max(size - n, 0), size - 1
        else:
            start = int(first)
            end = int(last) if last else size - 1
    except ValueError:
        return None
    if size == 0 or start >= size or start > end:
        return "bad"
    return start, min(end, size - 1)


def _iter_file(path: Path, start: int, length: int) -> Iterator[bytes]:
    with path.open("rb") as f:
        f.seek(start)
        remaining = length
        while remaining > 0:
            chunk = f.read(min(CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def file_response(path: Path, content_type: str, range_header: str | None, head: bool) -> Response:
    size = path.stat().st_size
    base = {"Accept-Ranges": "bytes", "Cache-Control": "no-cache"}
    rng = parse_range(range_header, size)
    if rng == "bad":
        return Response(status_code=416, headers={**base, "Content-Range": f"bytes */{size}"})
    if rng is None:
        start, end, status = 0, size - 1, 200
    else:
        start, end = rng  # type: ignore[misc]
        status = 206
        base["Content-Range"] = f"bytes {start}-{end}/{size}"
    length = end - start + 1 if size else 0
    base["Content-Length"] = str(length)
    if head:
        return Response(status_code=status, headers=base, media_type=content_type)
    return StreamingResponse(
        _iter_file(path, start, length), status_code=status, headers=base, media_type=content_type
    )
