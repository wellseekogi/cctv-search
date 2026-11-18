"""API 서버 실행.

  python -m api --config configs/retrieval.yaml

서비스는 127.0.0.1, worker 1 로 실행합니다(안내서 D 3).
"""
from __future__ import annotations

import argparse
import sys

import uvicorn

from retrieval.config import ConfigError, load_config

from .app import create_app


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m api", description="CCTV 검색 API 서버")
    p.add_argument("--config", default="configs/retrieval.yaml")
    p.add_argument("--host", default=None, help="기본값은 설정의 api.host (127.0.0.1)")
    p.add_argument("--port", type=int, default=None)
    args = p.parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"[입력 오류] {e}", file=sys.stderr)
        return 2
    app = create_app(config)
    health = app.state.service.health()
    print(f"[준비 상태] {health['status']} (model_ready={health['model_ready']}, index_ready={health['index_ready']})")
    for g in health["galleries"]:
        for problem in g["problems"]:
            print(f"  - {g['gallery_id']}: {problem}")
    uvicorn.run(app, host=args.host or config.api.host, port=args.port or config.api.port, workers=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
