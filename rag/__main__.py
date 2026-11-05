import argparse
import ipaddress
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path


def protect_local_configs():
    try:
        result = subprocess.run(["git", "rev-parse", "--git-path", "info/exclude"],
                                capture_output=True, text=True, check=False)
    except OSError:
        return
    if result.returncode == 0:
        path = Path(result.stdout.strip())
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if "*.local.json" not in text.splitlines():
            path.parent.mkdir(parents=True, exist_ok=True)
            # Local excludes persist when switching to older experiment branches.
            path.write_text(text.rstrip() + "\n*.local.json\n", encoding="utf-8")


def read_key(path):
    value = os.environ.get("CCTV_RAG_API_KEY")
    if path:
        value = json.loads(Path(path).read_text(encoding="utf-8"))["api_key"]
    if value is not None and (not isinstance(value, str) or len(value) < 32 or "\n" in value or "\r" in value):
        raise ValueError("Use an API key with at least 32 single-line characters")
    return value


def main():
    parser = argparse.ArgumentParser(description="Local vector retrieval and Korean frame recommendations")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--manifest", default="configs/rag-model.json")
    download.add_argument("--assets-root", default="models/local-rag")
    keygen = commands.add_parser("keygen")
    keygen.add_argument("--output", default="configs/rag.local.json")
    serve = commands.add_parser("serve")
    serve.add_argument("--index", required=True)
    serve.add_argument("--embedding-config", default="configs/embedding.siglip2.yaml")
    serve.add_argument("--manifest", default="configs/rag-model.json")
    serve.add_argument("--assets-root", default="models/local-rag")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--llm-port", type=int, default=8766)
    serve.add_argument("--api-key-file")
    serve.add_argument("--no-llm", action="store_true")
    serve.add_argument("--log-dir", default="runs/local-rag-runtime")
    ask = commands.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--base-url", default="http://127.0.0.1:8765")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--api-key-file")
    args = parser.parse_args()
    try:
        if args.command == "download":
            from .assets import ensure_assets
            assets = ensure_assets(args.manifest, args.assets_root)
            print(json.dumps({"status": "success", **{k: v for k, v in assets.items() if k != "manifest"}}, ensure_ascii=False))
            return 0
        if args.command == "keygen":
            path = Path(args.output)
            if not path.name.endswith(".local.json"):
                raise ValueError("Use a .local.json filename excluded from Git and source packages")
            protect_local_configs()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                json.dump({"api_key": secrets.token_urlsafe(32)}, stream)
                stream.write("\n")
            print(json.dumps({"status": "success", "key_file": str(path), "secret_printed": False}))
            return 0
        if args.command == "ask":
            from .client import recommend_frames
            result = recommend_frames(args.question, top_k=args.top_k, base_url=args.base_url,
                                      api_key=read_key(args.api_key_file))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        api_key = read_key(args.api_key_file)
        try:
            loopback = ipaddress.ip_address(args.host).is_loopback
        except ValueError:
            loopback = args.host.lower() == "localhost"
        if not loopback and api_key is None:
            raise ValueError("LAN serving requires --api-key-file or CCTV_RAG_API_KEY")
        if not args.no_llm and args.port == args.llm_port:
            raise ValueError("RAG and internal LLM ports must differ")
        if not (Path(args.index) / "index_manifest.json").is_file():
            raise FileNotFoundError("Select the directory containing a published index_manifest.json")
        from embedding import create_encoder, load_config
        from .assets import ensure_assets
        from .engine import FrameRecommender
        from .http_server import make_server
        from .runtime import LocalLLM

        print(json.dumps({"status": "loading", "index": args.index}), flush=True)
        encoder = create_encoder(load_config(args.embedding_config))
        engine = FrameRecommender(args.index, encoder)
        llm, server = None, None
        try:
            if not args.no_llm:
                assets = ensure_assets(args.manifest, args.assets_root)
                llm = LocalLLM(assets["server_path"], assets["model_path"],
                               port=args.llm_port, log_dir=args.log_dir).start()
                engine.llm = llm
            server = make_server(engine, args.host, args.port, api_key)
            print(json.dumps({"status": "ready", "host": args.host, "port": server.server_port,
                              **engine.health()}, ensure_ascii=False), flush=True)
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if server is not None:
                server.server_close()
            if llm is not None:
                llm.close()
        return 0
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
