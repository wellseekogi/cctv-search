import argparse
import importlib
import json
import sys
from pathlib import Path

from .contracts import SCHEMA, write_json, write_jsonl
from .config import create_encoder, load_config
from .pipeline import build_index, validate_index


def main():
    parser = argparse.ArgumentParser(description="C 담당 이미지·임베딩 인덱스 생성")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    for name in ("samples", "manifest", "data-root", "output", "cache", "frame-reader", "decoder-id"):
        build.add_argument("--" + name, required=True)
    build.add_argument("--config", default="configs/embedding.yaml")
    check = commands.add_parser("validate")
    check.add_argument("run_dir")
    args = parser.parse_args()

    def record_fatal(exc):
        if args.command != "build":
            return
        output = Path(args.output)
        if not output.exists() or (output.is_dir() and not any(output.iterdir())):
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "embedding_status.json", {"schema_version": SCHEMA,
                       "stage": "embedding", "status": "failed", "error_id": "embedding-fatal"})
            write_jsonl(output / "errors.jsonl", [{"schema_version": SCHEMA,
                        "error_id": "embedding-fatal", "stage": "embedding",
                        "code": type(exc).__name__, "message": str(exc)}])

    try:
        if args.command == "validate":
            index = validate_index(args.run_dir)
            print(json.dumps({"status": "success", "row_count": index["row_count"]}))
            return 0
        config = load_config(args.config)
        module, name = args.frame_reader.split(":", 1)
        reader = getattr(importlib.import_module(module), name)
        encoder = create_encoder(config)
        status = build_index(args.samples, args.manifest, args.data_root, args.output, args.cache,
                             reader, encoder, args.decoder_id, config["crop_margin_ratio"],
                             config["batch_size_gpu"] if encoder.device.startswith("cuda") else config["batch_size_cpu"])
        print(json.dumps(status, ensure_ascii=False))
        return 0 if status["status"] == "success" else 1
    except (ValueError, KeyError, TypeError, FileNotFoundError, FileExistsError) as exc:
        record_fatal(exc)
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        # A missing index_manifest keeps consumers from opening a failed publication.
        record_fatal(exc)
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
