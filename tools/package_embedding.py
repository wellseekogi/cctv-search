"""Package C's source and verified example bundles, excluding weights and environments."""

import argparse
import tempfile
import zipfile
from pathlib import Path

from embedding.contracts import file_sha256, write_json
from embedding.pipeline import validate_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_zip")
    parser.add_argument("--run", action="append", default=[], help="Verified run directory, relative to project")
    parser.add_argument("--bundle", action="store_true", help="Include the local git-history.bundle")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    selected = [root / "README.md", root / ".gitignore", root / ".gitattributes", root / "requirements-embedding.txt"]
    selected.append(root / "requirements-siglip2.txt")
    if args.bundle:
        selected.append(root / "git-history.bundle")
    for run in args.run:
        path = (root / run).resolve()
        if not path.is_relative_to(root / "runs") or not path.is_dir():
            raise ValueError("Select an existing directory within runs/")
    for folder in ["embedding", "rag", "configs", "fixtures", "tests", "tools", "docs", *args.run]:
        selected.extend(p for p in (root / folder).rglob("*") if p.is_file()
                        and "__pycache__" not in p.parts and "cache" not in p.parts
                        and not p.name.endswith(".local.json") and not p.name.startswith(".env")
                        and p.name != "local.yaml")
    hashes = {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(selected)}
    manifest = root / "source_hashes.json"
    write_json(manifest, {"files": hashes})
    output = Path(args.output_zip).resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh ZIP path")
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in selected + [manifest]:
            archive.write(path, (Path(root.name) / path.relative_to(root)).as_posix())
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP checksum failed")
        with tempfile.TemporaryDirectory(dir=root) as temp:
            archive.extractall(temp)
            extracted = Path(temp) / root.name
            for relative, sha in hashes.items():
                if file_sha256(extracted / relative) != sha:
                    raise ValueError("Packaged file hash mismatch")
            for run in args.run:
                for index in (extracted / run).rglob("index_manifest.json"):
                    validate_index(index.parent)
    print(f"PASS: {len(selected) + 1} files, ZIP CRC and extracted SHA256/index checks; {output}")


if __name__ == "__main__":
    main()
