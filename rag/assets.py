"""Download pinned local model/runtime assets and verify cache integrity."""

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


def _sha(stream):
    value = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(chunk)
    return value.hexdigest()


def _file_sha(path):
    with path.open("rb") as stream:
        return _sha(stream)


def _inside(root, *parts):
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Asset path escapes its root")
    return path


def _validate(record):
    name = record["filename"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError("Asset filename must be a plain basename")
    if name.split(".", 1)[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)],
                                   *[f"LPT{i}" for i in range(1, 10)]}:
        raise ValueError("Reserved Windows asset filename")
    parsed = urllib.parse.urlsplit(record["url"])
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Asset URL must use HTTPS without credentials")
    if not re.fullmatch(r"[a-f0-9]{64}", record["sha256"]):
        raise ValueError("Pin a full SHA256 for each asset")
    if type(record["bytes"]) is not int or record["bytes"] <= 0:
        raise ValueError("Asset size must be a positive integer")


def _download(record, path):
    if path.is_file() and path.stat().st_size == record["bytes"] and _file_sha(path) == record["sha256"]:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".part",
                                         delete=False) as output:
            candidate = Path(output.name)
            size, value = 0, hashlib.sha256()
            with urllib.request.urlopen(record["url"], timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > record["bytes"]:
                        raise ValueError("Asset download exceeds pinned size")
                    value.update(chunk)
                    output.write(chunk)
            if size != record["bytes"] or value.hexdigest() != record["sha256"]:
                raise ValueError("Asset download size or SHA256 mismatch")
        os.replace(candidate, path)
    finally:
        if candidate is not None:
            candidate.unlink(missing_ok=True)


def _members(archive):
    members, seen, servers = [], set(), []
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        relative = PurePosixPath(name)
        mode = info.external_attr >> 16
        if (not name or relative.is_absolute() or ".." in relative.parts or ":" in name
                or not relative.parts or stat.S_ISLNK(mode)):
            raise ValueError("Unsafe runtime ZIP member")
        if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError("Unsupported runtime ZIP member type")
        key = relative.as_posix().casefold()
        if key in seen:
            raise ValueError("Duplicate runtime ZIP member")
        seen.add(key)
        members.append((info, relative))
        if not info.is_dir() and relative.name.casefold() == "llama-server.exe":
            servers.append(relative)
    if len(servers) != 1:
        raise ValueError("Runtime ZIP must contain exactly one llama-server.exe")
    return members, servers[0]


def _runtime(zip_path, folder):
    with zipfile.ZipFile(zip_path) as archive:
        members, server = _members(archive)

        def matches(info, relative):
            target = _inside(folder, *relative.parts)
            if info.is_dir():
                return target.is_dir()
            if not target.is_file() or target.stat().st_size != info.file_size:
                return False
            with archive.open(info) as source:
                return _file_sha(target) == _sha(source)

        if not all(matches(info, relative) for info, relative in members):
            folder.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=folder.parent, prefix=".runtime-") as temporary:
                stage = Path(temporary).resolve()
                for info, relative in members:
                    target = _inside(stage, *relative.parts)
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(info) as source, target.open("wb") as output:
                            shutil.copyfileobj(source, output, length=1024 * 1024)
                folder.mkdir(parents=True, exist_ok=True)
                for info, relative in members:
                    target = _inside(folder, *relative.parts)
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(_inside(stage, *relative.parts), target)
        return _inside(folder, *server.parts)


def ensure_assets(manifest_path="configs/rag-model.json", root="models/local-rag"):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    model, runtime = manifest["model"], manifest["runtime"]
    _validate(model)
    _validate(runtime)
    tag = runtime["tag"]
    if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        raise ValueError("Runtime tag must be a plain directory name")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    model_path = _inside(root, "model", model["filename"])
    runtime_zip = _inside(root, "downloads", runtime["filename"])
    _download(model, model_path)
    _download(runtime, runtime_zip)
    server_path = _runtime(runtime_zip, _inside(root, "runtime", tag))
    return {"model_path": str(model_path), "server_path": str(server_path), "manifest": manifest,
            "model_sha256": model["sha256"], "runtime_sha256": runtime["sha256"]}
