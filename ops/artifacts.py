# ops/artifacts.py
from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from typing import Optional, Tuple


ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "/tmp/dspu_artifacts")


@dataclass
class Artifact:
    ref: str
    path: str
    sha256: str
    size_bytes: int


def _ensure_dir() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)


def _sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def put_bytes(data: bytes, *, ext: str) -> Artifact:
    """
    Store bytes in ARTIFACT_DIR, content-addressed by sha256.
    Returns an artifact ref like: artifact://<sha256>.<ext>
    """
    _ensure_dir()
    ext = (ext or "").lstrip(".") or "bin"
    digest = _sha256(data)
    filename = f"{digest}.{ext}"
    path = os.path.join(ARTIFACT_DIR, filename)

    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)

    size = os.path.getsize(path)
    ref = f"artifact://{filename}"
    return Artifact(ref=ref, path=path, sha256=digest, size_bytes=size)


def get_path(ref: str) -> str:
    """
    Resolve artifact://... refs to filesystem paths.
    """
    if not ref.startswith("artifact://"):
        raise ValueError(f"Unsupported ref (expected artifact://...): {ref}")
    _ensure_dir()
    filename = ref[len("artifact://") :]
    path = os.path.join(ARTIFACT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {ref} (path={path})")
    return path


def read_bytes(ref: str) -> bytes:
    path = get_path(ref)
    with open(path, "rb") as f:
        return f.read()


def b64_to_bytes(s: str) -> bytes:
    return base64.b64decode(s.encode("utf-8"))


def bytes_to_b64(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")
