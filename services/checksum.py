"""Deterministic SHA256 fingerprint for a model directory.

Hashed byte stream (algorithm id ``sha256-dir-v1``)::

    MAGIC | file_count(u64be)
    then, for each file sorted by UTF-8 bytes of its normalised relative path:
        len(path)(u64be) | path | file_size(u64be) | file contents

Every variable-length field is length-prefixed, so file boundaries are
unambiguous (file ``a`` containing ``bc`` can never collide with file ``ab``
containing ``c``). Paths are POSIX-style (``/``) and Unicode-NFC normalised,
so the same directory hashes identically on Linux, macOS and Windows.

NOTE: this is NOT compatible with the checksum produced by the original
script (which had no framing and used OS-specific path separators). Versions
registered earlier carry a legacy ``model_sha256`` that will never match.
"""
from __future__ import annotations

import hashlib
import os
import struct
import unicodedata
from pathlib import Path, PurePath
from typing import Dict, List, Tuple, Union

CHECKSUM_ALGORITHM = "sha256-dir-v1"
DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1 MiB
_MAGIC = b"ct2-dir-sha256-v1\x00"


def normalize_relative_path(relative: PurePath) -> str:
    """POSIX separators + NFC normalisation (macOS stores NFD names)."""
    return unicodedata.normalize("NFC", relative.as_posix())


def _collect_files(directory: Path) -> List[Tuple[str, Path]]:
    entries: Dict[str, Path] = {}
    for root, dirnames, filenames in os.walk(directory, followlinks=False):
        root_path = Path(root)
        for name in dirnames:
            if (root_path / name).is_symlink():
                raise ValueError(
                    f"Symlinked directories are not supported: {root_path / name}"
                )
        for name in filenames:
            abs_path = root_path / name
            rel = normalize_relative_path(abs_path.relative_to(directory))
            if rel in entries:
                raise ValueError(
                    f"Two files normalise to the same relative path: {rel!r}"
                )
            entries[rel] = abs_path
    return sorted(
        entries.items(), key=lambda kv: kv[0].encode("utf-8", "surrogateescape")
    )


def calculate_directory_sha256(
    directory: Union[str, Path], chunk_size: int = DEFAULT_CHUNK_SIZE
) -> str:
    """Return the hex SHA256 of all files (paths + contents) under ``directory``.

    Files are streamed in ``chunk_size`` pieces; the result does not depend on
    ``chunk_size``. Empty sub-directories are not part of the fingerprint.
    """
    directory = Path(directory)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    files = _collect_files(directory)
    digest = hashlib.sha256()
    digest.update(_MAGIC)
    digest.update(struct.pack(">Q", len(files)))

    for rel, abs_path in files:
        path_bytes = rel.encode("utf-8", "surrogateescape")
        size = abs_path.stat().st_size
        digest.update(struct.pack(">Q", len(path_bytes)))
        digest.update(path_bytes)
        digest.update(struct.pack(">Q", size))

        read = 0
        with abs_path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
                read += len(chunk)
        if read != size:
            raise RuntimeError(f"File changed while hashing: {abs_path}")

    return digest.hexdigest()
