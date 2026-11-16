import hashlib
import os
import struct
import unicodedata
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from services.checksum import (
    CHECKSUM_ALGORITHM,
    calculate_directory_sha256,
    normalize_relative_path,
)


def make_dir(root: Path, files: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


FILES = {"model.bin": b"\x00\x01weights", "shared_vocabulary.json": b'["a","b"]', "sub/x.txt": b"x"}


def test_identical_directories_same_checksum_regardless_of_creation_order(tmp_path):
    a = make_dir(tmp_path / "a", FILES)
    b = make_dir(tmp_path / "b", dict(reversed(list(FILES.items()))))
    assert calculate_directory_sha256(a) == calculate_directory_sha256(b)


def test_checksum_is_deterministic_and_hex(tmp_path):
    d = make_dir(tmp_path / "d", FILES)
    first = calculate_directory_sha256(d)
    assert first == calculate_directory_sha256(d)
    assert len(first) == 64 and int(first, 16) >= 0


def test_matches_independent_reference_implementation(tmp_path):
    d = make_dir(tmp_path / "d", {"a.bin": b"hello", "b/c.txt": b"world"})
    ref = hashlib.sha256()
    ref.update(b"ct2-dir-sha256-v1\x00")
    ref.update(struct.pack(">Q", 2))
    for rel, content in [(b"a.bin", b"hello"), (b"b/c.txt", b"world")]:
        ref.update(struct.pack(">Q", len(rel)) + rel + struct.pack(">Q", len(content)) + content)
    assert calculate_directory_sha256(d) == ref.hexdigest()
    assert CHECKSUM_ALGORITHM == "sha256-dir-v1"


def test_modified_file_changes_checksum(tmp_path):
    d = make_dir(tmp_path / "d", FILES)
    before = calculate_directory_sha256(d)
    (d / "model.bin").write_bytes(b"\x00\x01weightz")
    assert calculate_directory_sha256(d) != before


def test_single_bit_flip_changes_checksum(tmp_path):
    d = make_dir(tmp_path / "d", {"model.bin": b"\x00" * 100})
    before = calculate_directory_sha256(d)
    (d / "model.bin").write_bytes(b"\x00" * 50 + b"\x01" + b"\x00" * 49)
    assert calculate_directory_sha256(d) != before


@pytest.mark.parametrize(
    "other",
    [
        {**FILES, "extra.txt": b""},                                   # added empty file
        {k: v for k, v in FILES.items() if k != "sub/x.txt"},          # removed file
        {("renamed.bin" if k == "model.bin" else k): v for k, v in FILES.items()},  # renamed
    ],
)
def test_different_contents_or_names_change_checksum(tmp_path, other):
    a = make_dir(tmp_path / "a", FILES)
    b = make_dir(tmp_path / "b", other)
    assert calculate_directory_sha256(a) != calculate_directory_sha256(b)


def test_file_boundaries_are_unambiguous(tmp_path):
    # naive "path + content" concatenation would give "abc" for both
    a = make_dir(tmp_path / "a", {"a": b"bc"})
    b = make_dir(tmp_path / "b", {"ab": b"c"})
    assert calculate_directory_sha256(a) != calculate_directory_sha256(b)


def test_content_cannot_be_shifted_between_files(tmp_path):
    a = make_dir(tmp_path / "a", {"f1": b"AB", "f2": b"C"})
    b = make_dir(tmp_path / "b", {"f1": b"A", "f2": b"BC"})
    assert calculate_directory_sha256(a) != calculate_directory_sha256(b)


def test_large_file_is_chunked_and_chunk_size_independent(tmp_path):
    big = os.urandom(1024) * (9 * 1024)  # ~9 MiB
    d = make_dir(tmp_path / "d", {"model.bin": big})
    default = calculate_directory_sha256(d)
    assert default == calculate_directory_sha256(d, chunk_size=4096)
    assert default == calculate_directory_sha256(d, chunk_size=7 * 1024 * 1024 + 3)


def test_path_normalization_windows_and_posix_agree():
    assert normalize_relative_path(PureWindowsPath("sub\\dir\\f.txt")) == "sub/dir/f.txt"
    assert normalize_relative_path(PurePosixPath("sub/dir/f.txt")) == "sub/dir/f.txt"


def test_path_normalization_unicode_nfc():
    nfd = unicodedata.normalize("NFD", "café.txt")
    nfc = unicodedata.normalize("NFC", "café.txt")
    assert nfd != nfc
    assert normalize_relative_path(PurePosixPath(nfd)) == nfc


def test_nfc_and_nfd_filenames_hash_identically(tmp_path):
    nfd = unicodedata.normalize("NFD", "café.txt")
    nfc = unicodedata.normalize("NFC", "café.txt")
    a = make_dir(tmp_path / "a", {nfd: b"1"})
    b = make_dir(tmp_path / "b", {nfc: b"1"})
    assert calculate_directory_sha256(a) == calculate_directory_sha256(b)


def test_colliding_normalized_paths_raise(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / unicodedata.normalize("NFD", "é")).write_bytes(b"1")
    (d / unicodedata.normalize("NFC", "é")).write_bytes(b"2")
    if len(os.listdir(d)) < 2:
        pytest.skip("filesystem normalises filenames itself")
    with pytest.raises(ValueError, match="normalise"):
        calculate_directory_sha256(d)


def test_not_a_directory_raises(tmp_path):
    f = tmp_path / "f.txt"
    f.write_bytes(b"x")
    with pytest.raises(NotADirectoryError):
        calculate_directory_sha256(f)
    with pytest.raises(NotADirectoryError):
        calculate_directory_sha256(tmp_path / "missing")


def test_invalid_chunk_size_raises(tmp_path):
    with pytest.raises(ValueError):
        calculate_directory_sha256(make_dir(tmp_path / "d", FILES), chunk_size=0)


def test_symlinked_directory_is_rejected(tmp_path):
    d = make_dir(tmp_path / "d", FILES)
    target = make_dir(tmp_path / "target", {"x": b"1"})
    try:
        os.symlink(target, d / "link", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported here")
    with pytest.raises(ValueError, match="Symlinked"):
        calculate_directory_sha256(d)
