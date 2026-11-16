import json
import os
from pathlib import Path

import pytest

from services.model_validation import ModelValidationError, validate_model_directory


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "dummy-model"
    d.mkdir()
    (d / "model.bin").write_bytes(b"DUMMY-NOT-A-REAL-MODEL")
    (d / "shared_vocabulary.json").write_text(json.dumps(["<s>", "</s>", "hello"]))
    (d / "config.json").write_text(json.dumps({"layer_norm": True}))
    return d


def test_valid_model_passes(model_dir):
    result = validate_model_directory(model_dir)
    assert result.passed
    assert result.to_dict()["validation_status"] == "passed"
    assert result.missing_files == [] and result.validation_errors == []
    assert result.model_name == "dummy-model"


def test_result_dict_shape(model_dir):
    d = validate_model_directory(model_dir, model_name="translation-model").to_dict()
    assert d["model_name"] == "translation-model"
    assert {"validation_status", "missing_files", "validation_errors"} <= set(d)


def test_accepts_str_path(model_dir):
    assert validate_model_directory(str(model_dir)).passed


def test_missing_model_bin(model_dir):
    (model_dir / "model.bin").unlink()
    result = validate_model_directory(model_dir)
    assert not result.passed
    assert "model.bin" in result.missing_files


def test_empty_model_bin(model_dir):
    (model_dir / "model.bin").write_bytes(b"")
    result = validate_model_directory(model_dir)
    assert not result.passed
    assert any("model.bin is empty" in e for e in result.validation_errors)


def test_model_bin_is_a_directory(model_dir):
    (model_dir / "model.bin").unlink()
    (model_dir / "model.bin").mkdir()
    result = validate_model_directory(model_dir)
    assert not result.passed
    assert any("not a regular file" in e for e in result.validation_errors)


def test_missing_vocabulary(model_dir):
    (model_dir / "shared_vocabulary.json").unlink()
    result = validate_model_directory(model_dir)
    assert not result.passed
    assert result.missing_files == ["shared_vocabulary.json"]


def test_vocabulary_check_can_be_disabled(model_dir):
    (model_dir / "shared_vocabulary.json").unlink()
    assert validate_model_directory(model_dir, require_vocabulary=False).passed


def test_alternative_txt_vocabulary_layout(model_dir):
    (model_dir / "shared_vocabulary.json").unlink()
    (model_dir / "shared_vocabulary.txt").write_text("<s>\n</s>\nhello\n")
    assert validate_model_directory(model_dir).passed


def test_split_source_target_vocabulary_layout(model_dir):
    (model_dir / "shared_vocabulary.json").unlink()
    (model_dir / "source_vocabulary.json").write_text('["a"]')
    (model_dir / "target_vocabulary.json").write_text('["b"]')
    assert validate_model_directory(model_dir).passed


def test_partial_split_vocabulary_reports_missing_half(model_dir):
    (model_dir / "shared_vocabulary.json").unlink()
    (model_dir / "source_vocabulary.json").write_text('["a"]')
    result = validate_model_directory(model_dir)
    assert not result.passed
    assert result.missing_files == ["target_vocabulary.json"]


def test_config_json_is_optional(model_dir):
    (model_dir / "config.json").unlink()
    result = validate_model_directory(model_dir)
    assert result.passed
    assert any("config.json" in w for w in result.warnings)


@pytest.mark.parametrize("content", ["", "{not json"])
def test_present_but_broken_config_fails(model_dir, content):
    (model_dir / "config.json").write_text(content)
    result = validate_model_directory(model_dir)
    assert not result.passed
    assert any("config.json" in e for e in result.validation_errors)


def test_broken_vocabulary_json_fails(model_dir):
    (model_dir / "shared_vocabulary.json").write_text("[1, 2")
    result = validate_model_directory(model_dir)
    assert not result.passed


def test_tokenizer_optional_warning(model_dir):
    result = validate_model_directory(model_dir)
    assert result.passed and any("tokenizer" in w.lower() for w in result.warnings)


def test_tokenizer_required_when_requested(model_dir):
    result = validate_model_directory(model_dir, require_tokenizer=True)
    assert not result.passed and result.missing_files


def test_empty_tokenizer_file_fails(model_dir):
    (model_dir / "sp.model").write_bytes(b"")
    result = validate_model_directory(model_dir)
    assert not result.passed
    assert any("sp.model is empty" in e for e in result.validation_errors)


def test_nonempty_tokenizer_passes_without_warning(model_dir):
    (model_dir / "sp.model").write_bytes(b"spm")
    result = validate_model_directory(model_dir, require_tokenizer=True)
    assert result.passed


def test_nonexistent_directory(tmp_path):
    result = validate_model_directory(tmp_path / "nope")
    assert not result.passed
    assert any("does not exist" in e for e in result.validation_errors)


def test_path_is_a_file_not_directory(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    result = validate_model_directory(f)
    assert not result.passed
    assert any("not a directory" in e for e in result.validation_errors)


def test_empty_directory(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    result = validate_model_directory(d)
    assert not result.passed and "model.bin" in result.missing_files


def test_dangling_symlink_model_bin_is_reported(model_dir):
    (model_dir / "model.bin").unlink()
    try:
        os.symlink(model_dir / "does-not-exist", model_dir / "model.bin")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported here")
    result = validate_model_directory(model_dir)
    assert not result.passed
    assert any("model.bin" in e for e in result.validation_errors)


def test_unreadable_file_detected_via_open_failure(model_dir, monkeypatch):
    real_open = Path.open

    def fake_open(self, *args, **kwargs):
        if self.name == "model.bin":
            raise PermissionError(13, "Permission denied")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fake_open)
    result = validate_model_directory(model_dir)
    assert not result.passed
    assert any("model.bin is unreadable" in e for e in result.validation_errors)


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="chmod-based unreadability is not enforced for root / on Windows",
)
def test_unreadable_file_real_permissions(model_dir):
    path = model_dir / "model.bin"
    path.chmod(0o000)
    try:
        result = validate_model_directory(model_dir)
    finally:
        path.chmod(0o644)
    assert not result.passed
    assert any("unreadable" in e for e in result.validation_errors)


def test_model_validation_error_carries_result(model_dir):
    (model_dir / "model.bin").unlink()
    result = validate_model_directory(model_dir)
    err = ModelValidationError(result)
    assert err.result is result and "model.bin" in str(err)
