"""Structural validation of a converted CTranslate2 model directory.

What is checked (and why):

* ``model.bin`` is the only file every CTranslate2 model has -> always required.
* A vocabulary is required by default. Converters emit one of these layouts
  depending on version / model type (configurable via ``require_vocabulary``):
    - ``shared_vocabulary.json`` or ``shared_vocabulary.txt``
    - ``source_vocabulary.json|txt`` AND ``target_vocabulary.json|txt``
* ``config.json`` is OPTIONAL (not all converter versions write it). If it is
  present it must be non-empty, readable, valid JSON.
* Tokenizer files (e.g. SentencePiece ``*.model``) are NOT produced by the
  CTranslate2 converter, so they are optional. If found they must be
  non-empty and readable. Set ``require_tokenizer=True`` to enforce one.

Limits: this is a *structural* check. It cannot prove ``model.bin`` is a
loadable CTranslate2 model; that needs the ``ctranslate2`` runtime.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

MODEL_BIN = "model.bin"
CONFIG_JSON = "config.json"
VOCABULARY_LAYOUTS: Tuple[Tuple[str, ...], ...] = (
    ("shared_vocabulary.json",),
    ("shared_vocabulary.txt",),
    ("source_vocabulary.json", "target_vocabulary.json"),
    ("source_vocabulary.txt", "target_vocabulary.txt"),
)
TOKENIZER_SUFFIXES = (".model", ".spm")


@dataclass
class ValidationResult:
    model_name: str
    validation_status: str = "passed"  # "passed" | "failed"
    missing_files: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    files_checked: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.validation_status == "passed"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def _finalize(self) -> "ValidationResult":
        failed = bool(self.missing_files or self.validation_errors)
        self.validation_status = "failed" if failed else "passed"
        return self


class ModelValidationError(Exception):
    """Raised by callers that want to abort when validation fails."""

    def __init__(self, result: ValidationResult):
        self.result = result
        problems = [f"missing: {m}" for m in result.missing_files]
        problems += result.validation_errors
        super().__init__(
            f"Model validation failed for {result.model_name!r}: " + "; ".join(problems)
        )


def _check_file(path: Path, label: str, result: ValidationResult, *, as_json: bool = False) -> bool:
    """Existing-file checks: is a file, readable, non-empty, (valid JSON)."""
    result.files_checked.append(label)
    if not path.is_file():
        result.validation_errors.append(f"{label} is not a regular file")
        return False
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.read(1)  # surfaces permission / I/O errors
    except OSError as exc:
        result.validation_errors.append(f"{label} is unreadable: {exc}")
        return False
    if size == 0:
        result.validation_errors.append(f"{label} is empty")
        return False
    if as_json:
        try:
            with path.open("rb") as handle:
                json.loads(handle.read().decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            result.validation_errors.append(f"{label} is not valid JSON: {exc}")
            return False
    return True


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()  # dangling symlink => "present but broken"


def validate_model_directory(
    model_dir: Union[str, Path],
    model_name: Optional[str] = None,
    *,
    require_vocabulary: bool = True,
    require_tokenizer: bool = False,
) -> ValidationResult:
    """Validate ``model_dir`` and return a structured result (never raises for
    validation problems; inspect ``result.passed``)."""
    model_dir = Path(model_dir)
    result = ValidationResult(model_name=model_name or model_dir.name)

    if not model_dir.exists():
        result.validation_errors.append(f"Model directory does not exist: {model_dir}")
        return result._finalize()
    if not model_dir.is_dir():
        result.validation_errors.append(f"Model path is not a directory: {model_dir}")
        return result._finalize()
    try:
        next(model_dir.iterdir(), None)
    except OSError as exc:
        result.validation_errors.append(f"Model directory is unreadable: {exc}")
        return result._finalize()

    # model.bin (always required)
    model_bin = model_dir / MODEL_BIN
    if not _exists(model_bin):
        result.missing_files.append(MODEL_BIN)
    else:
        _check_file(model_bin, MODEL_BIN, result)

    # vocabulary (required by default; accept any known layout)
    if require_vocabulary:
        complete = [
            layout for layout in VOCABULARY_LAYOUTS
            if all(_exists(model_dir / name) for name in layout)
        ]
        if complete:
            for name in complete[0]:
                _check_file(model_dir / name, name, result, as_json=name.endswith(".json"))
        else:
            # report against the layout that is closest to complete
            # (most files already present; ties -> first listed layout)
            best = min(
                VOCABULARY_LAYOUTS,
                key=lambda layout: (
                    -sum(_exists(model_dir / n) for n in layout),
                    sum(not _exists(model_dir / n) for n in layout),
                ),
            )
            result.missing_files.extend(n for n in best if not _exists(model_dir / n))
            alternatives = " | ".join("+".join(layout) for layout in VOCABULARY_LAYOUTS)
            result.validation_errors.append(
                f"No complete vocabulary found; expected one of: {alternatives}"
            )

    # config.json (optional, but must be sane when present)
    config = model_dir / CONFIG_JSON
    if _exists(config):
        _check_file(config, CONFIG_JSON, result, as_json=True)
    else:
        result.warnings.append(f"{CONFIG_JSON} not present (optional for some converter versions)")

    # tokenizer (optional; sanity-check whatever is there)
    tokenizers = sorted(
        p for p in model_dir.iterdir()
        if p.suffix in TOKENIZER_SUFFIXES and (p.is_file() or p.is_symlink())
    )
    if tokenizers:
        for tok in tokenizers:
            _check_file(tok, tok.name, result)
    elif require_tokenizer:
        result.missing_files.append("tokenizer (*" + " | *".join(TOKENIZER_SUFFIXES) + ")")
    else:
        result.warnings.append(
            "No tokenizer model (*.model / *.spm) in the model directory; "
            "tokenization must be provided separately for evaluation/inference"
        )

    return result._finalize()
