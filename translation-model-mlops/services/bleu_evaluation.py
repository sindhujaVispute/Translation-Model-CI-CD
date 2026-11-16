"""Corpus BLEU evaluation with SacreBLEU.

Flow: load aligned source/reference files -> translate in batches through a
``Translator`` -> corpus BLEU -> ``EvaluationResult`` (score + metadata).

Failures raise ``EvaluationError`` / ``DatasetError``; a result object only
exists for an evaluation that actually completed.

Logging the result to MLflow is the job of ``mlflow_service``; this module has
no MLflow dependency.

IMPORTANT - information needed to evaluate a REAL model (nothing here guesses):
  * the tokenizer used in training (e.g. the SentencePiece model) -> ``tokenize``
  * the matching detokenizer                                         -> ``detokenize``
  * target-language prefix / special tokens, if the model needs them
  * beam size and other decoding options used for reporting
  * the SacreBLEU settings the team reports with (``BleuConfig``; the defaults
    below are SacreBLEU's own, and are recorded with every result)
"""
from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

from sacrebleu.metrics import BLEU

EVALUATION_TYPES = ("synthetic", "real")
METRIC_BLEU_AUTO = "bleu_auto"
METRIC_SENTENCE_COUNT = "evaluation_sentence_count"
SYNTHETIC_NOTICE = (
    "Synthetic evaluation: this score does NOT represent real model performance."
)


class DatasetError(ValueError):
    """The parallel evaluation data is missing, malformed or misaligned."""


class EvaluationError(RuntimeError):
    """Translation or scoring failed; no valid BLEU score exists."""


class Translator(Protocol):
    """Anything that turns a batch of source sentences into translations."""

    def translate_batch(self, sources: List[str]) -> List[str]:
        ...


@dataclass(frozen=True)
class BleuConfig:
    """SacreBLEU settings (defaults = SacreBLEU defaults, recorded in results)."""

    tokenize: str = "13a"
    lowercase: bool = False
    smooth_method: str = "exp"


@dataclass(frozen=True)
class ParallelData:
    sources: List[str]
    references: List[str]


@dataclass(frozen=True)
class EvaluationResult:
    bleu_score: float
    num_sentences: int
    dataset_version: str
    evaluation_type: str
    source_language: str
    target_language: str
    batch_size: int
    bleu_config: Dict[str, Any]
    sacrebleu_signature: str
    source_path: str
    reference_path: str

    @property
    def is_synthetic(self) -> bool:
        return self.evaluation_type == "synthetic"

    def metrics(self) -> Dict[str, float]:
        """Metric names/values logged to MLflow by ``mlflow_service``."""
        return {
            METRIC_BLEU_AUTO: self.bleu_score,
            METRIC_SENTENCE_COUNT: float(self.num_sentences),
        }

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["is_synthetic"] = self.is_synthetic
        if self.is_synthetic:
            data["note"] = SYNTHETIC_NOTICE
        return data


# --------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------
def _read_lines(path: Path, label: str) -> List[str]:
    if not path.is_file():
        raise DatasetError(f"{label} file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8-sig")  # tolerates a BOM
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetError(f"Cannot read {label} file {path} as UTF-8: {exc}") from exc
    lines = text.split("\n")  # not splitlines(): it also splits on U+2028 etc.
    if lines and lines[-1] == "":
        lines.pop()  # trailing newline
    return [line[:-1] if line.endswith("\r") else line for line in lines]


def load_parallel_dataset(
    source_path: Union[str, Path], reference_path: Union[str, Path]
) -> ParallelData:
    """Load and validate aligned one-sentence-per-line files."""
    sources = _read_lines(Path(source_path), "source")
    references = _read_lines(Path(reference_path), "reference")

    if not sources and not references:
        raise DatasetError("Evaluation dataset is empty")
    if len(sources) != len(references):
        raise DatasetError(
            f"Source/reference misaligned: {len(sources)} source lines vs "
            f"{len(references)} reference lines"
        )
    for label, lines in (("source", sources), ("reference", references)):
        blank = [i + 1 for i, line in enumerate(lines) if not line.strip()]
        if blank:
            shown = ", ".join(map(str, blank[:10])) + (" ..." if len(blank) > 10 else "")
            raise DatasetError(f"Blank {label} line(s) at line number(s): {shown}")
    return ParallelData(sources=sources, references=references)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate_bleu(
    translator: Translator,
    source_path: Union[str, Path],
    reference_path: Union[str, Path],
    *,
    dataset_version: str,
    evaluation_type: str,
    source_language: str,
    target_language: str,
    batch_size: int = 32,
    bleu_config: Optional[BleuConfig] = None,
) -> EvaluationResult:
    """Translate the dataset and compute corpus BLEU.

    ``evaluation_type`` must be ``"synthetic"`` or ``"real"`` (no default, so it
    is never recorded by accident). Raises ``DatasetError`` for bad data and
    ``EvaluationError`` if translation or scoring fails.
    """
    if evaluation_type not in EVALUATION_TYPES:
        raise ValueError(f"evaluation_type must be one of {EVALUATION_TYPES}, got {evaluation_type!r}")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    config = bleu_config or BleuConfig()

    data = load_parallel_dataset(source_path, reference_path)

    predictions: List[str] = []
    for start in range(0, len(data.sources), batch_size):
        batch = data.sources[start:start + batch_size]
        try:
            output = translator.translate_batch(batch)
        except Exception as exc:
            raise EvaluationError(
                f"Translation failed on sentences {start + 1}-{start + len(batch)}: {exc}"
            ) from exc
        if not isinstance(output, (list, tuple)) or len(output) != len(batch):
            got = len(output) if hasattr(output, "__len__") else type(output).__name__
            raise EvaluationError(
                f"Translator returned {got} outputs for {len(batch)} inputs "
                f"(sentences {start + 1}-{start + len(batch)})"
            )
        if not all(isinstance(item, str) for item in output):
            raise EvaluationError("Translator must return a list of strings")
        predictions.extend(output)

    try:
        metric = BLEU(
            tokenize=config.tokenize,
            lowercase=config.lowercase,
            smooth_method=config.smooth_method,
        )
        score = metric.corpus_score(predictions, [data.references])
        signature = str(metric.get_signature())
    except Exception as exc:
        raise EvaluationError(f"BLEU computation failed: {exc}") from exc

    return EvaluationResult(
        bleu_score=float(score.score),
        num_sentences=len(data.sources),
        dataset_version=dataset_version,
        evaluation_type=evaluation_type,
        source_language=source_language,
        target_language=target_language,
        batch_size=batch_size,
        bleu_config=asdict(config),
        sacrebleu_signature=signature,
        source_path=str(source_path),
        reference_path=str(reference_path),
    )


# --------------------------------------------------------------------------
# Translator configuration
# --------------------------------------------------------------------------
def load_translator(spec: str, model_dir: Union[str, Path], device: str = "cpu") -> Translator:
    """Build a translator from a ``package.module:callable`` spec.

    The callable is invoked as ``callable(model_dir, device)`` and must return an
    object with ``translate_batch``. This is where the team plugs in the real
    tokenizer/detokenizer (e.g. a function that returns a ``CTranslate2Translator``)
    without this project guessing any preprocessing. The spec is operator-supplied
    and imported/executed as code: only use trusted specs.
    """
    module_name, sep, func_name = (spec or "").partition(":")
    if not sep or not module_name or not func_name:
        raise EvaluationError(
            f"Invalid translator spec {spec!r}; expected 'package.module:callable'"
        )
    try:
        factory = getattr(importlib.import_module(module_name), func_name)
    except (ImportError, AttributeError) as exc:
        raise EvaluationError(f"Cannot load translator {spec!r}: {exc}") from exc
    try:
        translator = factory(Path(model_dir), device)
    except Exception as exc:
        raise EvaluationError(f"Translator factory {spec!r} failed: {exc}") from exc
    if not callable(getattr(translator, "translate_batch", None)):
        raise EvaluationError(f"Translator factory {spec!r} did not return an object with translate_batch()")
    return translator


# --------------------------------------------------------------------------
# Real-model adapter (UNTESTED against a real model: see README section 13)
# --------------------------------------------------------------------------
class CTranslate2Translator:
    """Adapter around ``ctranslate2.Translator``.

    ``tokenize`` / ``detokenize`` are REQUIRED and deliberately have no default:
    they must be the exact pre/post-processing the model was trained with.
    ``translate_options`` is passed straight to ``translate_batch``
    (e.g. ``{"beam_size": 4}``). Models needing a target prefix (multilingual
    models) are not supported yet.
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        tokenize: Callable[[str], List[str]],
        detokenize: Callable[[List[str]], str],
        *,
        device: str = "cpu",
        compute_type: str = "default",
        translate_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        import ctranslate2  # lazy: only needed for real-model evaluation

        self._translator = ctranslate2.Translator(
            str(model_dir), device=device, compute_type=compute_type
        )
        self._tokenize = tokenize
        self._detokenize = detokenize
        self._options = dict(translate_options or {})

    def translate_batch(self, sources: List[str]) -> List[str]:
        results = self._translator.translate_batch(
            [self._tokenize(s) for s in sources], **self._options
        )
        return [self._detokenize(r.hypotheses[0]) for r in results]
