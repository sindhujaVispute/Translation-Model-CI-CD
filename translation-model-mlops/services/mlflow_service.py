"""MLflow tracking + model registration for CTranslate2 models.

Uses only public ``MlflowClient`` APIs, an explicit tracking URI (no global
fluent state) and never touches MLflow's backend database directly. The
production artifact/database configuration is whatever the MLflow server is
already set up with: ``artifact_location`` is only passed when explicitly
given (tests do this).

Guarantees enforced here (not just by the caller):
  * a model whose validation did not pass is never registered;
  * a duplicate checksum never creates a run or version;
  * evaluation is never skipped silently: every request must declare exactly one of
    a completed evaluation, a failed one, or an explicit skip with a reason;
  * a failed evaluation is never recorded as completed: it is refused unless
    ``allow_failed_evaluation=True``, and then tagged ``evaluation_status=failed``;
  * connection and authentication problems raise ``MLflowConnectionError`` /
    ``MLflowAuthError``, distinct from "not found".
"""
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from mlflow.entities import Metric, Param, RunTag
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from services.bleu_evaluation import (
    METRIC_BLEU_AUTO,
    METRIC_SENTENCE_COUNT,
    EvaluationResult,
)
from services.checksum import CHECKSUM_ALGORITHM
from services.duplicate_detection import (
    DuplicateResult,
    MLflowAuthError,
    MLflowConnectionError,
    classify_mlflow_exception,
    find_duplicate,
)
from services.model_validation import ValidationResult

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT_NAME = "translation-model-registration"
METRIC_BLEU_MANUAL = "bleu"  # unchanged name: existing dashboards keep working
_MAX_TAG_CHARS = 500
SKIP_TRANSLATOR_NOT_CONFIGURED = "translator_not_configured"
SKIP_NOT_REQUESTED = "evaluation_not_requested"

__all__ = [
    "MLflowAuthError", "MLflowConnectionError", "RegistrationError",
    "RegistrationRequest", "RegistrationOutcome", "resolve_tracking_uri",
    "create_client", "get_or_create_experiment", "build_params", "build_metrics",
    "build_tags", "track_and_register",
]


class RegistrationError(RuntimeError):
    """Registration was refused (bad input) or MLflow rejected a request."""


@dataclass
class RegistrationRequest:
    registered_model_name: str
    model_dir: Path
    model_folder: str
    model_sha256: str
    validation: ValidationResult
    quantization: str = "int8"
    device: str = "cpu"
    source_language: str = "eng-latn"
    target_language: str = "unknown"
    training_checkpoint: str = "unknown"
    dataset_version: str = "unknown"
    manual_bleu: Optional[float] = None
    evaluation: Optional[EvaluationResult] = None
    evaluation_error: Optional[str] = None
    allow_failed_evaluation: bool = False
    evaluation_skipped: bool = False
    evaluation_skip_reason: Optional[str] = None  # required when skipped


@dataclass(frozen=True)
class RegistrationOutcome:
    status: str  # "registered" | "duplicate_skipped"
    registered_model_name: str
    version: Optional[str]
    run_id: Optional[str]
    source_uri: Optional[str]
    model_sha256: str
    duplicate: Optional[DuplicateResult] = None


# --------------------------------------------------------------------------
# Connection
# --------------------------------------------------------------------------
def resolve_tracking_uri(explicit: Optional[str] = None) -> str:
    """explicit argument > MLFLOW_TRACKING_URI env var > original local default."""
    return explicit or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI


def create_client(tracking_uri: Optional[str] = None) -> MlflowClient:
    """Client bound to an explicit URI (credentials, if any, come from MLflow's
    own MLFLOW_TRACKING_* environment variables; they are never read here)."""
    return MlflowClient(tracking_uri=resolve_tracking_uri(tracking_uri))


def _translate(exc: BaseException, action: str) -> Exception:
    mapped = classify_mlflow_exception(exc, action)
    if mapped is not None:
        return mapped
    return RegistrationError(
        f"MLflow rejected the request while {action} "
        f"({getattr(exc, 'error_code', type(exc).__name__)}): {exc}"
    )


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------
def get_or_create_experiment(
    client: Any, name: str, artifact_location: Optional[str] = None
) -> str:
    """Return the experiment id, creating the experiment if needed."""
    try:
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            try:
                return str(client.create_experiment(name, artifact_location=artifact_location))
            except MlflowException as exc:
                if getattr(exc, "error_code", None) != "RESOURCE_ALREADY_EXISTS":
                    raise
                experiment = client.get_experiment_by_name(name)  # lost a creation race
                if experiment is None:
                    raise
    except (MlflowException, OSError) as exc:
        raise _translate(exc, f"setting up experiment {name!r}") from exc
    if getattr(experiment, "lifecycle_stage", "active") == "deleted":
        raise RegistrationError(
            f"Experiment {name!r} is in the deleted state; restore it or choose another name"
        )
    return str(experiment.experiment_id)


# --------------------------------------------------------------------------
# What gets logged (pure functions, easy to test)
# --------------------------------------------------------------------------
def _evaluation_type(request: RegistrationRequest) -> str:
    """synthetic | automatic | manual | none. A manually supplied BLEU is only
    ever "manual"; it is never presented as an automatic result."""
    if request.evaluation is not None:
        return "synthetic" if request.evaluation.evaluation_type == "synthetic" else "automatic"
    return "manual" if request.manual_bleu is not None else "none"


def _evaluation_status(request: RegistrationRequest) -> str:
    if request.evaluation is not None:
        return "completed"
    if request.evaluation_error:
        return "failed"
    if request.evaluation_skipped:
        return "skipped"
    return "undeclared"


def _bleu_source(request: RegistrationRequest) -> str:
    manual, auto = request.manual_bleu is not None, request.evaluation is not None
    return {(True, True): "manual+automatic", (True, False): "manual",
            (False, True): "automatic", (False, False): "none"}[(manual, auto)]


def build_params(request: RegistrationRequest) -> Dict[str, str]:
    params = {
        "model_name": request.registered_model_name,
        "model_folder": request.model_folder,
        "training_framework": "OpenNMT",
        "model_format": "ctranslate2",
        "serving_runtime": "CTranslate2",
        "quantization": request.quantization,
        "device": request.device,
        "source_language": request.source_language,
        "target_language": request.target_language,
        "training_checkpoint": request.training_checkpoint,
        "dataset_version": request.dataset_version,
    }
    ev = request.evaluation
    if ev is not None:
        params.update({
            "eval_dataset_version": ev.dataset_version,
            "eval_batch_size": ev.batch_size,
            "eval_bleu_tokenize": ev.bleu_config["tokenize"],
            "eval_bleu_lowercase": ev.bleu_config["lowercase"],
            "eval_bleu_smooth_method": ev.bleu_config["smooth_method"],
        })
    return {k: str(v) for k, v in params.items()}


def build_metrics(request: RegistrationRequest) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if request.manual_bleu is not None:
        metrics[METRIC_BLEU_MANUAL] = float(request.manual_bleu)
    if request.evaluation is not None:
        metrics.update(request.evaluation.metrics())
    return metrics


def build_tags(request: RegistrationRequest, registration_status: str) -> Dict[str, str]:
    tags = {
        "model_type": "translation",
        "artifact_type": "ctranslate2_directory",
        "model_sha256": request.model_sha256,
        "model_sha256_algorithm": CHECKSUM_ALGORITHM,
        "evaluation_type": _evaluation_type(request),
        "evaluation_status": _evaluation_status(request),
        "bleu_source": _bleu_source(request),
        "registration_status": registration_status,
    }
    if request.evaluation is not None:
        tags["sacrebleu_signature"] = request.evaluation.sacrebleu_signature[:_MAX_TAG_CHARS]
    if request.evaluation_error:
        tags["evaluation_error"] = request.evaluation_error[:_MAX_TAG_CHARS]
    if request.evaluation_skipped:
        tags["evaluation_skip_reason"] = (request.evaluation_skip_reason or "")[:_MAX_TAG_CHARS]
    return tags


def _check_request(request: RegistrationRequest) -> None:
    if not request.validation.passed:
        raise RegistrationError(
            f"Refusing to register {request.registered_model_name!r}: model validation did not pass"
        )
    declared = sum([request.evaluation is not None, bool(request.evaluation_error),
                    request.evaluation_skipped])
    if declared > 1:
        raise ValueError(
            "Declare exactly one of `evaluation`, `evaluation_error`, `evaluation_skipped`"
        )
    if declared == 0:
        raise RegistrationError(
            "Evaluation status not declared: provide an evaluation result, or explicitly "
            "skip it with evaluation_skipped=True and a reason"
        )
    if request.evaluation_skipped and not request.evaluation_skip_reason:
        raise RegistrationError("A skipped evaluation must record a reason (evaluation_skip_reason)")
    if request.evaluation_error and not request.allow_failed_evaluation:
        raise RegistrationError(
            "Refusing to register: BLEU evaluation failed "
            f"({request.evaluation_error}). Fix the evaluation, or set "
            "allow_failed_evaluation=True to register it tagged evaluation_status=failed."
        )
    bleu = request.manual_bleu
    if bleu is not None and (not math.isfinite(bleu) or not 0.0 <= bleu <= 100.0):
        raise RegistrationError(f"Manual BLEU must be a finite number in [0, 100], got {bleu!r}")


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------
def _ensure_registered_model(client: Any, name: str) -> None:
    try:
        client.get_registered_model(name)
        logger.info("Registered model already exists: %s", name)
        return
    except MlflowException as exc:
        if getattr(exc, "error_code", None) != "RESOURCE_DOES_NOT_EXIST":
            raise
    logger.info("Creating registered model: %s", name)
    try:
        client.create_registered_model(
            name=name,
            description="CTranslate2 translation model originally trained using OpenNMT.",
        )
    except MlflowException as exc:
        if getattr(exc, "error_code", None) != "RESOURCE_ALREADY_EXISTS":
            raise  # already existing (created concurrently) is fine


def _wait_until_ready(client: Any, name: str, version: str, timeout: float, poll_interval: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        status = str(client.get_model_version(name, version).status)
        if status == "READY":
            return
        if status == "FAILED_REGISTRATION":
            raise RegistrationError(f"Model version {name} v{version} failed registration")
        if time.monotonic() >= deadline:
            raise RegistrationError(
                f"Model version {name} v{version} not READY after {timeout:.0f}s (status {status})"
            )
        time.sleep(poll_interval)


def track_and_register(
    client: Any,
    request: RegistrationRequest,
    *,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    artifact_location: Optional[str] = None,
    wait_timeout: float = 300.0,
    poll_interval: float = 1.0,
) -> RegistrationOutcome:
    """Log params/metrics/tags/artifacts and register a new model version.

    Raises ``RegistrationError`` (refused / rejected), ``MLflowConnectionError``
    or ``MLflowAuthError``. Returns ``status="duplicate_skipped"`` (creating
    nothing) when the checksum is already registered.
    """
    _check_request(request)

    duplicate = find_duplicate(client, request.model_sha256, request.registered_model_name)
    if duplicate.is_duplicate:
        logger.warning("Duplicate of %s v%s; nothing created", duplicate.model_name, duplicate.version)
        return RegistrationOutcome(
            status="duplicate_skipped",
            registered_model_name=request.registered_model_name,
            version=duplicate.version, run_id=None, source_uri=None,
            model_sha256=request.model_sha256, duplicate=duplicate,
        )

    experiment_id = get_or_create_experiment(client, experiment_name, artifact_location)

    try:
        run = client.create_run(
            experiment_id,
            tags=build_tags(request, "pending"),
            run_name=f"register-{request.model_folder}",
        )
    except (MlflowException, OSError) as exc:
        raise _translate(exc, "creating the run") from exc
    run_id = run.info.run_id
    logger.info("MLflow run: %s", run_id)

    version_number: Optional[str] = None
    try:
        now = int(time.time() * 1000)
        client.log_batch(
            run_id,
            metrics=[Metric(k, v, now, 0) for k, v in build_metrics(request).items()],
            params=[Param(k, v) for k, v in build_params(request).items()],
        )
        if request.evaluation is not None:
            client.log_dict(run_id, request.evaluation.to_dict(), "evaluation/evaluation_result.json")

        logger.info("Uploading model directory to MLflow...")
        client.log_artifacts(run_id, str(request.model_dir), artifact_path="model")
        source_uri = f"runs:/{run_id}/model"

        _ensure_registered_model(client, request.registered_model_name)
        version = client.create_model_version(
            name=request.registered_model_name,
            source=source_uri,
            run_id=run_id,
            tags={
                # "pending" until the version is confirmed READY; flipped below.
                **build_tags(request, "pending"),
                "quantization": request.quantization,
                "runtime": "CTranslate2",
            },
            description=f"CTranslate2 artifact for {request.model_folder}",
        )
        version_number = str(version.version)
        _wait_until_ready(client, request.registered_model_name, version_number,
                          wait_timeout, poll_interval)

        client.set_model_version_tag(request.registered_model_name, version_number,
                                     "registration_status", "registered")
        client.set_tag(run_id, "registration_status", "registered")
        client.set_terminated(run_id, "FINISHED")
    except BaseException as exc:
        _mark_failed(client, run_id, request.registered_model_name, version_number)
        if isinstance(exc, (MlflowException, OSError)):
            raise _translate(exc, "tracking/registering the model") from exc
        raise

    return RegistrationOutcome(
        status="registered",
        registered_model_name=request.registered_model_name,
        version=version_number, run_id=run_id, source_uri=source_uri,
        model_sha256=request.model_sha256,
    )


def _mark_failed(client: Any, run_id: str, model_name: str, version: Optional[str]) -> None:
    """Best effort: mark the run FAILED and, if a version was already created, tag it
    ``registration_status=failed`` (duplicate detection ignores such versions).
    Never hides the original error behind a cleanup error."""
    if version is not None:
        try:
            client.set_model_version_tag(model_name, version, "registration_status", "failed")
        except Exception as cleanup_exc:  # noqa: BLE001
            logger.warning("Could not tag %s v%s as failed: %s", model_name, version, cleanup_exc)
    try:
        client.set_tag(run_id, "registration_status", "failed")
        client.set_terminated(run_id, "FAILED")
    except Exception as cleanup_exc:  # noqa: BLE001
        logger.warning("Could not mark run %s as failed: %s", run_id, cleanup_exc)
