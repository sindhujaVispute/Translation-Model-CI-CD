"""Validate, evaluate and register a converted CTranslate2 model in MLflow.

Pipeline:
  validate -> preflight evaluation config -> SHA256 -> duplicate check
  -> BLEU evaluation (or explicit skip) -> MLflow tracking -> registration -> summary

Evaluation is never skipped silently. Exactly one of these happens:
  * automatic evaluation (needs --translator, dataset paths and --evaluation-type)
  * --skip-evaluation, recorded in MLflow with a reason
  * a clear error (exit 4) and nothing is registered

Exit codes: 0 ok / duplicate skipped, 2 validation failed (or bad CLI usage),
3 MLflow unreachable / auth, 4 evaluation misconfigured or failed, 5 registration refused/rejected.
"""
import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from services.bleu_evaluation import (
    DatasetError,
    EvaluationError,
    EvaluationResult,
    evaluate_bleu,
    load_translator,
)
from services.checksum import CHECKSUM_ALGORITHM, calculate_directory_sha256
from services.duplicate_detection import MLflowConnectionError, find_duplicate
from services.mlflow_service import (
    DEFAULT_EXPERIMENT_NAME,
    SKIP_NOT_REQUESTED,
    SKIP_TRANSLATOR_NOT_CONFIGURED,
    RegistrationError,
    RegistrationRequest,
    create_client,
    track_and_register,
)
from services.model_validation import validate_model_directory

EXIT_OK = 0
EXIT_VALIDATION_FAILED = 2
EXIT_MLFLOW_UNAVAILABLE = 3
EXIT_EVALUATION = 4
EXIT_REGISTRATION = 5

logger = logging.getLogger("register_ctranslate2_model")


class EvaluationConfigError(ValueError):
    """Automatic evaluation was requested/required but is not fully configured."""


def _bleu_value(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}")
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise argparse.ArgumentTypeError("BLEU must be a finite number between 0 and 100")
    return value


def build_parser() -> argparse.ArgumentParser:
    env = os.environ.get
    parser = argparse.ArgumentParser(
        description="Validate, evaluate and register a CTranslate2 model in MLflow."
    )
    # --- original arguments (unchanged) ---
    parser.add_argument("--model-dir", required=True,
                        help="Path to converted CTranslate2 model directory")
    parser.add_argument("--registered-model-name", default=env("MODEL_REGISTRY_NAME"),
                        required=not env("MODEL_REGISTRY_NAME"),
                        help="Defaults to $MODEL_REGISTRY_NAME when set")
    parser.add_argument("--model-folder", required=True)
    parser.add_argument("--quantization", default="int8")
    parser.add_argument("--source-language", default="eng-latn")
    parser.add_argument("--training-checkpoint", default="unknown")
    parser.add_argument("--dataset-version", default="unknown",
                        help="Version of the TRAINING dataset")
    parser.add_argument("--bleu", type=_bleu_value, default=None,
                        help="Manually supplied BLEU; logged as 'bleu', never as automatic")
    # --- new ---
    parser.add_argument("--target-language", default="unknown")
    parser.add_argument("--device", default="cpu")
    group = parser.add_argument_group("evaluation")
    group.add_argument("--skip-evaluation", action="store_true", default=False,
                       help="Explicitly skip automatic BLEU evaluation (recorded in MLflow)")
    group.add_argument("--translator", default=env("TRANSLATOR_FACTORY") or None,
                       metavar="MODULE:CALLABLE",
                       help="callable(model_dir, device) -> object with translate_batch(); "
                            "supplies the real tokenizer/detokenizer. Defaults to $TRANSLATOR_FACTORY")
    group.add_argument("--evaluation-source", help="Source sentences file (one per line)")
    group.add_argument("--evaluation-reference", help="Reference translations file")
    group.add_argument("--evaluation-type", choices=["synthetic", "real"], default=None,
                       help="Required for automatic evaluation: is the data synthetic or real?")
    group.add_argument("--evaluation-dataset-version",
                       default=env("EVALUATION_DATASET_VERSION", "unknown"))
    group.add_argument("--evaluation-batch-size", type=int, default=32)
    return parser


@dataclass
class EvaluationPlan:
    skip: bool
    skip_reason: Optional[str] = None


def plan_evaluation(args: argparse.Namespace) -> EvaluationPlan:
    """Decide, before touching MLflow, whether evaluation runs or is explicitly skipped."""
    if args.skip_evaluation:
        reason = SKIP_NOT_REQUESTED if args.translator else SKIP_TRANSLATOR_NOT_CONFIGURED
        return EvaluationPlan(skip=True, skip_reason=reason)

    if not args.translator:
        raise EvaluationConfigError(
            "No translator configured, so BLEU cannot be calculated. Provide --translator "
            "MODULE:CALLABLE (or set TRANSLATOR_FACTORY), or pass --skip-evaluation to register "
            "without evaluation."
        )
    missing = [flag for flag, value in (
        ("--evaluation-source", args.evaluation_source),
        ("--evaluation-reference", args.evaluation_reference),
        ("--evaluation-type", args.evaluation_type),
    ) if not value]
    if missing:
        raise EvaluationConfigError(
            "Automatic evaluation requires " + ", ".join(missing)
            + " (or pass --skip-evaluation)."
        )
    if args.evaluation_batch_size < 1:
        raise EvaluationConfigError("--evaluation-batch-size must be >= 1")
    return EvaluationPlan(skip=False)


def _print_summary(request: RegistrationRequest, outcome: Any) -> None:
    ev = request.evaluation
    print()
    print("======================================")
    print("MODEL REGISTRATION SUCCESSFUL")
    print("======================================")
    print(f"Registered Model : {outcome.registered_model_name}")
    print(f"Model Version    : {outcome.version}")
    print(f"Run ID           : {outcome.run_id}")
    print(f"Artifact URI     : {outcome.source_uri}")
    print(f"SHA256           : {outcome.model_sha256}")
    if ev is not None:
        label = " (SYNTHETIC - not real model quality)" if ev.is_synthetic else ""
        print(f"Evaluation       : completed, {ev.evaluation_type}, {ev.num_sentences} sentences")
        print(f"BLEU (automatic) : {ev.bleu_score:.2f}{label}")
    else:
        print(f"Evaluation       : SKIPPED ({request.evaluation_skip_reason})")
    if request.manual_bleu is not None:
        print(f"BLEU (manual)    : {request.manual_bleu:.2f}  [manually supplied]")
    print("======================================")


def main(argv=None, *, client: Any = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    model_dir = Path(args.model_dir).resolve()

    # 1. Validation ---------------------------------------------------------
    logger.info("Model directory: %s", model_dir)
    validation = validate_model_directory(model_dir, model_name=args.model_folder)
    for warning in validation.warnings:
        logger.warning(warning)
    if not validation.passed:
        logger.error("Model validation FAILED; nothing was registered.")
        for missing in validation.missing_files:
            logger.error("  missing: %s", missing)
        for error in validation.validation_errors:
            logger.error("  %s", error)
        return EXIT_VALIDATION_FAILED
    logger.info("Model validation passed.")

    # 2. Evaluation preflight (fail fast, before any network call) ----------
    try:
        plan = plan_evaluation(args)
    except EvaluationConfigError as exc:
        logger.error("Evaluation not configured: %s", exc)
        logger.error("Nothing was registered.")
        return EXIT_EVALUATION
    if plan.skip:
        logger.warning("Evaluation SKIPPED (%s); the model will be registered as unevaluated.",
                       plan.skip_reason)

    # 3. Checksum -----------------------------------------------------------
    logger.info("Calculating model directory SHA256 (%s)...", CHECKSUM_ALGORITHM)
    model_sha256 = calculate_directory_sha256(model_dir)
    logger.info("Model SHA256: %s", model_sha256)

    # 4. Duplicate detection (before the expensive evaluation) ---------------
    client = client or create_client()
    try:
        duplicate = find_duplicate(client, model_sha256, args.registered_model_name)
    except MLflowConnectionError as exc:
        logger.error("MLflow unavailable: %s", exc)
        return EXIT_MLFLOW_UNAVAILABLE
    if duplicate.is_duplicate:
        logger.warning("Duplicate artifact: identical to %s version %s. Registration skipped.",
                       duplicate.model_name, duplicate.version)
        return EXIT_OK

    # 5. Evaluation ---------------------------------------------------------
    evaluation: Optional[EvaluationResult] = None
    if not plan.skip:
        try:
            translator = load_translator(args.translator, model_dir, args.device)
            evaluation = evaluate_bleu(
                translator, args.evaluation_source, args.evaluation_reference,
                dataset_version=args.evaluation_dataset_version,
                evaluation_type=args.evaluation_type,
                source_language=args.source_language,
                target_language=args.target_language,
                batch_size=args.evaluation_batch_size,
            )
        except (EvaluationError, DatasetError) as exc:
            logger.error("BLEU evaluation FAILED: %s", exc)
            logger.error("Nothing was registered; the model was NOT recorded as evaluated.")
            return EXIT_EVALUATION
        logger.info("BLEU %.2f over %d sentences (%s evaluation)",
                    evaluation.bleu_score, evaluation.num_sentences, evaluation.evaluation_type)
        if evaluation.is_synthetic:
            logger.warning("Synthetic evaluation: score does not represent real model performance.")

    # 6. Tracking + registration -------------------------------------------
    request = RegistrationRequest(
        registered_model_name=args.registered_model_name,
        model_dir=model_dir,
        model_folder=args.model_folder,
        model_sha256=model_sha256,
        validation=validation,
        quantization=args.quantization,
        device=args.device,
        source_language=args.source_language,
        target_language=args.target_language,
        training_checkpoint=args.training_checkpoint,
        dataset_version=args.dataset_version,
        manual_bleu=args.bleu,
        evaluation=evaluation,
        evaluation_skipped=plan.skip,
        evaluation_skip_reason=plan.skip_reason,
    )
    try:
        outcome = track_and_register(
            client, request,
            experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT_NAME,
        )
    except MLflowConnectionError as exc:
        logger.error("MLflow unavailable during registration: %s", exc)
        return EXIT_MLFLOW_UNAVAILABLE
    except RegistrationError as exc:
        logger.error("Registration failed: %s", exc)
        return EXIT_REGISTRATION

    if outcome.status == "duplicate_skipped":  # lost a race with another registrar
        logger.warning("Duplicate of version %s detected at registration time; skipped.", outcome.version)
        return EXIT_OK
    _print_summary(request, outcome)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
