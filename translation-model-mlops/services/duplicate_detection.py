"""Detect already-registered model artifacts via the ``model_sha256`` tag.

Only public MlflowClient APIs are used; MLflow's backend DB is never touched.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

SHA256_TAG = "model_sha256"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NOT_FOUND = "RESOURCE_DOES_NOT_EXIST"
_PAGE_SIZE = 1000


class MLflowConnectionError(RuntimeError):
    """MLflow could not be reached or failed server-side (network / server
    error). Distinct from 'model not found', which is a normal outcome."""


class MLflowAuthError(MLflowConnectionError):
    """MLflow rejected the credentials or the permissions (401/403)."""


_AUTH_CODES = ("UNAUTHENTICATED", "PERMISSION_DENIED")
_CONNECTION_CODES = ("INTERNAL_ERROR", "TEMPORARILY_UNAVAILABLE")


def classify_mlflow_exception(exc: BaseException, action: str) -> Optional[MLflowConnectionError]:
    """Map a raw MLflow/OS error to MLflowAuthError / MLflowConnectionError, or
    ``None`` when it is neither (e.g. invalid parameter, already exists)."""
    code = getattr(exc, "error_code", None)
    if code in _AUTH_CODES:
        return MLflowAuthError(
            f"MLflow rejected the credentials/permissions while {action} ({code}): {exc}. "
            "Check the MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD or "
            "MLFLOW_TRACKING_TOKEN environment variables."
        )
    if code in _CONNECTION_CODES:
        return MLflowConnectionError(f"MLflow request failed while {action} ({code}): {exc}")
    if isinstance(exc, OSError):  # requests' connection/timeout errors derive from IOError
        return MLflowConnectionError(f"Could not reach MLflow server while {action}: {exc}")
    return None


@dataclass(frozen=True)
class DuplicateResult:
    is_duplicate: bool
    model_name: Optional[str] = None
    version: Optional[str] = None
    all_matches: List[Tuple[str, str]] = field(default_factory=list)
    versions_checked: int = 0
    failed_versions_ignored: int = 0


def _registration_failed(version: Any, tags: dict) -> bool:
    """A version that never became usable must not block re-registering the same
    artifact. PENDING versions still count (another registrar may be mid-flight)."""
    return (str(getattr(version, "status", "")) == "FAILED_REGISTRATION"
            or tags.get("registration_status") == "failed")


def _version_sort_key(version: str) -> Tuple[int, Any]:
    return (0, int(version)) if str(version).isdigit() else (1, str(version))


def find_duplicate(
    client: Any,
    model_sha256: str,
    registered_model_name: Optional[str] = None,
) -> DuplicateResult:
    """Look for a registered version whose ``model_sha256`` tag equals
    ``model_sha256``.

    ``registered_model_name`` given (default usage): only that model's versions
    are compared. ``None``: every registered model is scanned.

    Versions whose registration failed (``FAILED_REGISTRATION`` status or
    ``registration_status=failed`` tag) are ignored; pending ones still count.

    Raises ``MLflowConnectionError`` on connectivity/auth/server problems. A
    registered model that does not exist yet simply yields "no duplicate".
    """
    wanted = model_sha256.strip().lower()
    if not _SHA256_RE.match(wanted):
        raise ValueError(f"Not a valid SHA256 hex digest: {model_sha256!r}")

    filter_string = None
    if registered_model_name is not None:
        if "'" in registered_model_name or "\\" in registered_model_name:
            raise ValueError("Registered model name may not contain quotes or backslashes")
        filter_string = f"name = '{registered_model_name}'"

    matches: List[Tuple[str, str]] = []
    checked = 0
    ignored_failed = 0
    page_token = None
    try:
        while True:
            page = client.search_model_versions(
                filter_string=filter_string,
                max_results=_PAGE_SIZE,
                page_token=page_token,
            )
            for mv in page:
                checked += 1
                tags = mv.tags or {}
                tag = tags.get(SHA256_TAG)
                if not (tag and tag.strip().lower() == wanted):
                    continue
                if _registration_failed(mv, tags):
                    ignored_failed += 1
                    logger.info("Ignoring failed registration %s v%s with same checksum",
                                mv.name, mv.version)
                    continue
                matches.append((mv.name, str(mv.version)))
            page_token = getattr(page, "token", None)
            if not page_token:
                break
    except MlflowException as exc:
        if getattr(exc, "error_code", None) == _NOT_FOUND:
            logger.info("Registered model %r not found; no duplicates", registered_model_name)
            return DuplicateResult(is_duplicate=False, versions_checked=checked)
        raise (
            classify_mlflow_exception(exc, "checking for duplicates")
            or MLflowConnectionError(
                f"MLflow request failed ({getattr(exc, 'error_code', 'UNKNOWN')}): {exc}"
            )
        ) from exc
    except OSError as exc:
        raise classify_mlflow_exception(exc, "checking for duplicates") from exc

    if not matches:
        return DuplicateResult(is_duplicate=False, versions_checked=checked,
                               failed_versions_ignored=ignored_failed)

    matches.sort(key=lambda m: (m[0], _version_sort_key(m[1])))
    name, version = matches[0]
    return DuplicateResult(
        is_duplicate=True,
        model_name=name,
        version=version,
        all_matches=matches,
        versions_checked=checked,
        failed_versions_ignored=ignored_failed,
    )
