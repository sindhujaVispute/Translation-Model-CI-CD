import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    PERMISSION_DENIED,
    RESOURCE_DOES_NOT_EXIST,
    UNAUTHENTICATED,
)

from services.duplicate_detection import (
    MLflowAuthError,
    MLflowConnectionError,
    find_duplicate,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


class FakeVersion:
    def __init__(self, name, version, sha=None, tags=None, status=None):
        self.name = name
        self.version = str(version)
        if status is not None:
            self.status = status
        self.tags = tags if tags is not None else ({"model_sha256": sha} if sha else {})


class Page(list):
    token = None


class FakeClient:
    """Stand-in for MlflowClient.search_model_versions (no server needed)."""

    def __init__(self, versions=(), error=None, page_size=None):
        self.versions = list(versions)
        self.error = error
        self.page_size = page_size
        self.filters = []

    def search_model_versions(self, filter_string=None, max_results=1000, order_by=None, page_token=None):
        self.filters.append(filter_string)
        if self.error:
            raise self.error
        items = self.versions
        if filter_string:
            wanted = filter_string.split("'")[1]
            items = [v for v in items if v.name == wanted]
        start = int(page_token or 0)
        size = self.page_size or len(items) or 1
        page = Page(items[start:start + size])
        page.token = str(start + size) if start + size < len(items) else None
        return page


def test_new_checksum_is_not_duplicate():
    client = FakeClient([FakeVersion("m", 1, SHA_A)])
    result = find_duplicate(client, SHA_B, "m")
    assert not result.is_duplicate
    assert result.model_name is None and result.version is None
    assert result.versions_checked == 1


def test_existing_checksum_returns_name_and_version():
    client = FakeClient([FakeVersion("m", 1, SHA_A), FakeVersion("m", 2, SHA_B)])
    result = find_duplicate(client, SHA_B, "m")
    assert result.is_duplicate
    assert (result.model_name, result.version) == ("m", "2")


def test_changed_model_artifact_is_not_duplicate_of_old_version():
    client = FakeClient([FakeVersion("m", 1, SHA_A)])
    assert find_duplicate(client, SHA_A, "m").is_duplicate
    assert not find_duplicate(client, SHA_C, "m").is_duplicate


def test_comparison_is_case_and_whitespace_insensitive():
    client = FakeClient([FakeVersion("m", 1, SHA_A.upper() + " ")])
    assert find_duplicate(client, SHA_A, "m").is_duplicate


def test_default_scope_is_same_registered_model_only():
    client = FakeClient([FakeVersion("other", 1, SHA_A)])
    result = find_duplicate(client, SHA_A, "m")
    assert not result.is_duplicate
    assert client.filters == ["name = 'm'"]


def test_scope_all_models_when_name_is_none():
    client = FakeClient([FakeVersion("other", 3, SHA_A)])
    result = find_duplicate(client, SHA_A, None)
    assert result.is_duplicate and (result.model_name, result.version) == ("other", "3")
    assert client.filters == [None]


def test_versions_without_tag_are_ignored():
    client = FakeClient([FakeVersion("m", 1, tags={}), FakeVersion("m", 2, tags={"x": "y"})])
    assert not find_duplicate(client, SHA_A, "m").is_duplicate


def test_multiple_matches_lowest_version_first_numeric_order():
    client = FakeClient([FakeVersion("m", 10, SHA_A), FakeVersion("m", 2, SHA_A)])
    result = find_duplicate(client, SHA_A, "m")
    assert result.version == "2"
    assert result.all_matches == [("m", "2"), ("m", "10")]


def test_pagination_is_followed():
    versions = [FakeVersion("m", i, SHA_B) for i in range(1, 6)] + [FakeVersion("m", 6, SHA_A)]
    client = FakeClient(versions, page_size=2)
    result = find_duplicate(client, SHA_A, "m")
    assert result.is_duplicate and result.version == "6"
    assert result.versions_checked == 6
    assert len(client.filters) == 3


def test_empty_registry_is_not_duplicate():
    assert not find_duplicate(FakeClient([]), SHA_A, "m").is_duplicate


def test_model_not_found_error_means_no_duplicate():
    err = MlflowException("no such model", error_code=RESOURCE_DOES_NOT_EXIST)
    result = find_duplicate(FakeClient(error=err), SHA_A, "m")
    assert not result.is_duplicate


def test_server_unavailable_raises_connection_error():
    err = MlflowException("API request to http://x/api failed with exception ConnectionError", error_code=INTERNAL_ERROR)
    with pytest.raises(MLflowConnectionError):
        find_duplicate(FakeClient(error=err), SHA_A, "m")


def test_os_level_connection_error_raises_connection_error():
    with pytest.raises(MLflowConnectionError, match="Could not reach"):
        find_duplicate(FakeClient(error=ConnectionError("refused")), SHA_A, "m")


@pytest.mark.parametrize("code", [UNAUTHENTICATED, PERMISSION_DENIED])
def test_auth_errors_are_not_treated_as_not_found(code):
    err = MlflowException("denied", error_code=code)
    with pytest.raises(MLflowAuthError, match="MLFLOW_TRACKING"):
        find_duplicate(FakeClient(error=err), SHA_A, "m")


def test_plain_connection_error_is_not_auth_error():
    err = MlflowException("down", error_code=INTERNAL_ERROR)
    with pytest.raises(MLflowConnectionError) as info:
        find_duplicate(FakeClient(error=err), SHA_A, "m")
    assert not isinstance(info.value, MLflowAuthError)


@pytest.mark.parametrize("bad", ["", "xyz", "a" * 63, "g" * 64])
def test_invalid_checksum_rejected(bad):
    with pytest.raises(ValueError):
        find_duplicate(FakeClient(), bad, "m")


@pytest.mark.parametrize("name", ["it's", "a\\b"])
def test_unsafe_model_name_rejected(name):
    with pytest.raises(ValueError):
        find_duplicate(FakeClient(), SHA_A, name)


# ---- failed registrations must not block re-registration ---------------------
def test_failed_registration_status_is_not_a_duplicate():
    client = FakeClient([FakeVersion("m", 1, SHA_A, status="FAILED_REGISTRATION")])
    result = find_duplicate(client, SHA_A, "m")
    assert not result.is_duplicate and result.failed_versions_ignored == 1


def test_registration_status_failed_tag_is_not_a_duplicate():
    tags = {"model_sha256": SHA_A, "registration_status": "failed"}
    result = find_duplicate(FakeClient([FakeVersion("m", 1, tags=tags)]), SHA_A, "m")
    assert not result.is_duplicate and result.failed_versions_ignored == 1


def test_ready_version_with_same_checksum_is_still_a_duplicate():
    tags = {"model_sha256": SHA_A, "registration_status": "registered"}
    client = FakeClient([FakeVersion("m", 1, tags=tags, status="READY")])
    assert find_duplicate(client, SHA_A, "m").is_duplicate


def test_pending_version_still_counts_as_duplicate():
    # another registrar may be mid-flight: be conservative, do not double-register
    tags = {"model_sha256": SHA_A, "registration_status": "pending"}
    client = FakeClient([FakeVersion("m", 1, tags=tags, status="PENDING_REGISTRATION")])
    assert find_duplicate(client, SHA_A, "m").is_duplicate


def test_legacy_version_without_status_or_registration_tag_still_counts():
    assert find_duplicate(FakeClient([FakeVersion("m", 1, SHA_A)]), SHA_A, "m").is_duplicate


def test_failed_version_ignored_but_later_good_version_found():
    client = FakeClient([
        FakeVersion("m", 1, SHA_A, status="FAILED_REGISTRATION"),
        FakeVersion("m", 2, SHA_A, status="READY"),
    ])
    result = find_duplicate(client, SHA_A, "m")
    assert result.is_duplicate and result.version == "2"
    assert result.all_matches == [("m", "2")] and result.failed_versions_ignored == 1
