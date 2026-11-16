"""Unit tests use a fake client (no server). Integration tests at the bottom run
against a throw-away SQLite-backed MLflow store in tmp_path and are skipped when
mlflow is not installed. Nothing here ever reads MLFLOW_TRACKING_URI or touches a
production server."""
import json
from types import SimpleNamespace

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    PERMISSION_DENIED,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)

from tests.fakes import FakeClient
from services.bleu_evaluation import BleuConfig, EvaluationResult
from services.checksum import calculate_directory_sha256
from services.duplicate_detection import MLflowAuthError, MLflowConnectionError
from services.mlflow_service import (
    RegistrationError,
    RegistrationRequest,
    build_metrics,
    build_params,
    build_tags,
    get_or_create_experiment,
    resolve_tracking_uri,
    track_and_register,
)
from services.model_validation import validate_model_directory

SHA = "a" * 64


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "model.bin").write_bytes(b"DUMMY")
    (d / "shared_vocabulary.json").write_text('["a"]')
    return d


def make_eval(kind="synthetic"):
    return EvaluationResult(
        bleu_score=42.5, num_sentences=20, dataset_version="eval-v1", evaluation_type=kind,
        source_language="eng-latn", target_language="spa-latn", batch_size=8,
        bleu_config={"tokenize": "13a", "lowercase": False, "smooth_method": "exp"},
        sacrebleu_signature="nrefs:1|tok:13a", source_path="s", reference_path="r",
    )


def make_request(model_dir, **overrides):
    base = dict(
        registered_model_name="tm", model_dir=model_dir, model_folder="folder",
        model_sha256=SHA, validation=validate_model_directory(model_dir),
        target_language="spa-latn", manual_bleu=None, evaluation=None,
    )
    base.update(overrides)
    declares = any(k in overrides for k in ("evaluation", "evaluation_error", "evaluation_skipped"))
    if not declares:
        base.update(evaluation_skipped=True, evaluation_skip_reason="evaluation_not_requested")
    return RegistrationRequest(**base)


# --------------------------------------------------------------- experiment
def test_experiment_created_when_missing():
    c = FakeClient()
    assert get_or_create_experiment(c, "exp", artifact_location="file:///x") == "7"
    assert c.arg("create_experiment")[2] == {"artifact_location": "file:///x"}


def test_experiment_reused_when_present():
    c = FakeClient()
    c.experiments["exp"] = SimpleNamespace(experiment_id="3", lifecycle_stage="active")
    assert get_or_create_experiment(c, "exp") == "3"
    assert "create_experiment" not in c.names()


def test_experiment_creation_race_falls_back_to_existing():
    c = FakeClient()
    seen = {"n": 0}
    def get(name):
        seen["n"] += 1
        return None if seen["n"] == 1 else SimpleNamespace(experiment_id="9", lifecycle_stage="active")

    c.get_experiment_by_name = get
    c.fail["create_experiment"] = MlflowException("exists", error_code=RESOURCE_ALREADY_EXISTS)
    assert get_or_create_experiment(c, "exp") == "9"


def test_deleted_experiment_is_reported():
    c = FakeClient()
    c.experiments["exp"] = SimpleNamespace(experiment_id="3", lifecycle_stage="deleted")
    with pytest.raises(RegistrationError, match="deleted"):
        get_or_create_experiment(c, "exp")


def test_experiment_connection_error():
    c = FakeClient()
    c.fail["get_experiment_by_name"] = MlflowException("down", error_code=INTERNAL_ERROR)
    with pytest.raises(MLflowConnectionError):
        get_or_create_experiment(c, "exp")


# --------------------------------------------------- what gets logged (pure)
def test_params_cover_required_fields(model_dir):
    p = build_params(make_request(model_dir, quantization="int8", training_checkpoint="step_1000",
                                  dataset_version="train-v2"))
    assert p["model_name"] == "tm" and p["training_framework"] == "OpenNMT"
    assert p["model_format"] == "ctranslate2" and p["serving_runtime"] == "CTranslate2"
    assert p["quantization"] == "int8" and p["device"] == "cpu"
    assert p["source_language"] == "eng-latn" and p["target_language"] == "spa-latn"
    assert p["training_checkpoint"] == "step_1000" and p["dataset_version"] == "train-v2"
    assert all(isinstance(v, str) for v in p.values())


def test_params_include_evaluation_config(model_dir):
    p = build_params(make_request(model_dir, evaluation=make_eval()))
    assert p["eval_dataset_version"] == "eval-v1" and p["eval_bleu_tokenize"] == "13a"
    assert p["eval_batch_size"] == "8" and p["eval_bleu_lowercase"] == "False"


def test_metrics_manual_only_auto_only_both_none(model_dir):
    assert build_metrics(make_request(model_dir, manual_bleu=30.0)) == {"bleu": 30.0}
    assert build_metrics(make_request(model_dir, evaluation=make_eval())) == {
        "bleu_auto": 42.5, "evaluation_sentence_count": 20.0}
    both = build_metrics(make_request(model_dir, manual_bleu=30.0, evaluation=make_eval()))
    assert both["bleu"] == 30.0 and both["bleu_auto"] == 42.5
    assert build_metrics(make_request(model_dir)) == {}


def test_tags_synthetic_automatic(model_dir):
    t = build_tags(make_request(model_dir, evaluation=make_eval("synthetic")), "registered")
    assert t["model_sha256"] == SHA and t["model_sha256_algorithm"] == "sha256-dir-v1"
    assert t["model_type"] == "translation" and t["artifact_type"] == "ctranslate2_directory"
    assert t["evaluation_type"] == "synthetic" and t["evaluation_status"] == "completed"
    assert t["bleu_source"] == "automatic" and t["registration_status"] == "registered"


def test_real_dataset_evaluation_is_tagged_automatic(model_dir):
    t = build_tags(make_request(model_dir, evaluation=make_eval("real")), "registered")
    assert t["evaluation_type"] == "automatic" and t["evaluation_status"] == "completed"
    assert "evaluation_skip_reason" not in t


def test_skipped_with_manual_bleu_is_tagged_manual_not_automatic(model_dir):
    t = build_tags(make_request(model_dir, manual_bleu=31.0), "pending")
    assert (t["evaluation_type"], t["evaluation_status"], t["bleu_source"]) == ("manual", "skipped", "manual")
    assert t["evaluation_skip_reason"] == "evaluation_not_requested"
    assert "bleu_auto" not in build_metrics(make_request(model_dir, manual_bleu=31.0))


def test_skipped_without_bleu(model_dir):
    r = make_request(model_dir, evaluation_skipped=True, evaluation_skip_reason="translator_not_configured")
    t = build_tags(r, "pending")
    assert (t["evaluation_type"], t["evaluation_status"], t["bleu_source"]) == ("none", "skipped", "none")
    assert t["evaluation_skip_reason"] == "translator_not_configured"


def test_failed_evaluation_is_never_tagged_succeeded(model_dir):
    t = build_tags(make_request(model_dir, evaluation_error="boom", allow_failed_evaluation=True), "registered")
    assert t["evaluation_status"] == "failed" and t["evaluation_error"] == "boom"


def test_undeclared_evaluation_is_refused(model_dir):
    c = FakeClient()
    request = RegistrationRequest(
        registered_model_name="tm", model_dir=model_dir, model_folder="f", model_sha256=SHA,
        validation=validate_model_directory(model_dir))
    with pytest.raises(RegistrationError, match="not declared"):
        track_and_register(c, request, poll_interval=0)
    assert c.calls == []


def test_skip_requires_a_reason(model_dir):
    c = FakeClient()
    with pytest.raises(RegistrationError, match="reason"):
        track_and_register(c, make_request(model_dir, evaluation_skipped=True), poll_interval=0)
    assert c.calls == []


def test_skip_and_evaluation_together_rejected(model_dir):
    with pytest.raises(ValueError):
        track_and_register(FakeClient(), make_request(
            model_dir, evaluation=make_eval(), evaluation_skipped=True, evaluation_skip_reason="x"))


def test_skipped_registration_records_reason_on_run_and_version(model_dir):
    c = FakeClient()
    track_and_register(c, make_request(model_dir, evaluation_skipped=True,
                                       evaluation_skip_reason="translator_not_configured"), poll_interval=0)
    assert c.arg("create_run")[2]["tags"]["evaluation_skip_reason"] == "translator_not_configured"
    assert c.arg("create_model_version")[2]["tags"]["evaluation_status"] == "skipped"


# ----------------------------------------------------------- registration
def test_successful_registration_flow(model_dir):
    c = FakeClient()
    out = track_and_register(c, make_request(model_dir, manual_bleu=30.0, evaluation=make_eval()),
                             poll_interval=0)
    assert out.status == "registered" and out.version == "3" and out.run_id == "run123"
    assert out.source_uri == "runs:/run123/model"
    names = c.names()
    assert names.index("search_model_versions") < names.index("create_run") < names.index("log_batch")
    assert names.index("log_artifacts") < names.index("create_model_version") < names.index("set_terminated")
    assert c.arg("log_artifacts")[1] == ("run123", str(model_dir)) and c.arg("log_artifacts")[2] == {"artifact_path": "model"}
    assert c.arg("log_dict")[1][2] == "evaluation/evaluation_result.json"
    json.dumps(c.arg("log_dict")[1][1])
    metrics = {m.key: m.value for m in c.arg("log_batch")[2]["metrics"]}
    assert metrics == {"bleu": 30.0, "bleu_auto": 42.5, "evaluation_sentence_count": 20.0}
    params = {p.key: p.value for p in c.arg("log_batch")[2]["params"]}
    assert params["target_language"] == "spa-latn"
    mv_tags = c.arg("create_model_version")[2]["tags"]
    assert mv_tags["model_sha256"] == SHA
    assert mv_tags["registration_status"] == "pending"      # not "registered" until READY
    assert c.arg("set_model_version_tag")[1] == ("tm", "3", "registration_status", "registered")
    assert c.arg("set_terminated")[1] == ("run123", "FINISHED")
    assert c.arg("set_tag")[1] == ("run123", "registration_status", "registered")
    assert c.run_tags["registration_status"] == "pending"


def test_registered_model_created_only_when_missing(model_dir):
    c = FakeClient()
    track_and_register(c, make_request(model_dir), poll_interval=0)
    assert "create_registered_model" in c.names()
    c2 = FakeClient(); c2.registered.add("tm")
    track_and_register(c2, make_request(model_dir), poll_interval=0)
    assert "create_registered_model" not in c2.names()


def test_experiment_name_and_artifact_location_forwarded(model_dir):
    c = FakeClient()
    track_and_register(c, make_request(model_dir), experiment_name="my-exp",
                       artifact_location="file:///a", poll_interval=0)
    assert c.arg("create_experiment")[1] == ("my-exp",)
    assert c.arg("create_experiment")[2] == {"artifact_location": "file:///a"}


def test_invalid_model_is_never_registered(model_dir):
    (model_dir / "model.bin").unlink()
    c = FakeClient()
    with pytest.raises(RegistrationError, match="validation"):
        track_and_register(c, make_request(model_dir), poll_interval=0)
    assert c.calls == []


def test_duplicate_creates_nothing(model_dir):
    c = FakeClient()
    c.versions.append(SimpleNamespace(name="tm", version="5", tags={"model_sha256": SHA}))
    out = track_and_register(c, make_request(model_dir), poll_interval=0)
    assert out.status == "duplicate_skipped" and out.version == "5" and out.run_id is None
    assert c.names() == ["search_model_versions"]


def test_failed_evaluation_refused_by_default(model_dir):
    c = FakeClient()
    with pytest.raises(RegistrationError, match="evaluation failed"):
        track_and_register(c, make_request(model_dir, evaluation_error="translator crashed"), poll_interval=0)
    assert c.calls == []


def test_failed_evaluation_can_be_registered_when_explicitly_allowed(model_dir):
    c = FakeClient()
    track_and_register(c, make_request(model_dir, evaluation_error="x", allow_failed_evaluation=True), poll_interval=0)
    tags = c.arg("create_model_version")[2]["tags"]
    assert tags["evaluation_status"] == "failed"
    assert "bleu_auto" not in {m.key for m in c.arg("log_batch")[2]["metrics"]}


def test_evaluation_and_error_together_rejected(model_dir):
    with pytest.raises(ValueError):
        track_and_register(FakeClient(), make_request(model_dir, evaluation=make_eval(), evaluation_error="x"))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 101.0])
def test_invalid_manual_bleu_rejected(model_dir, bad):
    with pytest.raises(RegistrationError, match="Manual BLEU"):
        track_and_register(FakeClient(), make_request(model_dir, manual_bleu=bad))


def test_failure_mid_run_marks_run_failed_and_does_not_register(model_dir):
    c = FakeClient()
    c.fail["log_artifacts"] = MlflowException("down", error_code=INTERNAL_ERROR)
    with pytest.raises(MLflowConnectionError):
        track_and_register(c, make_request(model_dir), poll_interval=0)
    assert "create_model_version" not in c.names()
    assert ("run123", "FAILED") == c.arg("set_terminated")[1]
    assert ("run123", "registration_status", "failed") == c.arg("set_tag")[1]


def test_auth_error_is_distinct(model_dir):
    c = FakeClient()
    c.fail["create_model_version"] = MlflowException("denied", error_code=PERMISSION_DENIED)
    with pytest.raises(MLflowAuthError):
        track_and_register(c, make_request(model_dir), poll_interval=0)


def test_connection_error_when_server_unavailable_at_duplicate_check(model_dir):
    c = FakeClient()
    c.fail["search_model_versions"] = ConnectionError("refused")
    with pytest.raises(MLflowConnectionError):
        track_and_register(c, make_request(model_dir), poll_interval=0)
    assert "create_run" not in c.names()


def test_other_mlflow_rejection_is_registration_error(model_dir):
    c = FakeClient()
    c.fail["create_run"] = MlflowException("bad param", error_code=RESOURCE_ALREADY_EXISTS)
    with pytest.raises(RegistrationError, match="rejected"):
        track_and_register(c, make_request(model_dir), poll_interval=0)


def test_cleanup_error_does_not_mask_original(model_dir):
    c = FakeClient()
    c.fail["log_artifacts"] = MlflowException("down", error_code=INTERNAL_ERROR)
    c.fail["set_terminated"] = RuntimeError("cleanup also failed")
    with pytest.raises(MLflowConnectionError):
        track_and_register(c, make_request(model_dir), poll_interval=0)


def test_version_failed_registration_raises(model_dir):
    c = FakeClient(); c.version_status = "FAILED_REGISTRATION"
    with pytest.raises(RegistrationError, match="failed registration"):
        track_and_register(c, make_request(model_dir), poll_interval=0)
    assert ("run123", "FAILED") == c.arg("set_terminated")[1]


def test_version_never_ready_times_out(model_dir):
    c = FakeClient(); c.version_status = "PENDING_REGISTRATION"
    with pytest.raises(RegistrationError, match="not READY"):
        track_and_register(c, make_request(model_dir), wait_timeout=0, poll_interval=0)


def version_status_tags(c):
    return [call[1][2:] for call in c.calls if call[0] == "set_model_version_tag"]


def test_version_is_tagged_registered_only_after_ready_is_confirmed(model_dir):
    c = FakeClient()
    track_and_register(c, make_request(model_dir), poll_interval=0)
    names = c.names()
    assert names.index("get_model_version") < names.index("set_model_version_tag")
    assert version_status_tags(c) == [("registration_status", "registered")]


def test_failed_registration_tags_the_version_failed_never_registered(model_dir):
    c = FakeClient(); c.version_status = "FAILED_REGISTRATION"
    with pytest.raises(RegistrationError):
        track_and_register(c, make_request(model_dir), poll_interval=0)
    assert c.arg("create_model_version")[2]["tags"]["registration_status"] == "pending"
    assert version_status_tags(c) == [("registration_status", "failed")]
    assert c.arg("set_tag")[1] == ("run123", "registration_status", "failed")


def test_timeout_waiting_for_ready_tags_the_version_failed(model_dir):
    c = FakeClient(); c.version_status = "PENDING_REGISTRATION"
    with pytest.raises(RegistrationError, match="not READY"):
        track_and_register(c, make_request(model_dir), wait_timeout=0, poll_interval=0)
    assert ("registration_status", "registered") not in version_status_tags(c)
    assert version_status_tags(c) == [("registration_status", "failed")]


def test_failure_before_version_exists_does_not_touch_version_tags(model_dir):
    c = FakeClient()
    c.fail["log_artifacts"] = MlflowException("down", error_code=INTERNAL_ERROR)
    with pytest.raises(MLflowConnectionError):
        track_and_register(c, make_request(model_dir), poll_interval=0)
    assert version_status_tags(c) == []


def test_failure_tagging_the_version_registered_marks_it_failed(model_dir):
    c = FakeClient()
    calls = {"n": 0}
    original = c.set_model_version_tag

    def flaky(name, version, key, value):
        calls["n"] += 1
        if calls["n"] == 1:                      # the "registered" tag write fails
            raise MlflowException("down", error_code=INTERNAL_ERROR)
        return original(name, version, key, value)

    c.set_model_version_tag = flaky
    with pytest.raises(MLflowConnectionError):
        track_and_register(c, make_request(model_dir), poll_interval=0)
    assert version_status_tags(c) == [("registration_status", "failed")]


def test_version_tag_cleanup_error_does_not_mask_original(model_dir):
    c = FakeClient(); c.version_status = "FAILED_REGISTRATION"
    c.fail["set_model_version_tag"] = RuntimeError("cleanup also failed")
    with pytest.raises(RegistrationError, match="failed registration"):
        track_and_register(c, make_request(model_dir), poll_interval=0)


# ------------------------------------------------------------ tracking URI
def test_tracking_uri_resolution(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env:5000")
    assert resolve_tracking_uri("http://explicit:1") == "http://explicit:1"
    assert resolve_tracking_uri() == "http://env:5000"
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    assert resolve_tracking_uri() == "http://127.0.0.1:5000"


# ------------------------------------------- integration (real local store)
@pytest.fixture
def real(tmp_path):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    location = (tmp_path / "artifacts").as_uri()
    return MlflowClient(tracking_uri=uri), location


def _register(client, location, model_dir, **kw):
    request = make_request(model_dir, model_sha256=calculate_directory_sha256(model_dir), **kw)
    return track_and_register(client, request, experiment_name="it-exp", artifact_location=location,
                              poll_interval=0.2, wait_timeout=60)


def test_integration_register_new_model(real, model_dir):
    client, location = real
    out = _register(client, location, model_dir, manual_bleu=12.5, evaluation=make_eval("synthetic"))
    assert out.status == "registered" and out.version == "1"

    mv = client.get_model_version("tm", "1")
    assert mv.tags["model_sha256"] == calculate_directory_sha256(model_dir)
    assert mv.tags["evaluation_type"] == "synthetic"
    assert mv.tags["registration_status"] == "registered"

    run = client.get_run(out.run_id)
    assert run.data.params["model_format"] == "ctranslate2"
    assert run.data.metrics["bleu"] == 12.5 and run.data.metrics["bleu_auto"] == 42.5
    assert run.data.tags["registration_status"] == "registered"
    assert run.info.status == "FINISHED"
    files = {f.path for f in client.list_artifacts(out.run_id, "model")}
    assert {"model/model.bin", "model/shared_vocabulary.json"} <= files
    assert any(f.path.endswith("evaluation_result.json") for f in client.list_artifacts(out.run_id, "evaluation"))


def test_integration_duplicate_then_changed_model(real, model_dir):
    client, location = real
    first = _register(client, location, model_dir)
    again = _register(client, location, model_dir)
    assert again.status == "duplicate_skipped" and again.version == first.version
    assert len(client.search_model_versions("name = 'tm'")) == 1

    (model_dir / "model.bin").write_bytes(b"DUMMY-CHANGED")
    changed = _register(client, location, model_dir)
    assert changed.status == "registered" and changed.version == "2"


def test_integration_experiment_is_reused(real):
    client, location = real
    a = get_or_create_experiment(client, "same", location)
    assert get_or_create_experiment(client, "same", location) == a
