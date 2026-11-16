"""End-to-end tests of register_ctranslate2_model.main() with a fake MLflow client
and mock translators. No server, no real model, nothing registered anywhere."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, PERMISSION_DENIED

import register_ctranslate2_model as cli
from services.checksum import calculate_directory_sha256
from tests.fakes import FakeClient

ROOT = Path(__file__).resolve().parent.parent
SRC = str(ROOT / "mock_data" / "evaluation" / "source.txt")
REF = str(ROOT / "mock_data" / "evaluation" / "reference.txt")
ORACLE = "mock_data.mock_translator:create_translator"
FACTORY_CALLS = []


class _Raising:
    def translate_batch(self, sources):
        raise RuntimeError("tokenizer exploded")


def create_failing(model_dir, device):
    return _Raising()


def create_recording(model_dir, device):
    FACTORY_CALLS.append((str(model_dir), device))
    from mock_data.mock_translator import create_translator
    return create_translator(model_dir, device)


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "model.bin").write_bytes(b"DUMMY")
    (d / "shared_vocabulary.json").write_text('["a"]')
    return d


@pytest.fixture
def run(model_dir, monkeypatch):
    for key in ("MODEL_REGISTRY_NAME", "TRANSLATOR_FACTORY", "EVALUATION_DATASET_VERSION",
                "MLFLOW_EXPERIMENT_NAME", "MLFLOW_TRACKING_URI"):
        monkeypatch.delenv(key, raising=False)
    FACTORY_CALLS.clear()

    def _run(*extra, client=None, name=True):
        client = client or FakeClient()
        argv = ["--model-dir", str(model_dir), "--model-folder", "f"]
        if name:
            argv += ["--registered-model-name", "tm"]
        code = cli.main(argv + list(extra), client=client)
        return code, client

    return _run


def auto(*extra):
    return ["--translator", ORACLE, "--evaluation-source", SRC, "--evaluation-reference", REF,
            "--evaluation-type", "synthetic", *extra]


def version_tags(c):
    return c.arg("create_model_version")[2]["tags"]


def metrics(c):
    return {m.key: m.value for m in c.arg("log_batch")[2]["metrics"]}


def registered(c):
    return "create_model_version" in c.names()


# ------------------------------------------------------ automatic evaluation
def test_automatic_synthetic_evaluation_registers_with_completed_status(run):
    code, c = run(*auto("--target-language", "spa-latn", "--evaluation-dataset-version", "syn-v0"))
    assert code == cli.EXIT_OK and registered(c)
    tags = version_tags(c)
    assert (tags["evaluation_status"], tags["evaluation_type"], tags["bleu_source"]) == (
        "completed", "synthetic", "automatic")
    assert "evaluation_skip_reason" not in tags
    m = metrics(c)
    assert m["bleu_auto"] == pytest.approx(100.0) and m["evaluation_sentence_count"] == 20.0
    assert "bleu" not in m                      # nothing manual was supplied
    params = {p.key: p.value for p in c.arg("log_batch")[2]["params"]}
    assert params["target_language"] == "spa-latn" and params["eval_dataset_version"] == "syn-v0"


def test_real_evaluation_type_is_tagged_automatic(run):
    code, c = run("--translator", ORACLE, "--evaluation-source", SRC,
                  "--evaluation-reference", REF, "--evaluation-type", "real")
    assert code == 0 and version_tags(c)["evaluation_type"] == "automatic"


def test_manual_bleu_alongside_automatic_is_kept_separate(run):
    code, c = run(*auto("--bleu", "33.3"))
    assert code == 0
    m = metrics(c)
    assert m["bleu"] == 33.3 and m["bleu_auto"] == pytest.approx(100.0)
    assert version_tags(c)["bleu_source"] == "manual+automatic"


def test_translator_from_environment(run, monkeypatch):
    monkeypatch.setenv("TRANSLATOR_FACTORY", ORACLE)
    code, c = run("--evaluation-source", SRC, "--evaluation-reference", REF,
                  "--evaluation-type", "synthetic")
    assert code == 0 and version_tags(c)["evaluation_status"] == "completed"


# ------------------------------------------------------- not configured
def test_no_translator_and_no_skip_stops_with_error_and_registers_nothing(run):
    code, c = run()
    assert code == cli.EXIT_EVALUATION
    assert c.calls == []                         # failed fast, MLflow never contacted


def test_plan_error_message_is_actionable():
    args = cli.build_parser().parse_args(
        ["--model-dir", "x", "--registered-model-name", "m", "--model-folder", "f"])
    with pytest.raises(cli.EvaluationConfigError, match="--skip-evaluation"):
        cli.plan_evaluation(args)


@pytest.mark.parametrize("missing", ["--evaluation-source", "--evaluation-reference", "--evaluation-type"])
def test_incomplete_evaluation_config_fails_fast(run, missing):
    full = ["--translator", ORACLE, "--evaluation-source", SRC,
            "--evaluation-reference", REF, "--evaluation-type", "synthetic"]
    i = full.index(missing)
    code, c = run(*(full[:i] + full[i + 2:]))
    assert code == cli.EXIT_EVALUATION and c.calls == []


def test_evaluation_type_has_no_silent_default(run):
    code, c = run("--translator", ORACLE, "--evaluation-source", SRC, "--evaluation-reference", REF)
    assert code == cli.EXIT_EVALUATION and c.calls == []


def test_skip_evaluation_defaults_to_false():
    args = cli.build_parser().parse_args(
        ["--model-dir", "x", "--registered-model-name", "m", "--model-folder", "f"])
    assert args.skip_evaluation is False


# ------------------------------------------------------- explicit skip
def test_skip_without_translator_records_translator_not_configured(run):
    code, c = run("--skip-evaluation")
    assert code == 0 and registered(c)
    tags = version_tags(c)
    assert tags["evaluation_status"] == "skipped"
    assert tags["evaluation_skip_reason"] == "translator_not_configured"
    assert tags["evaluation_type"] == "none" and tags["bleu_source"] == "none"
    assert c.arg("create_run")[2]["tags"]["evaluation_skip_reason"] == "translator_not_configured"
    assert metrics(c) == {}


def test_skip_with_translator_records_evaluation_not_requested(run):
    code, c = run("--skip-evaluation", "--translator", "tests.test_register_cli:create_recording")
    assert code == 0
    assert version_tags(c)["evaluation_skip_reason"] == "evaluation_not_requested"
    assert FACTORY_CALLS == []                   # translator never even built


def test_skip_with_manual_bleu_is_manual_not_automatic(run):
    code, c = run("--skip-evaluation", "--bleu", "28.4")
    assert code == 0
    tags = version_tags(c)
    assert (tags["evaluation_status"], tags["evaluation_type"], tags["bleu_source"]) == (
        "skipped", "manual", "manual")
    assert metrics(c) == {"bleu": 28.4}          # never logged as bleu_auto


@pytest.mark.parametrize("bad", ["nan", "inf", "-1", "101", "abc"])
def test_invalid_manual_bleu_is_a_usage_error(run, bad):
    with pytest.raises(SystemExit) as info:
        run("--skip-evaluation", "--bleu", bad)
    assert info.value.code == 2


# ------------------------------------------------------- evaluation failure
def test_failing_translator_reports_failure_and_registers_nothing(run):
    code, c = run("--translator", "tests.test_register_cli:create_failing",
                  "--evaluation-source", SRC, "--evaluation-reference", REF,
                  "--evaluation-type", "synthetic")
    assert code == cli.EXIT_EVALUATION
    assert "create_run" not in c.names() and not registered(c)


def test_misaligned_dataset_reports_failure(run, tmp_path):
    short = tmp_path / "short.txt"
    short.write_text("only one line\n")
    code, c = run("--translator", ORACLE, "--evaluation-source", SRC,
                  "--evaluation-reference", str(short), "--evaluation-type", "synthetic")
    assert code == cli.EXIT_EVALUATION and "create_run" not in c.names()


@pytest.mark.parametrize("spec", ["nocolon", "no_such_module_xyz:f", ORACLE.split(":")[0] + ":missing"])
def test_unloadable_translator_reports_failure(run, spec):
    code, c = run("--translator", spec, "--evaluation-source", SRC,
                  "--evaluation-reference", REF, "--evaluation-type", "synthetic")
    assert code == cli.EXIT_EVALUATION and "create_run" not in c.names()


def test_failed_evaluation_never_leaves_an_evaluated_version(run):
    code, c = run("--translator", "tests.test_register_cli:create_failing", "--bleu", "40",
                  "--evaluation-source", SRC, "--evaluation-reference", REF,
                  "--evaluation-type", "synthetic")
    assert code == cli.EXIT_EVALUATION and not registered(c)   # --bleu does not rescue it


# ------------------------------------------------------- other pipeline rules
def test_invalid_model_stops_before_anything_else(run, model_dir):
    (model_dir / "model.bin").unlink()
    code, c = run(*auto())
    assert code == cli.EXIT_VALIDATION_FAILED and c.calls == []


def test_duplicate_skips_before_evaluation_and_creates_nothing(run, model_dir):
    c = FakeClient()
    sha = calculate_directory_sha256(model_dir)
    c.versions.append(SimpleNamespace(name="tm", version="4", tags={"model_sha256": sha}))
    code, c = run("--translator", "tests.test_register_cli:create_recording",
                  "--evaluation-source", SRC, "--evaluation-reference", REF,
                  "--evaluation-type", "synthetic", client=c)
    assert code == cli.EXIT_OK
    assert FACTORY_CALLS == []                   # evaluation never ran
    assert c.names() == ["search_model_versions"]


def test_changed_model_is_not_a_duplicate(run, model_dir):
    c = FakeClient()
    c.versions.append(SimpleNamespace(name="tm", version="4", tags={"model_sha256": "f" * 64}))
    code, c = run("--skip-evaluation", client=c)
    assert code == 0 and registered(c)


def test_mlflow_unavailable_exit_code(run):
    c = FakeClient()
    c.fail["search_model_versions"] = MlflowException("down", error_code=INTERNAL_ERROR)
    code, c = run("--skip-evaluation", client=c)
    assert code == cli.EXIT_MLFLOW_UNAVAILABLE and "create_run" not in c.names()


def test_auth_failure_exit_code(run):
    c = FakeClient()
    c.fail["create_model_version"] = MlflowException("denied", error_code=PERMISSION_DENIED)
    code, c = run("--skip-evaluation", client=c)
    assert code == cli.EXIT_MLFLOW_UNAVAILABLE
    assert ("run123", "FAILED") == c.arg("set_terminated")[1]


def test_registration_rejection_exit_code(run):
    c = FakeClient()
    c.version_status = "FAILED_REGISTRATION"
    code, _ = run("--skip-evaluation", client=c)
    assert code == cli.EXIT_REGISTRATION


# ------------------------------------------------------- backward compatibility
def test_original_arguments_still_accepted(run):
    code, c = run("--skip-evaluation", "--quantization", "int8", "--source-language", "eng-latn",
                  "--training-checkpoint", "step_5000", "--dataset-version", "train-v3", "--bleu", "27.1")
    assert code == 0
    params = {p.key: p.value for p in c.arg("log_batch")[2]["params"]}
    assert params["training_checkpoint"] == "step_5000" and params["dataset_version"] == "train-v3"
    assert params["quantization"] == "int8" and params["device"] == "cpu"


def test_registered_model_name_from_environment(run, monkeypatch):
    monkeypatch.setenv("MODEL_REGISTRY_NAME", "from-env")
    code, c = run("--skip-evaluation", name=False)
    assert code == 0 and c.arg("create_model_version")[1] == ("from-env",)


def test_experiment_name_from_environment(run, monkeypatch):
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "env-exp")
    code, c = run("--skip-evaluation")
    assert c.arg("create_experiment")[1] == ("env-exp",)
