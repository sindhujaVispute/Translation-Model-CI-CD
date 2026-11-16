#!/usr/bin/env python3
"""Real-environment verification for translation-model-mlops.

Run this ONCE on a machine with network access and Python 3.10:

    python3.10 scripts/verify_real_environment.py
    python scripts/verify_real_environment.py --python /path/to/python3.10

What it does (each item is reported PASS/FAIL, exit code 0 only if all pass):
  1. fresh venv from the target Python, `pip install -r requirements.txt`, full pytest run
     (the real-MLflow integration tests must RUN, not skip)
  2. starts a TEMPORARY local MLflow server: SQLite + temp artifact dir, 127.0.0.1,
     random free port. It never uses PostgreSQL, a configured tracking URI or credentials.
  3. registers the dummy model with the synthetic evaluation, then inspects MLflow records
  4. re-runs the identical command: registration must be skipped (duplicate); a changed
     model must become version 2
  5. stops the server: the CLI must exit with code 3
  6. confirms the interpreter is Python 3.10 (all of the above ran under it)

Only the standard library is used here, so this file itself runs on any Python 3.
"""
import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTERED_NAME = "verify-translation-model"
RESULTS = []

INSPECT = r'''
import json, sys
from mlflow.tracking import MlflowClient
c = MlflowClient(tracking_uri=sys.argv[1])
out = {"versions": []}
for v in c.search_model_versions("name = '%s'" % sys.argv[2]):
    run = c.get_run(v.run_id)
    out["versions"].append({
        "version": v.version, "status": str(v.status), "tags": dict(v.tags),
        "params": dict(run.data.params), "metrics": dict(run.data.metrics),
        "run_status": run.info.status, "run_tags": dict(run.data.tags),
        "model_artifacts": [f.path for f in c.list_artifacts(v.run_id, "model")],
        "evaluation_artifacts": [f.path for f in c.list_artifacts(v.run_id, "evaluation")],
    })
print("RESULT_JSON=" + json.dumps(out))
'''


def record(step, ok, detail=""):
    RESULTS.append((step, ok, detail))
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", step, (" - " + detail) if detail else ""), flush=True)
    return ok


def clean_env(**extra):
    """Environment with every MLflow/pipeline setting stripped, so nothing can leak to or
    from a production configuration (credentials included)."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("MLFLOW_") and k not in (
               "TRANSLATOR_FACTORY", "MODEL_REGISTRY_NAME", "EVALUATION_DATASET_VERSION")}
    env.update(extra)
    return env


def run(cmd, env=None, timeout=None):
    proc = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    return proc.returncode, proc.stdout


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def python_version(py):
    code, out = run([py, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"])
    return out.strip() if code == 0 else None


def parse_pytest_summary(output):
    """Return (passed, failed, skipped, error) counts from pytest's last summary line."""
    counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "errors": 0}
    for line in reversed(output.strip().splitlines()):
        if re.search(r"\b(passed|failed|skipped|error)", line) and " in " in line:
            for number, word in re.findall(r"(\d+) (passed|failed|skipped|errors?)", line):
                counts[word] += int(number)
            break
    return counts["passed"], counts["failed"], counts["skipped"], counts["error"] + counts["errors"]


def cli_command(venv_py, model_dir, *extra):
    return [venv_py, "register_ctranslate2_model.py", "--model-dir", model_dir,
            "--registered-model-name", REGISTERED_NAME, "--model-folder", "sample", *extra]


SYNTHETIC_ARGS = [
    "--translator", "mock_data.mock_translator:create_translator",
    "--evaluation-source", "mock_data/evaluation/source.txt",
    "--evaluation-reference", "mock_data/evaluation/reference.txt",
    "--evaluation-type", "synthetic", "--target-language", "spa-latn",
    "--evaluation-dataset-version", "synthetic-v0",
]


def inspect_versions(venv_py, uri):
    code, out = run([venv_py, "-c", INSPECT, uri, REGISTERED_NAME], env=clean_env())
    for line in out.splitlines():
        if line.startswith("RESULT_JSON="):
            return json.loads(line[len("RESULT_JSON="):])
    raise RuntimeError("could not inspect MLflow (exit %s):\n%s" % (code, out[-1500:]))


def start_server(venv_py, workdir, port, log_path):
    backend = "sqlite:///" + (workdir / "mlflow.db").as_posix()
    artifacts = workdir / "artifacts"
    artifacts.mkdir()
    # Safety guards: this must be a throw-away local store, never a configured server.
    assert backend.startswith("sqlite:///") and "postgres" not in backend.lower()
    cmd = [venv_py, "-m", "mlflow", "server", "--host", "127.0.0.1", "--port", str(port),
           "--backend-store-uri", backend, "--serve-artifacts",
           "--artifacts-destination", str(artifacts)]
    log = open(str(log_path), "w")
    proc = subprocess.Popen([str(c) for c in cmd], cwd=str(ROOT), env=clean_env(),
                            stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + 120
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        try:
            if urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2).status == 200:
                return proc, log
        except Exception:
            time.sleep(1)
    proc.kill()
    log.close()
    raise RuntimeError("MLflow server did not start; log tail:\n" + log_path.read_text()[-2000:])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python", default=shutil.which("python3.10"),
                        help="Python 3.10 interpreter (default: python3.10 on PATH)")
    parser.add_argument("--venv", help="reuse/create this venv instead of a temporary one")
    parser.add_argument("--allow-any-python", action="store_true",
                        help="dry run only: continue on a non-3.10 Python (step 6 still FAILS)")
    args = parser.parse_args()

    if not args.python:
        record("6. Python 3.10 available", False,
               "python3.10 not found; install it or pass --python /path/to/python3.10")
        return summarize()
    version = python_version(args.python)
    ok310 = version == "3.10"
    record("6. Interpreter is Python 3.10", ok310, "found %s at %s" % (version, args.python))
    if not ok310 and not args.allow_any_python:
        return summarize()

    workdir = Path(tempfile.mkdtemp(prefix="tmmlops-verify-"))
    server = log = None
    try:
        # ---- step 1: install + full test run -------------------------------------------------
        venv = Path(args.venv) if args.venv else workdir / "venv"
        if not venv.exists():
            code, out = run([args.python, "-m", "venv", venv])
            if not record("1a. create virtualenv", code == 0, out[-300:] if code else ""):
                return summarize()
        venv_py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        code, out = run([venv_py, "-m", "pip", "install", "-r", "requirements.txt"], timeout=1800)
        if not record("1b. pip install -r requirements.txt", code == 0,
                      "" if code == 0 else out.strip().splitlines()[-1] if out.strip() else "pip failed"):
            return summarize()
        code, out = run([venv_py, "-m", "pytest", "-v", "-rs", "-p", "no:cacheprovider"],
                        env=clean_env(), timeout=1800)
        passed, failed, skipped, errors = parse_pytest_summary(out)
        integration_skipped = bool(re.search(r"SKIPPED.*mlflow", out, re.I))
        record("1c. full pytest run", code == 0 and not failed and not errors,
               "%d passed, %d failed, %d errors, %d skipped" % (passed, failed, errors, skipped))
        record("1d. real-MLflow integration tests actually ran (not skipped)", not integration_skipped)
        if code != 0:
            print("\n--- pytest output (tail) ---\n" + out[-3500:])

        # ---- step 2: temporary MLflow server -------------------------------------------------
        port = free_port()
        uri = "http://127.0.0.1:%d" % port
        try:
            server, log = start_server(venv_py, workdir, port, workdir / "server.log")
        except Exception as exc:  # noqa: BLE001
            record("2. temporary MLflow server (SQLite + temp artifacts)", False, str(exc))
            return summarize()
        record("2. temporary MLflow server (SQLite + temp artifacts)", True, uri)

        env = clean_env(MLFLOW_TRACKING_URI=uri, MLFLOW_EXPERIMENT_NAME="verification-experiment")
        model = str(ROOT / "mock_data" / "sample_model")

        # ---- step 3: register synthetic model --------------------------------------------------
        code, out = run(cli_command(venv_py, model, *SYNTHETIC_ARGS), env=env, timeout=600)
        record("3a. CLI registers the synthetic model (exit 0)",
               code == 0 and "MODEL REGISTRATION SUCCESSFUL" in out, "exit=%s" % code)
        if code != 0:
            print(out[-2500:])
        versions = inspect_versions(venv_py, uri)["versions"]
        v1 = versions[0] if len(versions) == 1 else {}
        tags, params, metrics = v1.get("tags", {}), v1.get("params", {}), v1.get("metrics", {})
        checks = {
            "exactly one version": len(versions) == 1,
            "version READY": v1.get("status") == "READY",
            "model_sha256 tag (64 hex)": bool(re.fullmatch(r"[0-9a-f]{64}", tags.get("model_sha256", ""))),
            "evaluation_status=completed": tags.get("evaluation_status") == "completed",
            "evaluation_type=synthetic": tags.get("evaluation_type") == "synthetic",
            "bleu_source=automatic": tags.get("bleu_source") == "automatic",
            "registration_status=registered": tags.get("registration_status") == "registered",
            "params (format/target lang)": params.get("model_format") == "ctranslate2"
                                           and params.get("target_language") == "spa-latn",
            "metric bleu_auto present (~100 on oracle)": metrics.get("bleu_auto", 0) > 99.9,
            "metric evaluation_sentence_count=20": metrics.get("evaluation_sentence_count") == 20,
            "no manual 'bleu' metric": "bleu" not in metrics,
            "run FINISHED": v1.get("run_status") == "FINISHED",
            "model artifacts uploaded": "model/model.bin" in v1.get("model_artifacts", []),
            "evaluation_result.json logged": any(p.endswith("evaluation_result.json")
                                                 for p in v1.get("evaluation_artifacts", [])),
        }
        record("3b. MLflow records correct", all(checks.values()),
               "; ".join(k for k, ok in checks.items() if not ok) or "%d checks" % len(checks))

        # ---- step 4: duplicate detection --------------------------------------------------------
        code, out = run(cli_command(venv_py, model, *SYNTHETIC_ARGS), env=env, timeout=600)
        count = len(inspect_versions(venv_py, uri)["versions"])
        record("4a. identical model: registration skipped (exit 0, still 1 version)",
               code == 0 and "Duplicate artifact" in out and count == 1, "exit=%s versions=%s" % (code, count))
        changed = workdir / "model_changed"
        shutil.copytree(model, str(changed))
        with open(str(changed / "model.bin"), "ab") as handle:
            handle.write(b"-changed")
        code, out = run(cli_command(venv_py, str(changed), *SYNTHETIC_ARGS), env=env, timeout=600)
        count = len(inspect_versions(venv_py, uri)["versions"])
        record("4b. changed model registers as a new version (2 versions)",
               code == 0 and count == 2, "exit=%s versions=%s" % (code, count))

        # ---- step 5: server failure ------------------------------------------------------------
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
        server = None
        # MLflow retries with backoff by default (can take minutes); keep the test quick.
        fail_env = dict(env, MLFLOW_HTTP_REQUEST_MAX_RETRIES="1", MLFLOW_HTTP_REQUEST_TIMEOUT="5")
        started = time.time()
        code, out = run(cli_command(venv_py, model, "--skip-evaluation"), env=fail_env, timeout=600)
        record("5. server stopped: CLI exits with code 3", code == 3,
               "exit=%s after %.0fs" % (code, time.time() - started))
        if code != 3:
            print(out[-2000:])
    finally:
        if server is not None:
            server.kill()
        shutil.rmtree(str(workdir), ignore_errors=True)
    return summarize()


def summarize():
    print("\n==== SUMMARY ====")
    for step, ok, detail in RESULTS:
        print("%-4s %s%s" % ("PASS" if ok else "FAIL", step, (" - " + detail) if detail else ""))
    ok = bool(RESULTS) and all(r[1] for r in RESULTS)
    print("\nOVERALL: " + ("ALL CHECKS PASSED" if ok else "NOT VERIFIED / FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
