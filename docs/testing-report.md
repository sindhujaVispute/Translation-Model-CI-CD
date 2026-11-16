# Testing report

## Verification status at a glance

| Item | Status |
|---|---|
| Unit and CLI tests, offline run | **183 passed, 0 failed, 4 skipped** (see caveat below) |
| Run with real `pytest` | **Not done** |
| Run against real `mlflow` | **Not done** (the 3 integration tests were skipped) |
| Run against real `sacrebleu` | **Not done** |
| Python 3.10 | **Not done** (tests ran on Python 3.12.3) |
| Live MLflow server (register, duplicate, outage) | **Not done** |
| Real CTranslate2 model / real dataset | **Not available**; dummy model and synthetic data only |
| Coverage measurement | **Not measured** (`pytest-cov` was not run) |

**Caveat.** The development sandbox had no network access, so `pytest`, `mlflow`, `sacrebleu`
and `ctranslate2` could not be installed. The 183 results come from a throw-away runner that
provides small stand-ins for `pytest` (fixtures, `parametrize`, `raises`, `monkeypatch`),
for `mlflow` (exception classes and entity types only) and for `sacrebleu` (a minimal 4-gram BLEU
with brevity penalty and exp smoothing). The stand-ins are not part of this repository. Those
results show the project's own logic behaves as designed against the *assumptions made* about the real
libraries; they do not show that those assumptions are right. Treat the real-environment run
below as the actual acceptance test.

## Results by file (offline run)

| File | Passed | Skipped | Covers |
|---|---:|---:|---|
| `tests/test_checksum.py` | 18 | 0 | determinism and creation-order independence; independent reference implementation of the byte format; modified/renamed/added/removed files; file-boundary ambiguity; chunk-size independence on a ~9 MiB file; Windows/POSIX path and NFC/NFD normalisation; collisions; symlinked directories; invalid input |
| `tests/test_validation.py` | 25 | 1 | valid model; missing/empty `model.bin`; `model.bin` as a directory; missing, partial and alternative vocabulary layouts; optional/broken `config.json`; tokenizer rules; non-existent path; path is a file; empty directory; dangling symlink; unreadable file (simulated) |
| `tests/test_duplicate_detection.py` | 28 | 0 | new vs existing vs changed checksum; scope (same model / all models); pagination; not-found vs connection vs auth errors; invalid checksum/name; failed registrations ignored, pending counted, legacy versions counted |
| `tests/test_bleu_evaluation.py` | 35 | 0 | dataset loading and every malformed-data case; known BLEU values; batching; translator contract and failures; metadata; synthetic labelling; translator-factory loading; `CTranslate2Translator` glue (fake module) |
| `tests/test_mlflow.py` | 43 | 3 | experiment create/reuse/race/deleted; params, metrics, tags builders; full registration call sequence; refusal rules; duplicate creates nothing; failure cleanup; auth/connection mapping; status lifecycle (`pending` -> `registered`/`failed`); tracking-URI resolution; **3 integration tests against a temporary SQLite MLflow store (skipped)** |
| `tests/test_register_cli.py` | 34 | 0 | end-to-end `main()` with a fake client: automatic, not configured, skip (both reasons), manual BLEU, evaluation failure, duplicate before evaluation, MLflow down, auth failure, registration rejection, original arguments, environment defaults |
| **Total** | **183** | **4** | |

### Skipped tests
- `test_integration_register_new_model`, `test_integration_duplicate_then_changed_model`, `test_integration_experiment_is_reused`: need the real `mlflow` package.
- `test_unreadable_file_real_permissions`: `chmod`-based; not enforceable when running as root (a monkeypatch-based unreadable-file test does run).

### Known BLEU values (derived by hand, then asserted)
- One wrong word out of nine (n-gram precisions 8/9, 7/8, 6/7, 5/6, no brevity penalty): `100 * (5/9)^(1/4) = 86.33`.
- Eight-token prefix of a nine-token reference (all precisions 1): `100 * exp(1 - 9/8) = 88.25`.
- Perfect predictions: 100. Fully disjoint predictions: 0.

Both non-trivial values avoid zero n-gram precisions so they do not depend on smoothing details. They
passed against the stand-in BLEU; whether the real SacreBLEU gives identical values is unconfirmed.

## Mutation checks
To check that tests can fail, known bugs were reintroduced by hand (not with a mutation-testing
tool) and the suite re-run. Each was detected by at least one test, and the code was restored after each.

| Area | Reintroduced defect |
|---|---|
| Checksum | no length-prefix framing; no NFC normalisation |
| BLEU | ignore translator output-count mismatch; ignore source/reference misalignment; ignore `batch_size`; use `splitlines()` |
| MLflow service | skip validation guard; skip duplicate guard; allow failed evaluation silently; tag failed evaluation as completed; skip failed-run cleanup; treat auth errors as plain connection errors |
| CLI | missing translator silently becomes a skip; evaluation failure still registers; skip reason always `evaluation_not_requested`; manual BLEU tagged `automatic`; manual BLEU logged as `bleu_auto`; no required `--evaluation-type` |
| Status / duplicates | version created already tagged `registered`; failed version never tagged `failed`; `registered` written before `READY`; failed versions counted as duplicates; pending versions wrongly ignored |

23 defects, 23 detected.

## Defects found during development
- A vocabulary-layout heuristic chose the wrong layout for a half-present `source_`/`target_` vocabulary (found by a test, fixed).
- Version tagged `registered` before `READY` was confirmed, and no version-level failure tag (found by review, fixed, tested).
- A failed registration permanently blocked re-registration of the same artifact (found by review, fixed, tested).
- Other failures during development were gaps in the stand-in runner, not project defects.

## Limitations of these tests
- The fake MLflow client accepts the calls the code makes; if a real MLflow signature or error code differs from what was assumed, only the real-environment run will reveal it. Specific assumptions: `search_model_versions(page_token=...)`, `create_run(run_name=...)`, `log_batch`, `log_dict`, `get_model_version(...).status` values, `set_model_version_tag`, and error-code names such as `RESOURCE_DOES_NOT_EXIST`.
- A SQLite store cannot reproduce a down server or bad credentials; those paths are covered by simulated errors only until the live run.
- The mock translator returns the references, so a ~100 BLEU on synthetic data proves plumbing, not model quality.
- Windows behaviour (path normalisation is tested with pure-path objects; real Windows filesystems are not).

## How to verify in a real environment
On a machine with network access and Python 3.10:

```bash
python3.10 scripts/verify_real_environment.py        # or --python /path/to/python3.10
pip freeze > requirements.lock
```
The script reports PASS/FAIL for: fresh venv and `pip install -r requirements.txt`; the full test
suite with the integration tests actually running; a temporary SQLite-backed MLflow server on
127.0.0.1 (never PostgreSQL or a configured URI); registering the synthetic model and inspecting
14 properties of the MLflow records; a repeat run being skipped as a duplicate and a changed model
becoming version 2; and exit code 3 after the server is stopped. It exits 0 only if all pass.
Its own server-start and inspection code has not been executed yet, so a first-run failure may be
in the script rather than the project.

Expected: 183 test cases collected by real `pytest` (a different count is worth investigating) and
the 3 integration tests running instead of skipping. Record the outcome in this file when done.
