# translation-model-mlops

A standalone pipeline that registers OpenNMT-trained, CTranslate2-converted translation
models in MLflow, with validation, duplicate detection and automatic BLEU evaluation.

```
Model arrival -> validation -> evaluation preflight -> SHA256 -> duplicate check
  -> BLEU evaluation (or explicit skip) -> MLflow tracking -> registration -> summary
```

## Status

| | |
|---|---|
| Tests (offline run) | **183 passed, 0 failed, 4 skipped** |
| Verified with real `pytest`, `mlflow`, `sacrebleu`, Python 3.10, a live MLflow server | **No** |
| Real CTranslate2 model / real evaluation dataset | Not available yet (dummy model + synthetic data) |

The offline run used stand-ins for `pytest`, `mlflow` and `sacrebleu` because the development
sandbox had no network. It shows the project's own logic works against the assumptions made about those
libraries, not that the assumptions are right. Run `scripts/verify_real_environment.py` (section
"First run on your machine") before relying on this, and see [`docs/testing-report.md`](docs/testing-report.md)
for exactly what was and was not verified.

Documentation: [`docs/architecture.md`](docs/architecture.md) ·
[`docs/implementation-details.md`](docs/implementation-details.md) ·
[`docs/testing-report.md`](docs/testing-report.md)

## 1. Purpose
Replace manual registration with a repeatable pipeline: reject bad models, never register
the same artifact twice, calculate BLEU automatically, and record in MLflow exactly what was
(or was not) evaluated.

## 2. Architecture
| Module | Responsibility |
|---|---|
| `register_ctranslate2_model.py` | CLI; orchestrates the pipeline; exit codes |
| `services/model_validation.py` | structural validation of the model directory |
| `services/checksum.py` | deterministic directory SHA256 (`sha256-dir-v1`) |
| `services/duplicate_detection.py` | find an existing version with the same `model_sha256` tag |
| `services/bleu_evaluation.py` | dataset loading, batched translation, SacreBLEU, translator loading |
| `services/mlflow_service.py` | experiment, params/metrics/tags/artifacts, registration |
| `mock_data/` | dummy model, synthetic dataset, development-only mock translator |

## 3. Installation
```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip freeze > requirements.lock        # pin what actually resolved
```
`ctranslate2` is only needed to evaluate a real model (see section 13).

## 4. Environment configuration
Copy `.env.example` to `.env` (git-ignored). Variables: `MLFLOW_TRACKING_URI`,
`MLFLOW_EXPERIMENT_NAME`, `MODEL_REGISTRY_NAME` (default for `--registered-model-name`),
`EVALUATION_DATASET_VERSION`, optional `TRANSLATOR_FACTORY`. Credentials are never stored
here; MLflow reads `MLFLOW_TRACKING_USERNAME/PASSWORD/TOKEN` from the environment itself.

## 5. Model directory requirements
`model.bin` (required, non-empty); a vocabulary (`shared_vocabulary.json|txt` or
`source_/target_vocabulary.json|txt`); `config.json` optional (valid JSON if present);
tokenizer files (`*.model`/`*.spm`) optional (non-empty if present).

## 6. Validation
Checks existence, directory type, required files, empty files, unreadable files and JSON
validity, and returns a structured result. Invalid models stop the pipeline (exit 2).
It is structural only: it cannot prove `model.bin` loads in CTranslate2.

## 7. Checksum
Sorted, POSIX/NFC-normalised relative paths; length-prefixed paths and sizes (unambiguous file
boundaries); chunked reads. **Not compatible** with the original script's checksum, so versions
registered before this pipeline are never detected as duplicates.

## 8. Duplicate detection
Reads the `model_sha256` tag of versions of the same registered model through
`MlflowClient.search_model_versions`. A duplicate creates no run and no version (exit 0).
Versions whose registration **failed** (status `FAILED_REGISTRATION` or tag `registration_status=failed`)
are ignored so the artifact can be registered again; pending versions still count.
Connection/auth failures (exit 3) are distinct from "model not found" (treated as no duplicate).

## 9. BLEU evaluation
Aligned UTF-8 files, one sentence per line (blank lines and count mismatches are errors).
Translation goes through any object with `translate_batch(list[str]) -> list[str]`.
Scoring uses SacreBLEU defaults (`13a`, cased, `exp` smoothing), recorded with every result.

**Evaluation is never skipped silently.** Exactly one of:

| Situation | Behaviour |
|---|---|
| `--translator` + dataset paths + `--evaluation-type` | automatic BLEU, `evaluation_status=completed` |
| no translator configured | **exit 4**, nothing registered |
| `--skip-evaluation` | skipped; reason recorded (`translator_not_configured` or `evaluation_not_requested`) |
| `--bleu` supplied | logged as metric `bleu`, `bleu_source=manual`; never as automatic |
| evaluation fails | **exit 4**, nothing registered, never recorded as evaluated |

`--skip-evaluation` defaults to off. `--evaluation-type` (`synthetic`/`real`) has no default.

## 10. MLflow tracking and registration
* **Params:** model_name, model_folder, training_framework, model_format, serving_runtime,
  quantization, device, source/target language, training_checkpoint, dataset_version,
  eval_dataset_version, eval_batch_size, eval_bleu_*.
* **Metrics:** `bleu` (manual only), `bleu_auto`, `evaluation_sentence_count`.
* **Tags (run and version):** `model_sha256`, `model_sha256_algorithm`, `model_type`,
  `artifact_type`, `evaluation_status` (`completed|skipped|failed`),
  `evaluation_type` (`automatic|synthetic|manual|none`), `evaluation_skip_reason`,
  `bleu_source` (`automatic|manual|manual+automatic|none`), `registration_status`.
* **Artifacts:** the model directory under `model/`; `evaluation/evaluation_result.json`.
* `registration_status` goes `pending` -> `registered` only after MLflow reports the version `READY`;
  any failure after the run is created marks the run FAILED and, if a version exists, tags it `failed`.
* The service itself refuses registration unless the evaluation outcome is declared.

## 11. Running tests
```bash
python -m pytest -v --cov=services
```
183 test cases are expected (a different count is worth investigating). Unit tests need no server.
The last 3 tests in `tests/test_mlflow.py` use a temporary SQLite-backed MLflow store and skip when
`mlflow` is missing; they never use `MLFLOW_TRACKING_URI`. Coverage has not been measured yet.

## 12. Example CLI commands
Use a **local test MLflow**, never production, with dummy models.
```bash
# Automatic evaluation (development: mock translator + synthetic data)
python register_ctranslate2_model.py --model-dir mock_data/sample_model \
  --registered-model-name translation-model-dev --model-folder sample \
  --translator mock_data.mock_translator:create_translator \
  --evaluation-source mock_data/evaluation/source.txt \
  --evaluation-reference mock_data/evaluation/reference.txt \
  --evaluation-type synthetic --target-language spa-latn

# Explicitly skip evaluation (recorded in MLflow with a reason)
python register_ctranslate2_model.py --model-dir mock_data/sample_model \
  --registered-model-name translation-model-dev --model-folder sample --skip-evaluation

# Skip, with a manually supplied BLEU (logged as manual, never automatic)
python register_ctranslate2_model.py ... --skip-evaluation --bleu 27.4
```
Exit codes: `0` registered / duplicate skipped, `2` validation failed or bad usage,
`3` MLflow unreachable or auth failure, `4` evaluation misconfigured or failed,
`5` registration refused or rejected.

> **Migration:** the old command without any evaluation flags now exits with code 4.
> Existing automation must add `--skip-evaluation` (or configure a translator).

## 13. Integrating the real CTranslate2 model
Write a factory `callable(model_dir, device) -> translator` in your own package, and pass
`--translator your_pkg.module:create_translator` (or set `TRANSLATOR_FACTORY`). You must supply:
the tokenizer and detokenizer used in training (e.g. the SentencePiece model), any
target-language prefix or special tokens, decoding options (beam size...), and the SacreBLEU
settings you report with. `services.bleu_evaluation.CTranslate2Translator` is a thin adapter
(requires `pip install ctranslate2`; **untested against a real model**; no target-prefix support).
The factory is imported and executed as code: only use trusted specs.

## 14. Integrating the real evaluation dataset
Provide `--evaluation-source` / `--evaluation-reference` (UTF-8, line-aligned),
`--evaluation-type real` and `--evaluation-dataset-version <version>`. The mock data in
`mock_data/` is synthetic: BLEU computed on it says nothing about model quality.

## First run on your machine
One command performs the whole real-environment verification (network + Python 3.10 required):
```bash
python3.10 scripts/verify_real_environment.py          # or: --python /path/to/python3.10
```
It builds a fresh venv, installs `requirements.txt`, runs the full test suite (the real-MLflow
integration tests must run, not skip), starts a **temporary** MLflow server (SQLite + temp artifact
dir on 127.0.0.1; never PostgreSQL or your configured URI; MLFLOW_* variables are stripped from
its environment), registers the synthetic model and checks the MLflow records, re-runs it to
confirm the duplicate is skipped, then stops the server and confirms exit code 3.
It exits 0 only if every check passes. Afterwards run `pip freeze > requirements.lock`.

Note: by default MLflow retries failed requests with backoff, so an unreachable server can take
minutes to give up. The script lowers `MLFLOW_HTTP_REQUEST_MAX_RETRIES`/`_TIMEOUT` for the exit-3
check; set the same variables in automation if you want a fast failure.
