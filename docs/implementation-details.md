# Implementation details

Statements about MLflow/SacreBLEU behaviour below describe what the code *assumes*. They were
written without network access and have **not** been confirmed against the real packages
(see `testing-report.md`).

## 1. Checksum (`services/checksum.py`)
Algorithm id `sha256-dir-v1`. The hashed byte stream is:

```
b"ct2-dir-sha256-v1\0" | file_count (u64, big-endian)
for each file, sorted by the UTF-8 bytes of its normalised relative path:
    len(path) (u64) | path | file size (u64) | file contents
```
- Paths are POSIX-style (`/`) and Unicode NFC-normalised, so Linux, macOS (NFD names) and Windows agree.
- Every variable-length field is length-prefixed, so file boundaries are unambiguous.
- Files are streamed in 1 MiB chunks; the result is independent of chunk size. A file whose size changes while hashing raises an error.
- Two files that normalise to the same path raise `ValueError`; symlinked directories are rejected; empty sub-directories are not part of the fingerprint.
- **Incompatibility:** the original script's checksum had no framing and used OS-specific separators. Versions registered with it carry a different `model_sha256` and will never be detected as duplicates.

## 2. Validation (`services/model_validation.py`)
| File | Rule |
|---|---|
| `model.bin` | required, regular file, readable, non-empty |
| vocabulary | one complete layout required: `shared_vocabulary.json`, or `.txt`, or `source_vocabulary` + `target_vocabulary` (`.json`/`.txt`); `.json` must parse; checkable via `require_vocabulary=False` |
| `config.json` | optional; if present: non-empty, readable, valid JSON |
| tokenizer (`*.model`, `*.spm`) | optional (warning if absent); if present: non-empty, readable; `require_tokenizer=True` makes it mandatory |

The optional/required split reflects the author's understanding of CTranslate2 converter output
(not checked against a real converted model). Validation is structural only: it cannot prove that
`model.bin` loads. The result is a dataclass (`model_name`, `validation_status`, `missing_files`,
`validation_errors`, `warnings`, `files_checked`). When no vocabulary layout is complete, the
missing files are reported against the layout closest to complete.

## 3. Duplicate detection (`services/duplicate_detection.py`)
- Queries `client.search_model_versions(filter_string="name = '<model>'", max_results=1000, page_token=...)` and follows pagination. With `registered_model_name=None` every registered model is scanned.
- Compares the lower-cased `model_sha256` tag. Model names containing `'` or `\` are rejected (they cannot be safely embedded in the filter).
- **Failed registrations are ignored**: status `FAILED_REGISTRATION` or tag `registration_status=failed`. **Pending versions still count** (another registrar may be mid-flight). Legacy versions with neither status nor tag still count.
- Several matches: the lowest version number is returned first; all are in `all_matches`.
- Error classification (`classify_mlflow_exception`): error codes `UNAUTHENTICATED`/`PERMISSION_DENIED` -> `MLflowAuthError`; `INTERNAL_ERROR`/`TEMPORARILY_UNAVAILABLE`/`OSError` -> `MLflowConnectionError`; `RESOURCE_DOES_NOT_EXIST` while searching -> "no duplicate".
- Limitation: check-then-create is not atomic. Two registrars racing on the same new artifact could both pass the check; the service re-checks right before creating the run, which narrows but does not remove the window.

## 4. BLEU evaluation (`services/bleu_evaluation.py`)
- **Dataset**: UTF-8 (BOM tolerated), one sentence per line, CRLF tolerated. Split on `\n` only (not `splitlines()`, which also splits on U+2028 and would misalign data). Errors (`DatasetError`): missing file, undecodable file, empty dataset, unequal line counts, blank/whitespace-only lines (reported by line number).
- **Translation**: sources are sent in `batch_size` chunks. A translator exception, a wrong output count or non-string output raises `EvaluationError` naming the sentence range.
- **Scoring**: `sacrebleu.metrics.BLEU(tokenize, lowercase, smooth_method).corpus_score(...)`. `BleuConfig` defaults are SacreBLEU's own (`13a`, cased, `exp`) and are recorded with the SacreBLEU signature.
- **Result** (`EvaluationResult`): score, sentence count, dataset version, languages, batch size, config, signature, paths, synthetic/real. Synthetic results carry a note that the score does not represent real model performance. `metrics()` gives `bleu_auto` and `evaluation_sentence_count`.
- `evaluation_type` (`synthetic`/`real`) has **no default**. In MLflow tags `real` is recorded as `automatic`.
- **Translator loading** (`load_translator`): spec `package.module:callable`, called as `callable(model_dir, device)`; must return an object with `translate_batch`. The spec is imported and executed as code: trusted input only.
- **`CTranslate2Translator`**: thin adapter over `ctranslate2.Translator.translate_batch`; `tokenize`/`detokenize` are mandatory with no defaults; `translate_options` are passed through. It has only been tested against a fake module. Target-prefix (multilingual) models are not supported.
- Information needed for the real model: training tokenizer + detokenizer, special/target-language tokens, decoding options, reported SacreBLEU settings, the real dataset and its version.

## 5. MLflow service (`services/mlflow_service.py`)
Uses `MlflowClient` only (`get_experiment_by_name`, `create_experiment`, `create_run`, `log_batch`, `log_dict`, `log_artifacts`, `get_registered_model`, `create_registered_model`, `create_model_version`, `get_model_version`, `set_model_version_tag`, `set_tag`, `set_terminated`).

Pre-conditions enforced by `track_and_register` (raise before anything is created):
1. validation passed;
2. exactly one of: evaluation result, evaluation error, explicit skip (with reason);
3. a failed evaluation is refused unless `allow_failed_evaluation=True`;
4. manual BLEU is finite and in [0, 100];
5. duplicate check: a duplicate returns `status="duplicate_skipped"` and creates nothing.

Call sequence and status transitions:
1. get/create experiment (a creation race is tolerated; a deleted experiment raises `RegistrationError`);
2. `create_run` with tags (`registration_status=pending`);
3. `log_batch` params/metrics; `log_dict` evaluation JSON; `log_artifacts` model directory to `model/`;
4. get/create registered model (concurrent creation tolerated);
5. `create_model_version(source="runs:/<run>/model")` with `registration_status=pending`;
6. poll `get_model_version` until `READY` (fails on `FAILED_REGISTRATION` or timeout);
7. only then tag the version `registration_status=registered`, tag the run, and end the run `FINISHED`.

On any exception after run creation (including `KeyboardInterrupt`): the version (if it exists) is tagged `failed`, the run is tagged `failed` and ended `FAILED`; cleanup errors are logged and never mask the original error. MLflow errors are translated: auth -> `MLflowAuthError`, connection/server -> `MLflowConnectionError`, anything else -> `RegistrationError`.

Not done: a failed evaluation leaves no MLflow record (no audit run); large models are uploaded with a single `log_artifacts` call (no resume).

## 6. CLI (`register_ctranslate2_model.py`)
Original arguments are unchanged. New: `--target-language`, `--device`, `--skip-evaluation` (default off), `--translator`, `--evaluation-source`, `--evaluation-reference`, `--evaluation-type`, `--evaluation-dataset-version`, `--evaluation-batch-size`. `--registered-model-name` falls back to `$MODEL_REGISTRY_NAME`.

| Situation | Result |
|---|---|
| translator + dataset paths + `--evaluation-type` | automatic BLEU, `evaluation_status=completed` |
| no translator, no `--skip-evaluation` | exit 4, MLflow not contacted |
| `--skip-evaluation` | skipped; reason `evaluation_not_requested` (translator configured) or `translator_not_configured` |
| `--bleu` | metric `bleu`, `bleu_source=manual`; never `bleu_auto` |
| evaluation fails | exit 4, nothing registered, `--bleu` does not rescue it |

Breaking change: the previous command without evaluation flags now exits 4; add `--skip-evaluation`.

## 7. Other points
- `--device` is recorded as a parameter and passed to the translator factory; it does not select hardware by itself.
- MLflow's HTTP client retries with backoff by default, so an unreachable server can take minutes before exit 3. `MLFLOW_HTTP_REQUEST_MAX_RETRIES` / `MLFLOW_HTTP_REQUEST_TIMEOUT` shorten this.
- `mock_data/` is synthetic; BLEU computed on it (and by the oracle mock translator, ~100) demonstrates plumbing only.
