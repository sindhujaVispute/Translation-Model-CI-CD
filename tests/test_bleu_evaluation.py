import json
import math
import sys
import types
from pathlib import Path

import pytest

from services.bleu_evaluation import (
    METRIC_BLEU_AUTO,
    METRIC_SENTENCE_COUNT,
    BleuConfig,
    CTranslate2Translator,
    DatasetError,
    EvaluationError,
    evaluate_bleu,
    load_parallel_dataset,
)

EVAL_DIR = Path(__file__).resolve().parent.parent / "mock_data" / "evaluation"
SRC = EVAL_DIR / "source.txt"
REF = EVAL_DIR / "reference.txt"

KW = dict(dataset_version="synthetic-v0", evaluation_type="synthetic",
          source_language="eng-latn", target_language="spa-latn")


class MockTranslator:
    """Returns controlled predictions: a dict source->output or a function."""

    def __init__(self, outputs):
        self.outputs = outputs
        self.batch_sizes = []

    def translate_batch(self, sources):
        self.batch_sizes.append(len(sources))
        if callable(self.outputs):
            return self.outputs(sources)
        return [self.outputs[s] for s in sources]


def write_pair(tmp_path, sources, references, src_bytes=None):
    s, r = tmp_path / "s.txt", tmp_path / "r.txt"
    s.write_bytes(src_bytes if src_bytes is not None else ("\n".join(sources) + "\n").encode())
    r.write_bytes(("\n".join(references) + "\n").encode())
    return s, r


def oracle(src_path=SRC, ref_path=REF):
    pairs = dict(zip(src_path.read_text(encoding="utf-8").splitlines(),
                     ref_path.read_text(encoding="utf-8").splitlines()))
    return MockTranslator(pairs)


# ---- dataset loading ------------------------------------------------------
def test_synthetic_dataset_loads_and_is_aligned():
    data = load_parallel_dataset(SRC, REF)
    assert len(data.sources) == len(data.references) == 20
    assert data.sources[0] == "The cat sleeps on the sofa."


def test_empty_dataset(tmp_path):
    s, r = tmp_path / "s.txt", tmp_path / "r.txt"
    s.write_text(""); r.write_text("")
    with pytest.raises(DatasetError, match="empty"):
        load_parallel_dataset(s, r)


def test_missing_reference_file(tmp_path):
    s = tmp_path / "s.txt"; s.write_text("a\n")
    with pytest.raises(DatasetError, match="reference file not found"):
        load_parallel_dataset(s, tmp_path / "nope.txt")


def test_missing_source_file(tmp_path):
    r = tmp_path / "r.txt"; r.write_text("a\n")
    with pytest.raises(DatasetError, match="source file not found"):
        load_parallel_dataset(tmp_path / "nope.txt", r)


def test_misaligned_line_counts(tmp_path):
    s, r = write_pair(tmp_path, ["a", "b", "c"], ["x", "y"])
    with pytest.raises(DatasetError, match="3 source lines vs 2 reference"):
        load_parallel_dataset(s, r)


def test_blank_line_is_malformed(tmp_path):
    s, r = write_pair(tmp_path, ["a", "  ", "c"], ["x", "y", "z"])
    with pytest.raises(DatasetError, match="Blank source line.*2"):
        load_parallel_dataset(s, r)


def test_blank_reference_line_is_malformed(tmp_path):
    s, r = write_pair(tmp_path, ["a", "b"], ["x", ""])
    with pytest.raises(DatasetError, match="Blank reference line"):
        load_parallel_dataset(s, r)


def test_invalid_utf8(tmp_path):
    s, r = write_pair(tmp_path, ["a"], ["x"], src_bytes=b"\xff\xfe\xfa\n")
    with pytest.raises(DatasetError, match="UTF-8"):
        load_parallel_dataset(s, r)


def test_crlf_and_bom_handled(tmp_path):
    s, r = tmp_path / "s.txt", tmp_path / "r.txt"
    s.write_bytes(b"\xef\xbb\xbfhello\r\nworld\r\n")
    r.write_bytes(b"hola\r\nmundo")
    data = load_parallel_dataset(s, r)
    assert data.sources == ["hello", "world"] and data.references == ["hola", "mundo"]


def test_unicode_line_separator_does_not_split_lines(tmp_path):
    s, r = write_pair(tmp_path, ["a\u2028b", "c"], ["x", "y"])
    assert len(load_parallel_dataset(s, r).sources) == 2


# ---- BLEU values (controlled predictions) ---------------------------------
def test_perfect_predictions_score_100():
    result = evaluate_bleu(oracle(), SRC, REF, **KW)
    assert result.bleu_score == pytest.approx(100.0)


def test_completely_wrong_predictions_score_0(tmp_path):
    s, r = write_pair(tmp_path, ["one", "two"], ["alpha beta gamma", "delta epsilon"])
    result = evaluate_bleu(MockTranslator(lambda b: ["zzz"] * len(b)), s, r, **KW)
    assert result.bleu_score == pytest.approx(0.0)


def test_known_bleu_value_one_word_wrong(tmp_path):
    # n-gram precisions 8/9, 7/8, 6/7, 5/6, BP=1 -> BLEU = 100 * (5/9) ** 0.25
    s, r = write_pair(tmp_path, ["src"], ["the quick brown fox jumps over the lazy cat"])
    t = MockTranslator({"src": "the quick brown fox jumps over the lazy dog"})
    result = evaluate_bleu(t, s, r, **KW)
    assert result.bleu_score == pytest.approx(100 * (5 / 9) ** 0.25, abs=1e-6)


def test_known_bleu_value_brevity_penalty(tmp_path):
    # all precisions 1.0, hyp 8 tokens vs ref 9 -> BP = exp(1 - 9/8)
    s, r = write_pair(tmp_path, ["src"], ["the quick brown fox jumps over the lazy cat"])
    t = MockTranslator({"src": "the quick brown fox jumps over the lazy"})
    result = evaluate_bleu(t, s, r, **KW)
    assert result.bleu_score == pytest.approx(100 * math.exp(1 - 9 / 8), abs=1e-6)


def test_lowercase_option_changes_score(tmp_path):
    s, r = write_pair(tmp_path, ["src"], ["the quick brown fox jumps"])
    t = MockTranslator({"src": "The Quick Brown Fox Jumps"})
    cased = evaluate_bleu(t, s, r, **KW)
    folded = evaluate_bleu(t, s, r, bleu_config=BleuConfig(lowercase=True), **KW)
    assert cased.bleu_score < 1.0 and folded.bleu_score == pytest.approx(100.0)
    assert folded.bleu_config["lowercase"] is True


# ---- batching, translator contract, failures ------------------------------
def test_batching_respects_batch_size():
    t = oracle()
    result = evaluate_bleu(t, SRC, REF, batch_size=8, **KW)
    assert t.batch_sizes == [8, 8, 4]
    assert result.batch_size == 8


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_batch_size(bad):
    with pytest.raises(ValueError):
        evaluate_bleu(oracle(), SRC, REF, batch_size=bad, **KW)


def test_invalid_evaluation_type():
    with pytest.raises(ValueError, match="evaluation_type"):
        evaluate_bleu(oracle(), SRC, REF, **{**KW, "evaluation_type": "maybe"})


def test_translator_exception_is_wrapped():
    def boom(batch):
        raise RuntimeError("model crashed")

    with pytest.raises(EvaluationError, match="model crashed"):
        evaluate_bleu(MockTranslator(boom), SRC, REF, **KW)


def test_translator_wrong_output_count():
    with pytest.raises(EvaluationError, match="outputs for"):
        evaluate_bleu(MockTranslator(lambda b: b[:-1]), SRC, REF, **KW)


def test_translator_non_string_output():
    with pytest.raises(EvaluationError, match="strings"):
        evaluate_bleu(MockTranslator(lambda b: [None] * len(b)), SRC, REF, **KW)


def test_dataset_errors_propagate_from_evaluate(tmp_path):
    s, r = write_pair(tmp_path, ["a", "b"], ["x"])
    with pytest.raises(DatasetError):
        evaluate_bleu(oracle(), s, r, **KW)


# ---- metadata -------------------------------------------------------------
def test_result_metadata_and_metrics():
    result = evaluate_bleu(oracle(), SRC, REF, **KW)
    assert result.num_sentences == 20
    assert result.dataset_version == "synthetic-v0"
    assert (result.source_language, result.target_language) == ("eng-latn", "spa-latn")
    assert result.bleu_config == {"tokenize": "13a", "lowercase": False, "smooth_method": "exp"}
    assert result.sacrebleu_signature
    assert result.is_synthetic
    assert result.metrics() == {METRIC_BLEU_AUTO: result.bleu_score, METRIC_SENTENCE_COUNT: 20.0}


def test_synthetic_result_is_clearly_labelled_and_json_serialisable():
    d = evaluate_bleu(oracle(), SRC, REF, **KW).to_dict()
    assert d["is_synthetic"] is True and "NOT represent real model performance" in d["note"]
    json.dumps(d)


def test_real_evaluation_has_no_synthetic_note():
    d = evaluate_bleu(oracle(), SRC, REF, **{**KW, "evaluation_type": "real"}).to_dict()
    assert d["is_synthetic"] is False and "note" not in d


# ---- CTranslate2 adapter glue (fake ctranslate2; real model NOT tested) ----
def test_ctranslate2_adapter_uses_supplied_tokenizer_and_options(monkeypatch):
    calls = {}

    class FakeResult:
        def __init__(self, tokens):
            self.hypotheses = [tokens]

    class FakeCT2Translator:
        def __init__(self, path, device, compute_type):
            calls["init"] = (path, device, compute_type)

        def translate_batch(self, tokens, **opts):
            calls["opts"] = opts
            return [FakeResult([t.upper() for t in toks]) for toks in tokens]

    fake = types.ModuleType("ctranslate2")
    fake.Translator = FakeCT2Translator
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)

    tr = CTranslate2Translator(
        "/models/x", tokenize=str.split, detokenize=" ".join,
        device="cpu", translate_options={"beam_size": 3},
    )
    assert tr.translate_batch(["a b", "c"]) == ["A B", "C"]
    assert calls["init"] == ("/models/x", "cpu", "default")
    assert calls["opts"] == {"beam_size": 3}


# ---- translator factory loading ---------------------------------------------
def _good_factory(model_dir, device):
    return MockTranslator(lambda b: list(b))


def _bad_factory(model_dir, device):
    raise RuntimeError("cannot load model")


def _not_a_translator(model_dir, device):
    return object()


def test_load_translator_success_passes_model_dir_and_device():
    from services.bleu_evaluation import load_translator

    tr = load_translator("tests.test_bleu_evaluation:_good_factory", "/m", "cpu")
    assert tr.translate_batch(["a"]) == ["a"]


@pytest.mark.parametrize("spec", ["", "nocolon", ":f", "mod:", "no_such_module_xyz:f",
                                  "tests.test_bleu_evaluation:missing"])
def test_load_translator_bad_specs(spec):
    from services.bleu_evaluation import load_translator

    with pytest.raises(EvaluationError):
        load_translator(spec, "/m")


def test_load_translator_factory_failure_and_wrong_type():
    from services.bleu_evaluation import load_translator

    with pytest.raises(EvaluationError, match="cannot load model"):
        load_translator("tests.test_bleu_evaluation:_bad_factory", "/m")
    with pytest.raises(EvaluationError, match="translate_batch"):
        load_translator("tests.test_bleu_evaluation:_not_a_translator", "/m")
