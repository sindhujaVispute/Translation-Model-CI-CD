"""DEVELOPMENT ONLY - a fake translator for exercising the pipeline.

It "translates" by looking each sentence up in the synthetic reference file, so
it scores ~100 BLEU on the synthetic data by construction. That demonstrates the
plumbing only and says NOTHING about any real model. Always pair it with
``--evaluation-type synthetic``.

    --translator mock_data.mock_translator:create_translator
"""
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent / "evaluation"


class OracleTranslator:
    def __init__(self, sources, references):
        self._lookup = dict(zip(sources, references))

    def translate_batch(self, sources):
        return [self._lookup[s] for s in sources]  # KeyError => unknown sentence


def create_translator(model_dir, device="cpu"):
    read = lambda name: (_EVAL_DIR / name).read_text(encoding="utf-8").splitlines()  # noqa: E731
    return OracleTranslator(read("source.txt"), read("reference.txt"))
