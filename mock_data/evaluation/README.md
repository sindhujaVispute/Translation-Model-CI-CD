# Synthetic evaluation data

`source.txt` / `reference.txt` are **synthetic**: 20 short hand-written
English/Spanish sentence pairs used only to exercise the evaluation code.
Any BLEU computed on them says nothing about real model quality.
To use a real dataset, pass its paths instead (one sentence per line, UTF-8,
line-aligned) and record its version.
