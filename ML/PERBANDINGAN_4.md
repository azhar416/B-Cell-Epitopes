# perbandingan_4 input contract

`perbandingan_4` consumes precomputed numerical CLS vectors and combines them
with the Liu feature transform and selected classifier. This project does not
generate those vectors. They must come from the same protein BERT encoder and
CLS extraction used to prepare the training and evaluation tables.

For separate input tables, provide:

- `bcell_bert_cls.csv`
- `sars_bert_cls.csv`

The files belong in `dataset/bert_embeddings/` by default, or in the directory
passed with `--bert-embedding-dir`. Each CSV must contain one row per unique
`peptide_seq`, then a contiguous ordered set of numeric columns named
`bert_cls_000`, `bert_cls_001`, and so on. Both files must have the same CLS
column schema. Missing, duplicate, non-numeric, or non-finite vectors are
rejected.

Example invocation:

```powershell
python -m ML.train --training-mode external_sars --feature-set perbandingan_4 --model xgboost --bert-embedding-dir dataset/bert_embeddings
```

The workspace's `EmbeddingML/artifacts/RF/embeddings.npy` contains 1024-value
mean-pooled ProtT5 vectors. That is a different encoder and pooling method than
the protein BERT CLS vectors required here, so the existing file cannot be
renamed or treated as a compatible input.

This experiment keeps the project's existing external evaluation protocol:
model selection uses grouped B-cell folds, and the cleaned SARS set is scored
once after fitting. This does not reproduce Liu et al.'s benchmark datasets or
their validation design; it compares the feature representation in this
project's dataset and protocol.
