# ProtT5 Embedding + Classical ML

Standalone workflow in `EmbeddingML/`: clean labeled peptide data, compute frozen ProtT5 peptide embeddings, then train and evaluate an SVM or Random Forest. This directory does not import from `ML/` or `TF/`.

## Data preparation

The loader uses the same labeled CSV schema and cleaning rules as the existing ML/TF pipelines:

- Requires `parent_protein_id`, `protein_seq`, 1-based `start_position`/`end_position`, `peptide_seq`, `target`, and the eight biological descriptor columns.
- Normalizes sequence case/whitespace and maps rare amino acids `B/U/Z/O` to `X`.
- Rejects invalid residues, labels, missing protein IDs, or invalid coordinates.
- Removes peptides longer than 20 residues, removes all examples of peptides with contradictory labels, and collapses exact duplicate peptide sequences.
- Retains the original CSV row number as `source_row_index` so artifacts can be traced to source records.

The ProtT5 encoder receives only the normalized peptide sequence, formatted as space-separated residues. It is frozen and mean-pools non-special residue embeddings to one vector per peptide. Optional biological-feature fusion adds the eight descriptor columns to the PLM vector; missing descriptors are imputed inside the model pipeline using training data only.

## Splits

- B-cell uses a deterministic protein-group holdout targeting 80:20 rows and similar positive-label prevalence. Entire `parent_protein_id` groups remain in one partition; the row fraction can vary slightly because protein group sizes differ.
- Current SARS data contains one protein, so protein-level holdout is impossible. SARS uses stratified peptide-level 80:20 and the report explicitly flags shared protein identity; this does not measure generalization to unseen proteins.
- Split assignments and exact indices are saved to `split_assignments.csv`.

## Run

Install the pipeline dependencies from the repository root:

```powershell
python -m pip install -r EmbeddingML/requirements.txt
```

The default is ProtT5 + RBF SVM on B-cell:

```powershell
python -m EmbeddingML.train
```

Train Random Forest, select SARS, or add the biological descriptors:

```powershell
python -m EmbeddingML.train --model random_forest
python -m EmbeddingML.train --dataset sars
python -m EmbeddingML.train --use-biological-features
```

Other options include `--device auto|cpu|cuda|cuda:0`, `--batch-size`, `--data-dir`, `--output-dir`, and `--force-recompute-embeddings`.

The first run downloads ProtT5 from Hugging Face and stores a reusable, sequence-validated embedding cache. Internet access and sufficient memory/GPU capacity are needed for first-time embedding generation. CPU embedding is supported but the XL encoder may be slow and memory intensive.

## Outputs

Artifacts are written to `EmbeddingML/artifacts/<dataset>/<model>/`:

- `model.joblib`, `metrics.json`, `cleaned_dataset.csv`
- `split_assignments.csv`, `training_predictions.csv`, `validation_predictions.csv`
- `evaluation_curves.png`, `confusion_matrix.png`

The shared per-dataset `prott5_embeddings.npz` cache stores embeddings together with ordered sequences and encoder/tokenizer identifiers. Cache reuse is rejected if the peptide order or model identifiers differ. SVM uses imputation and standardization fitted only on its training partition; Random Forest uses training-fitted imputation without scaling.

Validation metrics include ROC-AUC and average precision (PR-AUC) computed from probabilities, plus thresholded classification metrics at the configured threshold. SVM/RF comparisons use the same deterministic split when run with the same configuration.
