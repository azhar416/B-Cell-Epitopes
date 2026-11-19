"""Build the ProtT5 vectors that FS-5 (perbandingan_5) reads.

Run this once per dataset. The sweep itself never loads an encoder: it reads
the files written here, which is why ML/requirements.txt pins no torch and no
transformers. This script is a separate tool, deliberately outside that pinned
environment, so generating vectors cannot disturb the versions the sweep is
reproduced under.

    python -m ML.build_prott5_assets --datasets bcell sars

What it writes, per dataset, into ML/assets/perbandingan_5:

    <dataset>_prott5.npz    sequences, embeddings, model_name, tokenizer_name
    <dataset>_prott5.json   provenance: encoder, pooling, counts, SHA-256

The JSON exists because a committed binary whose origin has to be taken on
trust is worth less than one whose origin is recorded. Nothing here infers
what produced a file; the file says so.

Why embedding every row before the fold split is not leakage: the encoder is
frozen, pretrained on UniRef50, and never fitted to this data, and each
peptide is encoded on its own. The vector for one peptide is the same whether
it is computed alone or beside fourteen thousand others, so this is a fixed
function of the peptide exactly as peptide_length is. The transforms that DO
depend on other rows -- the imputer, and the scaler for logistic regression --
live inside the per-fold Pipeline, where scikit-learn fits them on the
training fold only.

Requires torch, transformers and sentencepiece, which the sweep does not:

    pip install torch transformers sentencepiece
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

import pandas as pd

from ML.dataset import (
    DATASET_FILENAMES,
    PROTTRANS_DIMENSION,
    PROTTRANS_MODEL_NAME,
    PROTTRANS_TOKENIZER_NAME,
    load_labeled_dataset,
)
from ML.pre_dataset import normalize_sequence

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "perbandingan_5"
POOLING = "mean over non-special ProtT5 residue tokens"


def _load_encoder(device_name: str):
    """Load the frozen encoder, or explain what is missing."""
    try:
        import torch
        from transformers import AutoTokenizer, T5EncoderModel
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit(
            "This tool needs torch, transformers and sentencepiece, which the "
            "sweep environment deliberately does not carry. Install them "
            "first:\n    pip install torch transformers sentencepiece"
        ) from error

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"--device {device_name} was asked for, but CUDA is absent.")

    tokenizer = AutoTokenizer.from_pretrained(
        PROTTRANS_TOKENIZER_NAME, do_lower_case=False
    )
    # float32 on CPU, float16 on GPU: the same choice the repository's earlier
    # cache was built under. The two agree to a cosine similarity of 0.9999998,
    # so a cache built either way is usable, but the JSON records which.
    model = T5EncoderModel.from_pretrained(
        PROTTRANS_MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return torch, tokenizer, model, device


def encode(sequences: list[str], batch_size: int,
           device_name: str) -> tuple[np.ndarray, str, str]:
    """Mean-pool each peptide over its residue tokens, special tokens excluded.

    Returns the vectors plus the RESOLVED device and dtype. Recording the
    request ("auto") instead would make the provenance useless: a GPU build
    and a CPU build would both say "auto", while their vectors differ by
    about 3e-4 because one is float16 and the other float32.
    """
    if not sequences:
        raise ValueError("Cannot encode an empty sequence list.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    torch, tokenizer, model, device = _load_encoder(device_name)
    print(f"  encoder on {device}, batch {batch_size}")

    vectors = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            encoded = tokenizer(
                # ProtT5 expects residues separated by spaces.
                [" ".join(sequence) for sequence in batch],
                padding=True,
                truncation=False,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            special = encoded.pop("special_tokens_mask", None)
            inputs = {
                key: value.to(device)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask"}
            }
            hidden = model(**inputs).last_hidden_state
            if special is None:
                special = torch.zeros_like(inputs["input_ids"])
                for token_id in tokenizer.all_special_ids:
                    special |= inputs["input_ids"] == token_id
            residues = inputs["attention_mask"].bool() & ~special.to(device).bool()
            weights = residues.unsqueeze(-1).to(hidden.dtype)
            counts = weights.sum(dim=1)
            if (counts == 0).any():
                raise ValueError("A peptide tokenized to no residue tokens.")
            vectors.append(((hidden * weights).sum(dim=1) / counts).float().cpu().numpy())
            done = min(start + batch_size, len(sequences))
            if done % (batch_size * 25) == 0 or done == len(sequences):
                print(f"    {done:,}/{len(sequences):,}")
    return (
        np.concatenate(vectors, axis=0).astype(np.float32, copy=False),
        device.type,
        str(model.dtype).replace("torch.", ""),
    )


def peptide_list(dataset_name: str, data_dir: Path, label_free: bool,
                 max_peptide_length: int = 20) -> list[str]:
    """The peptides to embed, with or without opening the label column.

    label_free exists so phase 1 can build the SARS vectors without reading
    SARS labels: it takes every unique peptide string in the raw CSV and never
    touches `target`. A superset is safe because the loader left-merges the
    cache on peptide_seq and only requires that every cleaned row find a
    vector, not that the two sets match.
    """
    if not label_free:
        # feature_set="proposed" on purpose: asking for perbandingan_5 here
        # would require the very vectors this is about to write.
        frame = load_labeled_dataset(
            dataset_name, data_dir, max_peptide_length=max_peptide_length,
            feature_set="proposed",
        )
        sequences = frame["peptide_seq"].astype(str).tolist()
        if len(set(sequences)) != len(sequences):
            raise SystemExit(
                f"{dataset_name} has repeated peptide_seq values after cleaning."
            )
        print(f"{dataset_name}: {len(sequences):,} cleaned peptides (labels read)")
        return sequences

    csv_path = Path(data_dir) / DATASET_FILENAMES[dataset_name]
    if not csv_path.is_file():
        raise SystemExit(f"Dataset file not found: {csv_path}")
    raw = pd.read_csv(csv_path, usecols=["peptide_seq"])
    normalized = [
        normalize_sequence(value, row_label=index)
        for index, value in raw["peptide_seq"].items()
    ]
    unique = list(dict.fromkeys(normalized))
    # Cleaning drops anything longer than MAX_PEPTIDE_LENGTH, so embedding
    # those would be wasted work -- and expensive work, because one 393-residue
    # peptide in SARS pads its whole batch out to 393. Length is not a label,
    # so applying the same limit here keeps this path label-free. What is left
    # is still a superset of the cleaned peptides, which is all the loader
    # needs.
    sequences = [s for s in unique if len(s) <= max_peptide_length]
    dropped = len(unique) - len(sequences)
    print(f"{dataset_name}: {len(sequences):,} unique peptides of at most "
          f"{max_peptide_length} residues, from {len(raw):,} rows "
          f"({dropped:,} overlength skipped; labels NOT read)")
    return sequences


def build(dataset_name: str, data_dir: Path, batch_size: int, device: str,
          force: bool, label_free: bool = False,
          max_peptide_length: int = 20) -> Path:
    target = ASSET_DIR / f"{dataset_name}_prott5.npz"
    if target.is_file() and not force:
        print(f"{target.name} already present; skipping. Pass --force to rebuild.")
        return target

    sequences = peptide_list(dataset_name, data_dir, label_free,
                             max_peptide_length=max_peptide_length)

    embeddings, resolved_device, resolved_dtype = encode(
        sequences, batch_size, device
    )
    if embeddings.shape != (len(sequences), PROTTRANS_DIMENSION):
        raise SystemExit(f"Unexpected embedding shape {embeddings.shape}.")
    if not np.isfinite(embeddings).all():
        raise SystemExit("Encoder produced non-finite values.")

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        embeddings=embeddings,
        sequences=np.asarray(sequences, dtype=str),
        model_name=np.asarray(PROTTRANS_MODEL_NAME),
        tokenizer_name=np.asarray(PROTTRANS_TOKENIZER_NAME),
    )
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(".json").write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "encoder": PROTTRANS_MODEL_NAME,
                "tokenizer": PROTTRANS_TOKENIZER_NAME,
                "pooling": POOLING,
                "embedding_dimension": int(embeddings.shape[1]),
                "number_of_sequences": int(embeddings.shape[0]),
                "max_peptide_length": max_peptide_length,
                "peptide_source": (
                    "every unique peptide_seq in the raw CSV; labels not read"
                    if label_free else "cleaned dataset peptides"
                ),
                "device_requested": device,
                "device": resolved_device,
                "dtype": resolved_dtype,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "sha256": digest,
                "file": target.name,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  wrote {target.name}  {embeddings.shape}  sha256 {digest[:12]}")
    return target


def verify(dataset_names: list[str], data_dir: Path,
           labeled: bool = True) -> None:
    """Check the caches on their own, then optionally through the loader.

    The cache-only checks need no labels, so they run even when phase 1 built
    the SARS vectors label-free.
    """
    print()
    print("Verifying the caches:")
    caches = {}
    for dataset_name in dataset_names:
        path = ASSET_DIR / f"{dataset_name}_prott5.npz"
        with np.load(path, allow_pickle=False) as cache:
            sequences = cache["sequences"].astype(str)
            embeddings = cache["embeddings"].astype(np.float32)
            model_name = str(cache["model_name"].item())
            tokenizer_name = str(cache["tokenizer_name"].item())
        if (model_name, tokenizer_name) != (PROTTRANS_MODEL_NAME,
                                            PROTTRANS_TOKENIZER_NAME):
            raise SystemExit(f"{path.name} names a different encoder.")
        if embeddings.shape != (len(sequences), PROTTRANS_DIMENSION):
            raise SystemExit(f"{path.name} has shape {embeddings.shape}.")
        if not np.isfinite(embeddings).all():
            raise SystemExit(f"{path.name} carries non-finite values.")
        if len(set(sequences.tolist())) != len(sequences):
            raise SystemExit(f"{path.name} has duplicate peptide keys.")
        caches[dataset_name] = (sequences, embeddings)
        print(f"  {dataset_name}: {len(sequences):,} x {PROTTRANS_DIMENSION}, "
              "finite, unique keys, encoder matches")

    # A peptide in both datasets must carry the same vector: the encoder sees
    # only the peptide, so a difference would mean the two files were not built
    # the same way -- most likely on different devices, where float16 on GPU
    # and float32 on CPU disagree by about 3e-4. The external comparison would
    # then be made across two feature spaces.
    if {"bcell", "sars"} <= set(caches):
        left_keys, left_vectors = caches["bcell"]
        right_keys, right_vectors = caches["sars"]
        left_at = {key: index for index, key in enumerate(left_keys.tolist())}
        right_at = {key: index for index, key in enumerate(right_keys.tolist())}
        shared = sorted(set(left_at) & set(right_at))
        if shared:
            gap = float(max(
                np.abs(left_vectors[left_at[key]] - right_vectors[right_at[key]]).max()
                for key in shared
            ))
            print(f"  {len(shared)} peptides appear in both; "
                  f"largest vector difference {gap:.2e}")
            if gap > 1e-4:
                raise SystemExit(
                    "Shared peptides differ between the two files, so they are "
                    "not in one feature space. Rebuild both on one device."
                )

    if labeled:
        print("Verifying through ML.dataset.load_labeled_dataset:")
        for dataset_name in dataset_names:
            frame = load_labeled_dataset(
                dataset_name, data_dir, feature_set="perbandingan_5"
            )
            columns = [n for n in frame.columns if n.startswith("prott5_")]
            if len(columns) != PROTTRANS_DIMENSION:
                raise SystemExit(f"{dataset_name} produced {len(columns)} columns.")
            print(f"  {dataset_name}: {len(frame):,} rows x {len(columns)} columns")
    print("  -> usable by the sweep")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--datasets", nargs="+", default=["bcell", "sars"],
                        choices=["bcell", "sars"])
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "dataset")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto",
                        help="auto (default), cpu, or cuda")
    parser.add_argument("--force", action="store_true",
                        help="rebuild a cache that is already present")
    parser.add_argument(
        "--label-free", nargs="*", default=None, choices=["bcell", "sars"],
        help=(
            "embed these datasets from the raw CSV's unique peptides without "
            "reading their label column. Phase 1 uses '--label-free sars' so "
            "it can build the external vectors while leaving SARS labels "
            "untouched."
        ),
    )
    arguments = parser.parse_args()
    label_free = set(arguments.label_free or ())

    for dataset_name in arguments.datasets:
        build(dataset_name, arguments.data_dir, arguments.batch_size,
              arguments.device, arguments.force,
              label_free=dataset_name in label_free)
    # Round-tripping through the labeled loader would read the labels of every
    # dataset, which is the thing --label-free exists to avoid.
    verify(arguments.datasets, arguments.data_dir, labeled=not label_free)


if __name__ == "__main__":
    main()
