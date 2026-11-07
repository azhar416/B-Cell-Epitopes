"""Frozen ProtT5 embedding extraction with validated on-disk caching."""

import json
from pathlib import Path
from typing import Protocol

import numpy as np


class EmbeddingEncoder(Protocol):
    def encode(
        self,
        sequences: list[str],
        batch_size: int,
        device: str,
    ) -> np.ndarray: ...


def resolve_device(device_name: str) -> str:
    import torch

    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"DEVICE={device_name!r} was requested, but CUDA is unavailable."
        )
    return device_name


class ProtT5Encoder:
    def __init__(self, model_name: str, tokenizer_name: str, device: str):
        import torch
        from transformers import AutoTokenizer, T5EncoderModel

        self.device = torch.device(resolve_device(device))
        model_dtype = (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            do_lower_case=False,
        )
        self.model = T5EncoderModel.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(
        self,
        sequences: list[str],
        batch_size: int,
        device: str,
    ) -> np.ndarray:
        import torch

        if not sequences:
            raise ValueError("Cannot encode an empty sequence list.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if torch.device(device) != self.device:
            raise ValueError("The encoder device cannot change after model loading.")

        embeddings = []
        with torch.inference_mode():
            for start in range(0, len(sequences), batch_size):
                batch_sequences = sequences[start : start + batch_size]
                tokenized_sequences = [
                    " ".join(sequence) for sequence in batch_sequences
                ]
                encoded = self.tokenizer(
                    tokenized_sequences,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                    return_special_tokens_mask=True,
                )
                special_tokens_mask = encoded.pop("special_tokens_mask", None)
                model_inputs = {
                    key: value.to(self.device)
                    for key, value in encoded.items()
                    if key in {"input_ids", "attention_mask"}
                }
                outputs = self.model(**model_inputs)
                if special_tokens_mask is None:
                    special_tokens_mask = torch.zeros_like(
                        model_inputs["input_ids"]
                    )
                    for token_id in self.tokenizer.all_special_ids:
                        special_tokens_mask |= (
                            model_inputs["input_ids"] == token_id
                        )
                residue_mask = model_inputs["attention_mask"].bool()
                residue_mask &= ~special_tokens_mask.to(self.device).bool()
                weights = residue_mask.unsqueeze(-1).to(
                    outputs.last_hidden_state.dtype
                )
                pooled = (outputs.last_hidden_state * weights).sum(dim=1)
                residue_counts = weights.sum(dim=1)
                if (residue_counts == 0).any():
                    raise ValueError(
                        "ProtT5 tokenizer produced a sequence with no residue tokens."
                    )
                pooled = pooled / residue_counts
                embeddings.append(pooled.float().cpu().numpy())
        return np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)


def load_or_create_embeddings(
    sequences: list[str],
    *,
    model_name: str,
    tokenizer_name: str,
    cache_path: str | Path,
    batch_size: int,
    device: str,
    force_recompute: bool = False,
    encoder: EmbeddingEncoder | None = None,
) -> np.ndarray:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file() and not force_recompute:
        with np.load(cache_path, allow_pickle=False) as cache:
            cached_sequences = cache["sequences"].astype(str).tolist()
            cached_model = str(cache["model_name"].item())
            cached_tokenizer = str(cache["tokenizer_name"].item())
            cached_embeddings = cache["embeddings"].astype(np.float32)
        if (
            cached_sequences == sequences
            and cached_model == model_name
            and cached_tokenizer == tokenizer_name
        ):
            if cached_embeddings.shape[0] != len(sequences):
                raise ValueError("Cached embedding count does not match sequences.")
            print(f"Using cached ProtT5 embeddings: {cache_path}")
            return cached_embeddings
        print("Embedding cache does not match the current data/model; recomputing.")

    resolved_device = resolve_device(device)
    active_encoder = encoder or ProtT5Encoder(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        device=resolved_device,
    )
    embeddings = active_encoder.encode(
        sequences,
        batch_size=batch_size,
        device=resolved_device,
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(sequences):
        raise ValueError("Encoder returned embeddings with an invalid shape.")
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        sequences=np.asarray(sequences, dtype=str),
        model_name=np.asarray(model_name),
        tokenizer_name=np.asarray(tokenizer_name),
    )
    metadata = {
        "encoder": model_name,
        "tokenizer": tokenizer_name,
        "embedding_dimension": int(embeddings.shape[1]),
        "number_of_sequences": int(embeddings.shape[0]),
        "pooling": "mean over non-special ProtT5 residue tokens",
        "cache_file": str(cache_path),
    }
    cache_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return embeddings
