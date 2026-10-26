import torch
from torch import nn
from transformers import AutoModel, BertConfig


class ProteinTransformerForClassification(nn.Module):
    """Protein encoder with optional numeric-feature fusion for classification."""

    def __init__(
        self,
        model_name,
        model_config=None,
        use_lora=False,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_target_modules=("query", "value"),
        feature_fusion="sequence_only",
        num_features=8,
        cross_attention_heads=8,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            model_name,
            config=model_config,
        )
        self.use_lora = use_lora

        if use_lora:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as error:
                raise ImportError(
                    "LoRA requires PEFT. Install it with `pip install peft`."
                ) from error

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(lora_target_modules),
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.bert = get_peft_model(self.bert, lora_config)

        hidden_size = self.bert.config.hidden_size
        self.feature_fusion = feature_fusion.lower()
        if self.feature_fusion not in {
            "sequence_only",
            "weighted_concat",
            "cross_attention",
        }:
            raise ValueError(
                "feature_fusion must be 'sequence_only', 'weighted_concat', "
                "or 'cross_attention'."
            )
        self.num_features = num_features
        classifier_input_size = hidden_size
        if self.feature_fusion == "weighted_concat":
            self.feature_encoder = nn.Sequential(
                nn.Linear(num_features, hidden_size),
                nn.GELU(),
                nn.Dropout(p=0.1),
            )
            self.modality_weights = nn.Parameter(torch.zeros(2))
            classifier_input_size = hidden_size * 2
        elif self.feature_fusion == "cross_attention":
            if hidden_size % cross_attention_heads != 0:
                raise ValueError(
                    "hidden_size must be divisible by cross_attention_heads."
                )
            self.feature_value_projection = nn.Linear(1, hidden_size)
            self.feature_identity_embedding = nn.Embedding(
                num_features,
                hidden_size,
            )
            self.feature_cross_attention = nn.MultiheadAttention(
                hidden_size,
                num_heads=cross_attention_heads,
                dropout=0.1,
                batch_first=True,
            )
            classifier_input_size = hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(classifier_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        features=None,
    ):
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None and self.bert.config.model_type == "bert":
            model_inputs["token_type_ids"] = token_type_ids
        outputs = self.bert(**model_inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        if self.feature_fusion == "sequence_only":
            fused = cls_embedding
        else:
            if features is None:
                raise ValueError(
                    f"{self.feature_fusion} fusion requires descriptor features."
                )
            if features.ndim != 2 or features.shape[1] != self.num_features:
                raise ValueError(
                    f"features must have shape (batch, {self.num_features})."
                )
            if self.feature_fusion == "weighted_concat":
                feature_embedding = self.feature_encoder(features)
                weights = torch.softmax(self.modality_weights, dim=0)
                fused = torch.cat(
                    (
                        weights[0] * cls_embedding,
                        weights[1] * feature_embedding,
                    ),
                    dim=-1,
                )
            elif self.feature_fusion == "cross_attention":
                feature_ids = torch.arange(
                    self.num_features,
                    device=features.device,
                )
                feature_tokens = (
                    self.feature_value_projection(features.unsqueeze(-1))
                    + self.feature_identity_embedding(feature_ids).unsqueeze(0)
                )
                sequence_embeddings = outputs.last_hidden_state
                attended_features, _ = self.feature_cross_attention(
                    query=sequence_embeddings,
                    key=feature_tokens,
                    value=feature_tokens,
                    need_weights=False,
                )
                if attention_mask is None:
                    sequence_mask = torch.ones(
                        sequence_embeddings.shape[:2],
                        dtype=sequence_embeddings.dtype,
                        device=sequence_embeddings.device,
                    )
                else:
                    sequence_mask = attention_mask.unsqueeze(-1).to(
                        dtype=sequence_embeddings.dtype
                    )
                pooled_attended_features = (
                    attended_features * sequence_mask
                ).sum(dim=1) / sequence_mask.sum(dim=1).clamp_min(1)
                fused = torch.cat(
                    (cls_embedding, pooled_attended_features),
                    dim=-1,
                )
        return self.classifier(fused).squeeze(-1)


class ProtBertForClassification(ProteinTransformerForClassification):
    """ProtBert encoder with a binary classification head."""

    def __init__(self, model_name="Rostlab/prot_bert", **kwargs):
        bert_config = BertConfig.from_pretrained(model_name)
        super().__init__(
            model_name=model_name,
            model_config=bert_config,
            **kwargs,
        )


class ESM2ForClassification(ProteinTransformerForClassification):
    """ESM2 encoder with a binary classification head."""

    def __init__(self, model_name="facebook/esm2_t12_35M_UR50D", **kwargs):
        super().__init__(model_name=model_name, **kwargs)