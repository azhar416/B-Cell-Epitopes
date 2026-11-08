import torch
from torch import nn
from transformers import AutoModel, BertConfig, T5EncoderModel


def load_protein_encoder(model_type, model_name, model_config=None):
    """Load the sequence encoder used by the selected protein language model."""
    model_type = model_type.lower()
    if model_type == "prott5":
        return T5EncoderModel.from_pretrained(
            model_name,
            config=model_config,
        )
    if model_type in {"protbert", "esm2"}:
        return AutoModel.from_pretrained(
            model_name,
            config=model_config,
        )
    raise ValueError("model_type must be 'protbert', 'esm2', or 'prott5'.")


def pool_protein_embeddings(
    hidden_states,
    attention_mask,
    model_type,
    special_tokens_mask=None,
):
    """Pool encoder output using its classification token or ProtT5 residues."""
    if model_type.lower() != "prott5":
        return hidden_states[:, 0, :]

    if attention_mask is None:
        pooling_mask = torch.ones(
            hidden_states.shape[:2],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    else:
        pooling_mask = attention_mask.to(dtype=hidden_states.dtype)
    if special_tokens_mask is not None:
        pooling_mask = pooling_mask * (
            1 - special_tokens_mask.to(dtype=hidden_states.dtype)
        )
    pooling_mask = pooling_mask.unsqueeze(-1)
    return (hidden_states * pooling_mask).sum(dim=1) / pooling_mask.sum(
        dim=1
    ).clamp_min(1)


class ProteinTransformerForClassification(nn.Module):
    """Protein encoder with optional numeric-feature fusion for classification."""

    encoder_type = "esm2"

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
        classifier_type="pooled_mlp",
        cnn_channels=128,
        cnn_dropout=0.3,
        classifier_hidden_size=None,
        classifier_dropout=0.5,
        num_features=8,
        cross_attention_heads=8,
        msd_samples=1,
        msd_dropout=0.3,
        use_dipeptide_features=False,
        num_dipeptide_features=400,
    ):
        super().__init__()
        if msd_samples < 1:
            raise ValueError("msd_samples must be at least 1.")
        if not 0 <= msd_dropout < 1:
            raise ValueError("msd_dropout must be in the range [0, 1).")
        self.bert = load_protein_encoder(
            self.encoder_type,
            model_name,
            model_config=model_config,
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
            "se",
            "gmu",
        }:
            raise ValueError(
                "feature_fusion must be 'sequence_only', 'weighted_concat', "
                "'cross_attention', 'se', or 'gmu'."
            )
        self.num_features = num_features
        self.use_dipeptide_features = use_dipeptide_features
        self.num_dipeptide_features = num_dipeptide_features
        self.msd_samples = msd_samples
        self.msd_dropout = nn.Dropout(p=msd_dropout)
        self.classifier_type = classifier_type.lower()
        if self.classifier_type not in {
            "pooled_mlp",
            "cnn",
            "attention_pooling",
            "bigru",
        }:
            raise ValueError(
                "classifier_type must be 'pooled_mlp', 'cnn', "
                "'attention_pooling', or 'bigru'."
            )
        if classifier_hidden_size is None:
            classifier_hidden_size = hidden_size // 2
        if not isinstance(classifier_hidden_size, int) or classifier_hidden_size < 1:
            raise ValueError("classifier_hidden_size must be a positive integer or None.")
        if not 0 <= classifier_dropout < 1:
            raise ValueError("classifier_dropout must be in the range [0, 1).")
        if self.classifier_type == "cnn":
            if cnn_channels == "legacy":
                cnn_channels = max(16, hidden_size // 4)
            elif not isinstance(cnn_channels, int) or cnn_channels < 1:
                raise ValueError(
                    "cnn_channels must be a positive integer or 'legacy'."
                )
            if not 0 <= cnn_dropout < 1:
                raise ValueError("cnn_dropout must be in the range [0, 1).")
            self.cnn_channels = cnn_channels
            self.sequence_convolutions = nn.ModuleList(
                [
                    nn.Conv1d(
                        hidden_size,
                        self.cnn_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    )
                    for kernel_size in (3, 5)
                ]
            )
            self.sequence_convolution_projections = nn.ModuleList(
                [
                    nn.Linear(self.cnn_channels * 2, hidden_size)
                    for _ in self.sequence_convolutions
                ]
            )
            self.sequence_convolution_dropout = nn.Dropout(p=cnn_dropout)
            self.sequence_convolution_output = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
            )
        elif self.classifier_type == "attention_pooling":
            self.sequence_attention = nn.Linear(hidden_size, 1)
        elif self.classifier_type == "bigru":
            recurrent_size = max(1, hidden_size // 2)
            recurrent_output_size = recurrent_size * 2
            self.sequence_gru = nn.GRU(
                input_size=hidden_size,
                hidden_size=recurrent_size,
                batch_first=True,
                bidirectional=True,
            )
            self.sequence_gru_projection = (
                nn.Identity()
                if recurrent_output_size == hidden_size
                else nn.Linear(recurrent_output_size, hidden_size)
            )
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
        elif self.feature_fusion == "se":
            self.se_feature_encoder = nn.Sequential(
                nn.Linear(num_features, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
            )
        elif self.feature_fusion == "gmu":
            self.gmu_sequence_projection = nn.Linear(hidden_size, hidden_size)
            self.gmu_feature_projection = nn.Linear(num_features, hidden_size)
            self.gmu_gate = nn.Linear(hidden_size * 2, hidden_size)

        if self.use_dipeptide_features:
            self.dipeptide_feature_encoder = nn.Sequential(
                nn.Linear(num_dipeptide_features, hidden_size),
                nn.GELU(),
                nn.Dropout(p=0.3),
            )
            classifier_input_size += hidden_size

        self.classifier_hidden = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(classifier_input_size, classifier_hidden_size),
            nn.ReLU(),
        )
        self.classifier_output = nn.Linear(classifier_hidden_size, 1)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        features=None,
        special_tokens_mask=None,
        dipeptide_features=None,
    ):
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None and self.bert.config.model_type == "bert":
            model_inputs["token_type_ids"] = token_type_ids
        outputs = self.bert(**model_inputs)
        head_dtype = self.classifier_hidden[1].weight.dtype
        sequence_embeddings = outputs.last_hidden_state.to(dtype=head_dtype)
        if attention_mask is None:
            residue_mask = torch.ones(
                sequence_embeddings.shape[:2],
                dtype=torch.bool,
                device=sequence_embeddings.device,
            )
        else:
            residue_mask = attention_mask.to(
                device=sequence_embeddings.device,
                dtype=torch.bool,
            )
        if special_tokens_mask is not None:
            residue_mask = residue_mask & ~special_tokens_mask.to(
                device=sequence_embeddings.device,
                dtype=torch.bool,
            )
        cls_embedding = pool_protein_embeddings(
            sequence_embeddings,
            attention_mask,
            self.encoder_type,
            special_tokens_mask,
        )
        sequence_embedding = cls_embedding
        if self.classifier_type != "pooled_mlp":
            sequence_embedding = self._pool_sequence_for_classifier(
                sequence_embeddings,
                residue_mask,
            )
        if self.feature_fusion == "sequence_only":
            fused = sequence_embedding
        else:
            if features is None:
                raise ValueError(
                    f"{self.feature_fusion} fusion requires descriptor features."
                )
            if features.ndim != 2 or features.shape[1] != self.num_features:
                raise ValueError(
                    f"features must have shape (batch, {self.num_features})."
                )
            features = features.to(dtype=head_dtype)
            if self.feature_fusion == "weighted_concat":
                feature_embedding = self.feature_encoder(features)
                weights = torch.softmax(self.modality_weights, dim=0)
                fused = torch.cat(
                    (
                        weights[0] * sequence_embedding,
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
                    (sequence_embedding, pooled_attended_features),
                    dim=-1,
                )
            elif self.feature_fusion == "se":
                feature_gate = self.se_feature_encoder(features)
                fused = sequence_embedding * feature_gate
            elif self.feature_fusion == "gmu":
                sequence_projection = torch.tanh(
                    self.gmu_sequence_projection(sequence_embedding)
                )
                feature_projection = torch.tanh(
                    self.gmu_feature_projection(features)
                )
                gate = torch.sigmoid(
                    self.gmu_gate(
                        torch.cat(
                            (sequence_projection, feature_projection),
                            dim=-1,
                        )
                    )
                )
                fused = (
                    gate * sequence_projection
                    + (1 - gate) * feature_projection
                )

        if self.use_dipeptide_features:
            if dipeptide_features is None:
                raise ValueError(
                    "DPC-enabled models require dipeptide_features."
                )
            if (
                dipeptide_features.ndim != 2
                or dipeptide_features.shape[1] != self.num_dipeptide_features
            ):
                raise ValueError(
                    "dipeptide_features must have shape "
                    f"(batch, {self.num_dipeptide_features})."
                )
            dipeptide_embedding = self.dipeptide_feature_encoder(
                dipeptide_features.to(dtype=head_dtype)
            )
            fused = torch.cat((fused, dipeptide_embedding), dim=-1)
        elif dipeptide_features is not None:
            raise ValueError(
                "dipeptide_features were supplied but the model is not "
                "configured to use them."
            )

        hidden = self.classifier_hidden(fused)
        if self.training and self.msd_samples > 1:
            logits = torch.stack(
                [
                    self.classifier_output(self.msd_dropout(hidden))
                    for _ in range(self.msd_samples)
                ],
                dim=0,
            ).mean(dim=0)
        else:
            classifier_input = (
                self.msd_dropout(hidden) if self.training else hidden
            )
            logits = self.classifier_output(classifier_input)
        return torch.sigmoid(logits).squeeze(-1)

    def _pool_sequence_for_classifier(self, sequence_embeddings, residue_mask):
        if not residue_mask.any(dim=1).all():
            raise ValueError("Each sequence must contain at least one residue.")

        if self.classifier_type == "attention_pooling":
            scores = self.sequence_attention(sequence_embeddings).squeeze(-1)
            scores = scores.masked_fill(~residue_mask, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            return (sequence_embeddings * weights).sum(dim=1)

        if self.classifier_type == "cnn":
            sequence_input = sequence_embeddings * residue_mask.unsqueeze(-1)
            channels_first = sequence_input.transpose(1, 2)
            pooled_branches = []
            for convolution, projection in zip(
                self.sequence_convolutions,
                self.sequence_convolution_projections,
            ):
                activations = torch.relu(convolution(channels_first)).transpose(
                    1,
                    2,
                )
                masked_activations = activations.masked_fill(
                    ~residue_mask.unsqueeze(-1),
                    torch.finfo(activations.dtype).min,
                )
                maximum = masked_activations.max(dim=1).values
                mean = (
                    activations * residue_mask.unsqueeze(-1)
                ).sum(dim=1) / residue_mask.sum(dim=1, keepdim=True).clamp_min(1)
                pooled_branches.append(
                    torch.relu(
                        projection(
                            self.sequence_convolution_dropout(
                                torch.cat((maximum, mean), dim=-1)
                            )
                        )
                    )
                )
            return self.sequence_convolution_output(
                torch.cat(pooled_branches, dim=-1)
            )

        if self.classifier_type == "bigru":
            positions = torch.arange(
                1,
                sequence_embeddings.shape[1] + 1,
                device=residue_mask.device,
            ).unsqueeze(0)
            lengths = positions.masked_fill(~residue_mask, 0).max(dim=1).values
            packed = nn.utils.rnn.pack_padded_sequence(
                sequence_embeddings,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.sequence_gru(packed)
            recurrent_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=sequence_embeddings.shape[1],
            )
            recurrent_output = self.sequence_gru_projection(recurrent_output)
            weights = residue_mask.unsqueeze(-1).to(
                dtype=recurrent_output.dtype
            )
            return (recurrent_output * weights).sum(dim=1) / weights.sum(
                dim=1
            ).clamp_min(1)

        raise ValueError(f"Unsupported classifier_type: {self.classifier_type!r}.")


class ProtBertForClassification(ProteinTransformerForClassification):
    """ProtBert encoder with a binary classification head."""

    encoder_type = "protbert"

    def __init__(self, model_name="Rostlab/prot_bert", **kwargs):
        bert_config = BertConfig.from_pretrained(model_name)
        super().__init__(
            model_name=model_name,
            model_config=bert_config,
            **kwargs,
        )


class ESM2ForClassification(ProteinTransformerForClassification):
    """ESM2 encoder with a binary classification head."""

    encoder_type = "esm2"

    def __init__(self, model_name="facebook/esm2_t12_35M_UR50D", **kwargs):
        super().__init__(model_name=model_name, **kwargs)


class ProtT5ForClassification(ProteinTransformerForClassification):
    """ProtT5 encoder with a binary classification head."""

    encoder_type = "prott5"

    def __init__(
        self,
        model_name="Rostlab/prot_t5_xl_half_uniref50-enc",
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)