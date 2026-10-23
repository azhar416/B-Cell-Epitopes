import torch
from torch import nn
from transformers import AutoModel, BertConfig


class ProteinTransformerForClassification(nn.Module):
    """Protein language model encoder with a linear binary classification head."""

    def __init__(
        self,
        model_name,
        model_config=None,
        use_lora=False,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_target_modules=("query", "value"),
        feature_dim=0,
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
        if feature_dim < 0:
            raise ValueError("feature_dim must be non-negative.")
        self.feature_dim = feature_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size + feature_dim, hidden_size // 2),
            nn.GELU(),
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
        if self.feature_dim:
            if features is None:
                raise ValueError(
                    f"This model requires {self.feature_dim} additional features."
                )
            if features.ndim != 2 or features.shape != (
                cls_embedding.shape[0],
                self.feature_dim,
            ):
                raise ValueError(
                    "features must have shape "
                    f"({cls_embedding.shape[0]}, {self.feature_dim})."
                )
            cls_embedding = torch.cat(
                (cls_embedding, features.to(dtype=cls_embedding.dtype)),
                dim=-1,
            )
        elif features is not None:
            raise ValueError("This model was created without additional features.")
        return self.classifier(cls_embedding).squeeze(-1)


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

    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", **kwargs):
        super().__init__(model_name=model_name, **kwargs)