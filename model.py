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
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size // 2),
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
    ):
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None and self.bert.config.model_type == "bert":
            model_inputs["token_type_ids"] = token_type_ids
        outputs = self.bert(**model_inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
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

    def __init__(self, model_name="facebook/esm2_t12_35M_UR50D", **kwargs):
        super().__init__(model_name=model_name, **kwargs)