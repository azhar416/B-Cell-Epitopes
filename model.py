import torch
from torch import nn
from transformers import BertModel


class ProtBertForClassification(nn.Module):
    """ProtBert encoder with a binary classification head."""

    def __init__(
        self,
        model_name="Rostlab/prot_bert",
        dropout=0.3,
        use_lora=False,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_target_modules=("query", "value"),
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
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
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_embedding).squeeze(-1)