import torch
from torch import nn
from transformers import BertModel


class ProtBertForClassification(nn.Module):
    """ProtBert encoder with a binary classification head."""

    def __init__(
        self,
        model_name="Rostlab/prot_bert",
        dropout=0.3,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

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