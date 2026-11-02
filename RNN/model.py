import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from RNN.dataset import PAD_ID, VOCAB_SIZE


class BiLSTMClassifier(nn.Module):
    """BiLSTM over peptide plus protein context, fused with tabular features."""

    def __init__(
        self,
        tabular_feature_count=10,
        embedding_dim=64,
        hidden_size=128,
        num_layers=1,
        dropout=0.3,
    ):
        super().__init__()
        if tabular_feature_count < 1:
            raise ValueError("tabular_feature_count must be positive.")
        if embedding_dim < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("Embedding size, hidden size, and layers must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        self.embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=embedding_dim,
            padding_idx=PAD_ID,
        )
        self.sequence_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.sequence_dropout = nn.Dropout(dropout)
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_feature_count, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2 + 64, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, sequence, lengths, features):
        if sequence.ndim != 2:
            raise ValueError("sequence must have shape (batch, sequence_length).")
        if lengths.ndim != 1 or lengths.shape[0] != sequence.shape[0]:
            raise ValueError("lengths must contain one value per sequence.")
        if features.ndim != 2 or features.shape[0] != sequence.shape[0]:
            raise ValueError("features must have shape (batch, feature_count).")
        if features.shape[1] != self.tabular_encoder[0].in_features:
            raise ValueError(
                "features must contain "
                f"{self.tabular_encoder[0].in_features} tabular values."
            )

        embedded = self.embedding(sequence)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.sequence_encoder(packed)
        forward_last = hidden[-2]
        backward_last = hidden[-1]
        sequence_representation = self.sequence_dropout(
            torch.cat((forward_last, backward_last), dim=-1)
        )
        tabular_representation = self.tabular_encoder(features)
        combined = torch.cat(
            (sequence_representation, tabular_representation),
            dim=-1,
        )
        return self.classifier(combined).squeeze(-1)