from __future__ import annotations

import torch
from torch import nn


class RNNNextEvent(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, dropout: float, num_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.RNN(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        last_hidden = hidden[-1]
        return self.output(self.dropout(last_hidden))


class LSTMNextEvent(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, dropout: float, num_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        last_hidden = hidden[-1]
        return self.output(self.dropout(last_hidden))


class TransformerNextEvent(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        encoded = self.embedding(x) + self.position_embedding(positions)
        encoded = self.encoder(encoded)
        last_token = encoded[:, -1, :]
        return self.output(self.dropout(last_token))


def build_model(
    model_name: str,
    vocab_size: int,
    seq_len: int,
    embed_dim: int,
    hidden_dim: int,
    dropout: float,
    num_heads: int,
    num_layers: int,
) -> nn.Module:
    if model_name == "rnn":
        return RNNNextEvent(vocab_size, embed_dim, hidden_dim, dropout, num_layers)
    if model_name == "lstm":
        return LSTMNextEvent(vocab_size, embed_dim, hidden_dim, dropout, num_layers)
    if model_name == "transformer":
        return TransformerNextEvent(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_heads=num_heads,
            num_layers=num_layers,
        )
    raise ValueError(f"Unknown model: {model_name}")
