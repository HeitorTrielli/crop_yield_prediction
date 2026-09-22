"""Shared STNet temporal encoder (MLP + transformer + pooling), no decoder.

Used by DualSTNetRegression so spectral and climate trunks can share the
same architecture with different input_dim / sequence length.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .STNet import PositionalEncoding, linlayer
from .STNetRegression import AttentionPooling


class STNetEncoder(nn.Module):
    """Encode a padded sequence ``(x, mask, doy, weight)`` to a pooled vector."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_head: int = 16,
        n_layers: int = 1,
        d_inner: int = 128,
        activation: str = "relu",
        dropout: float = 0.2,
        max_len: int = 366,
        max_seq_len: int = 70,
        T: int = 1000,
        max_temporal_shift: int = 30,
        temporal_pooling: str = "ndvi",
        attn_pool_queries: int = 4,
    ):
        super().__init__()
        if temporal_pooling not in ("ndvi", "attention", "mean"):
            raise ValueError(
                f"temporal_pooling must be 'ndvi', 'attention', or 'mean', "
                f"got {temporal_pooling!r}"
            )
        self.input_dim = int(input_dim)
        self.max_seq_len = int(max_seq_len)
        self.temporal_pooling = temporal_pooling
        self.d_model = int(d_model)

        self.mlp_dim = [self.input_dim, 32, 64, d_model]
        layers = []
        for i in range(len(self.mlp_dim) - 1):
            layers.append(linlayer(self.mlp_dim[i], self.mlp_dim[i + 1]))
        self.mlp1 = nn.Sequential(*layers)

        self.inlayernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.position_enc = PositionalEncoding(
            d_model, max_len=max_len + 2 * max_temporal_shift, T=T
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_head, d_inner, dropout, activation, batch_first=True
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.transformerencoder = nn.TransformerEncoder(
            encoder_layer, n_layers, encoder_norm
        )

        if self.temporal_pooling == "attention":
            self.attn_pool = AttentionPooling(
                d_model, num_queries=attn_pool_queries, dropout=dropout
            )
            self.out_dim = self.attn_pool.out_dim
        else:
            self.attn_pool = None
            self.out_dim = d_model

    def forward(self, x, mask, doy, weight) -> torch.Tensor:
        x = x.permute((0, 2, 1))
        x = self.mlp1(x)
        x = x.permute((0, 2, 1))

        x = self.inlayernorm(x)
        x = self.dropout(x + self.position_enc(doy))
        x = self.transformerencoder(x, src_key_padding_mask=mask)

        if self.temporal_pooling == "attention":
            return self.attn_pool(x, mask, weight)

        if self.temporal_pooling == "mean":
            valid = (~mask).to(dtype=x.dtype)
            valid = valid / torch.clamp(valid.sum(1, keepdim=True), min=1e-8)
            return torch.bmm(valid.unsqueeze(1), x).squeeze(1)

        weight = self.dropout(weight)
        weight_sum = torch.clamp(weight.sum(1).unsqueeze(1), min=1e-8)
        weight = weight / weight_sum
        return torch.bmm(weight.unsqueeze(1), x).squeeze(1)
