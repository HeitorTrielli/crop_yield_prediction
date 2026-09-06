"""
Regression version of STNet for predicting continuous values (yield).
Adapted from STNet classification model.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .STNet import PositionalEncoding, linlayer


class AttentionPooling(nn.Module):
    """Learned multi-query attention pooling over the temporal axis.

    Replaces the fixed exp(NDVI)-weighted mean. Each of the ``num_queries``
    learnable queries attends over the transformer outputs and pools its own
    summary (e.g. green-up vs peak vs senescence windows); the concatenated
    summaries feed the decoder. The exp(NDVI) weight enters as a per-query
    log-prior with learnable gate (init 1.0 = trust it like the old pooling;
    the model can learn to ignore it).
    """

    def __init__(self, d_model: int, num_queries: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_queries = int(num_queries)
        self.queries = nn.Parameter(
            torch.randn(self.num_queries, d_model) * d_model**-0.5
        )
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = d_model**-0.5
        # per-query strength of the exp(NDVI) cloud/quality prior
        self.weight_gate = nn.Parameter(torch.ones(self.num_queries))
        self.dropout = nn.Dropout(dropout)
        self.out_dim = self.num_queries * d_model

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        # x: (B, T, D); mask: (B, T) True = padding; weight: (B, T) sums to 1
        k = self.key(x)
        v = self.value(x)
        logits = torch.einsum("qd,btd->bqt", self.queries.to(k.dtype), k) * self.scale
        prior = torch.log(weight.clamp_min(1e-6)).unsqueeze(1)  # (B, 1, T)
        logits = logits + self.weight_gate.view(1, -1, 1).to(logits.dtype) * prior
        logits = logits.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn)  # all-masked rows -> zeros, not NaN
        attn = self.dropout(attn)
        pooled = torch.einsum("bqt,btd->bqd", attn, v)
        return pooled.flatten(1)  # (B, Q*D)


class STNetRegression(nn.Module):
    """
    STNet model adapted for regression tasks.

    The decoder emits **z-scores** of the municipal training target
    (``(y − μ) / σ``). Bias 0 at init is climatology. Convert to original
    units with ``z * σ + μ`` (see ``denormalize_head_output``).

    Changes from classification version:
    - Output layer outputs num_outputs (default 1) instead of num_classes
    - No softmax activation
    - Returns continuous z-scores instead of class logits
    """

    def __init__(
        self,
        input_dim=10,
        num_outputs=1,
        d_model=128,
        n_head=16,
        n_layers=1,
        d_inner=128,
        activation="relu",
        dropout=0.2,
        max_len=366,
        max_seq_len=70,
        T=1000,
        max_temporal_shift=30,
        temporal_pooling="ndvi",
        attn_pool_queries=4,
    ):
        super(STNetRegression, self).__init__()
        self.modelname = "STNetRegression"
        self.max_seq_len = max_seq_len
        if temporal_pooling not in ("ndvi", "attention"):
            raise ValueError(
                f"temporal_pooling must be 'ndvi' or 'attention', got {temporal_pooling!r}"
            )
        self.temporal_pooling = temporal_pooling

        self.mlp_dim = [input_dim, 32, 64, d_model]
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
            decoder_in = self.attn_pool.out_dim
        else:
            self.attn_pool = None
            decoder_in = d_model

        # Regression decoder: LayerNorm (not BatchNorm) so train/eval use the
        # same normalization under pixel-chunked, variable-size batches.
        layers = []
        decoder = [decoder_in, 64, 32, num_outputs]
        for i in range(len(decoder) - 1):
            layers.append(nn.Linear(decoder[i], decoder[i + 1]))
            if i < (len(decoder) - 2):
                layers.extend(
                    [
                        nn.LayerNorm(decoder[i + 1]),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
        self.decoder = nn.Sequential(*layers)

    def forward(self, x, is_bert=False):
        x, mask, doy, weight = x

        x = x.permute((0, 2, 1))
        x = self.mlp1(x)
        x = x.permute((0, 2, 1))

        x = self.inlayernorm(x)
        x = self.dropout(x + self.position_enc(doy))

        x = self.transformerencoder(x, src_key_padding_mask=mask)

        # temporal pooling
        if not is_bert:
            if self.temporal_pooling == "attention":
                x = self.attn_pool(x, mask, weight)
            else:
                weight = self.dropout(weight)
                weight_sum = weight.sum(1).unsqueeze(1)
                weight_sum = torch.clamp(weight_sum, min=1e-8)  # Prevent division by zero
                weight /= weight_sum
                x = torch.bmm(weight.unsqueeze(1), x).squeeze(1)

        output = self.decoder(x)

        return output
