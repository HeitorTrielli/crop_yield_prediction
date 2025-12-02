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


class STNetRegression(nn.Module):
    """
    STNet model adapted for regression tasks.

    Changes from classification version:
    - Output layer outputs num_outputs (default 1) instead of num_classes
    - No softmax activation
    - Returns continuous values instead of class logits
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
    ):
        super(STNetRegression, self).__init__()
        self.modelname = "STNetRegression"
        self.max_seq_len = max_seq_len

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

        # Regression decoder: outputs num_outputs continuous values
        layers = []
        decoder = [d_model, 64, 32, num_outputs]
        for i in range(len(decoder) - 1):
            layers.append(nn.Linear(decoder[i], decoder[i + 1]))
            if i < (len(decoder) - 2):
                layers.extend(
                    [
                        nn.BatchNorm1d(decoder[i + 1]),
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

        # weight
        if not is_bert:
            weight = self.dropout(weight)
            weight_sum = weight.sum(1).unsqueeze(1)
            weight_sum = torch.clamp(weight_sum, min=1e-8)  # Prevent division by zero
            weight /= weight_sum
            x = torch.bmm(weight.unsqueeze(1), x).squeeze()

        output = self.decoder(x)

        return output
