"""Two STNet encoders (Sentinel megapixel + daily climate) fused in the decoder.

Soil is static, so it is never sent through either transformer. When the
spectral cube includes MapBiomas Solo sidecars, they are stripped before the
spectral encoder and concatenated with both pooled embeddings (late fusion).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .STNetEncoder import STNetEncoder


class DualSTNetRegression(nn.Module):
    """
    Dual-encoder yield head.

    Forward input is an 8-tuple::

        (x_s2, mask_s2, doy_s2, weight_s2, x_clim, mask_clim, doy_clim, weight_clim)

    ``x_s2`` is ``[B, T_s2, spectral_dim]`` or ``[B, T_s2, spectral_dim + soil_dim]``
    when soil sidecars are joined on the spectral cube.
    """

    def __init__(
        self,
        spectral_dim: int = 10,
        climate_dim: int = 5,
        num_outputs: int = 1,
        d_model: int = 128,
        n_head: int = 16,
        n_layers: int = 1,
        d_inner: int = 128,
        activation: str = "relu",
        dropout: float = 0.2,
        max_len: int = 366,
        max_seq_len: int = 45,
        climate_max_seq_len: int = 183,
        T: int = 1000,
        max_temporal_shift: int = 30,
        temporal_pooling: str = "ndvi",
        climate_pooling: str = "mean",
        attn_pool_queries: int = 4,
        soil_fusion: str = "late",
        soil_dim: int = 4,
        input_dim: int | None = None,
    ):
        super().__init__()
        self.modelname = "DualSTNetRegression"
        if soil_fusion not in ("late", "none"):
            raise ValueError(
                f"DualSTNet soil_fusion must be 'late' or 'none' "
                f"(soil is static; got {soil_fusion!r})"
            )
        self.soil_fusion = soil_fusion
        self.soil_dim = int(soil_dim)
        self.spectral_dim = int(spectral_dim)
        self.climate_dim = int(climate_dim)
        # Cube width seen by the dataloader (spectral ± soil). Used by inference helpers.
        if input_dim is None:
            extra = self.soil_dim if self.soil_fusion == "late" else 0
            self.input_dim = self.spectral_dim + extra
        else:
            self.input_dim = int(input_dim)
        self.max_seq_len = int(max_seq_len)
        self.climate_max_seq_len = int(climate_max_seq_len)

        enc_kw = dict(
            d_model=d_model,
            n_head=n_head,
            n_layers=n_layers,
            d_inner=d_inner,
            activation=activation,
            dropout=dropout,
            max_len=max_len,
            T=T,
            max_temporal_shift=max_temporal_shift,
            attn_pool_queries=attn_pool_queries,
        )
        self.spectral = STNetEncoder(
            input_dim=self.spectral_dim,
            max_seq_len=max_seq_len,
            temporal_pooling=temporal_pooling,
            **enc_kw,
        )
        clim_pool = climate_pooling
        if clim_pool == "ndvi":
            clim_pool = "mean"
        self.climate = STNetEncoder(
            input_dim=self.climate_dim,
            max_seq_len=climate_max_seq_len,
            temporal_pooling=clim_pool,
            **enc_kw,
        )

        decoder_in = self.spectral.out_dim + self.climate.out_dim
        if self.soil_fusion == "late":
            decoder_in = decoder_in + self.soil_dim

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

    def _split_soil(
        self, x_s2: torch.Tensor, mask_s2: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.soil_fusion != "late":
            return x_s2, None
        if x_s2.size(-1) < self.spectral_dim + self.soil_dim:
            return x_s2[..., : self.spectral_dim], None
        soil_seq = x_s2[..., self.spectral_dim : self.spectral_dim + self.soil_dim]
        if mask_s2 is not None and mask_s2.size(1) == x_s2.size(1):
            idx = (~mask_s2).to(dtype=torch.int64).argmax(dim=1)
        else:
            idx = torch.zeros(x_s2.size(0), dtype=torch.int64, device=x_s2.device)
        b = torch.arange(x_s2.size(0), device=x_s2.device)
        return x_s2[..., : self.spectral_dim], soil_seq[b, idx]

    def forward(self, x, is_bert: bool = False):
        if not isinstance(x, (tuple, list)) or len(x) != 8:
            raise ValueError(
                "DualSTNetRegression expects an 8-tuple "
                "(x_s2, mask_s2, doy_s2, weight_s2, x_clim, mask_clim, doy_clim, weight_clim)"
            )
        x_s2, mask_s2, doy_s2, w_s2, x_c, mask_c, doy_c, w_c = x
        x_s2, soil = self._split_soil(x_s2, mask_s2)
        if x_s2.size(-1) != self.spectral_dim:
            raise ValueError(
                f"Spectral cube has C={x_s2.size(-1)}, encoder expects {self.spectral_dim}"
            )
        if x_c.size(-1) != self.climate_dim:
            raise ValueError(
                f"Climate cube has C={x_c.size(-1)}, encoder expects {self.climate_dim}"
            )

        emb_s2 = self.spectral(x_s2, mask_s2, doy_s2, w_s2)
        emb_c = self.climate(x_c, mask_c, doy_c, w_c)
        parts = [emb_s2, emb_c]
        if soil is not None:
            parts.append(soil)
        return self.decoder(torch.cat(parts, dim=-1))
