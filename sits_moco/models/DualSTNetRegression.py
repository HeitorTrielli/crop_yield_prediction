"""Two STNet encoders (Sentinel megapixel + daily climate) fused in the decoder.

Soil is static, so it is never sent through either transformer. When the
spectral cube includes MapBiomas Solo sidecars, they are stripped before the
spectral encoder and concatenated with both pooled embeddings (late fusion).

Modality ablations may omit either trunk via ``use_spectral`` / ``use_climate``
(climate-only or spectral-only DualSTNet). At least one of spectral, climate,
or late soil must remain.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .STNetEncoder import STNetEncoder
from .regression_decoders import build_regression_decoder
from .recon_heads import (
    PooledSummaryRecon,
    SequenceReconHead,
    pooled_summary_dim,
    pooled_summary_target,
    _masked_mse,
)


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
        climate_d_model: int | None = None,
        climate_n_head: int | None = None,
        climate_d_inner: int | None = None,
        attn_pool_queries: int = 4,
        soil_fusion: str = "late",
        soil_dim: int = 4,
        input_dim: int | None = None,
        use_spectral: bool = True,
        use_climate: bool = True,
        decoder_type: str = "legacy",
        decoder_hidden_dims: str = "256,128",
        decoder_dropout: float | None = None,
        decoder_activation: str = "gelu",
        decoder_norm: str = "layer",
        decoder_num_experts: int = 4,
        recon_mode: str = "none",
        recon_hidden: int = 256,
        aux_yield_probe: str = "none",
    ):
        super().__init__()
        self.modelname = "DualSTNetRegression"
        self.recon_mode = str(recon_mode or "none").lower()
        if self.recon_mode not in {"none", "pooled", "sequence"}:
            raise ValueError(
                f"recon_mode must be none|pooled|sequence, got {recon_mode!r}"
            )
        self.aux_yield_probe_type = str(aux_yield_probe or "none").lower()
        if self.aux_yield_probe_type not in {"none", "linear", "mlp"}:
            raise ValueError(
                f"aux_yield_probe must be none|linear|mlp, got {aux_yield_probe!r}"
            )
        if soil_fusion not in ("late", "none"):
            raise ValueError(
                f"DualSTNet soil_fusion must be 'late' or 'none' "
                f"(soil is static; got {soil_fusion!r})"
            )
        self.soil_fusion = soil_fusion
        self.soil_dim = int(soil_dim)
        self.spectral_dim = int(spectral_dim)
        self.climate_dim = int(climate_dim)
        self.use_spectral = bool(use_spectral)
        self.use_climate = bool(use_climate)
        if not self.use_spectral and not self.use_climate and self.soil_fusion != "late":
            raise ValueError(
                "DualSTNet needs use_spectral, use_climate, and/or late soil fusion"
            )
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
        self.spectral: STNetEncoder | None = None
        self.climate: STNetEncoder | None = None
        if self.use_spectral:
            self.spectral = STNetEncoder(
                input_dim=self.spectral_dim,
                max_seq_len=max_seq_len,
                temporal_pooling=temporal_pooling,
                **enc_kw,
            )
        if self.use_climate:
            clim_pool = climate_pooling
            if clim_pool == "ndvi":
                clim_pool = "mean"
            clim_kw = dict(enc_kw)
            if climate_d_model is not None:
                clim_kw["d_model"] = int(climate_d_model)
            if climate_n_head is not None:
                clim_kw["n_head"] = int(climate_n_head)
            if climate_d_inner is not None:
                clim_kw["d_inner"] = int(climate_d_inner)
            if int(clim_kw["d_model"]) % int(clim_kw["n_head"]) != 0:
                raise ValueError(
                    "climate trunk d_model="
                    f"{clim_kw['d_model']} must be divisible by n_head="
                    f"{clim_kw['n_head']}"
                )
            self.climate = STNetEncoder(
                input_dim=self.climate_dim,
                max_seq_len=climate_max_seq_len,
                temporal_pooling=clim_pool,
                **clim_kw,
            )

        modality_dims: list[int] = []
        if self.spectral is not None:
            modality_dims.append(self.spectral.out_dim)
        if self.climate is not None:
            modality_dims.append(self.climate.out_dim)
        if self.soil_fusion == "late":
            modality_dims.append(self.soil_dim)
        decoder_in = sum(modality_dims)
        if decoder_in <= 0:
            raise ValueError("DualSTNet decoder has empty input")

        self.decoder_type = str(decoder_type).lower()
        if self.decoder_type == "legacy":
            # Keep both module structure and state-dict names checkpoint-compatible.
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
        else:
            self.decoder = build_regression_decoder(
                self.decoder_type,
                decoder_in,
                num_outputs,
                hidden_dims=decoder_hidden_dims,
                activation=decoder_activation,
                norm=decoder_norm,
                dropout=dropout if decoder_dropout is None else decoder_dropout,
                modality_dims=modality_dims,
                num_experts=decoder_num_experts,
            )

        # Parallel generative heads (advisor: keep feature signals in embedder).
        self.pooled_recon: PooledSummaryRecon | None = None
        self.spectral_recon: SequenceReconHead | None = None
        self.climate_recon: SequenceReconHead | None = None
        if self.recon_mode == "pooled":
            summary_dim = pooled_summary_dim(
                self.spectral_dim,
                self.climate_dim,
                self.soil_dim,
                use_spectral=self.spectral is not None,
                use_climate=self.climate is not None,
                use_soil=self.soil_fusion == "late",
            )
            if summary_dim <= 0:
                raise ValueError("pooled recon requested but no modalities available")
            self.pooled_recon = PooledSummaryRecon(
                decoder_in, summary_dim, hidden=int(recon_hidden)
            )
        elif self.recon_mode == "sequence":
            if self.spectral is not None:
                self.spectral_recon = SequenceReconHead(
                    self.spectral.d_model, self.spectral_dim
                )
            if self.climate is not None:
                self.climate_recon = SequenceReconHead(
                    self.climate.d_model, self.climate_dim
                )
            if self.spectral_recon is None and self.climate_recon is None:
                raise ValueError("sequence recon requires spectral and/or climate trunk")

        # Parallel yield probe: force fused embedding to stay yield-predictive.
        self.aux_yield_probe: nn.Module | None = None
        if self.aux_yield_probe_type == "linear":
            self.aux_yield_probe = nn.Linear(decoder_in, num_outputs)
        elif self.aux_yield_probe_type == "mlp":
            hidden = max(32, min(128, decoder_in // 2))
            self.aux_yield_probe = nn.Sequential(
                nn.Linear(decoder_in, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, num_outputs),
            )

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

    def encode(self, x, *, need_sequence: bool = False) -> dict:
        if not isinstance(x, (tuple, list)) or len(x) != 8:
            raise ValueError(
                "DualSTNetRegression expects an 8-tuple "
                "(x_s2, mask_s2, doy_s2, weight_s2, x_clim, mask_clim, doy_clim, weight_clim)"
            )
        x_s2, mask_s2, doy_s2, w_s2, x_c, mask_c, doy_c, w_c = x
        x_spec, soil = self._split_soil(x_s2, mask_s2)
        parts: list[torch.Tensor] = []
        spec_tokens = clim_tokens = None
        want_seq = bool(need_sequence) or self.recon_mode == "sequence"

        if self.spectral is not None:
            if x_spec.size(-1) != self.spectral_dim:
                raise ValueError(
                    f"Spectral cube has C={x_spec.size(-1)}, encoder expects {self.spectral_dim}"
                )
            if want_seq:
                pooled, spec_tokens = self.spectral(
                    x_spec, mask_s2, doy_s2, w_s2, return_sequence=True
                )
                parts.append(pooled)
            else:
                parts.append(self.spectral(x_spec, mask_s2, doy_s2, w_s2))
        if self.climate is not None:
            if x_c.size(-1) != self.climate_dim:
                raise ValueError(
                    f"Climate cube has C={x_c.size(-1)}, encoder expects {self.climate_dim}"
                )
            if want_seq:
                pooled, clim_tokens = self.climate(
                    x_c, mask_c, doy_c, w_c, return_sequence=True
                )
                parts.append(pooled)
            else:
                parts.append(self.climate(x_c, mask_c, doy_c, w_c))
        if soil is not None:
            parts.append(soil)
        if not parts:
            raise ValueError("DualSTNet forward produced no modality embeddings")
        fused = torch.cat(parts, dim=-1)
        return {
            "fused": fused,
            "x_spec": x_spec if self.spectral is not None else None,
            "mask_s2": mask_s2,
            "x_clim": x_c if self.climate is not None else None,
            "mask_c": mask_c,
            "soil": soil,
            "spec_tokens": spec_tokens,
            "clim_tokens": clim_tokens,
        }

    def forward(self, x, is_bert: bool = False):
        return self.decoder(self.encode(x)["fused"])

    def reconstruction_loss_from_encode(
        self,
        enc: dict,
        *,
        stopgrad_encoder: bool = False,
    ) -> torch.Tensor:
        """Parallel generative loss from an ``encode()`` result."""
        if self.recon_mode == "none":
            return enc["fused"].new_zeros(())

        if self.recon_mode == "pooled":
            assert self.pooled_recon is not None
            fused = enc["fused"].detach() if stopgrad_encoder else enc["fused"]
            pred = self.pooled_recon(fused)
            target = pooled_summary_target(
                enc["x_spec"],
                enc["mask_s2"],
                enc["x_clim"],
                enc["mask_c"],
                enc["soil"],
            ).to(dtype=pred.dtype)
            return F.mse_loss(pred, target)

        losses: list[torch.Tensor] = []
        if self.spectral_recon is not None and enc["spec_tokens"] is not None:
            tokens = (
                enc["spec_tokens"].detach()
                if stopgrad_encoder
                else enc["spec_tokens"]
            )
            pred = self.spectral_recon(tokens)
            target = enc["x_spec"].to(dtype=pred.dtype)
            valid = ~enc["mask_s2"] if enc["mask_s2"] is not None else None
            if valid is None:
                losses.append(F.mse_loss(pred, target))
            else:
                losses.append(_masked_mse(pred, target, valid))
        if self.climate_recon is not None and enc["clim_tokens"] is not None:
            tokens = (
                enc["clim_tokens"].detach()
                if stopgrad_encoder
                else enc["clim_tokens"]
            )
            pred = self.climate_recon(tokens)
            target = enc["x_clim"].to(dtype=pred.dtype)
            valid = ~enc["mask_c"] if enc["mask_c"] is not None else None
            if valid is None:
                losses.append(F.mse_loss(pred, target))
            else:
                losses.append(_masked_mse(pred, target, valid))
        if not losses:
            return enc["fused"].new_zeros(())
        return torch.stack(losses).mean()

    def reconstruction_loss(
        self,
        x,
        *,
        stopgrad_encoder: bool = False,
    ) -> torch.Tensor:
        """Parallel generative loss. Zero when ``recon_mode=none``."""
        if self.recon_mode == "none":
            device = x[0].device if isinstance(x, (tuple, list)) else "cpu"
            return torch.zeros((), device=device)
        enc = self.encode(x, need_sequence=self.recon_mode == "sequence")
        return self.reconstruction_loss_from_encode(
            enc, stopgrad_encoder=stopgrad_encoder
        )

    def aux_yield_probe_loss_from_encode(
        self,
        enc: dict,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """MSE from the parallel linear probe on fused embeddings.

        ``targets`` must already be in the same units as the main decoder
        (normally per-municipality z-scores). Returns 0 when probe is disabled.
        """
        if self.aux_yield_probe is None:
            return enc["fused"].new_zeros(())
        pred = self.aux_yield_probe(enc["fused"])
        tgt = targets
        if tgt.dim() == 1:
            tgt = tgt.unsqueeze(-1)
        if pred.shape != tgt.shape:
            tgt = tgt.reshape_as(pred)
        return F.mse_loss(pred.to(torch.float32), tgt.to(torch.float32))
