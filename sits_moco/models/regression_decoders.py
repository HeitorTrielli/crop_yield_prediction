"""Configurable regression heads for fused modality embeddings."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


def parse_hidden_dims(value: str | Sequence[int]) -> tuple[int, ...]:
    """Parse ``"256,128"`` or an integer sequence into validated widths."""
    if isinstance(value, str):
        dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        dims = tuple(int(part) for part in value)
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError(f"decoder_hidden_dims must contain positive widths; got {value!r}")
    return dims


def _activation(name: str) -> nn.Module:
    choices = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
    try:
        return choices[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unknown decoder activation {name!r}") from exc


def _norm(name: str, width: int) -> nn.Module:
    choices: dict[str, nn.Module] = {
        "layer": nn.LayerNorm(width),
        "batch": nn.BatchNorm1d(width),
        "none": nn.Identity(),
    }
    try:
        return choices[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown decoder norm {name!r}") from exc


def _hidden_stage(
    input_dim: int,
    output_dim: int,
    *,
    activation: str,
    norm: str,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        _norm(norm, output_dim),
        _activation(activation),
        nn.Dropout(dropout),
    )


class MLPDecoder(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        *,
        activation: str,
        norm: str,
        dropout: float,
    ):
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.append(
                _hidden_stage(
                    previous,
                    width,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                )
            )
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        super().__init__(*layers)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        activation: str,
        norm: str,
        dropout: float,
    ):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            _norm(norm, output_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            _norm(norm, output_dim),
        )
        self.skip = (
            nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        )
        self.activation = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.skip(x) + self.branch(x))


class ResidualDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        *,
        activation: str,
        norm: str,
        dropout: float,
    ):
        super().__init__()
        blocks: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            blocks.append(
                ResidualBlock(
                    previous,
                    width,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                )
            )
            previous = width
        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Linear(previous, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(x))


class LinearResidualDecoder(nn.Module):
    """Linear baseline plus a nonlinear correction."""

    def __init__(self, input_dim: int, output_dim: int, correction: nn.Module):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.correction = correction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.correction(x)


class GatedDecoder(nn.Module):
    """Feature-wise gating before an MLP regression head."""

    def __init__(self, input_dim: int, head: nn.Module):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Sigmoid())
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x * self.gate(x))


class ModalityAdditiveDecoder(nn.Module):
    """Add independent modality predictions, optionally with a joint correction."""

    def __init__(
        self,
        modality_dims: Sequence[int],
        output_dim: int,
        joint: nn.Module | None = None,
    ):
        super().__init__()
        self.modality_dims = tuple(int(dim) for dim in modality_dims)
        self.additive = nn.ModuleList(
            nn.Linear(dim, output_dim) for dim in self.modality_dims
        )
        self.joint = joint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = torch.split(x, self.modality_dims, dim=-1)
        output = sum(head(part) for head, part in zip(self.additive, parts))
        if self.joint is not None:
            output = output + self.joint(x)
        return output


class MixtureDecoder(nn.Module):
    """Soft mixture of MLP experts."""

    def __init__(self, input_dim: int, output_dim: int, experts: Sequence[nn.Module]):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = nn.Linear(input_dim, len(experts))
        self.output_dim = int(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = torch.stack([expert(x) for expert in self.experts], dim=1)
        weights = torch.softmax(self.gate(x), dim=-1).unsqueeze(-1)
        return (predictions * weights).sum(dim=1)


class GLUDecoder(nn.Sequential):
    """Gated-linear-unit hidden stages."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        *,
        norm: str,
        dropout: float,
    ):
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Linear(previous, 2 * width),
                    nn.GLU(dim=-1),
                    _norm(norm, width),
                    nn.Dropout(dropout),
                )
            )
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        super().__init__(*layers)


class DenseDecoder(nn.Module):
    """DenseNet-style MLP that preserves every preceding representation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        *,
        activation: str,
        norm: str,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        total_width = input_dim
        for width in hidden_dims:
            self.layers.append(
                _hidden_stage(
                    total_width,
                    width,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                )
            )
            total_width += width
        self.output = nn.Linear(total_width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=-1)))
        return self.output(torch.cat(features, dim=-1))


class HighwayBlock(nn.Module):
    def __init__(self, width: int, *, activation: str, dropout: float):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(width, width), _activation(activation), nn.Dropout(dropout)
        )
        self.gate = nn.Sequential(nn.Linear(width, width), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate(x)
        return gate * self.transform(x) + (1.0 - gate) * x


class HighwayDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        *,
        activation: str,
        norm: str,
        dropout: float,
    ):
        super().__init__()
        width = int(hidden_dims[0])
        self.input = _hidden_stage(
            input_dim,
            width,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )
        self.blocks = nn.Sequential(
            *[
                HighwayBlock(width, activation=activation, dropout=dropout)
                for _ in range(max(1, len(hidden_dims) - 1))
            ]
        )
        self.output = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(self.input(x)))


class FourierDecoder(nn.Module):
    """Learned sinusoidal features followed by a small MLP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        *,
        activation: str,
        norm: str,
        dropout: float,
    ):
        super().__init__()
        frequency_dim = int(hidden_dims[0])
        self.projection = nn.Linear(input_dim, frequency_dim)
        tail_dims = tuple(hidden_dims[1:]) or (frequency_dim,)
        self.head = MLPDecoder(
            2 * frequency_dim,
            output_dim,
            tail_dims,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase = self.projection(x)
        return self.head(torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1))


class QuadraticDecoder(nn.Module):
    """Linear prediction plus low-rank learned quadratic interactions."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int,
        correction: nn.Module | None = None,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.factor = nn.Linear(input_dim, rank, bias=False)
        self.quadratic = nn.Linear(rank, output_dim, bias=False)
        self.correction = correction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.linear(x) + self.quadratic(self.factor(x).square())
        if self.correction is not None:
            output = output + self.correction(x)
        return output


class MaxoutStage(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, pieces: int, norm: str, dropout: float):
        super().__init__()
        self.output_dim = int(output_dim)
        self.pieces = int(pieces)
        self.linear = nn.Linear(input_dim, output_dim * pieces)
        self.norm = _norm(norm, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.linear(x).reshape(-1, self.output_dim, self.pieces)
        return self.dropout(self.norm(projected.max(dim=-1).values))


class MaxoutDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        *,
        pieces: int,
        norm: str,
        dropout: float,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.append(MaxoutStage(previous, width, pieces, norm, dropout))
            previous = width
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(previous, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.layers(x))


class EnsembleDecoder(nn.Module):
    """Fixed average of independently initialized MLP experts."""

    def __init__(self, experts: Sequence[nn.Module]):
        super().__init__()
        self.experts = nn.ModuleList(experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([expert(x) for expert in self.experts], dim=0).mean(dim=0)


class ModalityBilinearDecoder(nn.Module):
    """Additive modality heads plus low-rank pairwise interactions."""

    def __init__(self, modality_dims: Sequence[int], output_dim: int, rank: int):
        super().__init__()
        self.modality_dims = tuple(int(dim) for dim in modality_dims)
        self.additive = nn.ModuleList(
            nn.Linear(dim, output_dim) for dim in self.modality_dims
        )
        pairs = [
            (left, right)
            for left in range(len(self.modality_dims))
            for right in range(left + 1, len(self.modality_dims))
        ]
        self.pairs = pairs
        self.left = nn.ModuleList(
            nn.Linear(self.modality_dims[i], rank, bias=False) for i, _ in pairs
        )
        self.right = nn.ModuleList(
            nn.Linear(self.modality_dims[j], rank, bias=False) for _, j in pairs
        )
        self.interaction = nn.ModuleList(
            nn.Linear(rank, output_dim, bias=False) for _ in pairs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = torch.split(x, self.modality_dims, dim=-1)
        output = sum(head(part) for head, part in zip(self.additive, parts))
        for k, (left, right) in enumerate(self.pairs):
            product = self.left[k](parts[left]) * self.right[k](parts[right])
            output = output + self.interaction[k](product)
        return output


class ModalityAttentionDecoder(nn.Module):
    """Treat modality embeddings as tokens and learn their interactions."""

    def __init__(
        self,
        modality_dims: Sequence[int],
        output_dim: int,
        hidden_dims: Sequence[int],
        *,
        activation: str,
        norm: str,
        dropout: float,
    ):
        super().__init__()
        self.modality_dims = tuple(int(dim) for dim in modality_dims)
        width = int(hidden_dims[0])
        heads = 4 if width % 4 == 0 else 1
        self.projections = nn.ModuleList(
            nn.Linear(dim, width) for dim in self.modality_dims
        )
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(width)
        tail_dims = tuple(hidden_dims[1:])
        if tail_dims:
            self.head = MLPDecoder(
                width,
                output_dim,
                tail_dims,
                activation=activation,
                norm=norm,
                dropout=dropout,
            )
        else:
            self.head = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = torch.split(x, self.modality_dims, dim=-1)
        tokens = torch.stack(
            [projection(part) for projection, part in zip(self.projections, parts)],
            dim=1,
        )
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        return self.head(self.norm(tokens + attended).mean(dim=1))


def build_regression_decoder(
    decoder_type: str,
    input_dim: int,
    output_dim: int,
    *,
    hidden_dims: str | Sequence[int] = (256, 128),
    activation: str = "gelu",
    norm: str = "layer",
    dropout: float = 0.1,
    modality_dims: Sequence[int] = (),
    num_experts: int = 4,
) -> nn.Module:
    """Build a decoder. The legacy head is intentionally handled by its model."""
    kind = decoder_type.lower()
    dims = parse_hidden_dims(hidden_dims)

    def mlp() -> MLPDecoder:
        return MLPDecoder(
            input_dim,
            output_dim,
            dims,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )

    if kind == "linear":
        return nn.Linear(input_dim, output_dim)
    if kind == "mlp":
        return mlp()
    if kind == "residual":
        return ResidualDecoder(
            input_dim,
            output_dim,
            dims,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )
    if kind == "linear_residual":
        return LinearResidualDecoder(input_dim, output_dim, mlp())
    if kind == "gated":
        return GatedDecoder(input_dim, mlp())
    if kind == "glu":
        return GLUDecoder(
            input_dim, output_dim, dims, norm=norm, dropout=dropout
        )
    if kind == "dense":
        return DenseDecoder(
            input_dim,
            output_dim,
            dims,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )
    if kind == "highway":
        return HighwayDecoder(
            input_dim,
            output_dim,
            dims,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )
    if kind == "fourier":
        return FourierDecoder(
            input_dim,
            output_dim,
            dims,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )
    if kind in {"quadratic", "quadratic_mlp"}:
        correction = mlp() if kind == "quadratic_mlp" else None
        return QuadraticDecoder(input_dim, output_dim, dims[0], correction)
    if kind == "maxout":
        if num_experts < 2:
            raise ValueError("maxout decoder requires decoder_num_experts >= 2")
        return MaxoutDecoder(
            input_dim,
            output_dim,
            dims,
            pieces=num_experts,
            norm=norm,
            dropout=dropout,
        )
    if kind in {"modality_additive", "modality_residual"}:
        if not modality_dims or sum(modality_dims) != input_dim:
            raise ValueError(
                f"{kind} requires modality_dims summing to {input_dim}; "
                f"got {tuple(modality_dims)}"
            )
        joint = mlp() if kind == "modality_residual" else None
        return ModalityAdditiveDecoder(modality_dims, output_dim, joint)
    if kind == "mixture":
        if num_experts < 2:
            raise ValueError("mixture decoder requires decoder_num_experts >= 2")
        return MixtureDecoder(input_dim, output_dim, [mlp() for _ in range(num_experts)])
    if kind == "ensemble":
        if num_experts < 2:
            raise ValueError("ensemble decoder requires decoder_num_experts >= 2")
        return EnsembleDecoder([mlp() for _ in range(num_experts)])
    if kind in {"modality_bilinear", "modality_attention"}:
        if not modality_dims or sum(modality_dims) != input_dim:
            raise ValueError(
                f"{kind} requires modality_dims summing to {input_dim}; "
                f"got {tuple(modality_dims)}"
            )
        if kind == "modality_bilinear":
            return ModalityBilinearDecoder(modality_dims, output_dim, dims[0])
        return ModalityAttentionDecoder(
            modality_dims,
            output_dim,
            dims,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )
    raise ValueError(f"Unknown decoder type {decoder_type!r}")
