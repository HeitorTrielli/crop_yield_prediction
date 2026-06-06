from .uscrops import USCrops, MoCoDataset, BERTDataset
from .feature_layout import (
    feature_layout_choices,
    normalize_feature_layout,
    resolve_feature_layout,
)
from .pixel_transform import PixelTransform, SPECTRAL_MEAN, SPECTRAL_STD
from .datautils import *


def __getattr__(name: str):
    if name == "USCropsAggregatedNPY":
        from .uscrops_aggregated_npy_polars import USCropsAggregatedNPY

        return USCropsAggregatedNPY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")