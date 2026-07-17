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
    if name in {
        "DEFAULT_MIN_COVERAGE_RATIO",
        "DEFAULT_MAX_COVERAGE_RATIO",
        "COVERAGE_RATIO_COL",
        "filter_yield_df_by_coverage",
        "filter_yield_pandas_by_coverage",
    }:
        from . import uscrops_aggregated_npy_polars as _agg

        return getattr(_agg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")