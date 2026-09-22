"""
Backward-compat pickle alias for older checkpoints.

Training used to live under ``yield_utils``; that module was renamed to
``utils_aggregated``. Checkpoints (e.g. productivity_soil_sidecar_moco) still
pickle ``criterion`` as ``yield_utils.AggregatedMSELoss``.
"""

from utils_aggregated import AggregatedMSELoss

__all__ = ["AggregatedMSELoss"]
