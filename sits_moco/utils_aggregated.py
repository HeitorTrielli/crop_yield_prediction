"""Compatibility shim — prefer ``import yield_utils``."""
from yield_utils import *  # noqa: F401,F403
from yield_utils import (  # noqa: F401 — common explicit re-exports for IDEs
    HEAD_OUTPUT_RAW,
    HEAD_OUTPUT_ZSCORE,
    TARGET_SPECS,
    AggregatedMSELoss,
    aggregate_municipality_from_pixel_chunks,
    denormalize_head_output,
    pixel_pool_for_head,
    regression_metrics,
    resolve_head_output,
    resolve_inference_target,
    resolve_model_kwargs,
    run_training_vram_probe,
    soil_fusion_from_state_dict,
    stnet_regression_input_dim_from_state_dict,
)
