LOO pixel / muni-mean result exports.

Layout:
  <study_trial>/
    looYYYY/
      pixel_by_pixel/     — pixel→muni aggregation pipeline
        predictions/
        figures/          (scatter, heatmaps, …)
        params.json
        training/
      muni_mean/          — municipal-mean (average-pixel) pipeline
        predictions/
        figures/productivity_scatter/
        figures/municipal_heatmaps/

Trials:
  pr_rs_trial_063     — best PR+RS year-LOO DualSTNet (both pipelines)
  pr_rs_sp_trial_004  — best PR+RS+SP year-LOO DualSTNet (muni_mean only)
  pr_only_trial_044   — PR-only (lowest mean loss)
  pr_only_trial_005   — PR-only (best mean R² excl. 2022)
