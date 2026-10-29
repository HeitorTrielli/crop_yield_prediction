LOO pixel / muni-mean result exports.

Layout:
  <study_trial>/
    looYYYY/
      pixel_by_pixel/     — pixel→muni aggregation pipeline
        predictions/
        figures/          (scatter, heatmaps, …)
        params.json
      muni_mean/          — municipal-mean (average-pixel) pipeline
        predictions/
        figures/productivity_scatter/
        figures/municipal_heatmaps/
        params.json

Trials:
  pr_rs_trial_063     — best PR+RS year-LOO DualSTNet (both pipelines)
  pr_rs_sp_trial_004  — best PR+RS+SP year-LOO DualSTNet (muni_mean only)
  pr_only_trial_044   — PR-only (lowest mean loss)
  pr_only_trial_005   — PR-only (best mean R² excl. 2022)

  pr_rs_top3_year_loo_ablations/ — PR+RS top-3 year-LOO modality ablations (18 trials, one study)
  pr_only_modality_ablations/    — older PR-only modality grids (separate YAMLs; not year-LOO)
  pr_only_ablation_top3_year_loo/ — best year-LOO trial per single-drop ablation (nosoil/noclimate/nosentinel)
    (pixel_by_pixel + muni_mean under each best_* folder)

  pr_only_decoder_muni_rank_trial_018 — PR-only decoder+muni_rank_corr winner (mlp 256,256,128, aux_weight=0.1; breaks trial_005 ceiling partially)
  pr_only_decoder_muni_rank_wide_trial_003 — PR-only wide aux winner (val_r2≈0.171). (aux=muni_rank_corr@0.2)
  pr_rs_sp_go_sc_decoder_muni_rank_wide_trial_005 — Five-state wide aux winner (val_r2≈0.449). (aux=muni_rank_corr@0.5)
