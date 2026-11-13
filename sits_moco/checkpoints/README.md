# Canonical municipal-mean (average-pixel) checkpoints

Simple DualSTNet models (**no auxiliary / recon heads**): Sentinel-2 + daily climate trunks, late soil fusion, year leave-one-out 2020–2024.

| Bundle | Scope | Source study / trial | Arch |
|--------|--------|----------------------|------|
| `parana_muni_mean_trial005_year_loo/` | Paraná | `productivity_muni_mean_dual_stnet_wide_soy_pr_0.5_1.5_top100_year_loo` / `trial_005` | d=128, h=4, d_inner=64, dropout=0.4, L=1 |
| `five_state_muni_mean_trial044_year_loo/` | PR+RS+SP+GO+SC | `…_pr_rs_sp_go_sc_0.5_1.5_top100_year_loo` / `trial_044` | d=128, h=4, d_inner=256, dropout=0.3, L=1 |

Each `looYYYY/` folder has:

- `model_best.pth` — early-stopped checkpoint for that holdout year  
- `extra_scaler.json` — fold input scaler  
- `config.json`, `params.json`, `testlog.csv`

`results/` remains gitignored; these copies are the tracked release set.
