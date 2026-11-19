# Separating Physical Hydrology from Multi-Sensor Artifacts in the SRYR (2003–2024)

Reproducibility code and derived diagnostic tables for:
**"Separating Physical Hydrology from Multi-Sensor Artifacts in a Permafrost-Underlain Alpine Headwater (Source Region of the Yellow River, 2003–2024)"**

## Repository Structure

- `code/01_analysis/`
  - `38_jms_rigorous_reanalysis.py` — GRACE/GRACE-FO 6-cap native footprint audit, segmented regression (`Eq. 1`), AR(1)-bootstrap Pettitt, depth-normalized GLDAS-2.1 subsurface audit, cross-sensor elevation warming & snow-masked NDVI validation, Pyper–Peterman `N_eff`, non-LAI ET coupling check, leakage-free walk-forward ML evaluation, and non-steady-state Budyko closure (`Eq. 3–4`).
  - `39_jms_breakthrough_probe.py` — Distributed-lag precipitation memory kernel, observed vs. theoretical Budyko runoff elasticity, and thermal-gradient limiting-factor regression.
  - `41_greening_residual_probe.py` — Climate-explained vs. unexplained permafrost greening residual with moving-block bootstrap confidence intervals and IGBP grassland/savanna mask checks.
  - `42_jms_round2_supplements.py` — Within-elevation-stratum (`3,800–4,300 m`, `≥4,300 m`) and baseline-NDVI-matched permafrost controls, `ST4` thermal threshold sensitivity (`−0.5, 0.0, +0.5 °C`), `TerraClimate` + `MODIS` snow cross-sensor checks, Ridge collinearity diagnostics, 2,000-draw joint Budyko Monte Carlo uncertainty propagation, orthogonal precipitation lag-band TreeSHAP (`RF` & `XGBoost`), and GRACE `AR(1)` `ρ` sensitivity sweeps.
- `code/02_figures/`
  - `45_jms_figures_publication.py` — Generates all seven 450–600 DPI publication-grade manuscript figures (`Fig1`–`Fig7`) and `Graphical_Abstract.png` directly from `results_repaired/`.
- `results_repaired/` — All derived basin, zonal, pixel-panel, Budyko, walk-forward, SHAP, and Monte Carlo diagnostic tables (`.csv`, `.npz`) used in Tables 1–6 and Figures 1–7.
- `data/boundary/yr_source_boundary.geojson` — Source Region of the Yellow River catchment polygon above Tangnaihai gauge.

## Quick Reproduction

```bash
pip install -r requirements.txt
python code/01_analysis/42_jms_round2_supplements.py
python code/02_figures/45_jms_figures_publication.py
```

All primary satellite and reanalysis products (JPL GRACE/GRACE-FO RL06.3 v04 CRI Mascons, GLDAS-2.1/2.2, GPM IMERG V07, ERA5-Land, TerraClimate, PML-V2.2a, MODIS MOD13A1/A3/MOD15A2H/MOD11A2/MOD16A2GF/MCD12Q1, MERIT DEM) are publicly available via NASA GES DISC, JPL Tellus, Copernicus C3S, and Google Earth Engine. Annual Tangnaihai runoff records (2016–2024) are published in the Yellow River Water Resources Bulletins (YRCC, 2017–2025).
