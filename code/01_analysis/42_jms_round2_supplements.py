#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
42_jms_round2_supplements.py
============================
Computes Round-2 ARS supplementary robustness and uncertainty-propagation diagnostics
from cached project outputs in results_repaired/:
  1. jms_permafrost_stratum_controls.csv:
     - 2-km GEE zonal & 5-km pixel within-elevation-belt controls (3,800-4,300 m, >=4,300 m)
       and baseline-NDVI-matched control (0.35-0.55),
     - ST4 thermal threshold sensitivity (-0.5, 0.0, +0.5 C),
     - Cross-sensor TerraClimate Tmean + MODIS snow substitution,
     - Unpenalized OLS vs. Ridge (L2) regularization comparison for n=21 extended model.
  2. jms_budyko_mc_uncertainty.csv & updated jms_table6_budyko_nonsteady.csv:
     - Joint 2,000-draw Monte Carlo error propagation through P_eff and E_wb,
     - Period-flux w_WB uncertainty, 2018-2019 (excl. 2020) sensitivity, P_TC sensitivity,
       and Fig. 6d waterfall component +/-1 SD bounds.
  3. jms_shap_orthogonal_check.csv:
     - TreeSHAP shares on RandomForest and XGBoost for cumulative and non-overlapping
       precipitation lag blocks (0-3, 4-6, 7-12, 13-24 mo).
  4. jms_grace_rho_sensitivity.csv:
     - Sensitivity of GRACE/GRACE-FO phase SNR and window trend 95% MC CIs across
       temporal error autocorrelation rho in {0.10, 0.30, 0.50, 0.60} and FO offsets.
"""

import json
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
import shap
from matplotlib.path import Path as MPath
from scipy.ndimage import zoom
from scipy.optimize import brentq
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

BASE = '/Users/yueliangmanle/Desktop/小论文中文2'
RES = os.path.join(BASE, 'results_repaired')
GEO = os.path.join(BASE, '高级图设计包/fig1_研究区/yr_source_boundary.geojson')


def load_pixel_masks():
    """Load cached 5-km pixel panels, resampled MERIT DEM, ST4 climatology, and basin mask."""
    t_npz = np.load(os.path.join(RES, 'jms_terrain_grid.npz'))
    dem_raw = t_npz['dem']
    st4 = np.load(os.path.join(RES, 'jms_st4_clim_grid.npz'))['st4_clim']
    dem_5k = zoom(dem_raw, (st4.shape[0] / dem_raw.shape[0], st4.shape[1] / dem_raw.shape[1]), order=1)

    p_npz = np.load(os.path.join(RES, '_cache_pixel_panel.npz'))
    e_npz = np.load(os.path.join(RES, '_cache_pixel_extra.npz'))
    ndvi, T, P = p_npz['ndvi'].copy(), p_npz['T'].copy(), p_npz['P'].copy()
    SM, SC, LC = e_npz['SM'].copy(), e_npz['SC'].copy(), e_npz['LC'].copy()
    for arr in (ndvi, T, P, SM, SC, LC):
        arr[arr <= -900] = np.nan

    ny, nx = st4.shape
    lons = np.linspace(95.5 + 8.0 / (2 * nx), 103.5 - 8.0 / (2 * nx), nx)
    lats = np.linspace(36.5 - 4.5 / (2 * ny), 32.0 + 4.5 / (2 * ny), ny)
    lo, la = np.meshgrid(lons, lats)
    with open(GEO) as f:
        gj = json.load(f)
    rings = [np.array(gj['coordinates'][0])] if gj['type'] == 'Polygon' else [np.array(p[0]) for p in gj['coordinates']]
    inside = np.zeros(lo.size, dtype=bool)
    pts = np.column_stack([lo.ravel(), la.ravel()])
    for r in rings:
        inside |= MPath(r[:, :2]).contains_points(pts)
    inside = inside.reshape(lo.shape)

    valid = (
        inside
        & np.isfinite(st4)
        & np.all(np.isfinite(ndvi), axis=0)
        & np.all(np.isfinite(T), axis=0)
        & np.all(np.isfinite(P), axis=0)
        & np.all(np.isfinite(SM), axis=0)
        & np.all(np.isfinite(SC), axis=0)
        & (ndvi.mean(axis=0) > 0.05)
    )
    return dem_5k, st4, ndvi, T, P, SM, SC, LC, valid


def fit_residual_block_boot(yrs, series_dict, use_cols, rng, alpha_ridge=None, n_boot=1000):
    """Fit detrended climate-to-NDVI sensitivity (OLS or standardized Ridge) with 3-yr block CIs."""
    def dt(v):
        return v - np.polyval(np.polyfit(yrs, v, 1), yrs)

    y_obs = series_dict['nd']
    tr_nd = float(np.polyfit(yrs, y_obs, 1)[0] * 10.0)
    slopes = {k: float(np.polyfit(yrs, series_dict[k], 1)[0] * 10.0) for k in use_cols}
    X_raw = np.column_stack([dt(series_dict[k]) for k in use_cols])
    y_dt = dt(y_obs)

    def estimate_explained(X_mat, y_vec):
        if alpha_ridge is None:
            fit = sm.OLS(y_vec, sm.add_constant(X_mat)).fit()
            betas = fit.params[1:]
            r2 = float(fit.rsquared)
        else:
            sx = X_mat.std(axis=0, ddof=1)
            sx = np.where(sx > 1e-12, sx, 1.0)
            Xs = (X_mat - X_mat.mean(axis=0)) / sx
            yc = y_vec - y_vec.mean()
            mdl = Ridge(alpha=alpha_ridge, fit_intercept=False).fit(Xs, yc)
            betas = mdl.coef_ / sx
            pred = X_mat @ betas + (y_vec.mean() - np.mean(X_mat @ betas))
            ss_res = np.sum((y_vec - pred) ** 2)
            ss_tot = np.sum((y_vec - y_vec.mean()) ** 2)
            r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
        expl = float(sum(betas[i] * slopes[k] for i, k in enumerate(use_cols)))
        return expl, r2

    expl_pt, r2_pt = estimate_explained(X_raw, y_dt)
    unexpl_pt = 100.0 * (1.0 - expl_pt / tr_nd)

    boot_vals = []
    n_yrs = len(yrs)
    n_blocks = int(np.ceil(n_yrs / 3))
    for _ in range(n_boot):
        starts = rng.integers(0, n_yrs - 2, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + 3) for s in starts])[:n_yrs]
        try:
            ex_b, _ = estimate_explained(X_raw[idx], y_dt[idx])
            boot_vals.append(100.0 * (1.0 - ex_b / tr_nd))
        except Exception:
            pass
    ci_lo, ci_hi = np.percentile(boot_vals, [2.5, 97.5])
    return tr_nd, expl_pt, r2_pt, unexpl_pt, float(ci_lo), float(ci_hi)


def run_permafrost_stratum_controls():
    """Diagnostic 1: Within-elevation controls, ST4 thresholds, TerraClimate/MODIS & Ridge."""
    print('\n=== [1] Permafrost Stratum Controls, Thresholds & Collinearity Diagnostics ===')
    rng = np.random.default_rng(11)
    yrs = np.arange(2004, 2025)

    rows = []
    # (A) 2-km GEE zonal combo means
    cdf = pd.read_csv(os.path.join(RES, '_cache_gee_combo_2004_2024.csv'))
    combo_meta = [
        (14, '<3800m', 'Permafrost (ST4<0C)'),
        (15, '<3800m', 'Seasonally Frozen (ST4>=0C)'),
        (24, '3800-4300m', 'Permafrost (ST4<0C)'),
        (25, '3800-4300m', 'Seasonally Frozen (ST4>=0C)'),
        (34, '>=4300m', 'Permafrost (ST4<0C)'),
        (35, '>=4300m', 'Seasonally Frozen (ST4>=0C)'),
    ]
    for zc, elev_lbl, th_lbl in combo_meta:
        s = cdf[cdf['combo_zone'] == zc].sort_values('year')
        akm = float(s['area_km2'].iloc[0])
        nd0 = float(s[s['year'] <= 2008]['NDVI_GS_NoSnow'].mean())
        nd1 = float(s[s['year'] >= 2020]['NDVI_GS_NoSnow'].mean())
        rel = (nd1 - nd0) / nd0 * 100.0
        fit = sm.OLS(s['NDVI_GS_NoSnow'].values, sm.add_constant(s['year'].values - 2004)).fit(
            cov_type='HAC', cov_kwds={'maxlags': 3}
        )
        rows.append({
            'Analysis': '2km_GEE_Combo_Trend',
            'Elevation_Stratum': elev_lbl,
            'Thermal_Class': th_lbl,
            'Model': 'NDVI_GS_NoSnow HAC',
            'n_unit': round(akm, 1),
            'Mean_Elev_m': np.nan,
            'Base_NDVI_04_08': round(nd0, 3),
            'NDVI_trend_dec': round(float(fit.params[1] * 10.0), 4),
            'Rel_change_pct': round(rel, 2),
            'HAC_p': round(float(fit.pvalues[1]), 5),
            'Explained_dec': np.nan,
            'R2': np.nan,
            'Unexpl_pct': np.nan,
            'Unexpl_CI95_lo': np.nan,
            'Unexpl_CI95_hi': np.nan,
        })

    # (B) 5-km pixel-panel within-stratum & baseline-NDVI-matched controls
    dem_5k, st4, ndvi, T, P, SM, SC, _, valid = load_pixel_masks()
    base_nd = ndvi[:5].mean(axis=0)
    ts = pd.read_csv(os.path.join(RES, 'jms_cross_sensor_annual_2004_2024.csv'))

    stratum_specs = [
        ('Pixel_Elevation_Stratum', '3800-4300m', (dem_5k >= 3800) & (dem_5k < 4300)),
        ('Pixel_Elevation_Stratum', '>=4300m', dem_5k >= 4300),
        ('Pixel_Elev_BaseNDVI_Matched', '3800-4300m & NDVI0 0.35-0.55',
         (dem_5k >= 3800) & (dem_5k < 4300) & (base_nd >= 0.35) & (base_nd <= 0.55)),
        ('Pixel_Basin_All', 'All elevations', np.ones_like(valid, dtype=bool)),
    ]
    for ana, el_lbl, el_mask in stratum_specs:
        for th_lbl, th_mask in [('Permafrost (ST4<0C)', st4 < 0), ('Seasonally Frozen (ST4>=0C)', st4 >= 0)]:
            m = valid & el_mask & th_mask
            if m.sum() < 20:
                continue
            sdict = {k: v[:, m].mean(axis=1) for k, v in dict(nd=ndvi, T=T, P=P, SM=SM, SC=SC).items()}
            for mdl_lbl, cols, alpha in [
                ('OLS T+P', ['T', 'P'], None),
                ('OLS T+P+SM+SC', ['T', 'P', 'SM', 'SC'], None),
                ('Ridge(alpha=1) T+P+SM+SC', ['T', 'P', 'SM', 'SC'], 1.0),
            ]:
                if ana != 'Pixel_Basin_All' and alpha is not None:
                    continue
                tr, ex, r2, un_pct, lo, hi = fit_residual_block_boot(yrs, sdict, cols, rng, alpha_ridge=alpha)
                rows.append({
                    'Analysis': ana,
                    'Elevation_Stratum': el_lbl,
                    'Thermal_Class': th_lbl,
                    'Model': mdl_lbl,
                    'n_unit': int(m.sum()),
                    'Mean_Elev_m': round(float(dem_5k[m].mean()), 1),
                    'Base_NDVI_04_08': round(float(base_nd[m].mean()), 3),
                    'NDVI_trend_dec': round(tr, 4),
                    'Rel_change_pct': np.nan,
                    'HAC_p': np.nan,
                    'Explained_dec': round(ex, 4),
                    'R2': round(r2, 3),
                    'Unexpl_pct': round(un_pct, 1),
                    'Unexpl_CI95_lo': round(lo, 1),
                    'Unexpl_CI95_hi': round(hi, 1),
                })

    # (C) Thermal threshold sensitivity (-0.5, 0.0, +0.5 C)
    tot_pix = int(valid.sum())
    for thr in (-0.5, 0.0, 0.5):
        for th_lbl, m_sub in [(f'ST4 < {thr:+.1f}C', valid & (st4 < thr)),
                              (f'ST4 >= {thr:+.1f}C', valid & (st4 >= thr))]:
            sdict = {k: v[:, m_sub].mean(axis=1) for k, v in dict(nd=ndvi, T=T, P=P).items()}
            tr, ex, r2, un_pct, lo, hi = fit_residual_block_boot(yrs, sdict, ['T', 'P'], rng)
            rows.append({
                'Analysis': f'ST4_Threshold_{thr:+.1f}C',
                'Elevation_Stratum': f'{m_sub.sum() / tot_pix * 100:.1f}% basin',
                'Thermal_Class': th_lbl,
                'Model': 'OLS T+P',
                'n_unit': int(m_sub.sum()),
                'Mean_Elev_m': round(float(dem_5k[m_sub].mean()), 1),
                'Base_NDVI_04_08': round(float(base_nd[m_sub].mean()), 3),
                'NDVI_trend_dec': round(tr, 4),
                'Rel_change_pct': np.nan,
                'HAC_p': np.nan,
                'Explained_dec': round(ex, 4),
                'R2': round(r2, 3),
                'Unexpl_pct': round(un_pct, 1),
                'Unexpl_CI95_lo': round(lo, 1),
                'Unexpl_CI95_hi': round(hi, 1),
            })

    # (D) Cross-sensor replacement: TerraClimate-adjusted Tmean + MODIS snow cover
    for th_lbl, th_mask, zid in [
        ('Permafrost (ST4<0C)', valid & (st4 < 0), 4),
        ('Seasonally Frozen (ST4>=0C)', valid & (st4 >= 0), 5),
    ]:
        sdict = {k: v[:, th_mask].mean(axis=1) for k, v in dict(nd=ndvi, T=T, P=P, SM=SM, SC=SC).items()}
        sub_z = ts[ts['zone_id'] == zid].sort_values('year')
        sdict['TC_T'] = sdict['T'] - (sub_z['ERA5_T2m'].values - sub_z['TC_Tmean'].values)
        sdict['MOD_SC'] = sub_z['MOD_Snow_GS_Pct'].values
        for mdl_lbl, cols in [
            ('CrossSensor TC_T + GPM_P', ['TC_T', 'P']),
            ('CrossSensor TC_T + P + SM + MODIS_SC', ['TC_T', 'P', 'SM', 'MOD_SC']),
        ]:
            tr, ex, r2, un_pct, lo, hi = fit_residual_block_boot(yrs, sdict, cols, rng)
            rows.append({
                'Analysis': 'CrossSensor_TC_MODIS',
                'Elevation_Stratum': 'All elevations',
                'Thermal_Class': th_lbl,
                'Model': mdl_lbl,
                'n_unit': int(th_mask.sum()),
                'Mean_Elev_m': round(float(dem_5k[th_mask].mean()), 1),
                'Base_NDVI_04_08': round(float(base_nd[th_mask].mean()), 3),
                'NDVI_trend_dec': round(tr, 4),
                'Rel_change_pct': np.nan,
                'HAC_p': np.nan,
                'Explained_dec': round(ex, 4),
                'R2': round(r2, 3),
                'Unexpl_pct': round(un_pct, 1),
                'Unexpl_CI95_lo': round(lo, 1),
                'Unexpl_CI95_hi': round(hi, 1),
            })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(RES, 'jms_permafrost_stratum_controls.csv')
    out_df.to_csv(out_path, index=False)
    print(f'Saved: {out_path} ({len(out_df)} rows)')
    print(out_df[out_df['Analysis'].isin(['2km_GEE_Combo_Trend', 'Pixel_Elevation_Stratum', 'Pixel_Elev_BaseNDVI_Matched', 'CrossSensor_TC_MODIS'])][
        ['Analysis', 'Elevation_Stratum', 'Thermal_Class', 'Model', 'n_unit', 'NDVI_trend_dec', 'Unexpl_pct', 'Unexpl_CI95_lo', 'Unexpl_CI95_hi']
    ].to_string(index=False))
    return out_df


def solve_budyko_w(ep, phi):
    if not (np.isfinite(ep) and np.isfinite(phi)) or ep <= 0.01 or ep >= 0.999 or phi <= 0:
        return np.nan
    try:
        return float(brentq(lambda w: (1.0 + phi ** (-w)) ** (-1.0 / w) - ep, 0.05, 25.0))
    except Exception:
        return np.nan


def cy_ep(phi, w):
    return phi / ((1.0 + phi ** w) ** (1.0 / w))


def run_budyko_mc_propagation():
    """Diagnostic 2: Propagate sigma_P, sigma_dS, sigma_R through Budyko w_WB and waterfall."""
    print('\n=== [2] Budyko Non-Steady-State Monte Carlo Error Propagation ===')
    t6 = pd.read_csv(os.path.join(RES, 'jms_table6_budyko_nonsteady.csv'))
    bl = pd.read_csv(os.path.join(RES, 'jms_budyko_2004_2024_nonsteady.csv'))
    df_m = pd.read_csv(os.path.join(RES, 'basin_merged_monthly_2003_2024.csv'))

    p_tc_yr = df_m[df_m['year'].between(2016, 2024)].groupby('year')['TC_P'].sum()
    pet_yr = bl[bl['year'].between(2016, 2024)].set_index('year')['PET_PML']

    rng = np.random.default_rng(2026)
    N_MC = 2000
    n_yrs = len(t6)
    w_draws = np.full((N_MC, n_yrs), np.nan)
    ep_draws = np.full((N_MC, n_yrs), np.nan)

    p_gpm_arr = t6['P_GPM_mm'].values
    p_tc_arr = np.array([float(p_tc_yr.loc[int(y)]) for y in t6['Year']])
    sig_p_arr = np.abs(p_gpm_arr - p_tc_arr)
    ds_arr = t6['dS_G22_mm'].values
    r_arr = t6['R_obs_mm'].values
    pet_arr = np.array([float(pet_yr.loc[int(y)]) for y in t6['Year']])

    # Store full-matrix MC draws across all 9 years simultaneously so annual and period metrics match
    p_sim_all = p_gpm_arr[None, :] + rng.normal(0.0, sig_p_arr[None, :], size=(N_MC, n_yrs))
    ds_sim_all = ds_arr[None, :] + rng.normal(0.0, 18.0, size=(N_MC, n_yrs))
    r_sim_all = r_arr[None, :] + rng.normal(0.0, 0.05 * r_arr[None, :], size=(N_MC, n_yrs))
    peff_sim_all = p_sim_all - ds_sim_all
    ewb_sim_all = peff_sim_all - r_sim_all
    ep_draws = ewb_sim_all / peff_sim_all

    for k in range(N_MC):
        for j in range(n_yrs):
            w_draws[k, j] = solve_budyko_w(ep_draws[k, j], pet_arr[j] / peff_sim_all[k, j])

    w_sd, w_68lo, w_68hi, w_95lo, w_95hi = [], [], [], [], []
    w_fix_lo, w_fix_hi = [], []
    ep_68lo, ep_68hi = [], []
    for j, r in t6.iterrows():
        wj = w_draws[np.isfinite(w_draws[:, j]), j]
        epj = ep_draws[np.isfinite(ep_draws[:, j]), j]
        w_sd.append(round(float(np.std(wj)), 3))
        w_68lo.append(round(float(np.percentile(wj, 16)), 3))
        w_68hi.append(round(float(np.percentile(wj, 84)), 3))
        w_95lo.append(round(float(np.percentile(wj, 2.5)), 3))
        w_95hi.append(round(float(np.percentile(wj, 97.5)), 3))
        ep_68lo.append(round(float(np.percentile(epj, 16)), 4))
        ep_68hi.append(round(float(np.percentile(epj, 84)), 4))
        phi0 = pet_arr[j] / r['P_eff_mm']
        w_fix_lo.append(round(solve_budyko_w((r['E_wb_mm'] - r['E_wb_unc_mm']) / r['P_eff_mm'], phi0), 3))
        w_fix_hi.append(round(solve_budyko_w((r['E_wb_mm'] + r['E_wb_unc_mm']) / r['P_eff_mm'], phi0), 3))

    t6['P_TC_mm'] = np.round(p_tc_arr, 1)
    t6['w_WB_mc_sd'] = w_sd
    t6['w_WB_ci68_lo'] = w_68lo
    t6['w_WB_ci68_hi'] = w_68hi
    t6['w_WB_ci95_lo'] = w_95lo
    t6['w_WB_ci95_hi'] = w_95hi
    t6['w_WB_fixedPeff_lo'] = w_fix_lo
    t6['w_WB_fixedPeff_hi'] = w_fix_hi
    t6['Ewb_over_Peff_ci68_lo'] = ep_68lo
    t6['Ewb_over_Peff_ci68_hi'] = ep_68hi

    t6_path = os.path.join(RES, 'jms_table6_budyko_nonsteady.csv')
    t6.to_csv(t6_path, index=False)
    print(f'Updated: {t6_path}')

    # Period-mean-flux & Waterfall uncertainty
    idx_pk = np.array([2, 3, 4])      # 2018-2020
    idx_dd = np.array([5, 6, 7, 8])   # 2021-2024
    idx_no20 = np.array([2, 3])       # 2018-2019 excluding 2020

    w_pk_mc, w_dd_mc, d_w_mc, clim_mc, ds_mc, lnd_mc = [], [], [], [], [], []
    for k in range(N_MC):
        w1 = solve_budyko_w(ewb_sim_all[k, idx_pk].mean() / peff_sim_all[k, idx_pk].mean(),
                            pet_arr[idx_pk].mean() / peff_sim_all[k, idx_pk].mean())
        w2 = solve_budyko_w(ewb_sim_all[k, idx_dd].mean() / peff_sim_all[k, idx_dd].mean(),
                            pet_arr[idx_dd].mean() / peff_sim_all[k, idx_dd].mean())
        if not (np.isfinite(w1) and np.isfinite(w2)):
            continue
        w_pk_mc.append(w1)
        w_dd_mc.append(w2)
        d_w_mc.append(w2 - w1)
        phi1 = pet_arr[idx_pk].mean() / peff_sim_all[k, idx_pk].mean()
        phi2 = pet_arr[idx_dd].mean() / peff_sim_all[k, idx_dd].mean()
        rp1 = r_sim_all[k, idx_pk].mean() / p_sim_all[k, idx_pk].mean()
        rp2 = r_sim_all[k, idx_dd].mean() / p_sim_all[k, idx_dd].mean()
        rp_c = (1.0 - ds_sim_all[k, idx_pk].mean() / p_sim_all[k, idx_pk].mean()) * (1.0 - cy_ep(phi2, w1))
        rp_d = (1.0 - ds_sim_all[k, idx_dd].mean() / p_sim_all[k, idx_dd].mean()) * (1.0 - cy_ep(phi2, w1))
        clim_mc.append(rp_c - rp1)
        ds_mc.append(rp_d - rp_c)
        lnd_mc.append(rp2 - rp_d)

    peff_tc = p_tc_arr - ds_arr
    ewb_tc = peff_tc - r_arr
    w_tc_pk = solve_budyko_w(ewb_tc[idx_pk].mean() / peff_tc[idx_pk].mean(), pet_arr[idx_pk].mean() / peff_tc[idx_pk].mean())
    w_tc_dd = solve_budyko_w(ewb_tc[idx_dd].mean() / peff_tc[idx_dd].mean(), pet_arr[idx_dd].mean() / peff_tc[idx_dd].mean())
    w_no20_pk = solve_budyko_w(t6['E_wb_mm'].values[idx_no20].mean() / t6['P_eff_mm'].values[idx_no20].mean(),
                               pet_arr[idx_no20].mean() / t6['P_eff_mm'].values[idx_no20].mean())

    dec = pd.read_csv(os.path.join(RES, 'jms_budyko_decomposition.csv')).iloc[0]
    mc_rows = [
        {'Metric': 'w_WB_Peak_2018_2020', 'Point_Estimate': round(float(dec['wWB_Peak']), 3),
         'MC_SD': round(float(np.std(w_pk_mc)), 3),
         'CI95_lo': round(float(np.percentile(w_pk_mc, 2.5)), 3),
         'CI95_hi': round(float(np.percentile(w_pk_mc, 97.5)), 3),
         'Note': 'Joint MC propagation of sigma_P, sigma_dS=18mm, sigma_R=5%'},
        {'Metric': 'w_WB_Drawdown_2021_2024', 'Point_Estimate': round(float(dec['wWB_Drawdown']), 3),
         'MC_SD': round(float(np.std(w_dd_mc)), 3),
         'CI95_lo': round(float(np.percentile(w_dd_mc, 2.5)), 3),
         'CI95_hi': round(float(np.percentile(w_dd_mc, 97.5)), 3),
         'Note': 'Joint MC propagation'},
        {'Metric': 'Delta_w_WB_Drawdown_minus_Peak', 'Point_Estimate': round(float(dec['wWB_Drawdown'] - dec['wWB_Peak']), 3),
         'MC_SD': round(float(np.std(d_w_mc)), 3),
         'CI95_lo': round(float(np.percentile(d_w_mc, 2.5)), 3),
         'CI95_hi': round(float(np.percentile(d_w_mc, 97.5)), 3),
         'Note': 'Invariance test across periods'},
        {'Metric': 'w_WB_Peak_2018_2019_excl_2020', 'Point_Estimate': round(float(w_no20_pk), 3),
         'MC_SD': np.nan, 'CI95_lo': np.nan, 'CI95_hi': np.nan,
         'Note': 'Sensitivity excluding extreme wet 2020'},
        {'Metric': 'Mean_E_wb_using_P_TC_2016_2024_mm', 'Point_Estimate': round(float(ewb_tc.mean()), 1),
         'MC_SD': np.nan, 'CI95_lo': np.nan, 'CI95_hi': np.nan,
         'Note': f'w_WB_TC Peak={w_tc_pk:.3f}, Drawdown={w_tc_dd:.3f}'},
        {'Metric': 'Waterfall_Climate_Aridity_Effect', 'Point_Estimate': round(float(dec['Climate_Aridity_Effect']), 4),
         'MC_SD': round(float(np.std(clim_mc)), 4),
         'CI95_lo': round(float(np.percentile(clim_mc, 2.5)), 4),
         'CI95_hi': round(float(np.percentile(clim_mc, 97.5)), 4),
         'Note': '2018-2020 to 2021-2024 R/P decomposition'},
        {'Metric': 'Waterfall_Storage_Release_Effect', 'Point_Estimate': round(float(dec['Storage_Release_Effect']), 4),
         'MC_SD': round(float(np.std(ds_mc)), 4),
         'CI95_lo': round(float(np.percentile(ds_mc, 2.5)), 4),
         'CI95_hi': round(float(np.percentile(ds_mc, 97.5)), 4),
         'Note': '2018-2020 to 2021-2024 R/P decomposition'},
        {'Metric': 'Waterfall_Landscape_wWB_Residual_Effect', 'Point_Estimate': round(float(dec['Landscape_wWB_Residual_Effect']), 4),
         'MC_SD': round(float(np.std(lnd_mc)), 4),
         'CI95_lo': round(float(np.percentile(lnd_mc, 2.5)), 4),
         'CI95_hi': round(float(np.percentile(lnd_mc, 97.5)), 4),
         'Note': '2018-2020 to 2021-2024 R/P decomposition'},
    ]
    mc_df = pd.DataFrame(mc_rows)
    mc_path = os.path.join(RES, 'jms_budyko_mc_uncertainty.csv')
    mc_df.to_csv(mc_path, index=False)
    print(f'Saved: {mc_path}')
    print(mc_df.to_string(index=False))
    return t6, mc_df


def run_shap_orthogonal_check():
    """Diagnostic 3: RF + XGBoost TreeSHAP on cumulative & orthogonal precipitation lag bands."""
    print('\n=== [3] TreeSHAP Model Alignment (RF vs XGBoost) & Orthogonal Lag Bands ===')
    df = pd.read_csv(os.path.join(RES, 'basin_merged_monthly_2003_2024.csv'))
    raw_base = df[(df['ym'] >= 200302) & (df['ym'] <= 202409)].copy().sort_values('ym_dt').reset_index(drop=True)
    raw_cols = ['lwe_basin', 'GPM_P', 'ERA5_T2m', 'PML_ET_tot', 'NDVI', 'PML_Ec']

    tr_raw = raw_base[raw_base['year'] < 2024]
    clim_tbl = tr_raw.groupby('month')[raw_cols].mean()
    w = raw_base[raw_base['year'] < 2024].copy()
    for c in raw_cols:
        w[f'anom_{c}'] = w[c] - w['month'].map(clim_tbl[c])
    w['P_cum3'] = w['anom_GPM_P'].rolling(3).sum()
    w['P_cum6'] = w['anom_GPM_P'].rolling(6).sum()
    w['P_cum12'] = w['anom_GPM_P'].rolling(12).sum()
    w['P_cum24'] = w['anom_GPM_P'].rolling(24).sum()
    w['P_band_0_3'] = w['P_cum3']
    w['P_band_4_6'] = w['P_cum6'] - w['P_cum3']
    w['P_band_7_12'] = w['P_cum12'] - w['P_cum6']
    w['P_band_13_24'] = w['P_cum24'] - w['P_cum12']
    w['ET_cum3'] = w['anom_PML_ET_tot'].rolling(3).sum()
    w['T2m_cum3'] = w['anom_ERA5_T2m'].rolling(3).mean()
    w['NDVI_lag1'] = w['anom_NDVI'].shift(1)
    w['NDVI_lag2'] = w['anom_NDVI'].shift(2)
    w['NDVI_lag3'] = w['anom_NDVI'].shift(3)
    w['Ec_cum3'] = w['anom_PML_Ec'].rolling(3).sum()
    w['lwe_lag1'] = w['anom_lwe_basin'].shift(1)

    exo24_names = [
        'anom_GPM_P', 'P_cum3', 'P_cum6', 'anom_ERA5_T2m', 'T2m_cum3',
        'anom_PML_ET_tot', 'ET_cum3', 'anom_NDVI', 'NDVI_lag1', 'NDVI_lag2',
        'NDVI_lag3', 'anom_PML_Ec', 'Ec_cum3', 'P_cum12', 'P_cum24',
    ]
    orth_names = [
        'P_band_0_3', 'P_band_4_6', 'P_band_7_12', 'P_band_13_24',
        'anom_ERA5_T2m', 'T2m_cum3', 'anom_PML_ET_tot', 'ET_cum3',
        'anom_NDVI', 'NDVI_lag1', 'NDVI_lag2', 'NDVI_lag3', 'anom_PML_Ec', 'Ec_cum3',
    ]
    sub = w.dropna(subset=['anom_lwe_basin'] + exo24_names + ['lwe_lag1']).copy()
    y = sub['anom_lwe_basin'].values

    rows = []
    for fset_name, fnames in [('Cumulative_exo24', exo24_names), ('Orthogonal_lag_bands', orth_names)]:
        X = StandardScaler().fit_transform(sub[fnames].values)
        rf = RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=5, random_state=42).fit(X, y)
        xg = xgb.XGBRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, random_state=42
        ).fit(X, y)
        for mname, mdl in [('RandomForest', rf), ('XGBoost', xg)]:
            sv = shap.TreeExplainer(mdl).shap_values(X)
            mean_abs = np.mean(np.abs(sv), axis=0)
            shares = mean_abs / np.sum(mean_abs) * 100.0
            for i, f in enumerate(fnames):
                dir_c = float(np.corrcoef(sub[f].values, sv[:, i])[0, 1])
                rows.append({
                    'FeatureSet': fset_name,
                    'Model': mname,
                    'n_train': len(sub),
                    'feature': f,
                    'mean_abs_shap': round(float(mean_abs[i]), 5),
                    'share_pct': round(float(shares[i]), 2),
                    'dir_corr': round(dir_c, 3),
                })

    sh_df = pd.DataFrame(rows)
    out_path = os.path.join(RES, 'jms_shap_orthogonal_check.csv')
    sh_df.to_csv(out_path, index=False)
    print(f'Saved: {out_path} ({len(sh_df)} rows)')
    for (fs, md), grp in sh_df.groupby(['FeatureSet', 'Model']):
        p_feats = grp[grp['feature'].str.contains('P_cum12|P_cum24|P_band_7_12|P_band_13_24')]['share_pct'].sum()
        p_all = grp[grp['feature'].str.contains('GPM_P|P_cum|P_band')]['share_pct'].sum()
        print(f'  {fs:<22} | {md:<13}: long_P={p_feats:.1f}%, all_P={p_all:.1f}%')
    return sh_df


def run_grace_rho_sensitivity():
    """Diagnostic 4: GRACE/GRACE-FO phase SNR & MC slope sensitivity across rho and FO bias."""
    print('\n=== [4] GRACE AR(1) rho & Inter-Mission FO Offset Sensitivity ===')
    df = pd.read_csv(os.path.join(RES, 'basin_merged_monthly_2003_2024.csv'))
    gv = df[df['lwe_basin'].notna() & (df['ym'] >= 200302) & (df['ym'] <= 202409)].copy().reset_index(drop=True)
    gv['anom_lwe'] = gv['lwe_basin'] - gv.groupby('month')['lwe_basin'].transform('mean')

    pre_s = gv[gv['ym'] <= 201706]
    dd_s = gv[gv['ym'] >= 202101]
    post_s = gv[gv['ym'] >= 201805]
    rec_s = gv[gv['ym'] >= 202209]

    diff_dd = float(dd_s['lwe_basin'].mean() - pre_s['lwe_basin'].mean())
    se_pre_iid = float(np.sqrt(np.sum(pre_s['unc_basin'] ** 2)) / len(pre_s))
    se_dd_iid = float(np.sqrt(np.sum(dd_s['unc_basin'] ** 2)) / len(dd_s))

    rng = np.random.default_rng(2026)
    rows = []
    for rho in (0.10, 0.30, 0.50, 0.60):
        infl = np.sqrt((1.0 + rho) / (1.0 - rho))
        for fo in (0.0, 0.5, 1.0):
            se_pre = se_pre_iid * infl
            se_dd = np.sqrt((se_dd_iid * infl) ** 2 + fo ** 2)
            snr = diff_dd / np.sqrt(se_pre ** 2 + se_dd ** 2)
            rows.append({
                'Diagnostic': 'Phase_SNR_2021_2024_vs_2003_2017',
                'rho': rho,
                'FO_bias_cm': fo,
                'Estimate': round(diff_dd, 3),
                'SNR': round(snr, 2),
                'CI95_lo': np.nan,
                'CI95_hi': np.nan,
            })

        for wname, sub in [('Slope_2018_05_to_2024_09', post_s), ('Slope_2022_09_to_2024_09', rec_s)]:
            t_x = sub['t_yr'].values - sub['t_yr'].values.mean()
            denom = np.sum(t_x ** 2)
            y0 = sub['anom_lwe'].values
            sig = sub['unc_basin'].values
            mc = []
            for _ in range(2000):
                w = rng.normal(0.0, 1.0, size=len(sub))
                e = np.empty(len(sub))
                e[0] = w[0]
                for i in range(1, len(sub)):
                    e[i] = rho * e[i - 1] + np.sqrt(1.0 - rho ** 2) * w[i]
                yc = y0 + e * sig
                mc.append(float(np.sum(t_x * (yc - np.mean(yc))) / denom))
            mc = np.array(mc)
            rows.append({
                'Diagnostic': wname,
                'rho': rho,
                'FO_bias_cm': 0.0,
                'Estimate': round(float(mc.mean()), 3),
                'SNR': np.nan,
                'CI95_lo': round(float(np.percentile(mc, 2.5)), 3),
                'CI95_hi': round(float(np.percentile(mc, 97.5)), 3),
            })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(RES, 'jms_grace_rho_sensitivity.csv')
    out_df.to_csv(out_path, index=False)
    print(f'Saved: {out_path} ({len(out_df)} rows)')
    print(out_df.to_string(index=False))
    return out_df


def main():
    run_permafrost_stratum_controls()
    run_budyko_mc_propagation()
    run_shap_orthogonal_check()
    run_grace_rho_sensitivity()
    print('\n>>> ALL ROUND-2 SUPPLEMENTARY DIAGNOSTICS COMPLETED CLEANLY!')


if __name__ == '__main__':
    main()
