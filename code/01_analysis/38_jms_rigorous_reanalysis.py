#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
38_jms_rigorous_reanalysis.py
=============================
Comprehensive post-peer-review quantitative audit and repair pipeline resolving every
Tier 0, Tier 1, and Tier 2 methodological, cryospheric, causal, and water-balance finding:
  1. GRACE exact 218-month ledger (144 pre + 10-mo gap [2017-07..2018-04] + 74 post [2018-05..2024-09]),
     6-native 3x3-deg cap discretization & exterior leakage diagnostics, AR(1)/block-bootstrap
     Monte Carlo + +-0.5 cm bias sensitivity, and passive linear storage-discharge null model.
  2. Exact Eq. (2) piecewise regression identity (beta1 + beta3 == post_slope), t/df reporting,
     and block-bootstrap Pettitt p-values.
  3. GLDAS-2.1 depth-normalized trends (mm a^-1 cm^-1) & conceptual reframing.
  4. Live GEE cross-sensor high-elevation validation (2004-2024): ERA5-Land T2m vs TerraClimate Tmean
     vs MODIS MOD11A2 LST vs GLDAS-2.1 Tair; MODIS MOD13A1 snow-masked NDVI (SummaryQA<=1 & Jul-Aug)
     and MOD10A1 NDSI snow cover; plus thermally mapped permafrost domain (ST4_clim < 0 C).
  5. Pyper-Peterman (1998) N_eff & Hamed-Rao (1998) MMK significance corrections across Table 2,
     memory windows k=1..12, and non-LAI ET checks (ERA5-Land & TerraClimate vs PML-V2.2a).
  6. Zero-leakage expanding walk-forward evaluation (in-fold seasonal climatology, matched N_test,
     all 4 algorithms across Exogenous [0-6 mo], Exogenous-Long [12-24 mo], Exogenous+AR(1), Persistence).
  7. Non-steady-state Choudhury-Yang Budyko (P_eff = P - dS_G22) for E_wb & E_PML, propagated E_wb
     uncertainty bounds, 2004-2017 baseline benchmarking, elasticity decomposition, and km^3 volumes.
"""

import os
import json
import time
import calendar
import warnings
warnings.filterwarnings('ignore')

import ee
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq, minimize
import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

OUT_DIR = '/Users/yueliangmanle/Desktop/小论文中文2/results_repaired'
PROC_DIR = '/Users/yueliangmanle/Desktop/小论文中文2/data/processed'
GEO_PATH = '/Users/yueliangmanle/Desktop/小论文中文2/高级图设计包/fig1_研究区/yr_source_boundary.geojson'
CELLS_PATH = '/Users/yueliangmanle/Desktop/小论文中文2/投稿包/04_数据/处理后数据/basin_cells_classified.csv'
os.makedirs(OUT_DIR, exist_ok=True)

AREA_KM2 = 123800.0
TANGNAIHAI_E8M3 = {
    2016: 136.4, 2017: 186.1, 2018: 291.5, 2019: 310.3,
    2020: 321.6, 2021: 222.9, 2022: 170.9, 2023: 220.7, 2024: 224.57,
}


# ---------------------------------------------------------------------------
# Generic Statistical & Autocorrelation-Adjusted Helpers
# ---------------------------------------------------------------------------
def retry_gee(fn, retries=6, delay=3.0):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))
            try:
                ee.Initialize(project='round-fusion-508306-p4')
            except Exception:
                pass


def ols_hac(t_yr, y, maxlags=4):
    m = np.isfinite(t_yr) & np.isfinite(y)
    t = np.asarray(t_yr[m], dtype=float)
    v = np.asarray(y[m], dtype=float)
    n = len(v)
    if n < 6:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0
    X = sm.add_constant(t - t[0])
    fit = sm.OLS(v, X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    ci = fit.conf_int()
    return (
        float(fit.params[1]), float(fit.bse[1]), float(fit.tvalues[1]),
        float(fit.pvalues[1]), float(ci[1, 0]), float(ci[1, 1]), n - 2
    )


def hamed_rao_mmk(y):
    """Hamed & Rao (1998) variance-corrected Mann-Kendall Z & p with Sen slope."""
    v = np.asarray(y, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 8:
        return np.nan, np.nan, np.nan
    i_idx, j_idx = np.triu_indices(n, k=1)
    S = float(np.sum(np.sign(v[j_idx] - v[i_idx])))
    _, counts = np.unique(v, return_counts=True)
    var_s0 = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18.0
    if var_s0 <= 0:
        return 0.0, 1.0, 1.0
    # Detrend via Sen slope before computing rank autocorrelation
    slopes = (v[j_idx] - v[i_idx]) / (j_idx - i_idx)
    sen = float(np.median(slopes))
    res = v - sen * np.arange(n)
    ranks = stats.rankdata(res)
    rk_c = ranks - np.mean(ranks)
    denom = np.sum(rk_c ** 2)
    ess_sum = 0.0
    max_lag = min(n - 3, max(1, n // 3))
    if denom > 0:
        for lag in range(1, max_lag + 1):
            r_k = float(np.sum(rk_c[:-lag] * rk_c[lag:]) / denom)
            # Include lag if statistically distinguishable at 95% two-sided
            if abs(r_k) > 1.96 / np.sqrt(n):
                ess_sum += (n - lag) * (n - lag - 1) * (n - lag - 2) * r_k
    n_ns = 1.0 + (2.0 / (n * (n - 1) * (n - 2))) * ess_sum
    n_ns = max(0.5, n_ns)
    var_s = var_s0 * n_ns
    Z = (S - 1.0) / np.sqrt(var_s) if S > 0 else ((S + 1.0) / np.sqrt(var_s) if S < 0 else 0.0)
    p = float(2.0 * stats.norm.sf(abs(Z)))
    return float(Z), p, float(n_ns)


def pyper_peterman_neff(x, y, max_lag=None):
    """Pyper & Peterman (1998) effective sample size N_eff for autocorrelated pairs."""
    m = np.isfinite(x) & np.isfinite(y)
    xv = np.asarray(x[m], dtype=float)
    yv = np.asarray(y[m], dtype=float)
    n = len(xv)
    if n < 10:
        return float(n)
    xc = xv - np.mean(xv)
    yc = yv - np.mean(yv)
    dx = np.sum(xc ** 2)
    dy = np.sum(yc ** 2)
    if dx <= 0 or dy <= 0:
        return float(n)
    if max_lag is None:
        max_lag = max(1, min(n // 4, 24))
    sum_ac = 0.0
    for k in range(1, max_lag + 1):
        rx_k = np.sum(xc[:-k] * xc[k:]) / dx
        ry_k = np.sum(yc[:-k] * yc[k:]) / dy
        sum_ac += ((n - k) / n) * rx_k * ry_k
    inv_neff = (1.0 / n) + (2.0 / n) * sum_ac
    neff = 1.0 / max(inv_neff, 1e-9)
    return float(np.clip(neff, 4.0, float(n)))


def pearson_with_neff(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    xv, yv = np.asarray(x[m], dtype=float), np.asarray(y[m], dtype=float)
    n = len(xv)
    r, p_iid = stats.pearsonr(xv, yv)
    neff = pyper_peterman_neff(xv, yv)
    df_eff = max(2.0, neff - 2.0)
    t_eff = r * np.sqrt(df_eff / max(1.0 - r ** 2, 1e-12))
    p_eff = float(2.0 * stats.t.sf(abs(t_eff), df=df_eff))
    t_iid = r * np.sqrt((n - 2.0) / max(1.0 - r ** 2, 1e-12))
    return float(r), n, float(t_iid), float(p_iid), float(neff), float(t_eff), p_eff


def partial_corr_with_neff(x, y, Z):
    df_tmp = pd.DataFrame({'x': x, 'y': y})
    for k in range(Z.shape[1]):
        df_tmp[f'z{k}'] = Z[:, k]
    df_tmp = df_tmp.dropna()
    n = len(df_tmp)
    Z_mat = sm.add_constant(df_tmp[[f'z{k}' for k in range(Z.shape[1])]].values)
    rx = sm.OLS(df_tmp['x'].values, Z_mat).fit().resid
    ry = sm.OLS(df_tmp['y'].values, Z_mat).fit().resid
    r, p_iid = stats.pearsonr(rx, ry)
    neff = pyper_peterman_neff(rx, ry)
    df_iid = max(2, n - Z.shape[1] - 2)
    df_eff = max(2.0, neff - Z.shape[1] - 2.0)
    t_iid = r * np.sqrt(df_iid / max(1.0 - r ** 2, 1e-12))
    t_eff = r * np.sqrt(df_eff / max(1.0 - r ** 2, 1e-12))
    p_eff = float(2.0 * stats.t.sf(abs(t_eff), df=df_eff))
    return float(r), n, int(df_iid), float(t_iid), float(p_iid), float(neff), float(df_eff), float(t_eff), p_eff


def pettitt_with_ar_boot(dates, y, n_boot=1000, seed=2026):
    m = np.isfinite(y)
    d = np.asarray(dates[m])
    v = np.asarray(y[m], dtype=float)
    n = len(v)
    ranks = stats.rankdata(v)
    cum_r = np.cumsum(ranks)
    idx = np.arange(1, n + 1)
    U = 2.0 * cum_r[:-1] - idx[:-1] * (n + 1.0)
    k_idx = int(np.argmax(np.abs(U)))
    K = float(np.abs(U[k_idx]))
    p_iid = min(1.0, float(2.0 * np.exp((-6.0 * K ** 2) / (n ** 3 + n ** 2))))
    # AR(1) stationary null bootstrap for realistic serial-correlation p-value
    phi = float(np.corrcoef(v[:-1], v[1:])[0, 1]) if n > 5 else 0.0
    phi = np.clip(phi, -0.95, 0.95)
    res = v[1:] - (np.mean(v) + phi * (v[:-1] - np.mean(v)))
    rng = np.random.default_rng(seed)
    ge_count = 0
    for _ in range(n_boot):
        sim = np.empty(n)
        sim[0] = v[0]
        draws = rng.choice(res, size=n - 1, replace=True)
        for i in range(1, n):
            sim[i] = np.mean(v) + phi * (sim[i - 1] - np.mean(v)) + draws[i - 1]
        rk_s = stats.rankdata(sim)
        U_s = 2.0 * np.cumsum(rk_s)[:-1] - idx[:-1] * (n + 1.0)
        if np.max(np.abs(U_s)) >= K:
            ge_count += 1
    p_ar = float((ge_count + 1) / (n_boot + 1))
    return str(pd.Timestamp(d[k_idx]).strftime('%Y-%m')), K, p_iid, p_ar, n


# ---------------------------------------------------------------------------
# 1. GRACE 6 Native 3-deg Cap Spatial Resolution, Leakage, Correlated MC, & Linear Reservoir Null
# ---------------------------------------------------------------------------
def audit_grace_6_caps_and_null(df):
    print('\n=== [1] GRACE 6-Cap Native Resolution, Correlated MC & Linear Reservoir Null ===')
    g_grid = pd.read_csv(os.path.join(PROC_DIR, 'grace_monthly_grid.csv'))
    cells = pd.read_csv(CELLS_PATH)
    g_grid['lat_r'] = g_grid['lat'].round(4)
    g_grid['lon_r'] = g_grid['lon'].round(4)
    cells['lat_r'] = cells['lat'].round(4)
    cells['lon_r'] = cells['lon'].round(4)

    # Identify unique native 3x3-deg spherical caps via a low-collision monthly signature
    m202006 = g_grid[g_grid['ym'] == 202006].merge(cells[['lat_r', 'lon_r', 'intersects', 'norm_weight']], on=['lat_r', 'lon_r'], how='left')
    m202006['cap_sig'] = m202006['lwe'].round(6)
    inter = m202006[m202006['intersects'] == True].copy()
    caps_summary = []
    cap_map = {}
    for ci, (sig, grp) in enumerate(inter.groupby('cap_sig'), start=1):
        tot_cells = int((m202006['cap_sig'] == sig).sum())
        in_cells = len(grp)
        w_sum = float(grp['norm_weight'].sum())
        cap_map[sig] = f'Cap_{ci}'
        caps_summary.append({
            'Cap_ID': f'Cap_{ci}',
            'Center_Lon': round(float(grp['lon'].mean()), 2),
            'Center_Lat': round(float(grp['lat'].mean()), 2),
            'Subgrid_Cells_In_Basin': in_cells,
            'Total_Cells_In_Domain': tot_cells,
            'Interior_Area_Weight': round(w_sum, 4),
            'Exterior_Fraction_In_144Grid': round(1.0 - in_cells / max(tot_cells, 1), 3),
        })
    cap_df = pd.DataFrame(caps_summary)
    cap_df.to_csv(os.path.join(OUT_DIR, 'jms_grace_cap_diagnostics.csv'), index=False)
    print(cap_df.to_string(index=False))

    # Build monthly cap-level series for exact correlated sigma_basin calculation
    sig_lookup = m202006[['lat_r', 'lon_r', 'cap_sig']].copy()
    g_m = g_grid.merge(sig_lookup, on=['lat_r', 'lon_r'], how='inner')
    g_m = g_m.merge(cells[['lat_r', 'lon_r', 'intersects', 'norm_weight']], on=['lat_r', 'lon_r'], how='left')
    g_in = g_m[g_m['intersects'] == True].copy()
    cap_weights = g_in.groupby('cap_sig')['norm_weight'].sum() / len(g_m['ym'].unique())

    # Compare uncertainty aggregation across the 6 native caps:
    # (a) naive weighted mean unc (what unc_basin stores)
    # (b) independent 6-cap quadrature sqrt(sum((w_c * unc_c)^2))
    # (c) fully correlated upper bound sum(w_c * unc_c)
    cap_ts = g_in.groupby(['ym', 'cap_sig']).agg(lwe=('lwe', 'first'), unc=('unc', 'first'), w=('norm_weight', 'sum')).reset_index()
    cap_ts = cap_ts[(cap_ts['ym'] >= 200302) & (cap_ts['ym'] <= 202409)].dropna(subset=['lwe'])
    unc_comp = cap_ts.groupby('ym').apply(
        lambda s: pd.Series({
            'sigma_quad6': float(np.sqrt(np.sum((s['w'] * s['unc']) ** 2))),
            'sigma_linear': float(np.sum(s['w'] * s['unc'])),
        })
    ).reset_index()

    gv = df[df['lwe_basin'].notna() & (df['ym'] >= 200302) & (df['ym'] <= 202409)].copy().reset_index(drop=True)
    gv = gv.merge(unc_comp, on='ym', how='left')
    gv['anom_lwe'] = gv['lwe_basin'] - gv.groupby('month')['lwe_basin'].transform('mean')
    phi_ar1 = float(np.corrcoef(gv['anom_lwe'].iloc[:-1], gv['anom_lwe'].iloc[1:])[0, 1])

    # Month ledger verification
    n_pre = int((gv['ym'] <= 201706).sum())
    n_gap_valid = int(((gv['ym'] >= 201707) & (gv['ym'] <= 201804)).sum())
    n_post_may18 = int((gv['ym'] >= 201805).sum())
    n_post_jun18 = int((gv['ym'] >= 201806).sum())
    print(f'[Month Ledger] Total={len(gv)}, Pre(<=2017-06)={n_pre}, Gap(2017-07..2018-04)={n_gap_valid}, Post(>=2018-05)={n_post_may18}, Post(>=2018-06)={n_post_jun18}')

    # Correlated Error & Inter-Mission Bias Propagation on Phase SNR and Post-2018 Slope
    rng = np.random.default_rng(2026)
    phase_rows = []
    pre_s = gv[gv['ym'] <= 201706]
    # AR(1)-adjusted phase-mean SE: sigma_mean_eff = sqrt((sum sigma_t^2)/N^2) * sqrt((1+phi_err)/(1-phi_err))
    # Use conservative error-aliasing correlation phi_err = 0.50 (approx 3-month GRACE Stokes smoothing) and +-0.5 cm GRACE/GRACE-FO inter-mission bias
    phi_err = 0.50
    inflation_ar = np.sqrt((1.0 + phi_err) / (1.0 - phi_err))
    bias_fo_sd = 0.50  # conservative 0.5 cm GRACE vs GRACE-FO baseline offset uncertainty (1-sigma)

    def phase_se(sub_df, sigma_col='unc_basin', include_fo_bias=False):
        se_iid = float(np.sqrt(np.sum(sub_df[sigma_col] ** 2)) / len(sub_df))
        se_corr = se_iid * inflation_ar
        if include_fo_bias:
            se_corr = float(np.sqrt(se_corr ** 2 + bias_fo_sd ** 2))
        return se_iid, se_corr

    se_pre_iid, se_pre_corr = phase_se(pre_s, 'unc_basin', include_fo_bias=False)
    for pname, mask, is_fo in [
        ('Pre-transition (2003-02..2017-06)', gv['ym'] <= 201706, False),
        ('Pluvial Peak (2018-05..2020-12)', (gv['ym'] >= 201805) & (gv['ym'] <= 202012), True),
        ('Post-peak Drawdown (2021-01..2024-09)', gv['ym'] >= 202101, True),
        ('Post-transition Full (2018-05..2024-09)', gv['ym'] >= 201805, True),
        ('Latest Year (2024-01..2024-09)', gv['year'] == 2024, True),
    ]:
        s = gv[mask]
        se_iid, se_corr = phase_se(s, 'unc_basin', include_fo_bias=is_fo)
        diff_jpl = float(s['lwe_basin'].mean() - pre_s['lwe_basin'].mean())
        diff_anom = float(s['anom_lwe'].mean() - pre_s['anom_lwe'].mean())
        comb_iid = float(np.sqrt(se_iid ** 2 + se_pre_iid ** 2))
        comb_corr = float(np.sqrt(se_corr ** 2 + se_pre_corr ** 2))
        phase_rows.append({
            'Phase': pname, 'n_months': len(s),
            'Mean_JPL_cm': round(float(s['lwe_basin'].mean()), 3),
            'Mean_Anom_cm': round(float(s['anom_lwe'].mean()), 3),
            'Mean_Unc_cm': round(float(s['unc_basin'].mean()), 3),
            'Mean_Quad6Cap_Unc_cm': round(float(s['sigma_quad6'].mean()), 3),
            'SE_iid_cm': round(se_iid, 3),
            'SE_corr_with_FObias_cm': round(se_corr, 3),
            'Diff_vs_Pre_JPL_cm': round(diff_jpl, 3),
            'Diff_vs_Pre_Anom_cm': round(diff_anom, 3),
            'SNR_iid': round(diff_jpl / comb_iid, 2),
            'SNR_corr_FObias': round(diff_jpl / comb_corr, 2),
        })
    ph_df = pd.DataFrame(phase_rows)
    ph_df.to_csv(os.path.join(OUT_DIR, 'jms_grace_uncertainty_robust.csv'), index=False)
    print(ph_df.to_string(index=False))

    # Post-transition sub-window trend tests + Correlated AR(1) & Block-Bootstrap Monte Carlo
    sub_mc_rows = []
    for wname, mask, apply_fo_step in [
        ('Full (2003-02..2024-09)', gv['ym'] >= 200302, True),
        ('Pre-gap (2003-02..2017-06)', gv['ym'] <= 201706, False),
        ('Post-gap (2018-05..2024-09)', gv['ym'] >= 201805, False),
        ('Post-gap (2018-06..2024-09)', gv['ym'] >= 201806, False),
        ('Post-peak Drawdown (2020-09..2024-09)', gv['ym'] >= 202009, False),
        ('Recent Stabilization (2022-09..2024-09)', gv['ym'] >= 202209, False),
    ]:
        s = gv[mask].copy().reset_index(drop=True)
        sl_ols, se_o, t_o, p_o, lo_o, hi_o, df_o = ols_hac(s['t_yr'].values, s['anom_lwe'].values, maxlags=4)
        X = sm.add_constant(s['t_yr'].values - s['t_yr'].values[0])
        fit_wls = sm.WLS(s['anom_lwe'].values, X, weights=1.0 / (s['unc_basin'].values ** 2)).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
        # Correlated AR(1) noise MC (phi=0.50) + N(0, 0.5) FO inter-mission step for Full window
        t_x = s['t_yr'].values - s['t_yr'].values.mean()
        denom = np.sum(t_x ** 2)
        y0 = s['anom_lwe'].values
        sig = s['unc_basin'].values
        is_post = (s['ym'].values >= 201805).astype(float)
        mc_iid, mc_corr = [], []
        for _ in range(2000):
            # iid draw
            z_iid = rng.normal(0.0, sig)
            mc_iid.append(float(np.sum(t_x * (y0 + z_iid - np.mean(y0 + z_iid))) / denom))
            # AR(1) correlated error draw (phi=0.50)
            w = rng.normal(0.0, 1.0, size=len(s))
            e_unit = np.empty(len(s))
            e_unit[0] = w[0]
            for i in range(1, len(s)):
                e_unit[i] = 0.50 * e_unit[i - 1] + np.sqrt(1.0 - 0.50 ** 2) * w[i]
            step_err = rng.normal(0.0, bias_fo_sd) * is_post if apply_fo_step else 0.0
            y_c = y0 + e_unit * sig + step_err
            mc_corr.append(float(np.sum(t_x * (y_c - np.mean(y_c))) / denom))
        mc_iid, mc_corr = np.array(mc_iid), np.array(mc_corr)
        sub_mc_rows.append({
            'Window': wname, 'n': len(s), 'df': df_o,
            'OLS_HAC_slope': round(sl_ols, 3), 'OLS_t': round(t_o, 2), 'OLS_p': round(p_o, 6),
            'WLS_HAC_slope': round(float(fit_wls.params[1]), 3), 'WLS_p': round(float(fit_wls.pvalues[1]), 6),
            'MC_iid_mean': round(float(mc_iid.mean()), 3),
            'MC_corr_mean': round(float(mc_corr.mean()), 3),
            'MC_corr_95CI': f'[{np.percentile(mc_corr, 2.5):+.3f}, {np.percentile(mc_corr, 97.5):+.3f}]',
            'MC_corr_same_sign_pct': round(float(np.mean(np.sign(mc_corr) == np.sign(mc_corr.mean())) * 100), 1),
        })
    sub_mc_df = pd.DataFrame(sub_mc_rows)
    sub_mc_df.to_csv(os.path.join(OUT_DIR, 'jms_grace_wls_montecarlo_robust.csv'), index=False)
    print(sub_mc_df.to_string(index=False))

    # Passive Linear Storage-Discharge Reservoir Null Model: dS/dt = P' - c_off - S/tau (calibrated strictly on 2003-2017!)
    df_cont = df[(df['ym'] >= 200302) & (df['ym'] <= 202409)].copy().reset_index(drop=True)
    df_cont['lwe_anom_full'] = df_cont['anom_lwe_basin']
    p_anom_cm = df_cont['anom_GPM_P'].values / 10.0  # mm/mo -> cm/mo
    m_pre_cal = (df_cont['ym'].values <= 201706) & np.isfinite(df_cont['lwe_anom_full'].values)

    def sim_reservoir(params):
        tau_mo, c_off, s0 = params
        tau_mo = max(1.5, tau_mo)
        decay = np.exp(-1.0 / tau_mo)
        gain = (1.0 - decay) * tau_mo
        s = np.empty(len(p_anom_cm))
        cur = s0
        for i in range(len(p_anom_cm)):
            cur = cur * decay + (p_anom_cm[i] - c_off) * gain
            s[i] = cur
        return s

    def loss_pre(params):
        s_sim = sim_reservoir(params)
        diff = s_sim[m_pre_cal] - df_cont['lwe_anom_full'].values[m_pre_cal]
        return float(np.mean(diff ** 2))

    # Multi-start grid over physical residence times tau = 2..36 months to avoid local minima
    best_opt = None
    for tau_init in [3.0, 6.0, 12.0, 18.0, 24.0]:
        cand = minimize(loss_pre, x0=[tau_init, 0.0, -1.0], bounds=[(1.5, 48.0), (-2.0, 2.0), (-5.0, 5.0)])
        if best_opt is None or cand.fun < best_opt.fun:
            best_opt = cand
    opt = best_opt
    s_null = sim_reservoir(opt.x)
    df_cont['TWSA_null_reservoir_cm'] = s_null
    m_post_val = (df_cont['ym'].values >= 201805) & np.isfinite(df_cont['lwe_anom_full'].values)
    r_pre_null, _ = stats.pearsonr(df_cont.loc[m_pre_cal, 'lwe_anom_full'], s_null[m_pre_cal])
    r_post_null, _ = stats.pearsonr(df_cont.loc[m_post_val, 'lwe_anom_full'], s_null[m_post_val])
    rmse_post_null = float(np.sqrt(np.mean((df_cont.loc[m_post_val, 'lwe_anom_full'] - s_null[m_post_val]) ** 2)))
    sl_null_post, _, _, p_null_post, _, _, _ = ols_hac(df_cont.loc[m_post_val, 't_yr'].values, s_null[m_post_val], maxlags=4)
    obs_post_mean21_24 = float(df_cont.loc[(df_cont['year'] >= 2021) & np.isfinite(df_cont['lwe_anom_full']), 'lwe_anom_full'].mean())
    null_post_mean21_24 = float(df_cont.loc[df_cont['year'] >= 2021, 'TWSA_null_reservoir_cm'].mean())
    null_summary = pd.DataFrame([{
        'Calib_Tau_months': round(float(opt.x[0]), 2),
        'Pre_2003_2017_r': round(float(r_pre_null), 3),
        'OOS_2018_2024_r': round(float(r_post_null), 3),
        'OOS_2018_2024_RMSE_cm': round(rmse_post_null, 3),
        'Null_Post2018_Slope_cma': round(sl_null_post, 3),
        'Null_Post2018_p': round(p_null_post, 5),
        'Obs_2021_2024_Anom_cm': round(obs_post_mean21_24, 3),
        'Null_2021_2024_Anom_cm': round(null_post_mean21_24, 3),
        'Obs_minus_Null_2021_2024_cm': round(obs_post_mean21_24 - null_post_mean21_24, 3),
    }])
    null_summary.to_csv(os.path.join(OUT_DIR, 'jms_linear_reservoir_null.csv'), index=False)
    df_cont[['ym', 'ym_dt', 't_yr', 'anom_lwe_basin', 'TWSA_null_reservoir_cm']].to_csv(
        os.path.join(OUT_DIR, 'jms_linear_reservoir_series.csv'), index=False
    )
    print('\n[Linear Reservoir Null Model OOS 2018-2024]:')
    print(null_summary.to_string(index=False))
    return cap_df, ph_df, sub_mc_df, null_summary


# ---------------------------------------------------------------------------
# 2. Exact Equation (2) Piecewise Consistency & Block-Bootstrap Pettitt
# ---------------------------------------------------------------------------
def reconcile_table3_segmented_eq2(df):
    print('\n=== [2] Exact Eq.(2) Piecewise Consistency & Complete Table 3 ===')
    targets = [
        ('GRACE Basin TWSA (cm a⁻¹)', 'anom_lwe_basin', 1.0, '2024-09-01', True),
        ('GLDAS-2.2 TWSA [matched 2024-09] (cm a⁻¹)', 'anom_G22_TWS_tavg', 0.1, '2024-09-01', False),
        ('GLDAS-2.2 TWSA [full 2024-12] (cm a⁻¹)', 'anom_G22_TWS_tavg', 0.1, '2024-12-01', False),
        ('GLDAS-2.1 Proxy [matched 2024-09] (cm a⁻¹)', 'anom_G21_TWS_proxy', 0.1, '2024-09-01', False),
        ('GLDAS-2.1 Proxy [full 2024-12] (cm a⁻¹)', 'anom_G21_TWS_proxy', 0.1, '2024-12-01', False),
        ('MODIS NDVI Anomaly (×10⁻³ a⁻¹)', 'anom_NDVI', 1000.0, '2024-12-01', False),
    ]
    rows = []
    t0_split = 2017.50  # Exact July 2017 breakpoint t0
    for label, col, sc, end_dt, is_grace in targets:
        sub_all = df[df['ym_dt'] <= end_dt].dropna(subset=[col]).copy()
        # Exclude 10-month GRACE inter-mission window uniformly in piecewise comparison only for GRACE; for continuous LSM keep masked vs pooled clearly aligned
        sub_subsamples = sub_all[(sub_all['ym'] <= 201706) | (sub_all['ym'] >= 201805)].copy()
        y_f = sub_all[col].values * sc
        t_f = sub_all['t_yr'].values

        sl_full, _, t_full, p_full, lo_f, hi_f, df_full = ols_hac(t_f, y_f, maxlags=4)
        # Pooled Eq. (2) strictly evaluated on matching Pre (<=2017-06) + Post (>=2018-05) mask for hydrological series (and continuous 2017-07 split for NDVI)
        fit_df = sub_all if label.startswith('MODIS') else sub_subsamples
        yf = fit_df[col].values * sc
        tf = fit_df['t_yr'].values
        I = (fit_df['ym'].values >= (201707 if label.startswith('MODIS') else 201805)).astype(float)
        tr = tf - t0_split
        X = np.column_stack([np.ones_like(tr), tr, I, tr * I])
        fit_pw = sm.OLS(yf, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
        b1 = float(fit_pw.params[1])
        b3 = float(fit_pw.params[3])
        post_pw = b1 + b3
        wald_post = fit_pw.wald_test(np.array([[0.0, 1.0, 0.0, 1.0]]), scalar=True)
        # Independent subsample slopes (Pre <=2017-06, Post >=2018-05)
        s_pre = sub_all[sub_all['ym'] <= 201706]
        s_post = sub_all[sub_all['ym'] >= 201805]
        sl_pre_s, _, t_pre_s, p_pre_s, _, _, _ = ols_hac(s_pre['t_yr'].values, s_pre[col].values * sc, maxlags=4)
        sl_post_s, _, t_post_s, p_post_s, lo_p, hi_p, _ = ols_hac(s_post['t_yr'].values, s_post[col].values * sc, maxlags=4)
        cp_dt, K_pt, p_pt_iid, p_pt_ar, n_pt = pettitt_with_ar_boot(sub_all['ym_dt'].values, y_f, n_boot=800)

        rows.append({
            'Series': label, 'N_total': len(sub_all), 'N_pre': len(s_pre), 'N_post_May18': len(s_post),
            'Full_HAC': round(sl_full, 4), 'Full_CI': f'[{lo_f:.3f}, {hi_f:.3f}]', 'Full_t': round(t_full, 2), 'Full_p': p_full, 'df_full': df_full,
            'Eq2_Beta1_Pre': round(b1, 4), 'Eq2_Beta3_Shift': round(b3, 4), 'Eq2_Shift_p': float(fit_pw.pvalues[3]),
            'Eq2_Post_Beta1PlusBeta3': round(post_pw, 4), 'Eq2_Post_p': float(wald_post.pvalue),
            'Sub_Pre_HAC': round(sl_pre_s, 4), 'Sub_Pre_p': round(p_pre_s, 4),
            'Sub_Post_May18_HAC': round(sl_post_s, 4), 'Sub_Post_CI': f'[{lo_p:.3f}, {hi_p:.3f}]', 'Sub_Post_t': round(t_post_s, 2), 'Sub_Post_p': float(p_post_s),
            'Pettitt_Break': cp_dt, 'Pettitt_K': round(K_pt, 1), 'Pettitt_p_iid': p_pt_iid, 'Pettitt_p_ARboot': round(p_pt_ar, 4),
        })
    t3_df = pd.DataFrame(rows)
    t3_df.to_csv(os.path.join(OUT_DIR, 'jms_table3_reconciled.csv'), index=False)
    print(t3_df[['Series', 'N_total', 'N_pre', 'N_post_May18', 'Full_HAC', 'Eq2_Beta1_Pre', 'Eq2_Beta3_Shift', 'Eq2_Post_Beta1PlusBeta3', 'Sub_Post_May18_HAC', 'Pettitt_Break', 'Pettitt_p_ARboot']].to_string(index=False))
    return t3_df


# ---------------------------------------------------------------------------
# 3. Depth-Normalized GLDAS-2.1 Soil Moisture (mm a^-1 cm^-1) & GLDAS-2.2 Matched
# ---------------------------------------------------------------------------
def audit_subsurface_depth_normalized(df):
    print('\n=== [3] Depth-Normalized GLDAS-2.1 & GLDAS-2.2 Conceptual Subsurface Audit ===')
    post_g = df[(df['ym_dt'] >= '2018-05-01') & (df['ym_dt'] <= '2024-09-01')]
    post_e = df[(df['ym_dt'] >= '2018-05-01') & (df['ym_dt'] <= '2024-12-01')]
    rows = []
    # GLDAS-2.2 components (matched 2018-05..2024-09, n=77)
    g22_sl = {}
    for lbl, col in [('GLDAS-2.2 GWS (Assimilated EnKS)', 'anom_G22_GWS_tavg'),
                     ('GLDAS-2.2 RZSM (Assimilated EnKS)', 'anom_G22_SoilMoist_RZ_tavg'),
                     ('GLDAS-2.2 SWE (Assimilated EnKS)', 'anom_G22_SWE_tavg')]:
        sl, _, t_v, p_v, lo, hi, df_v = ols_hac(post_g['t_yr'].values, post_g[col].values, maxlags=4)
        g22_sl[lbl] = sl
        rows.append({'Model_Layer': lbl, 'Thickness_cm': np.nan, 'Slope_mma_05_09': round(sl, 3),
                     'Per_cm_mma_cm': np.nan, 'Share_pct': np.nan, 't': round(t_v, 2), 'df': df_v, 'p_HAC': round(p_v, 5)})
    tot22 = sum(g22_sl.values())
    for r in rows[:3]:
        r['Share_pct'] = round(r['Slope_mma_05_09'] / tot22 * 100.0, 1)

    # GLDAS-2.1 4 layers + total 0-200 cm + Qsb
    g21_layers = [
        ('GLDAS-2.1 SM 0–10 cm (Open-loop)', 'anom_G21_SM_0_10', 10.0),
        ('GLDAS-2.1 SM 10–40 cm (Open-loop)', 'anom_G21_SM_10_40', 30.0),
        ('GLDAS-2.1 SM 40–100 cm (Open-loop)', 'anom_G21_SM_40_100', 60.0),
        ('GLDAS-2.1 SM 100–200 cm (Open-loop)', 'anom_G21_SM_100_200', 100.0),
    ]
    sm_sl = []
    for lbl, col, thick in g21_layers:
        sl, _, t_v, p_v, lo, hi, df_v = ols_hac(post_g['t_yr'].values, post_g[col].values, maxlags=4)
        sm_sl.append((lbl, thick, sl, t_v, df_v, p_v))
    tot21_sm = sum(x[2] for x in sm_sl)
    for lbl, thick, sl, t_v, df_v, p_v in sm_sl:
        rows.append({'Model_Layer': lbl, 'Thickness_cm': thick, 'Slope_mma_05_09': round(sl, 3),
                     'Per_cm_mma_cm': round(sl / thick, 4), 'Share_pct': round(sl / tot21_sm * 100.0, 1),
                     't': round(t_v, 2), 'df': df_v, 'p_HAC': round(p_v, 5)})
    sl_tot, _, t_tot, p_tot, _, _, df_tot = ols_hac(post_g['t_yr'].values, post_g['anom_G21_SM_tot'].values, maxlags=4)
    rows.append({'Model_Layer': 'GLDAS-2.1 SM Total 0–200 cm', 'Thickness_cm': 200.0, 'Slope_mma_05_09': round(sl_tot, 3),
                 'Per_cm_mma_cm': round(sl_tot / 200.0, 4), 'Share_pct': 100.0, 't': round(t_tot, 2), 'df': df_tot, 'p_HAC': round(p_tot, 5)})
    # Qsb in mm/month/a -> also mm/a^2 (*12)
    sl_q, _, t_q, p_q, _, _, df_q = ols_hac(post_g['t_yr'].values, post_g['anom_G21_Qsb'].values * 12.0, maxlags=4)
    rows.append({'Model_Layer': 'GLDAS-2.1 Qsb Baseflow (mm a⁻²)', 'Thickness_cm': np.nan, 'Slope_mma_05_09': round(sl_q, 3),
                 'Per_cm_mma_cm': np.nan, 'Share_pct': np.nan, 't': round(t_q, 2), 'df': df_q, 'p_HAC': round(p_q, 5)})
    sub_df = pd.DataFrame(rows)
    sub_df.to_csv(os.path.join(OUT_DIR, 'jms_subsurface_normalized.csv'), index=False)
    print(sub_df.to_string(index=False))
    return sub_df


# ---------------------------------------------------------------------------
# 4. Live GEE Multi-Sensor Elevation Validation + Snow-Masked NDVI + Thermal Permafrost Mask
# ---------------------------------------------------------------------------
def gee_extract_cross_sensor_belts():
    print('\n=== [4] Live GEE Multi-Sensor Temperature Validation, Snow-Free NDVI & Thermal Permafrost ===')
    retry_gee(lambda: ee.Initialize(project='round-fusion-508306-p4'))
    with open(GEO_PATH) as f:
        poly = ee.Geometry(json.load(f))

    dem = ee.Image('MERIT/DEM/v1_0_3').select('dem')
    era5_ic = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
    tc_ic = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
    lst_ic = ee.ImageCollection('MODIS/061/MOD11A2')
    mod13_ic = ee.ImageCollection('MODIS/061/MOD13A1')
    pml_ic = ee.ImageCollection('projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a')
    gpm_ic = ee.ImageCollection('NASA/GPM_L3/IMERG_MONTHLY_V07')

    # Zone scheme A: 3 MERIT DEM Belts (1: <3800, 2: 3800-4300, 3: >=4300)
    # Zone scheme B: Thermally Defined Permafrost (MA ST4_clim < 0 C => 4) vs Seasonally Frozen (ST4_clim >= 0 C => 5)
    zone_elev = ee.Image(0).where(dem.lt(3800), 1).where(dem.gte(3800).And(dem.lt(4300)), 2).where(dem.gte(4300), 3)
    st4_clim = era5_ic.filterDate('2004-01-01', '2025-01-01').select('soil_temperature_level_4').mean().subtract(273.15).rename('st4_clim')
    zone_perm = ee.Image(5).where(st4_clim.lt(0), 4)
    # Single joint code: 10*elev + perm in {14, 15, 24, 25, 34, 35} -> ONE reduceRegion call per year!
    zone_combo = zone_elev.multiply(10).add(zone_perm).rename('zone')

    # Save exact ST4_clim 0.05-deg grid for Fig 1a permafrost isotherm overlay
    st4_grid_path = os.path.join(OUT_DIR, 'jms_st4_clim_grid.npz')
    if not os.path.exists(st4_grid_path):
        roi_rect = ee.Geometry.Rectangle([95.5, 32.0, 103.5, 36.5])
        samp = retry_gee(lambda: st4_clim.reproject(crs='EPSG:4326', scale=5000).sampleRectangle(region=roi_rect, defaultValue=-999.0).getInfo())
        arr_st4 = np.array(samp['properties']['st4_clim'], dtype=float)
        arr_st4[arr_st4 <= -900] = np.nan
        np.savez_compressed(st4_grid_path, st4_clim=arr_st4, extent=np.array([95.5, 103.5, 32.0, 36.5]))
        print(f'[Saved ST4_clim grid {arr_st4.shape} to {st4_grid_path}]')

    area_img = ee.Image.pixelArea().divide(1e6).rename('area_km2')
    area_combo_grp = retry_gee(lambda: area_img.addBands(zone_combo).reduceRegion(
        reducer=ee.Reducer.sum().group(1, 'zone'), geometry=poly, scale=2000, maxPixels=1e9
    ).getInfo())['groups']
    combo_areas = {int(g['zone']): float(g['sum']) for g in area_combo_grp if int(g['zone']) in (14, 15, 24, 25, 34, 35)}
    area_dict = {
        1: combo_areas.get(14, 0.0) + combo_areas.get(15, 0.0),
        2: combo_areas.get(24, 0.0) + combo_areas.get(25, 0.0),
        3: combo_areas.get(34, 0.0) + combo_areas.get(35, 0.0),
        4: combo_areas.get(14, 0.0) + combo_areas.get(24, 0.0) + combo_areas.get(34, 0.0),
        5: combo_areas.get(15, 0.0) + combo_areas.get(25, 0.0) + combo_areas.get(35, 0.0),
    }
    print('[Combo & Marginal Areas km2 (2-km mask)]:', {k: round(v, 1) for k, v in sorted(area_dict.items())})

    def prep_lst(im):
        qc_d = im.select('QC_Day').bitwiseAnd(3).lte(1)
        qc_n = im.select('QC_Night').bitwiseAnd(3).lte(1)
        d_c = im.select('LST_Day_1km').multiply(0.02).subtract(273.15).updateMask(qc_d)
        n_c = im.select('LST_Night_1km').multiply(0.02).subtract(273.15).updateMask(qc_n)
        return d_c.add(n_c).multiply(0.5).rename('MOD_LST')

    def prep_ndvi_nosnow(im):
        qa_ok = im.select('SummaryQA').lte(1)
        return im.select('NDVI').multiply(0.0001).updateMask(qa_ok).rename('NDVI_QA_NoSnow')

    def prep_snow_frac(im):
        return im.select('SummaryQA').eq(2).multiply(100.0).rename('MOD_Snow_Pct')

    band_names = [
        'ERA5_T2m', 'ERA5_ST4', 'ERA5_SnowPct', 'TC_Tmean', 'MOD_LST',
        'NDVI_GS_All', 'NDVI_GS_NoSnow', 'NDVI_JulAug', 'MOD_Snow_GS_Pct', 'MOD_Snow_MaySep_Pct',
        'PML_Ec', 'PML_Es', 'PML_Ei', 'PML_ET', 'ERA5_ET', 'TC_ET', 'GPM_P'
    ]

    cache_combo_path = os.path.join(OUT_DIR, '_cache_gee_combo_2004_2024.csv')
    if os.path.exists(cache_combo_path):
        cached = pd.read_csv(cache_combo_path)
        combo_recs = cached.to_dict('records')
        done_yrs = set(cached['year'].unique())
    else:
        combo_recs, done_yrs = [], set()

    for yr in range(2004, 2025):
        if yr in done_yrs:
            continue
        s_dt, e_dt = f'{yr}-01-01', f'{yr+1}-01-01'
        days_yr = 366.0 if calendar.isleap(yr) else 365.0

        era = era5_ic.filterDate(s_dt, e_dt)
        era_t = era.select(['temperature_2m', 'soil_temperature_level_4']).mean().subtract(273.15).rename(['ERA5_T2m', 'ERA5_ST4'])
        era_snow = era.filter(ee.Filter.calendarRange(5, 9, 'month')).select('snow_cover').mean().multiply(100.0).rename('ERA5_SnowPct')
        era_et = era.select('total_evaporation_sum').sum().multiply(-1000.0).rename('ERA5_ET')

        tc_yr = tc_ic.filterDate(s_dt, e_dt)
        tc_t = tc_yr.select(['tmmn', 'tmmx']).mean().multiply(0.1).reduce(ee.Reducer.mean()).rename('TC_Tmean')
        tc_et = tc_yr.select('aet').sum().multiply(0.1).rename('TC_ET')

        mod_lst = lst_ic.filterDate(s_dt, e_dt).map(prep_lst).mean().rename('MOD_LST')

        m13_gs = mod13_ic.filterDate(s_dt, e_dt).filter(ee.Filter.calendarRange(5, 9, 'month'))
        ndvi_all = m13_gs.select('NDVI').mean().multiply(0.0001).rename('NDVI_GS_All')
        ndvi_ns = m13_gs.map(prep_ndvi_nosnow).mean().rename('NDVI_GS_NoSnow')
        ndvi_ja = mod13_ic.filterDate(s_dt, e_dt).filter(ee.Filter.calendarRange(7, 8, 'month')).map(prep_ndvi_nosnow).mean().rename('NDVI_JulAug')
        snow_gs = m13_gs.map(prep_snow_frac).mean().rename('MOD_Snow_GS_Pct')
        snow_sh = mod13_ic.filterDate(s_dt, e_dt).filter(
            ee.Filter.Or(ee.Filter.calendarRange(5, 5, 'month'), ee.Filter.calendarRange(9, 9, 'month'))
        ).map(prep_snow_frac).mean().rename('MOD_Snow_MaySep_Pct')

        pml = pml_ic.filterDate(s_dt, e_dt).select(['Ec', 'Es', 'Ei', 'ET']).mean().multiply(0.01 * days_yr).unmask(0).rename(['PML_Ec', 'PML_Es', 'PML_Ei', 'PML_ET'])
        gpm = gpm_ic.filterDate(s_dt, e_dt).select('precipitation').mean().multiply(24.0 * days_yr).rename('GPM_P')

        img_base = ee.Image.cat([era_t, era_snow, tc_t, mod_lst, ndvi_all, ndvi_ns, ndvi_ja, snow_gs, snow_sh, pml, era_et, tc_et, gpm])
        grp = retry_gee(lambda im=img_base: im.addBands(zone_combo).reduceRegion(
            reducer=ee.Reducer.mean().repeat(len(band_names)).group(groupField=len(band_names), groupName='zone'),
            geometry=poly, scale=2000, maxPixels=1e9
        ).getInfo())['groups']
        for g in grp:
            z_c = int(g['zone'])
            if z_c not in combo_areas:
                continue
            rec = {'year': yr, 'combo_zone': z_c, 'area_km2': combo_areas[z_c]}
            for bi, bname in enumerate(band_names):
                val = g['mean'][bi]
                rec[bname] = float(val) if val is not None else np.nan
            combo_recs.append(rec)
        pd.DataFrame(combo_recs).to_csv(cache_combo_path, index=False)
        print(f'  GEE cross-sensor year {yr} cached')

    cdf = pd.DataFrame(combo_recs)
    # Recombine area-weighted marginal zone means for zones 1, 2, 3 (elev) and 4, 5 (thermal permafrost)
    records = []
    for yr in sorted(cdf['year'].unique()):
        sub_y = cdf[cdf['year'] == yr]
        for zid, zlbl, mask_fn in [
            (1, '<3800m', sub_y['combo_zone'] // 10 == 1),
            (2, '3800-4300m', sub_y['combo_zone'] // 10 == 2),
            (3, '>=4300m', sub_y['combo_zone'] // 10 == 3),
            (4, 'Permafrost (ST4<0C)', sub_y['combo_zone'] % 10 == 4),
            (5, 'Seasonally_Frozen (ST4>=0C)', sub_y['combo_zone'] % 10 == 5),
        ]:
            part = sub_y[mask_fn]
            w = part['area_km2'].values
            rec = {'year': int(yr), 'zone_id': zid, 'zone_label': zlbl, 'area_km2': round(float(w.sum()), 1)}
            for bname in band_names:
                vals = part[bname].values
                ok = np.isfinite(vals)
                rec[bname] = float(np.sum(vals[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else np.nan
            records.append(rec)

    ts_df = pd.DataFrame(records).sort_values(['zone_id', 'year']).reset_index(drop=True)
    ts_df.to_csv(os.path.join(OUT_DIR, 'jms_cross_sensor_annual_2004_2024.csv'), index=False)

    tot_elev_a = sum(area_dict[z] for z in (1, 2, 3))
    tot_perm_a = sum(area_dict[z] for z in (4, 5))
    sum_rows = []
    for zid in [1, 2, 3, 4, 5]:
        sub = ts_df[ts_df['zone_id'] == zid].sort_values('year')
        zlbl = sub['zone_label'].iloc[0]
        akm = area_dict[zid]
        tot_ref = tot_elev_a if zid <= 3 else tot_perm_a
        early = sub[sub['year'] <= 2008]
        late = sub[sub['year'] >= 2020]
        for var in band_names:
            sl, se, t_v, p_hac, lo, hi, df_v = ols_hac(sub['year'].values, sub[var].values, maxlags=3)
            i_idx, j_idx = np.triu_indices(len(sub), k=1)
            yv = sub[var].values
            xv = sub['year'].values
            sen = float(np.median((yv[j_idx] - yv[i_idx]) / (xv[j_idx] - xv[i_idx]))) if np.all(np.isfinite(yv)) else np.nan
            z_mmk, p_mmk, n_ns = hamed_rao_mmk(yv)
            e_m, l_m = float(early[var].mean()), float(late[var].mean())
            is_temp = var in ('ERA5_T2m', 'ERA5_ST4', 'TC_Tmean', 'MOD_LST')
            rel_pct = np.nan if is_temp else ((l_m - e_m) / abs(e_m) * 100.0 if abs(e_m) > 1e-6 else np.nan)
            ndp = 4 if 'NDVI' in var else 2
            sum_rows.append({
                'Zone': zlbl, 'Area_km2': round(akm, 0), 'Area_pct': round(akm / tot_ref * 100.0, 1),
                'Variable': var,
                'Mean_2004_2008': round(e_m, ndp), 'Mean_2020_2024': round(l_m, ndp),
                'Abs_Change': round(l_m - e_m, ndp),
                'Rel_Change_pct': round(rel_pct, 2) if np.isfinite(rel_pct) else np.nan,
                'HAC_dec': round(sl * 10.0, ndp), 'HAC_SE_dec': round(se * 10.0, ndp),
                'HAC_t': round(t_v, 2), 'df': df_v, 'HAC_p': round(p_hac, 5),
                'Sen_dec': round(sen * 10.0, ndp) if np.isfinite(sen) else np.nan,
                'MMK_Z': round(z_mmk, 2), 'MMK_p': round(p_mmk, 5),
            })
    sum_df = pd.DataFrame(sum_rows)
    sum_df.to_csv(os.path.join(OUT_DIR, 'jms_table4_reconciled.csv'), index=False)
    print('\n=== [Cross-Sensor Temperature & Snow-Free NDVI Comparison Across Elevation Belts] ===')
    show_vars = ['ERA5_T2m', 'TC_Tmean', 'MOD_LST', 'ERA5_ST4', 'NDVI_GS_All', 'NDVI_GS_NoSnow', 'NDVI_JulAug', 'MOD_Snow_MaySep_Pct', 'PML_Ec', 'PML_Es', 'ERA5_ET', 'GPM_P']
    print(sum_df[sum_df['Variable'].isin(show_vars) & sum_df['Zone'].isin(['<3800m', '3800-4300m', '>=4300m'])][['Zone', 'Variable', 'Mean_2004_2008', 'Mean_2020_2024', 'Abs_Change', 'Rel_Change_pct', 'HAC_dec', 'HAC_t', 'HAC_p', 'MMK_p']].to_string(index=False))
    print('\n=== [Thermal Permafrost vs Seasonally Frozen Comparison] ===')
    print(sum_df[sum_df['Zone'].isin(['Permafrost (ST4<0C)', 'Seasonally_Frozen (ST4>=0C)']) & sum_df['Variable'].isin(['ERA5_T2m', 'TC_Tmean', 'ERA5_ST4', 'NDVI_GS_NoSnow', 'PML_Ec', 'PML_Es', 'GPM_P'])][['Zone', 'Area_pct', 'Variable', 'Mean_2004_2008', 'Mean_2020_2024', 'Abs_Change', 'HAC_dec', 'HAC_p', 'MMK_p']].to_string(index=False))
    return ts_df, sum_df


# ---------------------------------------------------------------------------
# 5. Pyper-Peterman N_eff Audit, Non-LAI ET Checks & Leakage-Free Walk-Forward
# ---------------------------------------------------------------------------
def audit_stats_causality_and_walkforward(df):
    print('\n=== [5] Pyper-Peterman N_eff, Non-LAI ET Checks & Strictly Isolated Walk-Forward ===')
    # (A) Table 2 Cross-validation with N_eff & t-stats
    sub_g = df[df['lwe_basin'].notna() & (df['ym'] >= 200302) & (df['ym'] <= 202409)].copy()
    cv_rows = []
    for name, c1, c2, scale, cat in [
        ('GRACE vs GLDAS-2.2 TWSA (deseas.)', 'anom_lwe_basin', 'anom_G22_TWS_tavg', 0.1, 'TWSA'),
        ('GRACE vs GLDAS-2.1 Proxy (deseas.)', 'anom_lwe_basin', 'anom_G21_TWS_proxy', 0.1, 'TWSA'),
        ('GRACE vs TerraClimate Soil (deseas.)', 'anom_lwe_basin', 'anom_TC_SOIL', 0.1, 'TWSA'),
        ('GPM V07 vs TerraClimate P (deseas.)', 'anom_GPM_P', 'anom_TC_P', 1.0, 'Precip.'),
        ('GPM V07 vs ERA5-Land P (deseas.)', 'anom_GPM_P', 'anom_ERA5_P', 1.0, 'Precip.'),
        ('ERA5-Land vs PML-V2.2a ET (deseas.)', 'anom_ERA5_ET', 'anom_PML_ET_tot', 1.0, 'ET'),
        ('ERA5-Land vs TerraClimate ET (deseas.)', 'anom_ERA5_ET', 'anom_TC_ET', 1.0, 'ET'),
        ('ERA5-Land vs GLDAS-2.1 ET (deseas.)', 'anom_ERA5_ET', 'anom_G21_ET', 1.0, 'ET'),
        ('ERA5-Land vs MOD16A2GF ET (deseas.)', 'anom_ERA5_ET', 'anom_MOD16_ET', 1.0, 'ET'),
        ('MODIS NDVI vs EVI (deseas.)', 'anom_NDVI', 'anom_EVI', 1.0, 'Veg.'),
        ('MODIS NDVI vs LAI (deseas.)', 'anom_NDVI', 'anom_LAI', 1.0, 'Veg.'),
    ]:
        s_pair = sub_g if cat == 'TWSA' else df
        x, y = s_pair[c1].values, s_pair[c2].values * scale
        r, n, t_iid, p_iid, neff, t_eff, p_eff = pearson_with_neff(x, y)
        rmse = float(np.sqrt(np.mean((x - y) ** 2)))
        cv_rows.append({
            'Category': cat, 'Pair': name, 'Deseas_r': round(r, 4), 'RMSE': round(rmse, 3),
            'N': n, 't_iid': round(t_iid, 2), 'p_iid': p_iid,
            'N_eff_PP': round(neff, 1), 't_eff': round(t_eff, 2), 'p_eff_PP': p_eff,
        })
    t2_neff_df = pd.DataFrame(cv_rows)
    t2_neff_df.to_csv(os.path.join(OUT_DIR, 'jms_table2_neff.csv'), index=False)
    print(t2_neff_df[['Pair', 'Deseas_r', 'N', 'p_iid', 'N_eff_PP', 't_eff', 'p_eff_PP']].to_string(index=False))

    # (B) Rolling Memory Windows k=1..12 with Pyper-Peterman N_eff + Non-overlapping annual test
    mem_rows = []
    for k in [1, 2, 3, 4, 6, 9, 12]:
        tmp = df[df['ym'] <= 202409].copy()
        tmp['P_k'] = tmp['anom_GPM_P'].rolling(k).mean()
        tmp['NDVI_k'] = tmp['anom_NDVI'].rolling(k).mean()
        tmp['T_k'] = tmp['anom_ERA5_T2m'].rolling(k).mean()
        sub = tmp.dropna(subset=['anom_lwe_basin', 'P_k', 'NDVI_k', 'T_k'])
        r_p, n_p, _, pp_iid, neff_p, tp_eff, pp_eff = pearson_with_neff(sub['P_k'].values, sub['anom_lwe_basin'].values)
        r_part, n_c, df_iid, t_iid, p_iid, neff_c, df_eff, t_eff, p_eff = partial_corr_with_neff(
            sub['NDVI_k'].values, sub['anom_lwe_basin'].values, sub[['P_k', 'anom_ERA5_T2m']].values
        )
        mem_rows.append({
            'Window_k_mo': k, 'r_P_TWSA': round(r_p, 4), 'N_eff_P': round(neff_p, 1), 'p_P_eff': pp_eff,
            'partial_r_NDVI_TWSA': round(r_part, 4), 'N': n_c, 'df_iid': df_iid, 'p_iid': round(p_iid, 4),
            'N_eff_PP': round(neff_c, 1), 'df_eff': round(df_eff, 1), 't_eff': round(t_eff, 2), 'p_eff_PP': round(p_eff, 4),
        })
    # Non-overlapping annual check (complete years 2004-2023 [n=20])
    ann_df = df[(df['year'] >= 2004) & (df['year'] <= 2023)].groupby('year').agg(
        twsa=('anom_lwe_basin', 'mean'), p_ann=('GPM_P', 'sum'), t_ann=('ERA5_T2m', 'mean'),
        ndvi_gs=('NDVI', lambda s: float(s.iloc[4:9].mean()))
    ).dropna().reset_index()
    r_ann, n_a, df_a, t_a, p_a, neff_a, df_ea, t_ea, p_ea = partial_corr_with_neff(
        ann_df['ndvi_gs'].values, ann_df['twsa'].values, ann_df[['p_ann', 't_ann']].values
    )
    mem_rows.append({
        'Window_k_mo': 999, 'r_P_TWSA': round(stats.pearsonr(ann_df['p_ann'], ann_df['twsa'])[0], 4),
        'N_eff_P': float(n_a), 'p_P_eff': stats.pearsonr(ann_df['p_ann'], ann_df['twsa'])[1],
        'partial_r_NDVI_TWSA': round(r_ann, 4), 'N': n_a, 'df_iid': df_a, 'p_iid': round(p_a, 4),
        'N_eff_PP': round(neff_a, 1), 'df_eff': round(df_ea, 1), 't_eff': round(t_ea, 2), 'p_eff_PP': round(p_ea, 4),
    })
    mem_df = pd.DataFrame(mem_rows)
    mem_df.to_csv(os.path.join(OUT_DIR, 'jms_neff_memory.csv'), index=False)
    print('\n=== [Memory Window Sensitivity k=1..12 & Annual Non-Overlapping (999) with N_eff] ===')
    print(mem_df[['Window_k_mo', 'r_P_TWSA', 'partial_r_NDVI_TWSA', 'N', 'p_iid', 'N_eff_PP', 't_eff', 'p_eff_PP']].to_string(index=False))

    # (C) Non-LAI ET Partial Correlations vs PML-V2.2a (recomputing monthly anomalies strictly, also with N_eff!)
    et_rows = []
    for resp, grp_type in [
        ('anom_PML_Ei', 'PML-V2.2a (MODIS-LAI prescribed)'),
        ('anom_PML_Ec', 'PML-V2.2a (MODIS-LAI prescribed)'),
        ('anom_PML_Es', 'PML-V2.2a (MODIS-LAI prescribed)'),
        ('anom_PML_ET_tot', 'PML-V2.2a (MODIS-LAI prescribed)'),
        ('anom_ERA5_ET', 'Independent Non-LAI (ERA5-Land)'),
        ('anom_TC_ET', 'Independent Non-LAI (TerraClimate)'),
        ('anom_G21_ET', 'Independent Non-LAI (GLDAS-2.1)'),
    ]:
        for lag in [0, 1, 2, 3]:
            vx = df['anom_NDVI'].shift(lag)
            vy = df[resp]
            Z = np.column_stack([df['anom_GPM_P'].values, df['anom_ERA5_T2m'].values]) if lag == 0 else np.column_stack(
                [df['anom_GPM_P'].values, df['anom_ERA5_T2m'].values, df['anom_GPM_P'].shift(lag).fillna(0).values]
            )
            r, n, df_i, t_i, p_i, neff, df_e, t_e, p_e = partial_corr_with_neff(vx, vy, Z)
            et_rows.append({'Target': resp, 'Class': grp_type, 'Lag': lag, 'Partial_r': round(r, 4),
                            'p_iid': p_i, 'N_eff': round(neff, 1), 't_eff': round(t_e, 2), 'p_eff_PP': p_e})
    et_df = pd.DataFrame(et_rows)
    et_df.to_csv(os.path.join(OUT_DIR, 'jms_et_nonlai_check.csv'), index=False)
    print('\n=== [NDVI vs Coupled PML vs Independent Non-LAI ET Partitioning (with N_eff)] ===')
    print(et_df[et_df['Lag'].isin([0, 1])][['Target', 'Class', 'Lag', 'Partial_r', 'N_eff', 'p_iid', 'p_eff_PP']].to_string(index=False))

    # (D) Strictly Leakage-Free Walk-Forward (in-fold seasonal climatology!) & Matched N_test across ALL 4 Models & Tiers
    raw_base = df[(df['ym'] >= 200302) & (df['ym'] <= 202409)].copy().sort_values('ym_dt').reset_index(drop=True)
    # Prepare raw covariates before fold-wise deseasonalization
    raw_cols = ['lwe_basin', 'GPM_P', 'ERA5_T2m', 'PML_ET_tot', 'NDVI', 'PML_Ec']
    # Build lag/rolling features per test_yr using ONLY historical climatology (< test_yr)!
    model_defs = {
        'Ridge': lambda: Ridge(alpha=10.0),
        'Lasso': lambda: Lasso(alpha=0.05, max_iter=5000),
        'RandomForest': lambda: RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=5, random_state=42),
        'XGBoost': lambda: xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, random_state=42),
    }
    exo6_names = ['anom_GPM_P', 'P_cum3', 'P_cum6', 'anom_ERA5_T2m', 'T2m_cum3',
                  'anom_PML_ET_tot', 'ET_cum3', 'anom_NDVI', 'NDVI_lag1', 'NDVI_lag2', 'NDVI_lag3', 'anom_PML_Ec', 'Ec_cum3']
    exo24_names = exo6_names + ['P_cum12', 'P_cum24']
    ar_names = exo24_names + ['lwe_lag1']

    def build_fold_features(sub_slice, clim_df):
        w = sub_slice.copy()
        for c in raw_cols:
            w[f'anom_{c}'] = w[c] - w['month'].map(clim_df[c])
        w['P_cum3'] = w['anom_GPM_P'].rolling(3).sum()
        w['P_cum6'] = w['anom_GPM_P'].rolling(6).sum()
        w['P_cum12'] = w['anom_GPM_P'].rolling(12).sum()
        w['P_cum24'] = w['anom_GPM_P'].rolling(24).sum()
        w['ET_cum3'] = w['anom_PML_ET_tot'].rolling(3).sum()
        w['T2m_cum3'] = w['anom_ERA5_T2m'].rolling(3).mean()
        w['NDVI_lag1'] = w['anom_NDVI'].shift(1)
        w['NDVI_lag2'] = w['anom_NDVI'].shift(2)
        w['NDVI_lag3'] = w['anom_NDVI'].shift(3)
        w['Ec_cum3'] = w['anom_PML_Ec'].rolling(3).sum()
        w['lwe_lag1'] = w['anom_lwe_basin'].shift(1)
        return w

    # First determine matched common evaluation index (valid lwe_lag1 AND 24-mo features for year >= 2014)
    wf_store = {tier: {m: {'y': [], 'yh': [], 'dt': [], 'yr': []} for m in model_defs}
                for tier in ['Exogenous-Only (0–6 mo)', 'Exogenous-Long (0–24 mo)', 'Exogenous-Long + AR(1)']}
    pers_y, pers_yh = [], []

    last_fold_rf_shap = None
    last_fold_X = None

    for test_yr in range(2014, 2025):
        # Historical slice up to end of test_yr; climatology strictly computed on year < test_yr!
        tr_raw = raw_base[raw_base['year'] < test_yr]
        clim_tbl = tr_raw.groupby('month')[raw_cols].mean()
        fold_all = build_fold_features(raw_base[raw_base['year'] <= test_yr], clim_tbl)
        # Strict common valid filter across ALL tiers (including lwe_lag1 & P_cum24) so N_test matches!
        fold_valid = fold_all.dropna(subset=['anom_lwe_basin'] + ar_names).copy()
        tr = fold_valid[fold_valid['year'] < test_yr]
        te = fold_valid[fold_valid['year'] == test_yr]
        if len(te) == 0 or len(tr) < 40:
            continue
        pers_y.extend(te['anom_lwe_basin'].values)
        pers_yh.extend(te['lwe_lag1'].values)

        for tier, fcols in [
            ('Exogenous-Only (0–6 mo)', exo6_names),
            ('Exogenous-Long (0–24 mo)', exo24_names),
            ('Exogenous-Long + AR(1)', ar_names),
        ]:
            sc = StandardScaler()
            Xtr = sc.fit_transform(tr[fcols].values)
            Xte = sc.transform(te[fcols].values)
            for mname, mfn in model_defs.items():
                mod = mfn()
                mod.fit(Xtr, tr['anom_lwe_basin'].values)
                yh = mod.predict(Xte)
                wf_store[tier][mname]['y'].extend(te['anom_lwe_basin'].values)
                wf_store[tier][mname]['yh'].extend(yh)
                wf_store[tier][mname]['dt'].extend(te['ym_dt'].values)
                wf_store[tier][mname]['yr'].extend(te['year'].values)
                if test_yr == 2024 and tier == 'Exogenous-Long (0–24 mo)' and mname == 'RandomForest':
                    # TreeSHAP strictly on final training fold (<2024) evaluated on in-fold training+OOS covariates
                    exp = shap.TreeExplainer(mod)
                    sv = exp.shap_values(Xtr)
                    last_fold_rf_shap = (fcols, sv, tr[fcols].reset_index(drop=True))

    wf_rows = []
    for tier in ['Exogenous-Only (0–6 mo)', 'Exogenous-Long (0–24 mo)', 'Exogenous-Long + AR(1)']:
        for mname in model_defs:
            y = np.array(wf_store[tier][mname]['y'])
            yh = np.array(wf_store[tier][mname]['yh'])
            r, _ = stats.pearsonr(y, yh)
            wf_rows.append({
                'FeatureSet': tier, 'Model': mname, 'N_test': len(y),
                'RMSE_cm': round(float(np.sqrt(mean_squared_error(y, yh))), 3),
                'MAE_cm': round(float(mean_absolute_error(y, yh)), 3),
                'OOS_R2': round(float(r2_score(y, yh)), 3),
                'r': round(float(r), 3),
            })
    r_p, _ = stats.pearsonr(pers_y, pers_yh)
    wf_rows.append({
        'FeatureSet': 'Persistence Baseline', 'Model': 'Persistence (t − 1)', 'N_test': len(pers_y),
        'RMSE_cm': round(float(np.sqrt(mean_squared_error(pers_y, pers_yh))), 3),
        'MAE_cm': round(float(mean_absolute_error(pers_y, pers_yh)), 3),
        'OOS_R2': round(float(r2_score(pers_y, pers_yh)), 3),
        'r': round(float(r_p), 3),
    })
    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(os.path.join(OUT_DIR, 'jms_walkforward_clean.csv'), index=False)
    print('\n=== [Zero-Leakage Matched Walk-Forward Results (2014–2024, In-Fold Climatology)] ===')
    print(wf_df.to_string(index=False))

    # Save matched OOS predictions for Fig 5a (Exogenous-Only RF, Exogenous-Long RF, Exogenous+AR(1) Lasso)
    oos_df = pd.DataFrame({
        'ym_dt': wf_store['Exogenous-Long (0–24 mo)']['RandomForest']['dt'],
        'Obs_TWSA': wf_store['Exogenous-Long (0–24 mo)']['RandomForest']['y'],
        'Exo6_RF': wf_store['Exogenous-Only (0–6 mo)']['RandomForest']['yh'],
        'Exo24_RF': wf_store['Exogenous-Long (0–24 mo)']['RandomForest']['yh'],
        'ExoAR1_Lasso': wf_store['Exogenous-Long + AR(1)']['Lasso']['yh'],
    })
    oos_df.to_csv(os.path.join(OUT_DIR, 'jms_walkforward_oos_preds.csv'), index=False)

    # Save exact TreeSHAP from RandomForest (train < 2024, Exogenous-Long 0–24 mo)
    fcols, sv, Xtr_df = last_fold_rf_shap
    mean_abs = np.mean(np.abs(sv), axis=0)
    dirs = [float(np.corrcoef(Xtr_df[f].values, sv[:, i])[0, 1]) for i, f in enumerate(fcols)]
    shap_clean = pd.DataFrame({
        'feature': fcols, 'mean_abs_shap': mean_abs,
        'share_pct': mean_abs / np.sum(mean_abs) * 100.0,
        'dir_corr': dirs,
    }).sort_values('share_pct', ascending=False).reset_index(drop=True)
    shap_clean.to_csv(os.path.join(OUT_DIR, 'jms_shap_clean.csv'), index=False)
    print('\n=== [Clean RandomForest TreeSHAP Shares (Trained strictly on 2003–2023, Exogenous-Long 0–24 mo)] ===')
    print(shap_clean.to_string(index=False))
    return t2_neff_df, mem_df, et_df, wf_df, shap_clean


# ---------------------------------------------------------------------------
# 6. Non-Steady-State Budyko (P_eff = P - dS) & Propagated Uncertainty
# ---------------------------------------------------------------------------
def solve_budyko_w(ep, pet_p):
    if not (np.isfinite(ep) and np.isfinite(pet_p)) or ep <= 0 or ep >= 1.0 or pet_p <= 0:
        return np.nan
    try:
        return float(brentq(lambda w: (1.0 + pet_p ** (-w)) ** (-1.0 / w) - ep, 0.05, 20.0))
    except Exception:
        return np.nan


def audit_nonsteady_budyko(df):
    print('\n=== [6] Non-Steady-State Budyko (P_eff = P - dS) & Propagated Uncertainty (Table 6 Fixed) ===')
    rows = []
    for yr in range(2016, 2025):
        sub = df[df['year'] == yr]
        r_e8 = TANGNAIHAI_E8M3[yr]
        r_mm = r_e8 * 100000.0 / AREA_KM2
        p_gpm = float(sub['GPM_P'].sum())
        p_tc = float(sub['TC_P'].sum())
        # Note: reconcile exact PML_ET_tot vs component sum cleanly!
        e_pml = float(sub['PML_ET_tot'].sum())
        e_tc = float(sub['TC_ET'].sum())
        e_era5 = float(sub['ERA5_ET'].sum())
        e_g21 = float(sub['G21_ET'].sum())
        e_mod16 = float(sub['MOD16_ET'].sum())
        pet_pml = float(sub['PML_PET'].sum())
        ndvi_gs = float(sub[sub['month'].isin([5, 6, 7, 8, 9])]['NDVI'].mean())

        cur_dec = df[(df['year'] == yr) & (df['month'] == 12)]
        prev_dec = df[(df['year'] == yr - 1) & (df['month'] == 12)]
        ds_g22 = float(cur_dec['G22_TWS_tavg'].values[0] - prev_dec['G22_TWS_tavg'].values[0])

        # Propagated uncertainty sigma(E_wb) combining P spread (|P_GPM - P_TC|/sqrt(2)) & typical annual GRACE/G22 dS unc (~18 mm/yr)
        sig_p = abs(p_gpm - p_tc)
        sig_ds = 18.0
        sig_ewb = float(np.sqrt(sig_p ** 2 + sig_ds ** 2))

        e_wb = p_gpm - r_mm - ds_g22
        p_eff = p_gpm - ds_g22  # Effective water supply (Chen et al. 2013; Moussa & Lhomme 2016)
        ep_eff_wb = e_wb / p_eff
        ep_eff_pml = e_pml / p_eff
        pet_peff = pet_pml / p_eff
        w_wb_nonsteady = solve_budyko_w(ep_eff_wb, pet_peff)
        w_pml_nonsteady = solve_budyko_w(ep_eff_pml, pet_peff)
        w_pml_steady = solve_budyko_w(e_pml / p_gpm, pet_pml / p_gpm)

        regime = 'Dry' if r_mm < 155 else ('Wet' if r_mm > 220 else 'Normal')
        rows.append({
            'Year': yr, 'Regime': regime,
            'P_GPM_mm': round(p_gpm, 1), 'R_obs_mm': round(r_mm, 1), 'R_km3': round(r_e8 * 0.1, 2),
            'dS_G22_mm': round(ds_g22, 1), 'P_eff_mm': round(p_eff, 1),
            'E_wb_mm': round(e_wb, 1), 'E_wb_unc_mm': round(sig_ewb, 1),
            'E_TC_mm': round(e_tc, 1), 'E_PML_mm': round(e_pml, 1), 'E_ERA5_mm': round(e_era5, 1),
            'R_over_P': round(r_mm / p_gpm, 3),
            'Ewb_over_P': round(e_wb / p_gpm, 3), 'Ewb_over_Peff': round(ep_eff_wb, 3),
            'Epml_over_P': round(e_pml / p_gpm, 3),
            'PET_over_Peff': round(pet_peff, 3),
            'w_WB_nonsteady': round(w_wb_nonsteady, 3) if np.isfinite(w_wb_nonsteady) else np.nan,
            'w_PML_nonsteady': round(w_pml_nonsteady, 3) if np.isfinite(w_pml_nonsteady) else np.nan,
            'w_PML_steady': round(w_pml_steady, 3) if np.isfinite(w_pml_steady) else np.nan,
            'NDVI_GS': round(ndvi_gs, 4),
        })
    t6_df = pd.DataFrame(rows)
    t6_df.to_csv(os.path.join(OUT_DIR, 'jms_table6_budyko_nonsteady.csv'), index=False)
    print(t6_df[['Year', 'Regime', 'P_GPM_mm', 'R_obs_mm', 'R_km3', 'dS_G22_mm', 'P_eff_mm', 'E_wb_mm', 'E_wb_unc_mm', 'R_over_P', 'Ewb_over_Peff', 'w_WB_nonsteady', 'w_PML_steady']].to_string(index=False))

    # Full 2004-2024 long-term Budyko benchmark using effective supply P_eff = P_GPM - dS_G22
    dec_g22 = df[df['month'] == 12].set_index('year')['G22_TWS_tavg']
    ann_long = df[df['year'] >= 2004].groupby('year').agg(
        P_GPM=('GPM_P', 'sum'), E_PML=('PML_ET_tot', 'sum'), PET_PML=('PML_PET', 'sum'),
        E_TC=('TC_ET', 'sum'), NDVI_GS=('NDVI', lambda s: float(s.iloc[4:9].mean()))
    ).reset_index()
    ann_long['dS_G22'] = [float(dec_g22.loc[y] - dec_g22.loc[y - 1]) if (y - 1) in dec_g22.index else np.nan for y in ann_long['year']]
    ann_long['P_eff'] = ann_long['P_GPM'] - ann_long['dS_G22']
    ann_long['ep_pml'] = ann_long['E_PML'] / ann_long['P_GPM']
    ann_long['pet_p'] = ann_long['PET_PML'] / ann_long['P_GPM']
    ann_long['ep_pml_eff'] = ann_long['E_PML'] / ann_long['P_eff']
    ann_long['pet_peff'] = ann_long['PET_PML'] / ann_long['P_eff']
    ann_long['w_pml_steady'] = [solve_budyko_w(r.ep_pml, r.pet_p) for _, r in ann_long.iterrows()]
    ann_long['w_pml_nonsteady'] = [solve_budyko_w(r.ep_pml_eff, r.pet_peff) for _, r in ann_long.iterrows()]
    ann_long.to_csv(os.path.join(OUT_DIR, 'jms_budyko_2004_2024_nonsteady.csv'), index=False)

    # Phase summaries (distinguishing Mean-of-Ratios vs Ratio-of-Means explicitly!) & Budyko Decomposition
    phases = []
    for lbl, yrs_l in [('Baseline (2004–2017)', list(range(2004, 2018))),
                       ('Pre-Pluvial Dry (2016–2017)', [2016, 2017]),
                       ('Pluvial Peak (2018–2020)', [2018, 2019, 2020]),
                       ('Drawdown (2021–2024)', [2021, 2022, 2023, 2024]),
                       ('Gauge Record (2016–2024)', list(range(2016, 2025)))]:
        al_s = ann_long[ann_long['year'].isin(yrs_l)]
        t6_s = t6_df[t6_df['Year'].isin(yrs_l)]
        d = {
            'Phase': lbl,
            'P_GPM_mm': round(float(al_s['P_GPM'].mean()), 1),
            'dS_G22_mm': round(float(al_s['dS_G22'].mean()), 1),
            'P_eff_mm': round(float(al_s['P_eff'].mean()), 1),
            'E_PML_mm': round(float(al_s['E_PML'].mean()), 1),
            'PET_PML_mm': round(float(al_s['PET_PML'].mean()), 1),
            'w_PML_nonsteady_mean': round(float(al_s['w_pml_nonsteady'].mean()), 3),
            'w_PML_steady_mean': round(float(al_s['w_pml_steady'].mean()), 3),
        }
        if len(t6_s) == len(yrs_l):
            rm = float(t6_s['R_obs_mm'].mean())
            pm = float(t6_s['P_GPM_mm'].mean())
            peffm = float(t6_s['P_eff_mm'].mean())
            ewbm = float(t6_s['E_wb_mm'].mean())
            petm = float(al_s['PET_PML'].mean())
            d.update({
                'R_obs_mm': round(rm, 1), 'R_km3_yr': round(rm * AREA_KM2 / 1e6, 2),
                'E_wb_mm': round(ewbm, 1),
                'RP_mean_of_ratios': round(float(t6_s['R_over_P'].mean()), 3),
                'RP_ratio_of_means': round(rm / pm, 3),
                'EwbP_mean_of_ratios': round(float(t6_s['Ewb_over_P'].mean()), 3),
                'EwbP_ratio_of_means': round(ewbm / pm, 3),
                'w_WB_nonsteady_mean': round(float(t6_s['w_WB_nonsteady'].mean()), 3),
                'w_WB_nonsteady_flux': round(solve_budyko_w(ewbm / peffm, petm / peffm), 3),
            })
        phases.append(d)
    ph_bud_df = pd.DataFrame(phases)
    ph_bud_df.to_csv(os.path.join(OUT_DIR, 'jms_budyko_phase_summary.csv'), index=False)
    print('\n=== [Budyko Long-Term Phase Benchmarks (Mean-of-Ratios vs Ratio-of-Means)] ===')
    print(ph_bud_df.to_string(index=False))

    # Analytical decomposition of R_obs/P shift from Peak (2018-2020) to Drawdown (2021-2024):
    # Closure: R/P = 1 - E_wb/P - dS/P = (1 - dS/P) * [1 - F(PET/P_eff, w_wb)]
    p1 = ph_bud_df[ph_bud_df['Phase'] == 'Pluvial Peak (2018–2020)'].iloc[0]
    p2 = ph_bud_df[ph_bud_df['Phase'] == 'Drawdown (2021–2024)'].iloc[0]
    def cy_ep(phi, w):
        return phi / ((1.0 + phi ** w) ** (1.0 / w))
    phi1 = p1['PET_PML_mm'] / p1['P_eff_mm']
    phi2 = p2['PET_PML_mm'] / p2['P_eff_mm']
    w1, w2 = p1['w_WB_nonsteady_flux'], p2['w_WB_nonsteady_flux']
    rp1 = p1['RP_ratio_of_means']
    rp2 = p2['RP_ratio_of_means']
    # Step 1: hold dS/P fixed at Peak & w=w1, vary aridity phi1 -> phi2 (climate drying effect on P_eff)
    rp_clim = (1.0 - p1['dS_G22_mm'] / p1['P_GPM_mm']) * (1.0 - cy_ep(phi2, w1))
    # Step 2: update dS/P release effect
    rp_ds = (1.0 - p2['dS_G22_mm'] / p2['P_GPM_mm']) * (1.0 - cy_ep(phi2, w1))
    decomp = pd.DataFrame([{
        'RP_Peak_flux': rp1, 'RP_Drawdown_flux': rp2, 'Total_dRP': round(rp2 - rp1, 4),
        'Climate_Aridity_Effect': round(rp_clim - rp1, 4),
        'Storage_Release_Effect': round(rp_ds - rp_clim, 4),
        'Landscape_wWB_Residual_Effect': round(rp2 - rp_ds, 4),
        'wWB_Peak': w1, 'wWB_Drawdown': w2,
        'Volumetric_Runoff_Drop_km3_yr': round(p2['R_km3_yr'] - p1['R_km3_yr'], 2),
    }])
    decomp.to_csv(os.path.join(OUT_DIR, 'jms_budyko_decomposition.csv'), index=False)
    print('\n=== [Exact Non-Steady-State Budyko Runoff Ratio Decomposition] ===')
    print(decomp.to_string(index=False))
    return t6_df, ph_bud_df, decomp


def main():
    df = pd.read_csv(os.path.join(OUT_DIR, 'basin_merged_monthly_2003_2024.csv'))
    audit_grace_6_caps_and_null(df)
    reconcile_table3_segmented_eq2(df)
    audit_subsurface_depth_normalized(df)
    gee_extract_cross_sensor_belts()
    audit_stats_causality_and_walkforward(df)
    audit_nonsteady_budyko(df)
    print('\n>>> ALL JMS RIGOROUS REANALYSIS AUDITS COMPLETED CLEANLY!')


if __name__ == '__main__':
    main()
