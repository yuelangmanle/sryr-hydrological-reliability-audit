#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
39_jms_breakthrough_probe.py
Three pre-specified probes (no post-hoc fishing; every probe reports its null/negative outcome):
 A. Limiting-factor regime change across the thermally-defined permafrost gradient (pixel panel from GEE).
 B. Precipitation distributed-lag memory kernel for TWSA, calibrated on 2006-2017 only, projected into 2018-2024
    (block-bootstrap CI for the unexplained 'excess storage' residual).
 C. Observed runoff elasticity vs Budyko-implied elasticity at fixed landscape parameter.
"""
import os, json, time, calendar, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.optimize import brentq

OUT = '/Users/yueliangmanle/Desktop/小论文中文2/results_repaired'
GEO = '/Users/yueliangmanle/Desktop/小论文中文2/高级图设计包/fig1_研究区/yr_source_boundary.geojson'
rng = np.random.default_rng(2026)


# ---------------------------------------------------------------- B. memory kernel
def probe_B():
    print('\n=== [B] Precipitation distributed-lag memory kernel (calibrated 2006-2017 only) ===')
    df = pd.read_csv(os.path.join(OUT, 'basin_merged_monthly_2003_2024.csv'))
    df = df[(df['ym'] >= 200302) & (df['ym'] <= 202409)].sort_values('ym').reset_index(drop=True)
    pre_mask = df['ym'] <= 201706
    p_clim = df[pre_mask].groupby('month')['GPM_P'].mean()
    df['P_an'] = df['GPM_P'] - df['month'].map(p_clim)
    y_clim = df[pre_mask & df['lwe_basin'].notna()].groupby('month')['lwe_basin'].mean()
    df['Y'] = df['lwe_basin'] - df['month'].map(y_clim)
    wins = [(0, 2), (3, 5), (6, 11), (12, 17), (18, 23), (24, 35)]
    for a, b in wins:
        df[f'W_{a}_{b}'] = sum(df['P_an'].shift(j) for j in range(a, b + 1))
    feats = [f'W_{a}_{b}' for a, b in wins]
    ok = df[feats].notna().all(axis=1) & df['Y'].notna()
    tr = ok & (df['ym'] <= 201706)
    te = ok & (df['ym'] >= 201805)
    Xtr, ytr = df.loc[tr, feats].values / 10.0, df.loc[tr, 'Y'].values  # cm of precip
    mdl = LinearRegression(positive=True).fit(Xtr, ytr)
    pred_all = mdl.predict(df[feats].fillna(0).values / 10.0)
    df['Y_hat'] = np.where(df[feats].notna().all(axis=1), pred_all, np.nan)
    r_in = stats.pearsonr(df.loc[tr, 'Y'], df.loc[tr, 'Y_hat'])[0]
    r_out = stats.pearsonr(df.loc[te, 'Y'], df.loc[te, 'Y_hat'])[0]
    rmse_out = float(np.sqrt(np.mean((df.loc[te, 'Y'] - df.loc[te, 'Y_hat']) ** 2)))
    r2_out = 1 - np.sum((df.loc[te, 'Y'] - df.loc[te, 'Y_hat']) ** 2) / np.sum((df.loc[te, 'Y'] - df.loc[te, 'Y'].mean()) ** 2)
    kern = pd.DataFrame({'window_months': [f'{a}-{b}' for a, b in wins],
                         'coef_cm_per_cm_per_month': mdl.coef_,
                         'cum_response_cm_per_cm': [c * (b - a + 1) for c, (a, b) in zip(mdl.coef_, wins)]})
    kern['share_pct'] = kern['cum_response_cm_per_cm'] / kern['cum_response_cm_per_cm'].sum() * 100
    kern['cum_share_pct'] = kern['share_pct'].cumsum()
    print(kern.to_string(index=False))
    print(f'In-sample(2006-2017) r={r_in:.3f}; OUT-OF-SAMPLE(2018-05..2024-09) r={r_out:.3f}, R2={r2_out:.3f}, RMSE={rmse_out:.3f} cm, n={int(te.sum())}')

    def seg(mask_yrs):
        m = te & df['year'].isin(mask_yrs)
        return float(df.loc[m, 'Y'].mean()), float(df.loc[m, 'Y_hat'].mean())
    rows = []
    for lbl, yrs in [('2018-2020', [2018, 2019, 2020]), ('2021-2024', [2021, 2022, 2023, 2024])]:
        o, p = seg(yrs)
        rows.append({'Phase': lbl, 'Obs_cm': o, 'Pred_from_P_memory_cm': p, 'Excess_obs_minus_pred_cm': o - p})
    # slopes of obs and predicted post-2018 (cm/a)
    t = df.loc[te, 't_yr'].values
    sl_obs = np.polyfit(t, df.loc[te, 'Y'].values, 1)[0]
    sl_hat = np.polyfit(t, df.loc[te, 'Y_hat'].values, 1)[0]
    res = df.loc[te, 'Y'].values - df.loc[te, 'Y_hat'].values
    sl_res = np.polyfit(t, res, 1)[0]
    print(pd.DataFrame(rows).to_string(index=False))
    print(f'Post-2018 slope (cm/a): observed={sl_obs:+.3f}, predicted-from-precipitation-memory={sl_hat:+.3f}, residual={sl_res:+.3f}')

    # block bootstrap (12-month blocks) of training rows -> CI for excess storage 2021-2024 and post slope explained share
    idx_tr = np.where(tr.values)[0]
    nblk = int(np.ceil(len(idx_tr) / 12))
    ex_boot, sl_boot = [], []
    m2124 = (te & (df['year'] >= 2021)).values
    for _ in range(1000):
        starts = rng.integers(0, len(idx_tr) - 12, size=nblk)
        samp = np.concatenate([idx_tr[s:s + 12] for s in starts])[:len(idx_tr)]
        mm = LinearRegression(positive=True).fit(df.loc[samp, feats].values / 10.0, df.loc[samp, 'Y'].values)
        ph = mm.predict(df.loc[te, feats].values / 10.0)
        ex_boot.append(float(df.loc[m2124, 'Y'].mean() - mm.predict(df.loc[m2124, feats].values / 10.0).mean()))
        sl_boot.append(float(np.polyfit(t, ph, 1)[0]))
    ex_boot, sl_boot = np.array(ex_boot), np.array(sl_boot)
    print(f'Excess storage 2021-2024 (obs - P-memory): 95% block-bootstrap CI [{np.percentile(ex_boot,2.5):+.2f}, {np.percentile(ex_boot,97.5):+.2f}] cm; '
          f'predicted post-2018 slope 95% CI [{np.percentile(sl_boot,2.5):+.3f}, {np.percentile(sl_boot,97.5):+.3f}] cm/a (observed {sl_obs:+.3f})')
    kern.to_csv(os.path.join(OUT, 'probe_B_memory_kernel.csv'), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'probe_B_excess_storage.csv'), index=False)
    df[['ym', 'ym_dt', 't_yr', 'Y', 'Y_hat']].to_csv(os.path.join(OUT, 'probe_B_series.csv'), index=False)


# ---------------------------------------------------------------- C. runoff elasticity
def cy_E(P, PET, w):
    phi = PET / P
    return P * phi / ((1 + phi ** w) ** (1 / w))


def probe_C():
    print('\n=== [C] Runoff elasticity: observed vs Budyko-implied (fixed landscape w) ===')
    t6 = pd.read_csv(os.path.join(OUT, 'jms_table6_budyko_nonsteady.csv'))
    bl = pd.read_csv(os.path.join(OUT, 'jms_budyko_2004_2024_nonsteady.csv'))
    X = sm.add_constant(np.log(t6['P_GPM_mm'].values))
    f = sm.OLS(np.log(t6['R_obs_mm'].values), X).fit(cov_type='HC1')
    X2 = sm.add_constant(np.log(t6['P_eff_mm'].values))
    f2 = sm.OLS(np.log(t6['R_obs_mm'].values), X2).fit(cov_type='HC1')
    pet = float(bl[bl['year'].between(2016, 2024)]['PET_PML'].mean())
    p0 = float(t6['P_GPM_mm'].mean())
    w_fix = float(np.median(t6['w_WB_nonsteady']))
    eps = []
    for w in (1.1, w_fix, 1.5):
        h = 1.0
        R_hi = (p0 + h) - cy_E(p0 + h, pet, w)
        R_lo = (p0 - h) - cy_E(p0 - h, pet, w)
        eps.append((w, (R_hi - R_lo) / (2 * h) * p0 / (p0 - cy_E(p0, pet, w))))
    print(f'Observed elasticity d lnR / d lnP = {f.params[1]:.2f} (HC1 95% CI [{f.conf_int()[1,0]:.2f}, {f.conf_int()[1,1]:.2f}], n=9); '
          f'with P_eff = {f2.params[1]:.2f}')
    for w, e in eps:
        print(f'  Budyko-implied elasticity at fixed w={w:.2f}: {e:.2f}')
    r_drop = (169.4 - 248.6) / 248.6 * 100
    p_drop = (587.5 - 697.3) / 697.3 * 100
    print(f'Peak->Drawdown: P {p_drop:.1f}%, R {r_drop:.1f}%  => empirical amplification {r_drop/p_drop:.2f}x; runoff change = -9.8 km3/yr')
    pd.DataFrame({'obs_elasticity': [f.params[1]], 'obs_CI_lo': [f.conf_int()[1, 0]], 'obs_CI_hi': [f.conf_int()[1, 1]],
                  'w_fix': [w_fix], 'budyko_elasticity_wfix': [eps[1][1]]}).to_csv(os.path.join(OUT, 'probe_C_elasticity.csv'), index=False)


# ---------------------------------------------------------------- A. limiting factor regimes
def probe_A():
    print('\n=== [A] Limiting-factor regimes across the thermally-defined permafrost gradient ===')
    import ee
    from matplotlib.path import Path as MPath
    cache = os.path.join(OUT, '_cache_pixel_panel.npz')
    st4_npz = np.load(os.path.join(OUT, 'jms_st4_clim_grid.npz'))
    st4 = st4_npz['st4_clim']
    if os.path.exists(cache):
        d = np.load(cache)
        ndvi, T, P = d['ndvi'], d['T'], d['P']
    else:
        ee.Initialize(project='round-fusion-508306-p4')
        with open(GEO) as f:
            poly = ee.Geometry(json.load(f))
        rect = ee.Geometry.Rectangle([95.5, 32.0, 103.5, 36.5])
        m13 = ee.ImageCollection('MODIS/061/MOD13A2')
        proj = m13.first().select('NDVI').projection()
        era = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
        gpm = ee.ImageCollection('NASA/GPM_L3/IMERG_MONTHLY_V07')

        def ndvi_img(y):
            col = m13.filterDate(f'{y}-05-01', f'{y}-10-01')
            col = col.map(lambda im: im.select('NDVI').multiply(0.0001).updateMask(im.select('SummaryQA').lte(1)))
            return col.mean().setDefaultProjection(proj).reduceResolution(ee.Reducer.mean(), maxPixels=2048) \
                      .reproject(crs='EPSG:4326', scale=5000).rename(f'y{y}')

        def t_img(y):
            return era.filterDate(f'{y}-05-01', f'{y}-10-01').select('temperature_2m').mean().subtract(273.15) \
                      .reproject(crs='EPSG:4326', scale=5000).rename(f'y{y}')

        def p_img(y):
            col = gpm.filterDate(f'{y}-05-01', f'{y}-10-01').map(
                lambda im: im.select('precipitation').multiply(24.0).multiply(
                    ee.Date(im.get('system:time_start')).advance(1, 'month').difference(ee.Date(im.get('system:time_start')), 'day')))
            return col.sum().reproject(crs='EPSG:4326', scale=5000).rename(f'y{y}')

        def fetch(fn):
            img = ee.Image.cat([fn(y) for y in range(2004, 2025)]).toFloat().unmask(-999.0)
            for k in range(6):
                try:
                    s = img.sampleRectangle(region=rect, defaultValue=-999.0).getInfo()['properties']
                    return np.stack([np.array(s[f'y{y}'], dtype=float) for y in range(2004, 2025)])
                except Exception as e:
                    print('  retry', k, str(e)[:80]); time.sleep(4 * (k + 1)); ee.Initialize(project='round-fusion-508306-p4')
            raise RuntimeError('GEE sampleRectangle failed')
        ndvi, T, P = fetch(ndvi_img), fetch(t_img), fetch(p_img)
        np.savez_compressed(cache, ndvi=ndvi, T=T, P=P)
    print('panel shapes', ndvi.shape, T.shape, P.shape, 'st4', st4.shape)
    ny, nx = st4.shape
    if ndvi.shape[1:] != st4.shape:
        print('!! grid mismatch, aborting probe A'); return
    xmin, xmax, ymin, ymax = 95.5, 103.5, 32.0, 36.5
    lons = np.linspace(xmin + (xmax - xmin) / (2 * nx), xmax - (xmax - xmin) / (2 * nx), nx)
    lats = np.linspace(ymax - (ymax - ymin) / (2 * ny), ymin + (ymax - ymin) / (2 * ny), ny)
    lon2, lat2 = np.meshgrid(lons, lats)
    gj = json.load(open(GEO))
    rings = [np.array(gj['coordinates'][0])] if gj['type'] == 'Polygon' else [np.array(p[0]) for p in gj['coordinates']]
    pts = np.column_stack([lon2.ravel(), lat2.ravel()])
    inside = np.zeros(len(pts), bool)
    for r in rings:
        inside |= MPath(r[:, :2]).contains_points(pts)
    inside = inside.reshape(lon2.shape)
    for a in (ndvi, T, P):
        a[a <= -900] = np.nan
    valid = inside & np.isfinite(st4) & np.all(np.isfinite(ndvi), axis=0) & np.all(np.isfinite(T), axis=0) & np.all(np.isfinite(P), axis=0) & (ndvi.mean(axis=0) > 0.05)
    print('valid pixels in basin:', int(valid.sum()), ' of inside:', int(inside.sum()))
    s = st4[valid]
    print('ST4_clim in basin: min/median/max =', np.nanmin(s).round(2), np.nanmedian(s).round(2), np.nanmax(s).round(2),
          ' share<0C =', (s < 0).mean().round(3))
    yrs = np.arange(2004, 2025)

    def z(a): return (a - a.mean()) / a.std(ddof=1)

    def fit_bin(mask, label):
        nd = ndvi[:, mask].mean(axis=1); t_ = T[:, mask].mean(axis=1); p_ = P[:, mask].mean(axis=1)
        # detrended interannual anomalies (remove linear trend from each series) to avoid shared-trend confounding
        def dt(v):
            c = np.polyfit(yrs, v, 1); return v - np.polyval(c, yrs)
        Z = np.column_stack([z(dt(t_)), z(dt(p_))]); y = z(dt(nd))
        def coef(idx):
            f = sm.OLS(y[idx], sm.add_constant(Z[idx])).fit(); return f.params[1], f.params[2]
        bT, bP = coef(np.arange(len(yrs)))
        boots = []
        for _ in range(2000):  # moving-block bootstrap over years (block=3)
            st = rng.integers(0, len(yrs) - 2, size=int(np.ceil(len(yrs) / 3)))
            idx = np.concatenate([np.arange(a, a + 3) for a in st])[:len(yrs)]
            try: boots.append(coef(idx))
            except Exception: pass
        boots = np.array(boots)
        tr_nd = np.polyfit(yrs, nd, 1)[0] * 10; tr_t = np.polyfit(yrs, t_, 1)[0] * 10; tr_p = np.polyfit(yrs, p_, 1)[0] * 10
        # raw (non-standardised) sensitivities for trend attribution
        f_raw = sm.OLS(dt(nd), sm.add_constant(np.column_stack([dt(t_), dt(p_)]))).fit()
        att_T = f_raw.params[1] * tr_t; att_P = f_raw.params[2] * tr_p
        print(f'{label:<22} n_pix={int(mask.sum()):>4} | beta_T={bT:+.2f} [{np.percentile(boots[:,0],2.5):+.2f},{np.percentile(boots[:,0],97.5):+.2f}]'
              f' beta_P={bP:+.2f} [{np.percentile(boots[:,1],2.5):+.2f},{np.percentile(boots[:,1],97.5):+.2f}]'
              f' | NDVI trend {tr_nd:+.4f}/dec = T-part {att_T:+.4f} + P-part {att_P:+.4f} (dT={tr_t:+.2f} C/dec, dP={tr_p:+.1f} mm/dec)')
        return dict(Bin=label, n_pix=int(mask.sum()), beta_T=bT, beta_P=bP,
                    T_lo=np.percentile(boots[:, 0], 2.5), T_hi=np.percentile(boots[:, 0], 97.5),
                    P_lo=np.percentile(boots[:, 1], 2.5), P_hi=np.percentile(boots[:, 1], 97.5),
                    NDVI_trend_dec=tr_nd, T_attr=att_T, P_attr=att_P, dT=tr_t, dP=tr_p), boots

    full = np.zeros_like(valid)
    out_rows, bdict = [], {}
    st4v = np.where(valid, st4, np.nan)
    q = np.nanpercentile(st4v, [25, 50, 75])
    bins = [('ST4_clim < 0 C (permafrost)', valid & (st4v < 0)), ('ST4_clim >= 0 C (seasonal frozen)', valid & (st4v >= 0)),
            (f'Q1 coldest (<{q[0]:.1f})', valid & (st4v < q[0])), (f'Q2 ({q[0]:.1f}..{q[1]:.1f})', valid & (st4v >= q[0]) & (st4v < q[1])),
            (f'Q3 ({q[1]:.1f}..{q[2]:.1f})', valid & (st4v >= q[1]) & (st4v < q[2])), (f'Q4 warmest (>={q[2]:.1f})', valid & (st4v >= q[2])),
            ('Whole basin', valid)]
    for lbl, m in bins:
        r, b = fit_bin(m, lbl); out_rows.append(r); bdict[lbl] = b
    a, b_ = bdict['ST4_clim < 0 C (permafrost)'], bdict['ST4_clim >= 0 C (seasonal frozen)']
    n = min(len(a), len(b_))
    dT = a[:n, 0] - b_[:n, 0]; dP = a[:n, 1] - b_[:n, 1]
    print(f'Contrast (permafrost - seasonally frozen): d_beta_T={np.mean(dT):+.2f} [{np.percentile(dT,2.5):+.2f},{np.percentile(dT,97.5):+.2f}], '
          f'd_beta_P={np.mean(dP):+.2f} [{np.percentile(dP,2.5):+.2f},{np.percentile(dP,97.5):+.2f}]  (independent-bootstrap approx.)')
    pd.DataFrame(out_rows).to_csv(os.path.join(OUT, 'probe_A_limiting_factor.csv'), index=False)

    # pixel-wise robustness: per-pixel standardized regression -> median by ST4 class
    pix = []
    ii, jj = np.where(valid)
    for i, j in zip(ii, jj):
        def dt(v):
            c = np.polyfit(yrs, v, 1); return v - np.polyval(c, yrs)
        Zp = np.column_stack([z(dt(T[:, i, j])), z(dt(P[:, i, j]))]); yp = z(dt(ndvi[:, i, j]))
        f = sm.OLS(yp, sm.add_constant(Zp)).fit()
        pix.append((st4[i, j], f.params[1], f.params[2], f.rsquared))
    pix = pd.DataFrame(pix, columns=['st4', 'bT', 'bP', 'r2'])
    pix['cls'] = np.where(pix['st4'] < 0, 'perm', 'sfg')
    print('\nPixel-wise median standardized coefficients (detrended):')
    print(pix.groupby('cls')[['bT', 'bP', 'r2']].median().round(3).to_string())
    print('Mann-Whitney perm vs sfg  beta_T p=%.2g, beta_P p=%.2g' % (
        stats.mannwhitneyu(pix[pix.cls == 'perm'].bT, pix[pix.cls == 'sfg'].bT).pvalue,
        stats.mannwhitneyu(pix[pix.cls == 'perm'].bP, pix[pix.cls == 'sfg'].bP).pvalue))
    pix.to_csv(os.path.join(OUT, 'probe_A_pixelwise.csv'), index=False)


if __name__ == '__main__':
    probe_B()
    probe_C()
    probe_A()
    print('\n>>> breakthrough probes finished')
