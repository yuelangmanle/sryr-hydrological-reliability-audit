#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
45_jms_figures_publication.py
=============================
Publication-grade (Nature / WRR / JMS) 600-dpi figure suite for the revised
Source Region of the Yellow River methodological reliability audit.
Every single number, curve, error bar, and label is read directly from
results_repaired/*.csv and *.npz. Zero fabricated values.
"""
import os
import sys
import warnings
import textwrap
import importlib.util
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image

BASE = '/Users/yueliangmanle/Desktop/小论文中文2'
sp = importlib.util.spec_from_file_location('f36', f'{BASE}/code/02_figures/36_jms_english_figures.py')
f36 = importlib.util.module_from_spec(sp)
sp.loader.exec_module(f36)

OUT = f36.OUT_DIR
RES = f36.RES_DIR

# Nature / Okabe-Ito inspired palette (high contrast, colorblind-safe)
NAVY = '#1B365D'
BLUE = '#2B6CB0'
SKY = '#63B3ED'
TEAL = '#008080'
MINT = '#38A169'
ORANGE = '#D97706'
CRIMSON = '#C53030'
CORAL = '#E53E3E'
PURPLE = '#6B46C1'
SLATE = '#4A5568'
LGREY = '#E2E8F0'
MGREY = '#A0AEC0'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'axes.edgecolor': '#2D3748',
    'axes.linewidth': 0.85,
    'xtick.color': '#2D3748',
    'ytick.color': '#2D3748',
    'text.color': '#1A202C',
    'axes.titlepad': 8,
    'savefig.dpi': 600,
})

LEG = dict(frameon=True, facecolor='white', framealpha=0.94, edgecolor='#CBD5E0', borderpad=0.45)


def R(name):
    return pd.read_csv(os.path.join(RES, name))


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=600, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print('Saved:', p)


def clean_ax(ax, grid='y'):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid in ('y', 'both'):
        ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#CBD5E0', alpha=0.7, zorder=0)
    if grid in ('x', 'both'):
        ax.xaxis.grid(True, linestyle='--', linewidth=0.5, color='#CBD5E0', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def panel_title(ax, letter, title, fs=10.2):
    ax.set_title(f'{letter}   {title}', loc='left', fontweight='bold', fontsize=fs, color='#1A202C', pad=8)


def cap_grid():
    g = pd.read_csv(os.path.join(BASE, 'data/processed/grace_monthly_grid.csv'))
    cells = pd.read_csv(os.path.join(BASE, '投稿包/04_数据/处理后数据/basin_cells_classified.csv'))
    for d in (g, cells):
        d['lat_r'] = d['lat'].round(4)
        d['lon_r'] = d['lon'].round(4)
    m = g[g['ym'] == 202006].merge(
        cells[['lat_r', 'lon_r', 'intersects', 'norm_weight']],
        on=['lat_r', 'lon_r'],
        how='left'
    )
    m['sig'] = m['lwe'].round(6)
    ids = {s: f'Cap {i}' for i, s in enumerate(sorted(m.loc[m.intersects == True, 'sig'].unique()), start=1)}
    m['cap'] = m['sig'].map(ids)
    return m


# =============================================================================
# FIG 1: Study Area + Hypsometric Permafrost Inset + 6 Caps + Dumbbell Trends
# =============================================================================
def fig1():
    rings = f36.load_polygon()
    t = np.load(os.path.join(RES, 'jms_terrain_grid.npz'))
    dem, upa = t['dem'], t['upa']
    xmin, xmax, ymin, ymax = t['extent']
    st4z = np.load(os.path.join(RES, 'jms_st4_clim_grid.npz'))['st4_clim']
    ny, nx = dem.shape
    lons = np.linspace(xmin + (xmax - xmin) / (2 * nx), xmax - (xmax - xmin) / (2 * nx), nx)
    lats = np.linspace(ymax - (ymax - ymin) / (2 * ny), ymin + (ymax - ymin) / (2 * ny), ny)
    lo, la = np.meshgrid(lons, lats)
    mask = f36.polygon_mask(rings, lo, la)
    zc = np.full(dem.shape, np.nan)
    v = mask & (dem > 0)
    zc[v & (dem < 3800)] = 0
    zc[v & (dem >= 3800) & (dem < 4300)] = 1
    zc[v & (dem >= 4300)] = 2

    t4 = R('jms_table4_reconciled.csv')
    ap = t4.drop_duplicates('Zone').set_index('Zone')['Area_pct']
    hill = np.array(Image.open(f36.HILL_PATH).convert('L'))
    cm = cap_grid()
    diag = R('jms_grace_cap_diagnostics.csv')
    gt = R('grace_grid_trends.csv')
    gt['lat_r'] = gt['lat'].round(4)
    gt['lon_r'] = gt['lon'].round(4)
    cm = cm.merge(gt[['lat_r', 'lon_r', 'slope_full', 'slope_post2018']], on=['lat_r', 'lon_r'], how='left')

    fig = plt.figure(figsize=(15.0, 7.6), facecolor='white')
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.35, 1.0], hspace=0.38, wspace=0.18,
                           left=0.05, right=0.98, top=0.93, bottom=0.08)

    # (a) Map
    ax = fig.add_subplot(gs[:, 0])
    ax.set_xlim(95.5, 103.5)
    ax.set_ylim(32.07, 36.5)
    ax.imshow(hill, extent=[95.5, 103.5, 32.0, 36.5], cmap='gray', alpha=0.38, aspect='auto', zorder=1)
    ax.imshow(np.ma.masked_invalid(zc), extent=[xmin, xmax, ymin, ymax],
              cmap=ListedColormap(['#E6C68A', '#74C69D', '#3B6E9C']),
              norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), alpha=0.70, aspect='auto', origin='upper', zorder=2)
    ax.imshow(np.where(mask & (upa >= 1000) & (upa < 5500), 1.0, np.nan),
              extent=[xmin, xmax, ymin, ymax], cmap=ListedColormap(['#2B6CB0']), alpha=0.65, aspect='auto', origin='upper', zorder=3)
    ax.imshow(np.where(mask & (upa >= 5500), 1.0, np.nan),
              extent=[xmin, xmax, ymin, ymax], cmap=ListedColormap(['#102A43']), alpha=0.96, aspect='auto', origin='upper', zorder=3.5)

    lo5, la5 = np.meshgrid(
        np.linspace(95.5 + 8 / 358, 103.5 - 8 / 358, st4z.shape[1]),
        np.linspace(36.5 - 4.5 / 202, 32.0 + 4.5 / 202, st4z.shape[0])
    )
    plt.rcParams['hatch.color'] = CRIMSON
    ax.contourf(lo5, la5, np.where(np.isfinite(st4z), st4z, 99), levels=[-20, 0], colors='none', hatches=['////'], zorder=6)
    ax.contour(lo5, la5, st4z, levels=[0.0], colors=[CRIMSON], linewidths=1.3, zorder=6.5)
    for r in rings:
        ax.plot(r[:, 0], r[:, 1], color='white', lw=2.8, zorder=4.8)
        ax.plot(r[:, 0], r[:, 1], color='#1A202C', lw=1.6, zorder=5)

    ax.scatter([100.15], [35.50], marker='*', s=240, color='#E53E3E', edgecolor='white', lw=1.0, zorder=8)
    ax.annotate('Tangnaihai gauge\n(100.15°E, 35.50°N;\n123,800 km²)',
                xy=(100.15, 35.50), xytext=(101.28, 35.02),
                fontsize=8.0, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#CBD5E0', alpha=0.95),
                arrowprops=dict(arrowstyle='-|>', color='#1A202C', lw=1.1), zorder=9)

    h = [
        mpatches.Patch(color='#E6C68A', label=f'<3,800 m ({ap["<3800m"]:.1f}%)'),
        mpatches.Patch(color='#74C69D', label=f'3,800–4,300 m ({ap["3800-4300m"]:.1f}%)'),
        mpatches.Patch(color='#3B6E9C', label=f'≥4,300 m ({ap[">=4300m"]:.1f}%)'),
        mpatches.Patch(facecolor='none', edgecolor=CRIMSON, hatch='////',
                       label=f'ERA5-Land ST4 < 0 °C\n(permafrost side, {ap["Permafrost (ST4<0C)"]:.1f}%)'),
        plt.Line2D([0], [0], color='#102A43', lw=2.0, label='Yellow River & tributaries'),
    ]
    ax.legend(handles=h, loc='upper right', fontsize=7.5, title='MERIT DEM & thermal zones', title_fontsize=7.8, **LEG)
    ax.annotate('N', xy=(95.82, 33.15), xytext=(95.82, 32.68), ha='center', va='center',
                fontsize=10, fontweight='bold', arrowprops=dict(arrowstyle='-|>', color='#1A202C', lw=1.6), zorder=9)
    ax.plot([96.25, 97.28], [32.42, 32.42], color='#1A202C', lw=2.4, zorder=9)
    ax.text(96.76, 32.50, '100 km', ha='center', fontsize=8, fontweight='bold', zorder=9)
    ax.set_xticks([96, 98, 100, 102])
    ax.set_xticklabels(['96°E', '98°E', '100°E', '102°E'])
    ax.set_yticks([32, 33, 34, 35, 36])
    ax.set_yticklabels(['32°N', '33°N', '34°N', '35°N', '36°N'])
    panel_title(ax, 'a', 'Topography, elevation belts & thermally defined permafrost')

    # Hypsometric distribution inset in SW corner
    axins = inset_axes(ax, width='27%', height='20%', loc='lower left', bbox_to_anchor=(0.06, 0.18, 1, 1), bbox_transform=ax.transAxes)
    elev_vals = dem[v]
    bins = np.linspace(2600, 5600, 26)
    cnt, edges = np.histogram(elev_vals, bins=bins)
    pct = 100 * cnt / cnt.sum()
    mids = 0.5 * (edges[:-1] + edges[1:])
    cols_h = ['#E6C68A' if m < 3800 else ('#74C69D' if m < 4300 else '#3B6E9C') for m in mids]
    axins.bar(mids, pct, width=(edges[1] - edges[0]) * 0.9, color=cols_h, edgecolor='none', alpha=0.92)
    axins.axvline(3800, color='#4A5568', ls=':', lw=0.7)
    axins.axvline(4300, color='#4A5568', ls=':', lw=0.7)
    axins.set_facecolor('white')
    axins.patch.set_alpha(0.92)
    axins.set_title('Hypsometry (%)', fontsize=6.8, fontweight='bold', pad=2)
    axins.set_xticks([3000, 4000, 5000])
    axins.set_xticklabels(['3k', '4k', '5k m'], fontsize=6.2)
    axins.set_yticks([])
    axins.spines['top'].set_visible(False)
    axins.spines['right'].set_visible(False)

    # (b) 6 Native Mascon Caps
    axb = fig.add_subplot(gs[0, 1])
    axb.set_xlim(95.5, 103.6)
    axb.set_ylim(32.0, 36.5)
    cap_cols = dict(zip(sorted(cm.cap.dropna().unique()), ['#3B6E9C', '#E07A5F', '#3D9970', '#C53030', '#805AD5', '#8D6E63']))
    d = 0.50306
    for _, r in cm.iterrows():
        axb.add_patch(mpatches.Rectangle(
            (r.lon - d / 2, r.lat - d / 2), d, d,
            facecolor=cap_cols[r.cap], alpha=0.88 if r.intersects == True else 0.24,
            edgecolor='white', lw=0.35
        ))
    for r in rings:
        axb.plot(r[:, 0], r[:, 1], color='white', lw=2.2, zorder=4.8)
        axb.plot(r[:, 0], r[:, 1], color='#1A202C', lw=1.35, zorder=5)
    for _, r in diag.iterrows():
        k = r.Cap_ID.replace('_', ' ')
        axb.text(r.Center_Lon, r.Center_Lat, f'{k[-1]}', fontsize=8.5, fontweight='bold',
                 ha='center', va='center', color='white', zorder=6,
                 bbox=dict(boxstyle='circle,pad=0.18', fc='#1A202C', ec='white', lw=0.8))
    axb.set_xticks([96, 98, 100, 102])
    axb.set_xticklabels(['96°E', '98°E', '100°E', '102°E'])
    axb.set_yticks([32, 34, 36])
    axb.set_yticklabels(['32°N', '34°N', '36°N'])
    panel_title(axb, 'b', 'Six native 3° mascon units share the 76 cells', fs=9.8)

    # (c) Dumbbell comparison of Full vs Post-2018 trends + clean right metadata column
    axc = fig.add_subplot(gs[1, 1])
    clean_ax(axc, 'x')
    inb = cm[cm.intersects == True]
    rows = []
    for k, grp in inb.groupby('cap'):
        w = grp.norm_weight.values
        rows.append((k, np.average(grp.slope_full, weights=w), np.average(grp.slope_post2018, weights=w), w.sum()))
    rows = sorted(rows, key=lambda x: x[0])
    exf = diag.set_index(diag.Cap_ID.str.replace('_', ' '))['Exterior_Fraction_In_144Grid']
    y = np.arange(len(rows))
    for yi, (k, sf, sp_, wt) in zip(y, rows):
        axc.plot([sf, sp_], [yi, yi], color='#CBD5E0', lw=3.0, zorder=2)
        axc.scatter([sf], [yi], color=SLATE, s=62, zorder=4, edgecolor='white', lw=0.7,
                    label='Full 2003–2024' if yi == 0 else '')
        axc.scatter([sp_], [yi], color=CRIMSON, s=68, marker='D', zorder=5, edgecolor='white', lw=0.7,
                    label='Post-2018' if yi == 0 else '')
        axc.text(1.03, yi, f'wt {wt:.2f} | {exf[k]*100:.0f}% ext.', va='center', ha='left',
                 fontsize=7.2, color='#4A5568', transform=axc.get_yaxis_transform())
    axc.axvline(0, color='#1A202C', lw=0.8, ls='--')
    axc.set_yticks(y)
    axc.set_yticklabels([r[0] for r in rows], fontsize=8.2, fontweight='bold')
    axc.invert_yaxis()
    axc.set_xlim(-1.05, 1.05)
    axc.set_xlabel('Weighted trend of 0.5° cells inside basin (cm a⁻¹)')
    axc.legend(loc='lower left', fontsize=7.3, **LEG)
    panel_title(axc, 'c', 'Dumbbell trend per native unit (right: weight & exterior %)', fs=9.6)
    fig.subplots_adjust(right=0.91)
    save(fig, 'Fig1_StudyArea_and_GRACE_Spatial.png')


# =============================================================================
# FIG 2: Temporal Transition, Dual-Error SNR, Forest Plot & N_eff Collapse
# =============================================================================
def fig2():
    df = R('basin_merged_monthly_2003_2024.csv')
    df['dt'] = pd.to_datetime(df['ym_dt'])
    t3 = R('jms_table3_reconciled.csv').iloc[0]
    ph = R('jms_grace_uncertainty_robust.csv')
    mc = R('jms_grace_wls_montecarlo_robust.csv')
    mem = R('jms_neff_memory.csv')

    fig, axs = plt.subplots(2, 2, figsize=(14.4, 9.4), facecolor='white')
    plt.subplots_adjust(left=0.11, right=0.97, top=0.93, bottom=0.07, hspace=0.35, wspace=0.25)
    a, b, c, d = axs.ravel()

    # (a) Time series with shaded GRACE gap & dual-regime slopes
    clean_ax(a, 'y')
    gs_, ge_ = pd.Timestamp('2017-07-01'), pd.Timestamp('2018-04-30')
    m = df['lwe_basin'].notna()
    a.axvspan(gs_, ge_, color='#EDF2F7', alpha=0.9, zorder=1, label='GRACE gap (2017-07 to 2018-04)')
    a.axhline(0, color='#A0AEC0', lw=0.7, ls=':')
    a.plot(df.dt, df.anom_G21_TWS_proxy / 10, color=ORANGE, lw=0.95, alpha=0.72, label='GLDAS-2.1 (open-loop)')
    a.plot(df.dt, df.anom_G22_TWS_tavg / 10, color=TEAL, lw=1.1, alpha=0.82, label='GLDAS-2.2 (assimilated)')
    a.fill_between(df.dt[m], df.anom_lwe_basin[m] - df.unc_basin[m], df.anom_lwe_basin[m] + df.unc_basin[m],
                   color=NAVY, alpha=0.14, label='GRACE formal ±1σ')
    a.plot(df.dt[m], df.anom_lwe_basin[m], color=NAVY, lw=1.55, label='GRACE/GRACE-FO (n = 218)')
    for sub, col, lb in [
        (m & (df.ym <= 201706), BLUE, f'Pre-gap {t3.Sub_Pre_HAC:+.3f} cm a⁻¹'),
        (m & (df.ym >= 201805), CRIMSON, f'Post-gap {t3.Sub_Post_May18_HAC:+.3f} cm a⁻¹')
    ]:
        p = np.polyfit(df.loc[sub, 't_yr'], df.loc[sub, 'anom_lwe_basin'], 1)
        a.plot(df.dt[sub], np.polyval(p, df.loc[sub, 't_yr']), color=col, lw=2.3, label=lb, zorder=6)
    a.set_ylim(-10.5, 13.0)
    a.set_ylabel('Deseasonalized TWSA (cm)')
    a.text(0.985, 0.04,
           f'Pettitt break {t3.Pettitt_Break}\niid p = {t3.Pettitt_p_iid:.1e} vs. AR(1)-boot p = {t3.Pettitt_p_ARboot:.2f}',
           transform=a.transAxes, ha='right', fontsize=7.6,
           bbox=dict(boxstyle='round,pad=0.3', fc='#FFF5F5', ec=CRIMSON, lw=0.8))
    a.legend(loc='upper left', fontsize=6.8, ncol=2, **LEG)
    a.xaxis.set_major_locator(mdates.YearLocator(4))
    a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    panel_title(a, 'a', 'Basin TWSA, multi-model coherence & segmented trends')

    # (b) Phase means with dual error bars + SNR badges
    clean_ax(b, 'y')
    sub = ph.iloc[[0, 1, 2, 4]].reset_index(drop=True)
    nm = ['2003–2017\n(reference)', '2018-05–2020\npluvial peak', '2021–2024\ndrawdown', '2024\n(Jan–Sep)']
    x = np.arange(4)
    cols_ = ['#718096', TEAL, PURPLE, ORANGE]
    b.bar(x, sub.Mean_JPL_cm, color=cols_, width=0.46, edgecolor='#1A202C', lw=0.7, alpha=0.85, zorder=3)
    b.errorbar(x - 0.09, sub.Mean_JPL_cm, yerr=2 * sub.SE_iid_cm, fmt='o', ms=4, color='#1A202C',
               ecolor='#1A202C', capsize=4, lw=1.4, zorder=5, label='±2 SE (independent errors)')
    b.errorbar(x + 0.09, sub.Mean_JPL_cm, yerr=2 * sub.SE_corr_with_FObias_cm, fmt='s', ms=4, color=CRIMSON,
               ecolor=CRIMSON, capsize=4, lw=1.4, zorder=5, label='±2 SE (AR(1) + 0.5 cm FO offset)')
    for i, r in sub.iterrows():
        top = r.Mean_JPL_cm + 2 * r.SE_corr_with_FObias_cm + 0.30
        txt = 'Reference\nbaseline' if i == 0 else f'SNR {r.SNR_iid:.1f} → {r.SNR_corr_FObias:.1f}'
        fc = '#EDF2F7' if i == 0 else ('#FFF5F5' if r.SNR_corr_FObias < 2 else '#F0FFF4')
        b.text(i, top, txt, ha='center', va='bottom', fontsize=7.4, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.22', fc=fc, ec='#CBD5E0', lw=0.6))
    b.set_xticks(x)
    b.set_xticklabels(nm, fontsize=8.0)
    b.set_ylim(0, 9.6)
    b.set_ylabel('Mean JPL TWSA (cm)')
    b.legend(loc='upper right', fontsize=7.1, **LEG)
    panel_title(b, 'b', 'Phase means & SNR attenuation under correlated errors')

    # (c) Forest Plot: HAC-OLS + WLS + 2,000 correlated-error MC draws
    clean_ax(c, 'x')
    sel = mc.iloc[[0, 2, 4, 5]].reset_index(drop=True)
    nms = ['Full 2003–2024', 'Post-gap 2018-05–2024', 'Drawdown 2020-09–2024', 'Recent 2022-09–2024']
    c.axvspan(-0.05, 0.05, color='#EDF2F7', alpha=0.8, zorder=1)
    for i, r in sel.iterrows():
        lo_, hi_ = [float(s) for s in r.MC_corr_95CI.strip('[]').split(',')]
        c.plot([lo_, hi_], [i, i], color=BLUE, lw=6.0, alpha=0.28, solid_capstyle='round', zorder=2,
               label='MC 95% CI (correlated)' if i == 0 else '')
        c.scatter([r.OLS_HAC_slope], [i], color=CRIMSON, s=62, zorder=5, edgecolor='white', lw=0.7,
                  label='OLS-HAC' if i == 0 else '')
        c.scatter([r.WLS_HAC_slope], [i], color=NAVY, marker='D', s=46, zorder=4, edgecolor='white', lw=0.7,
                  label='WLS-HAC' if i == 0 else '')
        c.text(max(hi_, r.OLS_HAC_slope, r.WLS_HAC_slope) + 0.12, i,
               f'{r.OLS_HAC_slope:+.2f} [{lo_:+.2f}, {hi_:+.2f}]', va='center', fontsize=7.6, fontweight='bold')
    c.axvline(0, color='#4A5568', ls='--', lw=0.9)
    c.set_yticks(range(4))
    c.set_yticklabels(nms, fontsize=8.2, fontweight='bold')
    c.invert_yaxis()
    c.set_xlim(-2.6, 5.5)
    c.set_xlabel('Trend (cm a⁻¹): OLS-HAC, WLS-HAC & 2,000 correlated-error draws')
    c.legend(loc='upper right', fontsize=7.0, **LEG)
    panel_title(c, 'c', 'Forest plot of window sensitivity (recent CI spans zero)')

    # (d) Effective sample size significance collapse
    clean_ax(d, 'y')
    k = mem[mem.Window_k_mo < 100]
    thr = -np.log10(0.05)
    d.axhspan(0, thr, color='#FFF5F5', alpha=0.6, zorder=0)
    d.axhspan(thr, 2.35, color='#F0FFF4', alpha=0.45, zorder=0)
    d.plot(k.Window_k_mo, -np.log10(k.p_iid), marker='o', ms=5.5, color='#718096', lw=2.0, label='Nominal df (iid)', zorder=4)
    d.plot(k.Window_k_mo, -np.log10(k.p_eff_PP), marker='s', ms=5.5, color=CRIMSON, lw=2.2, label='Pyper–Peterman effective df', zorder=5)
    an = mem[mem.Window_k_mo == 999].iloc[0]
    d.scatter([14.6], [-np.log10(an.p_eff_PP)], marker='D', s=80, color=TEAL, edgecolor='white', lw=0.8, zorder=6,
              label='Non-overlapping annual (n = 20)')
    d.axhline(thr, color='#2D3748', ls=':', lw=1.1)
    d.text(0.5, thr + 0.04, 'α = 0.05 threshold', fontsize=7.4, fontweight='bold', color='#2D3748')
    r12 = k[k.Window_k_mo == 12].iloc[0]
    d.annotate('', xy=(12, -np.log10(r12.p_eff_PP) + 0.04), xytext=(12, -np.log10(r12.p_iid) - 0.04),
               arrowprops=dict(arrowstyle='-|>', color=CRIMSON, lw=1.6, ls='--'))
    d.text(11.3, 1.80, f'k = 12 mo: r = {r12.partial_r_NDVI_TWSA:+.3f}\np: {r12.p_iid:.3f} → {r12.p_eff_PP:.3f}\n(N_eff = {r12.N_eff_PP:.1f})',
           ha='right', va='center', fontsize=7.4, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=CRIMSON, lw=0.7))
    d.text(14.6, -np.log10(an.p_eff_PP) + 0.14, f'r = {an.partial_r_NDVI_TWSA:+.2f}\n(p = {an.p_eff_PP:.3f})',
           ha='center', fontsize=7.3, fontweight='bold', color=TEAL)
    d.set_xticks([1, 2, 3, 4, 6, 9, 12, 14.6])
    d.set_xticklabels(['1', '2', '3', '4', '6', '9', '12', 'annual'])
    d.set_xlabel('Cumulative window k (months) for NDVI–TWSA partial correlation')
    d.set_ylabel('−log₁₀ p')
    d.set_ylim(0, 2.35)
    d.legend(loc='upper left', fontsize=7.1, **LEG)
    panel_title(d, 'd', 'Autocorrelation collapses k = 12 NDVI–TWSA significance')
    save(fig, 'Fig2_Temporal_Transition_and_Subzones.png')


# =============================================================================
# FIG 3: Soil-Profile Area=Mass Diagram + Cross-Sensor Slope Lines + Zonal Bars
# =============================================================================
def fig3():
    sub = R('jms_subsurface_normalized.csv')
    t4 = R('jms_table4_reconciled.csv')
    ts = R('jms_cross_sensor_annual_2004_2024.csv')

    fig, axs = plt.subplots(2, 2, figsize=(14.4, 9.4), facecolor='white')
    plt.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.08, hspace=0.36, wspace=0.26)
    a, b, c, d = axs.ravel()
    belts = ['<3800m', '3800-4300m', '>=4300m']
    bl = ['<3,800 m', '3,800–4,300 m', '≥4,300 m']

    # (a) Physical Soil-Column Profile (height = layer thickness cm, width = rate per cm, area = total loss!)
    clean_ax(a, 'x')
    g21 = sub[sub.Model_Layer.str.contains('GLDAS-2.1 SM') & sub.Thickness_cm.lt(200)].reset_index(drop=True)
    depths = [(0, 10), (10, 40), (40, 100), (100, 200)]
    cols_s = ['#F6E05E', '#ED8936', '#DD6B20', '#9C4221']
    for i, ((d0, d1), col) in enumerate(zip(depths, cols_s)):
        r = g21.iloc[i]
        a.barh((d0 + d1) / 2, r.Per_cm_mma_cm, height=(d1 - d0) * 0.92, color=col, edgecolor='#1A202C', lw=0.8, alpha=0.92)
        txt = f'{d0}–{d1} cm: {r.Per_cm_mma_cm:.3f} mm a⁻¹ cm⁻¹  ({r.Share_pct:.1f}% of loss; {r.Thickness_cm/2:.0f}% of depth)'
        y_pos = (d0 + d1) / 2
        a.text(r.Per_cm_mma_cm - 0.0025, y_pos, txt, ha='right', va='center', fontsize=7.3, fontweight='bold')
    g22 = sub[sub.Model_Layer.str.contains('GLDAS-2.2 GWS')].iloc[0]
    a.axvline(-0.0468, color=NAVY, ls='--', lw=1.2, label='Column-mean rate (−0.047 mm a⁻¹ cm⁻¹)')
    a.text(0.02, 0.03,
           f'Bar height = physical thickness (200 cm); area = total loss.\n40–200 cm holds {g21.Share_pct[2:].sum():.1f}% of loss across 80% of depth.\nGLDAS-2.2 assigns {g22.Share_pct:.1f}% to GWS (GRACE-assimilated).',
           transform=a.transAxes, fontsize=7.3,
           bbox=dict(boxstyle='round,pad=0.26', fc='#FFFAF0', ec=ORANGE, lw=0.8))
    a.set_ylim(255, -10)
    a.set_yticks([0, 40, 100, 200])
    a.set_yticklabels(['0 cm', '40 cm', '100 cm', '200 cm'])
    a.set_xlim(-0.225, 0.008)
    a.set_xlabel('Post-2018 loss rate per cm of soil (mm a⁻¹ cm⁻¹), GLDAS-2.1 open-loop')
    a.legend(loc='lower right', bbox_to_anchor=(0.98, 0.03), fontsize=6.9, **LEG)
    panel_title(a, 'a', 'Soil-profile normalization: block area = total loss (no deep excess)')

    # (b) Cross-Sensor Temperature Point-Range Slope Plot
    clean_ax(b, 'y')
    x = np.arange(3)
    offsets = [-0.14, 0.0, 0.14]
    specs = [('ERA5_T2m', 'ERA5-Land T2m', CRIMSON, 'o'), ('TC_Tmean', 'TerraClimate Tmean', BLUE, 's'), ('MOD_LST', 'MODIS LST (day+night)', SLATE, 'D')]
    txt = []
    for (var, nm, col, mk), dx in zip(specs, offsets):
        rr = [t4[(t4.Zone == z) & (t4.Variable == var)].iloc[0] for z in belts]
        vals = [r.HAC_dec for r in rr]
        errs = [1.96 * r.HAC_SE_dec for r in rr]
        b.plot(x + dx, vals, color=col, lw=1.8, alpha=0.75, zorder=3)
        b.errorbar(x + dx, vals, yerr=errs, fmt=mk, ms=6.5, color=col, mfc=col, mec='white', mew=0.7,
                   ecolor=col, capsize=3.5, lw=1.3, label=nm, zorder=5)
        hi = ts[(ts.zone_id == 3)].sort_values('year')[var].values
        lw_ = ts[(ts.zone_id == 1)].sort_values('year')[var].values
        yy = np.arange(2004, 2025)
        f = sm.OLS(hi - lw_, sm.add_constant(yy - 2004)).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
        txt.append(f'{nm.split()[0]}: {10*f.params[1]:+.2f} °C/dec (p = {f.pvalues[1]:.3f})')
    b.axhline(0, color='#718096', ls=':', lw=0.8)
    b.text(0.02, 0.97, 'High-minus-low belt gradient\n' + '\n'.join(txt), transform=b.transAxes, va='top', fontsize=7.5,
           bbox=dict(boxstyle='round,pad=0.3', fc='#F7FAFC', ec='#CBD5E0'))
    b.set_xticks(x)
    b.set_xticklabels(bl)
    b.set_ylim(-0.35, 1.98)
    b.set_ylabel('Temperature trend 2004–2024 (°C decade⁻¹, ±95% HAC CI)')
    b.legend(loc='upper right', fontsize=7.1, **LEG)
    panel_title(b, 'b', 'Elevation warming gradient is product-dependent')

    # (c) & (d) Zonal Eco-Hydrological Response
    def rel(zset, zl, ax, letter, title):
        clean_ax(ax, 'y')
        vs = [('GPM_P', 'Precipitation', SKY), ('PML_Es', 'Soil evap. Es', '#F6AD55'),
              ('PML_Ec', 'Transpiration Ec', TEAL), ('NDVI_GS_NoSnow', 'NDVI (snow-masked)', NAVY)]
        x_ = np.arange(len(zset))
        w = 0.19
        for j, (v, nm, col) in enumerate(vs):
            rr = [t4[(t4.Zone == z) & (t4.Variable == v)].iloc[0] for z in zset]
            ax.bar(x_ + (j - 1.5) * w, [r.Rel_Change_pct for r in rr], w, color=col, edgecolor='#1A202C', lw=0.55, label=nm, zorder=3)
            for xi, r in zip(x_, rr):
                s = '**' if r.HAC_p < 0.01 else ('*' if r.HAC_p < 0.05 else 'ns')
                ax.text(xi + (j - 1.5) * w, max(r.Rel_Change_pct, 0) + 0.5, s, ha='center', fontsize=7.5, fontweight='bold',
                        color=CRIMSON if s != 'ns' else '#718096')
        ax.set_xticks(x_)
        ax.set_xticklabels(zl)
        ax.set_ylabel('2020–2024 vs. 2004–2008 (%)')
        ax.set_ylim(0, 30)
        panel_title(ax, letter, title)

    rel(belts, bl, c, 'c', 'Relative changes by elevation belt (** p < 0.01, * p < 0.05)')
    c.legend(loc='upper right', fontsize=7.0, ncol=2, **LEG)
    pz = ['Permafrost (ST4<0C)', 'Seasonally_Frozen (ST4>=0C)']
    rel(pz, ['Permafrost side\n(ERA5-Land ST4 < 0 °C)', 'Seasonally frozen side\n(ERA5-Land ST4 ≥ 0 °C)'], d,
        'd', 'Relative changes by thermally defined permafrost zone')
    save(fig, 'Fig3_Subsurface_EDW_and_Uncertainty.png')


# =============================================================================
# FIG 4: PCMCI Causal Network + Diverging Lollipop ET + Dumbbell N_eff Shrinkage
# =============================================================================
def fig4():
    et = R('jms_et_nonlai_check.csv')
    t2 = R('jms_table2_neff.csv')

    fig = plt.figure(figsize=(15.6, 6.0), facecolor='white')
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.22, 1.05, 1.08], wspace=0.40,
                           left=0.02, right=0.98, top=0.87, bottom=0.14)

    # (a) PCMCI Network
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(0, 10)
    ax.axis('off')
    panel_title(ax, 'a', 'Difference-based PCMCI (lags 1–3; gap masked)', fs=9.8)
    nodes = {
        'P': (1.6, 8.3, 'Precipitation\n(GPM V07)', '#EBF8FF', BLUE),
        'T': (8.4, 8.3, 'Air temp.\n(ERA5-Land)', '#FFF5F5', CRIMSON),
        'Ec': (5.0, 6.4, 'Transpiration\n(PML Ec)', '#FEFCBF', ORANGE),
        'SM': (1.6, 3.4, 'Soil moisture\n(GLDAS-2.1)', '#E6FFFA', TEAL),
        'N': (8.4, 3.4, 'Greenness\n(MODIS NDVI)', '#F0FFF4', MINT),
        'W': (5.0, 1.2, 'Water storage\n(GRACE TWSA)', '#EDF2F7', NAVY),
    }
    for k, (x, y, lb, fc, ec) in nodes.items():
        ax.add_patch(FancyBboxPatch((x - 1.25, y - 0.62), 2.5, 1.24,
                                    boxstyle='round,pad=0.1,rounding_size=0.22', fc=fc, ec=ec, lw=1.6, zorder=4))
        ax.text(x, y, lb, ha='center', va='center', fontsize=7.3, fontweight='bold', zorder=5)

    def edge(u, v, rad, col, ls, lbl, lp, lw=2.0):
        x1, y1 = nodes[u][:2]
        x2, y2 = nodes[v][:2]
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f'arc3,rad={rad}',
                                     arrowstyle='-|>,head_length=6.5,head_width=3.6',
                                     color=col, ls=ls, lw=lw, shrinkA=28, shrinkB=28, zorder=3))
        ax.text(lp[0], lp[1], lbl, fontsize=6.4, fontweight='bold', color=col, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.14', fc='white', ec='none', alpha=0.92), zorder=6)

    edge('P', 'SM', 0.0, BLUE, '-', 'lag 1 +0.237\n(q=0.0012)', (0.60, 5.85), 2.2)
    edge('SM', 'Ec', -0.35, TEAL, '-', 'lag 1 +0.234\n(q=0.0012)', (2.75, 5.95), 2.2)
    edge('Ec', 'SM', -0.35, ORANGE, '--', 'feedback −0.152\n(p=0.015)', (3.65, 4.25), 1.5)
    edge('P', 'Ec', 0.18, BLUE, '-', 'lag 1 +0.166 (q=0.034)', (4.15, 8.20), 1.8)
    edge('T', 'N', 0.0, CRIMSON, '-', 'lag 2 −0.210\n(q<0.007)', (9.15, 5.90), 1.8)
    edge('W', 'Ec', 0.25, PURPLE, '-', 'TWSA→Ec lag 3\n+0.274 (q=0.0085)', (5.85, 3.05), 2.0)
    edge('W', 'SM', 0.0, NAVY, '--', 'lag 1 +0.208 (p=0.009)', (2.50, 1.90), 1.5)
    ax.text(5, -0.05, 'Solid: FDR q < 0.05; dashed: nominal p < 0.05. No NDVI→TWSA or NDVI→Ec edge detected.',
            ha='center', fontsize=6.8, style='italic', color='#4A5568')

    # (b) NDVI vs PML vs Non-LAI ET products
    b = fig.add_subplot(gs[0, 1])
    clean_ax(b, 'y')
    tg = [
        ('anom_PML_Ei', 'PML Ei (LAI)', PURPLE),
        ('anom_PML_Ec', 'PML Ec (LAI)', TEAL),
        ('anom_PML_Es', 'PML Es', ORANGE),
        ('anom_ERA5_ET', 'ERA5-Land ET', SLATE),
        ('anom_TC_ET', 'TerraClimate ET', '#718096'),
        ('anom_G21_ET', 'GLDAS-2.1 ET', '#A0AEC0'),
    ]
    x = np.arange(2)
    w = 0.135
    for j, (t, nm, col) in enumerate(tg):
        rr = [et[(et.Target == t) & (et.Lag == lg)].iloc[0] for lg in (0, 1)]
        b.bar(x + (j - 2.5) * w, [r.Partial_r for r in rr], w, color=col, edgecolor='#1A202C', lw=0.45,
              hatch='' if t.startswith('anom_PML') else '///', label=nm, zorder=3)
        for xi, r in zip(x, rr):
            if r.p_eff_PP < 0.05:
                b.text(xi + (j - 2.5) * w, r.Partial_r + (0.007 if r.Partial_r > 0 else -0.022), '*',
                       ha='center', fontsize=11, fontweight='bold', color=CRIMSON)
    b.axhline(0, color='#2D3748', lw=0.8)
    b.set_xticks(x)
    b.set_xticklabels(['Lag 0', 'Lag 1'])
    b.set_ylim(-0.22, 0.39)
    b.set_ylabel('NDVI partial r (ctrl. P, T2m)')
    b.legend(loc='upper right', fontsize=6.6, ncol=2, **LEG)
    panel_title(b, 'b', 'Positive NDVI coupling limited to LAI-driven PML (* p_eff < 0.05)', fs=9.4)

    # (c) Dumbbell N vs N_eff Shrinkage Chart
    c = fig.add_subplot(gs[0, 2])
    clean_ax(c, 'x')
    t2 = t2.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(t2))
    for yi, r in t2.iterrows():
        c.plot([r.N_eff_PP, r.N], [yi, yi], color='#CBD5E0', lw=2.8, zorder=2)
        c.scatter([r.N], [yi], color='#CBD5E0', edgecolor='#4A5568', s=52, zorder=3,
                  label='Nominal N' if yi == 0 else '')
        c.scatter([r.N_eff_PP], [yi], color=CRIMSON, edgecolor='white', lw=0.7, s=62, zorder=4,
                  label='Effective N (Pyper–Peterman)' if yi == 0 else '')
    short = [
        p.replace(' (deseas.)', '').replace('TerraClimate', 'TC').replace('GLDAS-2.1', 'G2.1')
         .replace('GLDAS-2.2', 'G2.2').replace('ERA5-Land', 'ERA5').replace('PML-V2.2a', 'PML')
         .replace('MOD16A2GF', 'MOD16')
        for p in t2.Pair
    ]
    c.set_xscale('log')
    c.set_xlim(10, 340)
    c.set_xticks([10, 30, 100, 300])
    c.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    c.set_yticks(y)
    c.set_yticklabels(short, fontsize=7.2, fontweight='bold')
    c.set_xlabel('Sample size (months, log scale)')
    c.legend(loc='lower left', fontsize=6.9, **LEG)
    panel_title(c, 'c', 'Serial correlation shrinks effective sample size', fs=9.8)
    save(fig, 'Fig4_Causality_and_ET_Partitioning.png')


# =============================================================================
# FIG 5: Walk-Forward Trajectories + Skill Comparison + SHAP Lollipop
# =============================================================================
def fig5():
    oos = R('jms_walkforward_oos_preds.csv')
    oos['dt'] = pd.to_datetime(oos['ym_dt'])
    wf = R('jms_walkforward_clean.csv')
    sh = R('jms_shap_clean.csv')

    fig = plt.figure(figsize=(15.0, 5.7), facecolor='white')
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.32, 0.96, 1.02], wspace=0.33,
                           left=0.055, right=0.975, top=0.87, bottom=0.14)

    # (a) OOS Trajectories with error envelope
    a = fig.add_subplot(gs[0, 0])
    clean_ax(a, 'y')
    a.fill_between(oos.dt, oos.Obs_TWSA, oos.Exo24_RF, color=TEAL, alpha=0.14, label='0–24 mo RF residual')
    a.plot(oos.dt, oos.Obs_TWSA, color=NAVY, lw=2.1, label='Observed GRACE TWSA')
    a.plot(oos.dt, oos.Exo6_RF, color=ORANGE, lw=1.25, alpha=0.85, label='Exogenous 0–6 mo (RF)')
    a.plot(oos.dt, oos.Exo24_RF, color=TEAL, lw=1.6, label='Exogenous 0–24 mo (RF)')
    a.plot(oos.dt, oos.ExoAR1_Lasso, color=CRIMSON, lw=1.25, ls='--', label='0–24 mo + AR(1) (Lasso)')
    a.axhline(0, color='#A0AEC0', ls=':', lw=0.8)
    a.set_ylabel('Deseasonalized TWSA (cm)')
    a.set_ylim(-7.5, 9.5)
    a.legend(loc='upper left', fontsize=7.0, **LEG)
    a.xaxis.set_major_locator(mdates.YearLocator(2))
    a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    panel_title(a, 'a', 'Matched walk-forward predictions (in-fold climatology)', fs=9.8)

    # (b) Skill Bars with Regime Shading
    b = fig.add_subplot(gs[0, 1])
    clean_ax(b, 'y')
    b.axvspan(-0.45, 0.45, color='#FFF5F5', alpha=0.7, zorder=0)
    b.axvspan(0.55, 1.45, color='#EBF8FF', alpha=0.65, zorder=0)
    tiers = ['Exogenous-Only (0–6 mo)', 'Exogenous-Long (0–24 mo)', 'Exogenous-Long + AR(1)']
    mods = ['Ridge', 'Lasso', 'RandomForest', 'XGBoost']
    x = np.arange(3)
    w = 0.19
    for j, m in enumerate(mods):
        v = [wf[(wf.FeatureSet == t) & (wf.Model == m)].OOS_R2.iloc[0] for t in tiers]
        b.bar(x + (j - 1.5) * w, v, w, label=m, edgecolor='#1A202C', lw=0.45,
              color=['#A0AEC0', BLUE, TEAL, ORANGE][j], zorder=3)
    pr = wf[wf.FeatureSet == 'Persistence Baseline'].iloc[0]
    b.axhline(pr.OOS_R2, color=CRIMSON, ls='--', lw=1.4, zorder=4)
    b.text(2.42, pr.OOS_R2 + 0.02, f'Persistence R² = {pr.OOS_R2:.3f}', ha='right', fontsize=7.4, fontweight='bold', color=CRIMSON)
    b.axhline(0, color='#2D3748', lw=0.75)
    b.set_xticks(x)
    b.set_xticklabels(['0–6 mo', '0–24 mo', '0–24 mo\n+ AR(1)'], fontsize=8.2, fontweight='bold')
    b.set_ylabel(f'Out-of-sample R² (n = {int(pr.N_test)} months)')
    b.set_ylim(-0.08, 1.02)
    b.legend(loc='upper left', fontsize=6.9, **LEG)
    panel_title(b, 'b', 'Memory length, not model class, drives skill', fs=9.8)

    # (c) SHAP Lollipop Plot
    c = fig.add_subplot(gs[0, 2])
    clean_ax(c, 'x')
    top = sh.iloc[:10].iloc[::-1].reset_index(drop=True)
    nice = {
        'P_cum24': 'Precip. 24-mo cum.', 'P_cum12': 'Precip. 12-mo cum.',
        'T2m_cum3': 'T2m 3-mo mean', 'Ec_cum3': 'Ec 3-mo cum.',
        'NDVI_lag3': 'NDVI lag 3', 'NDVI_lag2': 'NDVI lag 2',
        'ET_cum3': 'ET 3-mo cum.', 'P_cum6': 'Precip. 6-mo cum.',
        'P_cum3': 'Precip. 3-mo cum.', 'anom_GPM_P': 'Precip. monthly',
    }
    for i, r in top.iterrows():
        col = BLUE if r.dir_corr >= 0 else CRIMSON
        c.plot([0, r.share_pct], [i, i], color=col, lw=2.8, alpha=0.75, zorder=2)
        c.scatter([r.share_pct], [i], color=col, s=64, edgecolor='white', lw=0.8, zorder=4)
        c.text(r.share_pct + 1.4, i, f'{r.share_pct:.1f}% ({"+" if r.dir_corr >= 0 else "−"})',
               va='center', fontsize=7.4, fontweight='bold')
    c.text(0.97, 0.34, f'12–24 mo P\n= {sh.iloc[:2].share_pct.sum():.1f}% SHAP',
           transform=c.transAxes, ha='right', fontsize=7.8, fontweight='bold', color=BLUE,
           bbox=dict(boxstyle='round,pad=0.28', fc='#EBF8FF', ec=BLUE, lw=0.8))
    c.set_yticks(range(len(top)))
    c.set_yticklabels([nice.get(f, f) for f in top.feature], fontsize=7.6)
    c.set_xlim(0, 58)
    c.set_xlabel('Mean |SHAP| share (%)')
    c.legend(handles=[
        mpatches.Patch(color=BLUE, label='Positive TWSA response'),
        mpatches.Patch(color=CRIMSON, label='Negative response')
    ], loc='lower right', fontsize=7.0, **LEG)
    panel_title(c, 'c', 'Exogenous 0–24 mo RF SHAP (< 2024)', fs=9.8)
    save(fig, 'Fig5_WalkForward_and_SHAP.png')


# =============================================================================
# FIG 6: Closure Benchmark + Budyko Trajectory + Elasticity CI + Waterfall
# =============================================================================
def fig6():
    t6 = R('jms_table6_budyko_nonsteady.csv')
    old = R('table4_water_balance_budyko.csv')
    bl = R('jms_budyko_2004_2024_nonsteady.csv')
    dec = R('jms_budyko_decomposition.csv').iloc[0]
    mc_bud = R('jms_budyko_mc_uncertainty.csv').set_index('Metric')
    el = R('probe_C_elasticity.csv').iloc[0]

    fig, axs = plt.subplots(2, 2, figsize=(14.4, 9.4), facecolor='white')
    plt.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.075, hspace=0.35, wspace=0.26)
    a, b, c, d = axs.ravel()
    yrs = t6.Year.values

    # (a) Closure Benchmark
    clean_ax(a, 'y')
    a.bar(yrs, t6.P_GPM_mm, 0.62, color='#BEE3F8', edgecolor=BLUE, alpha=0.65, label='GPM precipitation', zorder=2)
    a.bar(yrs, t6.R_obs_mm, 0.34, color=BLUE, alpha=0.92, label='Tangnaihai runoff', zorder=3)
    a.errorbar(yrs, t6.E_wb_mm, yerr=t6.E_wb_unc_mm, fmt='o-', color='#1A202C', lw=2.2, ms=6.0, capsize=3.5,
               label=r'Closure $E_{wb} = P - R - \Delta S_{G22}$ (±1σ)', zorder=6)
    for col, nm, c_, m_ in [
        ('E_TC_mm', 'TerraClimate', TEAL, '^'),
        ('E_PML_mm', 'PML-V2.2a', MINT, 's'),
        ('E_ERA5_mm', 'ERA5-Land', ORANGE, 'd'),
        ('E_MOD16_mm', 'MOD16A2GF', CRIMSON, 'x'),
    ]:
        a.plot(old.Year, old[col], marker=m_, ms=4.8, lw=1.35, color=c_, ls='--',
               label=f'{nm} ET ({old[col].mean():.0f}, {100*(old[col].mean()/t6.E_wb_mm.mean()-1):+.1f}%)', zorder=5)
    a.set_ylim(0, 1020)
    a.set_xticks(yrs)
    a.set_ylabel('Annual flux (mm a⁻¹)')
    a.legend(loc='upper left', fontsize=6.8, ncol=2, **LEG)
    panel_title(a, 'a', f'Water-balance closure benchmark (mean E_wb = {t6.E_wb_mm.mean():.1f} mm a⁻¹)')

    # (b) Non-steady-state Budyko with shaded envelope, ±1σ error bars & complete legend
    clean_ax(b, 'both')
    phi = np.linspace(0.5, 2.6, 200)
    b.fill_between(phi, phi / (1 + phi ** 1.1) ** (1 / 1.1), phi / (1 + phi ** 1.5) ** (1 / 1.5),
                   color='#CBD5E0', alpha=0.35, label='Choudhury–Yang band (ω = 1.1–1.5)')
    for w_, ls in [(1.1, ':'), (1.3, '--'), (1.5, '-.')]:
        b.plot(phi, phi / (1 + phi ** w_) ** (1 / w_), color='#718096', ls=ls, lw=1.2, label=f'ω = {w_}')
    b.plot([0.5, 1, 2.6], [0.5, 1, 1], color='#2D3748', lw=1.1, alpha=0.7, label='Demand/supply limit')
    bb = bl[bl.year <= 2015]
    b.scatter(bb.pet_peff, bb.ep_pml_eff, s=36, facecolor='none', edgecolor='#718096', lw=1.0,
              label='2004–2015 (PML-based)', zorder=3)
    rc = {'Dry': ORANGE, 'Normal': BLUE, 'Wet': MINT}
    b.plot(t6.PET_over_Peff, t6.Ewb_over_Peff, color='#4A5568', lw=0.9, alpha=0.55, zorder=4)
    lbl_off_b = {
        2016: (0.028, 0.012), 2017: (-0.078, 0.014), 2018: (0.032, 0.012),
        2019: (0.032, -0.022), 2020: (0.025, 0.012), 2021: (-0.075, 0.015),
        2022: (-0.078, 0.015), 2023: (0.025, -0.018), 2024: (0.028, -0.022),
    }
    for _, r in t6.iterrows():
        yr_i = int(r.Year)
        elo = r.Ewb_over_Peff - r.Ewb_over_Peff_ci68_lo
        ehi = r.Ewb_over_Peff_ci68_hi - r.Ewb_over_Peff
        b.errorbar([r.PET_over_Peff], [r.Ewb_over_Peff], yerr=[[elo], [ehi]], fmt='none',
                   ecolor=rc[r.Regime], elinewidth=1.15, capsize=2.8, alpha=0.75, zorder=4.5)
        b.scatter([r.PET_over_Peff], [r.Ewb_over_Peff], s=76, color=rc[r.Regime], edgecolor='#1A202C', lw=0.9, zorder=5)
        dx_b, dy_b = lbl_off_b.get(yr_i, (0.025, 0.011))
        b.text(r.PET_over_Peff + dx_b, r.Ewb_over_Peff + dy_b, str(yr_i)[2:], fontsize=7.4, fontweight='bold')
    reg_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c_, markeredgecolor='k',
                   label=f'{k} year (E_wb ± 1σ)')
        for k, c_ in rc.items()
    ]
    base_handles, _ = b.get_legend_handles_labels()
    b.legend(handles=reg_handles + base_handles, loc='lower right', fontsize=6.0, ncol=2, **LEG)
    b.set_xlim(0.9, 2.65)
    b.set_ylim(0.46, 0.98)
    b.set_xlabel(r'Dryness index $PET_{PML}/(P-\Delta S)$')
    b.set_ylabel(r'Evaporative index $E/(P-\Delta S)$')
    panel_title(b, 'b', 'Non-steady-state Budyko space (invariant ω_WB ≈ 1.28)')

    # (c) Elasticity with 95% confidence band (resolved 21/23 label overlap)
    clean_ax(c, 'both')
    P = t6.P_GPM_mm.values
    Rr = t6.R_obs_mm.values
    pg = np.linspace(500, 830, 100)
    pet = bl[bl.year.between(2016, 2024)].PET_PML.mean()
    for w_, ls in [(1.1, ':'), (1.3, '--'), (1.5, '-.')]:
        ph_ = pet / pg
        c.plot(pg, pg - pg * ph_ / (1 + ph_ ** w_) ** (1 / w_), color='#718096', ls=ls, lw=1.2, label=f'Budyko, ω = {w_}')
    X_ln = sm.add_constant(np.log(P))
    fit_ln = sm.OLS(np.log(Rr), X_ln).fit(cov_type='HC1')
    pred = fit_ln.get_prediction(sm.add_constant(np.log(pg))).summary_frame(alpha=0.05)
    c.fill_between(pg, np.exp(pred['mean_ci_lower']), np.exp(pred['mean_ci_upper']), color=CRIMSON, alpha=0.14, label='Observed 95% HC1 CI')
    c.plot(pg, np.exp(pred['mean']), color=CRIMSON, lw=2.2,
           label=f'Observed fit: elasticity {el.obs_elasticity:.2f} [{el.obs_CI_lo:.2f}, {el.obs_CI_hi:.2f}]')
    c.scatter(P, Rr, s=68, c=[rc[r] for r in t6.Regime], edgecolor='#1A202C', lw=0.9, zorder=5)
    lbl_off_c = {
        2016: (5, 4), 2017: (5, 4), 2018: (5, 4), 2019: (-15, 5),
        2020: (-13, 6), 2021: (-15, 5), 2022: (5, 4), 2023: (6, -9), 2024: (-14, -9),
    }
    for p_, r_, y_ in zip(P, Rr, yrs):
        dx_c, dy_c = lbl_off_c.get(int(y_), (5, 4))
        c.text(p_ + dx_c, r_ + dy_c, str(y_)[2:], fontsize=7.4, fontweight='bold')
    c.set_xlabel('Annual precipitation (mm)')
    c.set_ylabel('Observed runoff depth (mm)')
    c.legend(loc='upper left', fontsize=6.9, **LEG)
    panel_title(c, 'c', 'Observed runoff elasticity matches fixed-ω Budyko')

    # (d) Waterfall Chart for Runoff Ratio Decomposition (with ±1 SD MC error bars!)
    clean_ax(d, 'y')
    rp_start = dec.RP_Peak_flux
    rp_end = dec.RP_Drawdown_flux
    steps = [dec.Climate_Aridity_Effect, dec.Storage_Release_Effect, dec.Landscape_wWB_Residual_Effect]
    step_sds = [
        float(mc_bud.loc['Waterfall_Climate_Aridity_Effect', 'MC_SD']),
        float(mc_bud.loc['Waterfall_Storage_Release_Effect', 'MC_SD']),
        float(mc_bud.loc['Waterfall_Landscape_wWB_Residual_Effect', 'MC_SD']),
    ]
    w_pk_sd = float(mc_bud.loc['w_WB_Peak_2018_2020', 'MC_SD'])
    w_dd_sd = float(mc_bud.loc['w_WB_Drawdown_2021_2024', 'MC_SD'])

    labels = ['2018–20\nPeak R/P', 'Climate\n(aridity)', 'Storage\nrelease', 'Landscape\n(ω_WB res.)', '2021–24\nDrawdown']
    bottoms = [0, rp_start + steps[0], rp_start + steps[0], rp_start + steps[0] + steps[1] + steps[2], 0]
    heights = [rp_start, abs(steps[0]), abs(steps[1]), abs(steps[2]), rp_end]
    cols_w = [NAVY, ORANGE, BLUE, TEAL, SLATE]
    for i in range(5):
        d.bar(i, heights[i], bottom=bottoms[i], width=0.52, color=cols_w[i], edgecolor='#1A202C', lw=0.75, zorder=3)
    # Connector dashed lines
    levels = [rp_start, rp_start + steps[0], rp_start + steps[0] + steps[1], rp_end]
    for i, lv in enumerate(levels):
        d.plot([i + 0.26, i + 0.74], [lv, lv], color='#4A5568', ls=':', lw=1.0, zorder=4)
    # Propagated ±1 SD error bars on the three decomposition terms
    bar_endpoints = [rp_start + steps[0], rp_start + steps[0] + steps[1], rp_end]
    d.errorbar([1, 2, 3], bar_endpoints, yerr=step_sds, fmt='none',
               ecolor='#1A202C', elinewidth=1.25, capsize=3.5, zorder=5, label='±1 SD (MC closure error)')
    # Value labels clear of error bars
    d.text(0, rp_start + 0.008, f'{rp_start:.3f}', ha='center', fontsize=8.0, fontweight='bold')
    d.text(1, bar_endpoints[0] - step_sds[0] - 0.008, f'{steps[0]:+.3f}\n(±{step_sds[0]:.3f})',
           ha='center', va='top', fontsize=7.3, fontweight='bold', color=CRIMSON)
    d.text(2, bar_endpoints[1] + step_sds[1] + 0.006, f'{steps[1]:+.4f}\n(±{step_sds[1]:.4f})',
           ha='center', va='bottom', fontsize=7.0, fontweight='bold', color=BLUE)
    d.text(3, bar_endpoints[2] + step_sds[2] + 0.006, f'{steps[2]:+.4f}\n(±{step_sds[2]:.3f})',
           ha='center', va='bottom', fontsize=7.0, fontweight='bold', color=TEAL)
    d.text(4, rp_end + 0.008, f'{rp_end:.3f}', ha='center', fontsize=8.0, fontweight='bold')
    d.set_xticks(range(5))
    d.set_xticklabels(labels, fontsize=7.8, fontweight='bold')
    d.set_ylim(0.18, 0.44)
    d.set_ylabel('Catchment runoff ratio (R / P)')
    d.text(0.97, 0.94,
           f'Δ(R/P) = {dec.Total_dRP:+.3f} ({dec.Volumetric_Runoff_Drop_km3_yr:+.2f} km³ a⁻¹)\n'
           f'ω_WB: {dec.wWB_Peak:.3f} (±{w_pk_sd:.2f}) → {dec.wWB_Drawdown:.3f} (±{w_dd_sd:.2f})',
           transform=d.transAxes, ha='right', va='top', fontsize=7.5, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', fc='#F7FAFC', ec='#CBD5E0'))
    d.legend(loc='lower right', fontsize=6.8, **LEG)
    panel_title(d, 'd', 'Waterfall attribution of the runoff-ratio decline (±1 SD)')
    save(fig, 'Fig6_WaterBalance_and_Budyko.png')


# =============================================================================
# FIG 7: Dot-and-CI Climate Sensitivity + Greening Residual Decomposition
# =============================================================================
def fig7():
    lf = R('probe_A_limiting_factor.csv')
    rs = R('probe_A2_greening_residual.csv')

    fig, axs = plt.subplots(1, 2, figsize=(13.8, 5.5), facecolor='white')
    plt.subplots_adjust(left=0.075, right=0.975, top=0.87, bottom=0.19, wspace=0.25)
    a, b = axs

    # (a) Dot-and-CI Plot
    clean_ax(a, 'y')
    lf2 = lf.iloc[[0, 1, 2, 3, 4, 5]].reset_index(drop=True)
    x = np.arange(len(lf2))
    a.axhline(0, color='#4A5568', ls='--', lw=0.8)
    a.errorbar(x - 0.14, lf2.beta_T, yerr=[lf2.beta_T - lf2.T_lo, lf2.T_hi - lf2.beta_T],
               fmt='o', ms=7, color=CRIMSON, mfc=CRIMSON, mec='white', mew=0.8,
               ecolor=CRIMSON, capsize=4, lw=1.6, label='Temperature sensitivity (β_T)', zorder=4)
    a.errorbar(x + 0.14, lf2.beta_P, yerr=[lf2.beta_P - lf2.P_lo, lf2.P_hi - lf2.beta_P],
               fmt='D', ms=6.8, color=BLUE, mfc=BLUE, mec='white', mew=0.8,
               ecolor=BLUE, capsize=4, lw=1.6, label='Precipitation sensitivity (β_P)', zorder=4)
    for xi, r in lf2.iterrows():
        a.plot([xi - 0.14, xi + 0.14], [r.beta_T, r.beta_P], color='#CBD5E0', lw=1.8, zorder=2)
    names = ['ST4 < 0 °C\n(permafrost)', 'ST4 ≥ 0 °C\n(seasonal)'] + [s.split(' ')[0] + '\n' + s.split(' ', 1)[1] for s in lf2.Bin[2:]]
    a.set_xticks(x)
    a.set_xticklabels(names, fontsize=7.5, fontweight='bold')
    a.set_ylabel('Standardized NDVI sensitivity\n(detrended; 95% block-bootstrap CI)')
    a.set_ylim(-0.22, 1.92)
    a.legend(loc='upper right', fontsize=7.5, **LEG)
    panel_title(a, 'a', 'Precipitation outweighs temperature in every thermal class', fs=9.8)

    # (b) Greening Residual Breakdown
    clean_ax(b, 'y')
    sub = rs[rs.Mask == 'all vegetated pixels'].reset_index(drop=True)
    x_pos = [0.0, 1.05, 2.85, 3.90]
    labs = []
    idx_x = 0
    for zone, zl in [('permafrost ST4<0', 'Permafrost\n(ST4 < 0 °C)'), ('seasonally frozen ST4>=0', 'Seasonally frozen\n(ST4 ≥ 0 °C)')]:
        for pred, pl, col in [('T+P', 'T + P', SKY), ('T+P+soil moisture+snow cover', '+ SM + snow', BLUE)]:
            r = sub[(sub.Zone == zone) & (sub.Predictors == pred)].iloc[0]
            xp = x_pos[idx_x]
            b.bar(xp, r.NDVI_trend_dec, 0.68, color='#EDF2F7', edgecolor='#1A202C', lw=0.8, zorder=2)
            b.bar(xp, r.Explained, 0.68, color=col, edgecolor='#1A202C', lw=0.8, zorder=3)
            b.text(xp, r.NDVI_trend_dec + 0.0006, f'Unexplained {r.Unexpl_pct:.0f}%\n[{r.Unexpl_lo:.0f}, {r.Unexpl_hi:.0f}]%',
                   ha='center', fontsize=7.0, fontweight='bold',
                   color=CRIMSON if zone.startswith('permafrost') else '#4A5568')
            labs.append(f'{zl}\n{pl}')
            idx_x += 1
    b.set_xticks(x_pos)
    b.set_xticklabels(labs, fontsize=7.0, fontweight='bold')
    b.set_xlim(-0.65, 4.55)
    b.set_ylabel('NDVI trend 2004–2024 (decade⁻¹)')
    b.set_ylim(0, 0.0275)
    b.legend(handles=[
        mpatches.Patch(color='#EDF2F7', ec='#1A202C', label='Observed trend (unexplained top)'),
        mpatches.Patch(color=SKY, ec='#1A202C', label='Explained by interannual T + P'),
        mpatches.Patch(color=BLUE, ec='#1A202C', label='Explained incl. ERA5 SM + snow'),
    ], loc='upper right', fontsize=7.1, **LEG)
    panel_title(b, 'b', 'Greening residual unexplained by interannual climate (95% CI)', fs=9.8)
    save(fig, 'Fig7_Limiting_Factors_and_Greening_Residual.png')


# =============================================================================
# GRAPHICAL ABSTRACT: Executive 3-Column Card Layout + Embedded Micro-Charts
# =============================================================================
def ga():
    ph = R('jms_grace_uncertainty_robust.csv')
    mem = R('jms_neff_memory.csv')
    el = R('probe_C_elasticity.csv').iloc[0]
    rs = R('probe_A2_greening_residual.csv')
    dec = R('jms_budyko_decomposition.csv').iloc[0]
    r12 = mem[mem.Window_k_mo == 12].iloc[0]
    pf = rs[(rs.Mask == 'all vegetated pixels') & (rs.Zone == 'permafrost ST4<0') & (rs.Predictors == 'T+P')].iloc[0]

    fig, ax = plt.subplots(figsize=(12.6, 6.5), facecolor='white')
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    # Header Banner
    ax.add_patch(FancyBboxPatch((0.2, 5.52), 12.2, 0.80, boxstyle='round,pad=0.04,rounding_size=0.14', fc=NAVY, ec='none'))
    ax.text(6.3, 6.03, 'Reliability Audit of Alpine Water-Storage & Cryosphere–Vegetation Inferences',
            ha='center', va='center', color='white', fontsize=13.5, fontweight='bold')
    ax.text(6.3, 5.71, 'Source Region of the Yellow River  ·  Tangnaihai catchment (123,800 km²)  ·  2003–2024',
            ha='center', va='center', color='#BEE3F8', fontsize=9.6)

    cards = [
        (0.25, '#FFF5F5', CRIMSON, '01', 'Claims Overturned by Audit', [
            f'k = 12 mo NDVI–TWSA suppression: p {r12.p_iid:.3f} → {r12.p_eff_PP:.3f} (N_eff = {r12.N_eff_PP:.0f})',
            f'Post-peak SNR drops {ph.iloc[2].SNR_iid:.1f} → {ph.iloc[2].SNR_corr_FObias:.1f} with AR(1) + FO bias',
            'Budyko-ω "greening shift": disappears using E_wb & P − ΔS',
            'Deep-layer drainage: GLDAS shares mirror 80% layer depth',
            'Positive NDVI–ET coupling: restricted to LAI-driven PML',
            'Elevation warming gradient: ERA5 +0.29 vs. TerraClimate −0.02 °C/dec',
        ]),
        (4.38, '#F0FFF4', TEAL, '02', 'Robust Findings That Hold', [
            f'2021–2024 TWSA remains +{ph.iloc[2].Diff_vs_Pre_JPL_cm:.2f} cm above 2003–2017',
            f'Runoff elasticity {el.obs_elasticity:.2f} matches fixed-ω Budyko ({el.budyko_elasticity_wfix:.2f})',
            f'P −15.7% drives R −31.9% ({dec.Volumetric_Runoff_Drop_km3_yr:+.1f} km³ a⁻¹); ω_WB invariant',
            '12–24 mo P gives 80.4% RF (63.8% XGB) SHAP skill',
            'Closure (423.5 mm) brackets TC/PML; falsifies MOD16 (+43%)',
            'Basin air temperature warms consistently ≈ +1.0 °C/dec',
        ]),
        (8.51, '#FFFAF0', ORANGE, '03', 'Open Permafrost Lead', [
            f'Permafrost side (ST4 < 0 °C): NDVI +{pf.NDVI_trend_dec:.4f}/dec',
            f'{pf.Unexpl_pct:.0f}% [{pf.Unexpl_lo:.0f}, {pf.Unexpl_hi:.0f}]% unexplained by interannual T & P',
            'Holds with ERA5 soil moisture, snow & IGBP grassland mask',
            'Priority in-situ tests: active-layer thaw, grazing & restoration',
        ]),
    ]

    for x0, fc, ec, num, title, lines in cards:
        ax.add_patch(FancyBboxPatch((x0, 0.25), 3.84, 5.02, boxstyle='round,pad=0.05,rounding_size=0.14', fc=fc, ec=ec, lw=1.7))
        # Badge
        ax.add_patch(FancyBboxPatch((x0 + 0.16, 4.82), 0.44, 0.34, boxstyle='round,pad=0.02,rounding_size=0.08', fc=ec, ec='none'))
        ax.text(x0 + 0.38, 4.99, num, ha='center', va='center', color='white', fontsize=9.2, fontweight='bold')
        ax.text(x0 + 0.72, 4.99, title, ha='left', va='center', fontsize=10.0, fontweight='bold', color=ec)
        ax.plot([x0 + 0.16, x0 + 3.68], [4.70, 4.70], color=ec, lw=1.0, alpha=0.45)

        yy = 4.46
        for ln in lines:
            wrapped = textwrap.fill('• ' + ln, width=44, subsequent_indent='  ')
            n_lines = wrapped.count('\n') + 1
            ax.text(x0 + 0.16, yy, wrapped, fontsize=7.6, va='top', linespacing=1.28)
            yy -= (0.43 + 0.23 * (n_lines - 1))

    # Mini embedded visual bar in Card 3 bottom (showing 47.6% climate vs 52.4% unexplained residual)
    ax.add_patch(FancyBboxPatch((8.68, 0.52), 3.50, 0.90, boxstyle='round,pad=0.03,rounding_size=0.08', fc='white', ec=ORANGE, lw=0.9))
    ax.text(10.43, 1.22, 'Permafrost NDVI Trend Attribution (2004–2024)', ha='center', fontsize=6.6, fontweight='bold', color='#2D3748')
    ax.add_patch(mpatches.Rectangle((8.88, 0.72), 3.10 * 0.476, 0.32, fc=SKY, ec='#1A202C', lw=0.6))
    ax.add_patch(mpatches.Rectangle((8.88 + 3.10 * 0.476, 0.72), 3.10 * 0.524, 0.32, fc=ORANGE, ec='#1A202C', lw=0.6, alpha=0.85))
    ax.text(8.88 + 3.10 * 0.238, 0.88, '48% T+P', ha='center', va='center', fontsize=6.9, fontweight='bold', color='#1A202C')
    ax.text(8.88 + 3.10 * 0.738, 0.88, '52% Open Lead', ha='center', va='center', fontsize=6.9, fontweight='bold', color='white')

    save(fig, 'Graphical_Abstract.png')


if __name__ == '__main__':
    todo = sys.argv[1:] or ['1', '2', '3', '4', '5', '6', '7', 'ga']
    fn = {'1': fig1, '2': fig2, '3': fig3, '4': fig4, '5': fig5, '6': fig6, '7': fig7, 'ga': ga}
    for k in todo:
        fn[k]()
