#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
36_jms_english_figures.py
=========================
Generates all 600-DPI English publication figures (Fig. 1 - Fig. 6 + Graphical Abstract)
for Journal of Mountain Science (JMS) submission from verified results_repaired/ datasets.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from PIL import Image

OUT_DIR = '/Users/yueliangmanle/Desktop/小论文中文2/JMS_Submission_Package/Figures'
RES_DIR = '/Users/yueliangmanle/Desktop/小论文中文2/results_repaired'
GEO_PATH = '/Users/yueliangmanle/Desktop/小论文中文2/高级图设计包/fig1_研究区/yr_source_boundary.geojson'
HILL_PATH = '/Users/yueliangmanle/Desktop/小论文中文2/高级图设计包/fig1_研究区/dem_hillshade_base.png'
os.makedirs(OUT_DIR, exist_ok=True)

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['axes.linewidth'] = 0.85
mpl.rcParams['xtick.major.width'] = 0.75
mpl.rcParams['ytick.major.width'] = 0.75
mpl.rcParams['xtick.labelsize'] = 8.8
mpl.rcParams['ytick.labelsize'] = 8.8
mpl.rcParams['axes.titlesize'] = 10.2
mpl.rcParams['axes.labelsize'] = 9.3

NAVY = '#1B365D'
BLUE = '#2B6CB0'
TEAL = '#1B7F6E'
ORANGE = '#D95F02'
CRIMSON = '#C53030'
GOLD = '#B7791F'
PURPLE = '#6B46C1'
SLATE = '#4A5568'


def style_ax(ax, grid_axis='y'):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, linestyle='--', linewidth=0.5, alpha=0.35, color='#CBD5E0')
    ax.set_axisbelow(True)


def load_polygon():
    with open(GEO_PATH, 'r') as f:
        gj = json.load(f)
    if gj['type'] == 'Polygon':
        return [np.array(gj['coordinates'][0])]
    elif gj['type'] == 'MultiPolygon':
        return [np.array(p[0]) for p in gj['coordinates']]
    return []


def polygon_mask(rings, lon_2d, lat_2d):
    pts = np.column_stack([lon_2d.ravel(), lat_2d.ravel()])
    inside = np.zeros(len(pts), dtype=bool)
    for ring in rings:
        inside |= MplPath(ring[:, :2]).contains_points(pts)
    return inside.reshape(lon_2d.shape)


# =============================================================================
# FIGURE 1: Study Area Elevation Belts & Spatial GRACE TWSA Trends
# =============================================================================
def plot_fig1():
    rings = load_polygon()
    hill = np.array(Image.open(HILL_PATH).convert('L'))
    terrain = np.load(os.path.join(RES_DIR, 'jms_terrain_grid.npz'))
    dem = terrain['dem']
    upa = terrain['upa']
    xmin, xmax, ymin, ymax = terrain['extent']

    ny, nx = dem.shape
    lons = np.linspace(xmin + (xmax - xmin) / (2 * nx), xmax - (xmax - xmin) / (2 * nx), nx)
    lats = np.linspace(ymax - (ymax - ymin) / (2 * ny), ymin + (ymax - ymin) / (2 * ny), ny)
    lon2d, lat2d = np.meshgrid(lons, lats)
    mask = polygon_mask(rings, lon2d, lat2d)

    zclass = np.full_like(dem, np.nan, dtype=float)
    valid = mask & (dem > 0)
    zclass[valid & (dem < 3800)] = 0
    zclass[valid & (dem >= 3800) & (dem < 4300)] = 1
    zclass[valid & (dem >= 4300)] = 2

    df_cells = pd.read_csv(os.path.join(RES_DIR, 'grace_grid_trends.csv'))
    cells_in = df_cells[df_cells['intersects'] == True].copy()
    elev_sum = pd.read_csv(os.path.join(RES_DIR, 'jms_elevation_bands_summary.csv'))
    area_rows = elev_sum[['Zone', 'Area_km2', 'Area_pct']].drop_duplicates().copy()

    fig = plt.figure(figsize=(14.4, 7.6), facecolor='white')
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.44, 1.0], height_ratios=[1.0, 1.0],
                           wspace=0.22, hspace=0.30, left=0.055, right=0.955, top=0.93, bottom=0.08)

    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_xlim(95.5, 103.5)
    ax1.set_ylim(32.0, 36.5)

    ax1.imshow(hill, extent=[95.5, 103.5, 32.0, 36.5], cmap='gray', alpha=0.40, aspect='auto', zorder=1)
    cmap_belts = ListedColormap(['#E5C494', '#7FC97F', '#386CB0'])
    norm_belts = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_belts.N)
    ax1.imshow(np.ma.masked_invalid(zclass), extent=[xmin, xmax, ymin, ymax],
               cmap=cmap_belts, norm=norm_belts, alpha=0.68, aspect='auto', origin='upper', zorder=2)

    riv_sub = np.where(mask & (upa >= 1000) & (upa < 5500), 1.0, np.nan)
    riv_main = np.where(mask & (upa >= 5500), 1.0, np.nan)
    ax1.imshow(riv_sub, extent=[xmin, xmax, ymin, ymax], cmap=ListedColormap(['#2B6CB0']),
               alpha=0.65, aspect='auto', origin='upper', zorder=3)
    ax1.imshow(riv_main, extent=[xmin, xmax, ymin, ymax], cmap=ListedColormap(['#1A365D']),
               alpha=0.95, aspect='auto', origin='upper', zorder=3.5)

    for ring in rings:
        ax1.plot(ring[:, 0], ring[:, 1], color='#1A202C', lw=1.6, zorder=4)

    ax1.plot([95.9, 102.0], [34.5, 34.5], linestyle='--', color=CRIMSON, lw=1.5, zorder=5)
    ax1.text(97.25, 34.60, 'NW Permafrost Subzone (≥34.5°N)', color=CRIMSON,
             fontsize=8.3, fontweight='bold', ha='left', va='bottom',
             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.80), zorder=6)
    ax1.text(100.35, 33.42, 'SE Wetland–Meadow\nSubzone (<34.5°N)', color='#742A2A',
             fontsize=8.3, fontweight='bold', ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='none', alpha=0.80), zorder=6)

    # Tangnaihai outlet gauge placed cleanly NW of legend
    ax1.scatter([100.15], [35.50], marker='*', s=210, color='#E53E3E', edgecolor='white', lw=0.9, zorder=7)
    ax1.annotate('Tangnaihai Gauge\n(100.15°E, 35.50°N;\n123,800 km²)', xy=(100.15, 35.50),
                 xytext=(101.20, 34.92), fontsize=8.0, fontweight='bold', color='#1A202C',
                 bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#CBD5E0', lw=0.8, alpha=0.93),
                 arrowprops=dict(arrowstyle='->', color='#2D3748', lw=1.05), zorder=8)

    patches = [
        mpatches.Patch(color='#E5C494', label='<3,800 m (Lower valley, 28.0%)'),
        mpatches.Patch(color='#7FC97F', label='3,800–4,300 m (Alpine meadow, 30.3%)'),
        mpatches.Patch(color='#386CB0', label='≥4,300 m (Cryosphere belt, 41.8%)'),
        plt.Line2D([0], [0], color='#1A365D', lw=1.8, label='Yellow River mainstem & tributaries'),
    ]
    ax1.legend(handles=patches, loc='upper right', fontsize=7.6, frameon=True,
               facecolor='white', edgecolor='#CBD5E0', framealpha=0.94,
               title='MERIT DEM Elevation Belts (Mean 4,116.8 m)', title_fontsize=7.8)

    # Inset cleanly shifted inward so labels never touch main map spines or ticks
    ax_in = ax1.inset_axes([0.075, 0.085, 0.325, 0.245])
    ax_in.set_facecolor((1, 1, 1, 0.94))
    order_zones = ['<3800m', '3800-4300m', '>=4300m']
    lbls_short = ['<3,800 m', '3,800–\n4,300 m', '≥4,300 m']
    pcts = [float(area_rows.loc[area_rows['Zone'] == z, 'Area_pct'].iloc[0]) for z in order_zones]
    cols_in = ['#E5C494', '#7FC97F', '#386CB0']
    ax_in.bar(range(3), pcts, color=cols_in, edgecolor='#2D3748', linewidth=0.6, width=0.55)
    for bi, pval in enumerate(pcts):
        ax_in.text(bi, pval + 1.5, f'{pval:.1f}%', ha='center', va='bottom', fontsize=7.3, fontweight='bold')
    ax_in.set_xticks(range(3))
    ax_in.set_xticklabels(lbls_short, fontsize=6.9)
    ax_in.set_ylim(0, 54)
    ax_in.set_ylabel('Area (%)', fontsize=7.3)
    ax_in.set_title('Hypsometric Structure\n(2,572–6,221 m a.s.l.)', fontsize=7.5, fontweight='bold', pad=3)
    ax_in.spines['top'].set_visible(False)
    ax_in.spines['right'].set_visible(False)
    ax_in.tick_params(axis='both', labelsize=6.9)

    ax1.annotate('N', xy=(103.05, 32.80), xytext=(103.05, 32.35), ha='center', va='center',
                 fontsize=10, fontweight='bold', color='#1A202C',
                 arrowprops=dict(arrowstyle='-|>', color='#1A202C', lw=1.6, mutation_scale=14), zorder=8)
    ax1.plot([101.55, 102.58], [32.22, 32.22], color='#1A202C', lw=2.2, zorder=8)
    ax1.text(102.065, 32.29, '100 km', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1A202C')

    ax1.set_xticks([96, 98, 100, 102])
    ax1.set_xticklabels(['96°E', '98°E', '100°E', '102°E'])
    ax1.set_yticks([32, 33, 34, 35, 36])
    ax1.set_yticklabels(['32°N', '33°N', '34°N', '35°N', '36°N'])
    ax1.set_title('(a) SRYR Topography, Elevation Belts & River Network', fontweight='bold', loc='left', pad=8)

    def plot_mascon_panel(ax, col_val, col_p, title, vlim, cmap_name, add_note):
        ax.set_xlim(95.5, 103.6)
        ax.set_ylim(32.0, 36.5)
        cmap = plt.get_cmap(cmap_name)
        norm = TwoSlopeNorm(vmin=vlim[0], vcenter=0.0, vmax=vlim[1])
        d = 0.50306
        for _, row in cells_in.iterrows():
            val = float(row[col_val])
            pval = float(row[col_p])
            x0, y0 = float(row['lon']) - d / 2, float(row['lat']) - d / 2
            rect = mpatches.Rectangle((x0, y0), d, d, facecolor=cmap(norm(val)),
                                      edgecolor='#718096', linewidth=0.35, alpha=0.92, zorder=2)
            ax.add_patch(rect)
            if pval < 0.05:
                ax.scatter([float(row['lon'])], [float(row['lat'])], s=9, color='#1A202C', marker='.', zorder=4)
        for ring in rings:
            ax.plot(ring[:, 0], ring[:, 1], color='#1A202C', lw=1.4, zorder=5)
        ax.plot([95.9, 102.0], [34.5, 34.5], linestyle='--', color='#2D3748', lw=1.1, zorder=5)
        ax.scatter([100.15], [35.50], marker='*', s=110, color='#1A202C', edgecolor='white', lw=0.6, zorder=6)

        sm_m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm_m.set_array([])
        cbar = fig.colorbar(sm_m, ax=ax, orientation='vertical', fraction=0.044, pad=0.03)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(r'Trend ($\mathrm{cm\ a^{-1}}$)', fontsize=8.3)

        ax.text(0.03, 0.05, add_note + '\n(Dots: $p$ < 0.05; 76 intersected cells)', transform=ax.transAxes,
                fontsize=7.5, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.24', fc='white', ec='#CBD5E0', alpha=0.92), zorder=7)
        ax.set_xticks([96, 98, 100, 102])
        ax.set_xticklabels(['96°E', '98°E', '100°E', '102°E'])
        ax.set_yticks([32, 34, 36])
        ax.set_yticklabels(['32°N', '34°N', '36°N'])
        ax.set_title(title, fontweight='bold', loc='left', pad=7)

    plot_mascon_panel(
        fig.add_subplot(gs[0, 1]), 'slope_full', 'p_full',
        '(b) Full-Period GRACE TWSA Trend (2003–2024)',
        (-0.5, 0.9), 'RdBu', 'Basin: +0.190 cm a⁻¹\nNW: +0.392 | SE: +0.078'
    )
    plot_mascon_panel(
        fig.add_subplot(gs[1, 1]), 'slope_post2018', 'p_post',
        '(c) Post-Transition GRACE Drawdown (2018–2024)',
        (-1.05, 0.35), 'RdBu', 'Basin: −0.663 cm a⁻¹\nNW: −0.855 | SE: −0.557'
    )

    out = os.path.join(OUT_DIR, 'Fig1_StudyArea_and_GRACE_Spatial.png')
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print('Saved:', out)


# =============================================================================
# FIGURE 2: Multi-Scale Temporal Evolution & Abrupt Transition (2003-2024)
# =============================================================================
def plot_fig2():
    df = pd.read_csv(os.path.join(RES_DIR, 'basin_merged_monthly_2003_2024.csv'))
    df['dt'] = pd.to_datetime(df['ym_dt'])
    df['t'] = df['t_yr']

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.0), facecolor='white', sharex=True)
    plt.subplots_adjust(left=0.065, right=0.935, top=0.94, bottom=0.07, hspace=0.28, wspace=0.24)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    gap_start = pd.Timestamp('2017-07-01')
    gap_end = pd.Timestamp('2018-05-31')
    split_dt = pd.Timestamp('2017-05-01')

    def decorate_time(ax):
        ax.axvspan(gap_start, gap_end, color='#E2E8F0', alpha=0.65, zorder=0, label='GRACE gap (2017-07 to 2018-05)')
        ax.axvline(split_dt, color=CRIMSON, linestyle='--', linewidth=1.1, alpha=0.85, zorder=2)
        ax.axhline(0, color='#A0AEC0', linewidth=0.7, linestyle=':', zorder=1)
        style_ax(ax, 'y')

    # (a)
    decorate_time(ax_a)
    m_gr = df['lwe_basin'].notna()
    an_g22 = df['anom_G22_TWS_tavg'] / 10.0
    an_g21 = df['anom_G21_TWS_proxy'] / 10.0
    ax_a.plot(df['dt'], an_g21, color=GOLD, lw=1.0, alpha=0.82, label='GLDAS-2.1 non-assimilated ($r$ = 0.754)')
    ax_a.plot(df['dt'], an_g22, color=TEAL, lw=1.25, alpha=0.88, label='GLDAS-2.2 assimilated ($r$ = 0.877)')
    ax_a.fill_between(df.loc[m_gr, 'dt'],
                      df.loc[m_gr, 'anom_lwe_basin'] - df.loc[m_gr, 'unc_basin'],
                      df.loc[m_gr, 'anom_lwe_basin'] + df.loc[m_gr, 'unc_basin'],
                      color=NAVY, alpha=0.16, label='GRACE formal ±1σ band')
    ax_a.plot(df.loc[m_gr, 'dt'], df.loc[m_gr, 'anom_lwe_basin'], color=NAVY, lw=1.6, label='GRACE JPL TWSA ($n$ = 218)')

    pre = m_gr & (df['dt'] <= pd.Timestamp('2017-06-15'))
    post = m_gr & (df['dt'] >= pd.Timestamp('2018-06-01'))
    for sub_m, col_l, lbl_l in [(pre, BLUE, 'Pre-2017: +0.094 cm a⁻¹ ($p$ = 0.308)'),
                                (post, CRIMSON, 'Post-2018: −0.663 cm a⁻¹ ($p$ < 10⁻⁷)')]:
        x = df.loc[sub_m, 't'].values
        y = df.loc[sub_m, 'anom_lwe_basin'].values
        p = np.polyfit(x, y, 1)
        ax_a.plot(df.loc[sub_m, 'dt'], np.polyval(p, x), color=col_l, lw=2.0, label=lbl_l)

    ax_a.text(split_dt, 8.2, ' Pettitt shift:\n May 2017 ($p$ = 4.2×10⁻²⁰)', color=CRIMSON, fontsize=7.8,
              ha='right', va='top', fontweight='bold')
    ax_a.set_ylim(-7.0, 8.6)
    ax_a.set_ylabel('Deseasonalized TWSA (cm)')
    ax_a.set_title('(a) Basin-Mean TWSA Transition & Multi-Source Agreement', fontweight='bold', loc='left', pad=6)
    ax_a.legend(loc='lower left', fontsize=7.1, ncol=2, frameon=True, facecolor='white', framealpha=0.93, edgecolor='#CBD5E0')

    # (b)
    decorate_time(ax_b)
    ax_b.plot(df.loc[m_gr, 'dt'], df.loc[m_gr, 'anom_lwe_nw'], color=BLUE, lw=1.45,
              label='NW Cryosphere (≥34.5°N): +0.283 → −0.855 cm a⁻¹')
    ax_b.plot(df.loc[m_gr, 'dt'], df.loc[m_gr, 'anom_lwe_se'], color=ORANGE, lw=1.45,
              label='SE Wetland (<34.5°N): −0.011 → −0.557 cm a⁻¹')
    ax_b.hlines(-1.02, pd.Timestamp('2003-02-01'), pd.Timestamp('2017-06-01'), colors=SLATE, linestyles=':', lw=1.3)
    ax_b.hlines(3.46, pd.Timestamp('2018-06-01'), pd.Timestamp('2020-12-01'), colors=TEAL, linestyles='--', lw=1.45)
    ax_b.hlines(1.08, pd.Timestamp('2021-01-01'), pd.Timestamp('2024-09-01'), colors=PURPLE, linestyles='--', lw=1.45)
    ax_b.text(pd.Timestamp('2019-10-01'), 8.8, 'Pluvial Peak\n(2018–2020)', fontsize=7.6, color=TEAL,
              fontweight='bold', ha='center', va='top',
              bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=TEAL, lw=0.6, alpha=0.90))
    ax_b.text(pd.Timestamp('2022-11-01'), -5.6, '2021–2024 drawdown:\nstill +2.12 cm above 2003–2017', fontsize=7.5,
              color=PURPLE, fontweight='bold', ha='center',
              bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
    ax_b.set_ylim(-7.5, 9.8)
    ax_b.set_ylabel('Subzone TWSA Anomaly (cm)')
    ax_b.set_title('(b) Subzone Contrast: NW Cryosphere Amplification', fontweight='bold', loc='left', pad=6)
    ax_b.legend(loc='upper left', fontsize=7.4, frameon=True, facecolor='white', framealpha=0.93, edgecolor='#CBD5E0')

    # (c)
    decorate_time(ax_c)
    p12 = df['anom_GPM_P'].rolling(12, center=True, min_periods=6).mean()
    ax_c.fill_between(df['dt'], 0, p12, where=(p12 >= 0), color='#3182CE', alpha=0.30, label='12-mo smoothed GPM $P$ anomaly (+)')
    ax_c.fill_between(df['dt'], 0, p12, where=(p12 < 0), color='#E53E3E', alpha=0.25, label='12-mo smoothed GPM $P$ anomaly (−)')
    ax_c.set_ylim(-15, 22)
    ax_c.set_ylabel('Precipitation Anomaly (mm month⁻¹)')

    ax_c2 = ax_c.twinx()
    ax_c2.spines['top'].set_visible(False)
    t12 = df['anom_ERA5_T2m'].rolling(12, center=True, min_periods=6).mean()
    st4_12 = df['anom_ERA5_ST4'].rolling(12, center=True, min_periods=6).mean()
    ax_c2.plot(df['dt'], t12, color=ORANGE, lw=1.25, label='12-mo $T_{2m}$ (+0.899 °C dec⁻¹)')
    ax_c2.plot(df['dt'], st4_12, color=CRIMSON, lw=1.8, label='12-mo Deep Soil ST4 (+0.410 °C dec⁻¹, $p$ = 5.5×10⁻⁹)')
    ax_c2.set_ylim(-1.6, 2.3)
    ax_c2.set_ylabel('Temperature Anomaly (°C)')

    h1, l1 = ax_c.get_legend_handles_labels()
    h2, l2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=7.1, frameon=True, facecolor='white', framealpha=0.93, edgecolor='#CBD5E0')
    ax_c.set_title('(c) Pluvial Surge & Persistent Deep-Soil Warming (100–289 cm)', fontweight='bold', loc='left', pad=6)

    # (d)
    decorate_time(ax_d)
    ndvi_12 = df['anom_NDVI'].rolling(12, center=True, min_periods=6).mean() * 1000.0
    evi_12 = df['anom_EVI'].rolling(12, center=True, min_periods=6).mean() * 1000.0
    ax_d.plot(df['dt'], df['anom_NDVI'] * 1000.0, color='#CBD5E0', lw=0.75, alpha=0.9, label='Monthly NDVI anomaly')
    ax_d.plot(df['dt'], ndvi_12, color=TEAL, lw=2.0, label='12-mo NDVI (+0.915×10⁻³ a⁻¹, $p$ = 0.0031)')
    ax_d.plot(df['dt'], evi_12, color=BLUE, lw=1.45, linestyle='--', label='12-mo EVI (+0.570×10⁻³ a⁻¹, $p$ = 0.0064)')

    ax_d.text(0.98, 0.06,
              'Growing-season NDVI (May–Sep) step:\n'
              '2003–2018: 0.5193 → 2019–2024: 0.5422 (+4.40%)\n'
              'NW Cryosphere: +7.72% vs SE Meadow: +2.77%\n'
              'PML GPP trend: +3.73 gC m⁻² a⁻¹ ($p$ = 1.1×10⁻⁵)',
              transform=ax_d.transAxes, ha='right', va='bottom', fontsize=7.4, color='#1A202C',
              bbox=dict(boxstyle='round,pad=0.3', fc='#F0FFF4', ec=TEAL, lw=0.8, alpha=0.95))

    ax_d.set_ylim(-52, 56)
    ax_d.set_ylabel('Vegetation Index Anomaly (×10⁻³)')
    ax_d.set_title('(d) Persistent Alpine Greening & Productivity Surge', fontweight='bold', loc='left', pad=6)
    ax_d.legend(loc='upper left', fontsize=7.3, frameon=True, facecolor='white', framealpha=0.93, edgecolor='#CBD5E0')

    for ax in [ax_c, ax_d]:
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_xlabel('Year')

    out = os.path.join(OUT_DIR, 'Fig2_Temporal_Transition_and_Subzones.png')
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print('Saved:', out)


# =============================================================================
# FIGURE 3: Subsurface Disaggregation, Permafrost, EDW Cascade, Formal Error
# =============================================================================
def plot_fig3():
    elev = pd.read_csv(os.path.join(RES_DIR, 'jms_elevation_bands_summary.csv'))
    ph = pd.read_csv(os.path.join(RES_DIR, 'jms_grace_phase_uncertainty.csv'))
    mc = pd.read_csv(os.path.join(RES_DIR, 'jms_grace_wls_montecarlo.csv'))
    df_m = pd.read_csv(os.path.join(RES_DIR, 'basin_merged_monthly_2003_2024.csv'))

    fig = plt.figure(figsize=(14.6, 9.4), facecolor='white')
    gs = gridspec.GridSpec(2, 2, hspace=0.36, wspace=0.34, left=0.165, right=0.96, top=0.93, bottom=0.07)

    # (a) Subsurface vertical disaggregation (generous left margin so no clipping!)
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, 'x')
    labels_a = [
        'G2.2 Groundwater (73.3%)',
        'G2.2 Root-zone SM (22.6%)',
        'G2.2 Snow water (4.1%)',
        'G2.1 100–200 cm SM (37.3%)',
        'G2.1 40–100 cm SM (44.9%)',
        'G2.1 10–40 cm SM (11.0%)',
        'G2.1 0–10 cm SM (6.8%)'
    ]
    vals_a = [-4.932, -1.519, -0.274, -3.527, -4.241, -1.040, -0.639]
    cols_a = [NAVY, TEAL, '#63B3ED', '#B7791F', ORANGE, '#ED8936', '#FBD38D']
    y_pos = np.arange(len(labels_a))[::-1]
    ax_a.barh(y_pos, vals_a, color=cols_a, height=0.60, edgecolor='#2D3748', linewidth=0.55)
    ax_a.axhline(3.5, color='#718096', linestyle='--', lw=0.95)
    for yp, v in zip(y_pos, vals_a):
        ax_a.text(v - 0.12, yp, f'{v:.2f}', va='center', ha='right', fontsize=7.9, fontweight='bold', color='#1A202C')

    ax_a.text(-5.95, 4.15, 'GLDAS-2.2 Assimilated\n(GWS+RZSM = 95.9%)', fontsize=7.8, fontweight='bold',
              color=NAVY, ha='left', va='bottom',
              bbox=dict(boxstyle='round,pad=0.22', fc='#EBF8FF', ec=NAVY, lw=0.6, alpha=0.95))
    ax_a.text(-5.95, 0.05, 'Independent GLDAS-2.1\n(40–200 cm deep = 82.2%)', fontsize=7.8, fontweight='bold',
              color='#9C4221', ha='left', va='bottom',
              bbox=dict(boxstyle='round,pad=0.22', fc='#FFFAF0', ec=ORANGE, lw=0.6, alpha=0.95))

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(labels_a, fontsize=8.0)
    ax_a.set_ylim(-0.6, 6.75)
    ax_a.set_xlim(-6.2, 0.25)
    ax_a.set_xlabel('Post-2018 Trend (mm a⁻¹, 2018-06 to 2024-09)')
    ax_a.set_title('(a) Deep Subsurface & Groundwater Dominance of Drawdown', fontweight='bold', loc='left', pad=7)

    # (b) Subzone Deep Soil Warming ST4
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, 'y')
    df_full = df_m[(df_m['year'] >= 2004) & (df_m['year'] <= 2024)].copy()
    ann_st = df_full.groupby('year').agg(
        nw_st4=('nw_ERA5_ST4', 'mean'),
        se_st4=('se_ERA5_ST4', 'mean'),
        nw_frz=('nw_ERA5_ST4', lambda s: (s < 0).sum()),
        se_frz=('se_ERA5_ST4', lambda s: (s < 0).sum())
    ).reset_index()
    ax_b.plot(ann_st['year'], ann_st['nw_st4'], marker='o', ms=4, color=BLUE, lw=1.6,
              label='NW ST4 (+0.413 °C dec⁻¹; mean −0.11 °C)')
    ax_b.plot(ann_st['year'], ann_st['se_st4'], marker='s', ms=4, color=ORANGE, lw=1.6,
              label='SE ST4 (+0.408 °C dec⁻¹; mean +1.00 °C)')
    ax_b.axhline(0.0, color='#1A202C', linestyle='--', lw=0.95)
    ax_b.text(2004.3, 0.06, '0 °C Permafrost Thaw Threshold', fontsize=7.6, color='#2D3748', fontweight='bold')

    ax_b2 = ax_b.twinx()
    ax_b2.spines['top'].set_visible(False)
    ax_b2.bar(ann_st['year'] - 0.18, ann_st['nw_frz'], width=0.34, color=BLUE, alpha=0.22, label='NW freeze mo (mean 6.62)')
    ax_b2.bar(ann_st['year'] + 0.18, ann_st['se_frz'], width=0.34, color=ORANGE, alpha=0.22, label='SE freeze mo (−1.09 dec⁻¹)')
    ax_b2.set_ylim(0, 11.8)
    ax_b2.set_ylabel('Deep Soil Freeze Months (mo yr⁻¹)')
    ax_b.set_ylim(-1.1, 2.25)
    ax_b.set_xlim(2003.4, 2024.7)
    ax_b.set_xticks([2004, 2008, 2012, 2016, 2020, 2024])
    ax_b.set_ylabel('Deep Soil Temp. ST4 (100–289 cm, °C)')
    ax_b.set_title('(b) Subzone Deep Soil Warming & Freeze Retreat', fontweight='bold', loc='left', pad=7)
    h1, l1 = ax_b.get_legend_handles_labels()
    h2, l2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=7.1, frameon=True, facecolor='white', framealpha=0.93, edgecolor='#CBD5E0')

    # (c) 3-Elevation-Belt Cascade
    ax_c = fig.add_subplot(gs[1, 0])
    style_ax(ax_c, 'y')
    zones = ['<3800m', '3800-4300m', '>=4300m']
    z_labels = ['<3,800 m\n(28.0%)', '3,800–4,300 m\n(30.3%)', '≥4,300 m\n(41.8%)']
    def get_var(var, col):
        return [float(elev.loc[(elev['Zone'] == z) & (elev['Variable'] == var), col].iloc[0]) for z in zones]

    ndvi_pct = get_var('NDVI_GS', 'Rel_Change_pct')
    ec_pct = get_var('PML_Ec', 'Rel_Change_pct')
    es_pct = get_var('PML_Es', 'Rel_Change_pct')
    gpm_pct = get_var('GPM_P', 'Rel_Change_pct')
    t2m_sl = get_var('T2m', 'HAC_slope_per_10a')
    ec_p = get_var('PML_Ec', 'p_val')

    x = np.arange(3)
    w = 0.20
    ax_c.bar(x - 1.5 * w, gpm_pct, width=w, color='#63B3ED', edgecolor='#2D3748', lw=0.5, label='Precipitation $P$ (% change)')
    ax_c.bar(x - 0.5 * w, es_pct, width=w, color='#F6AD55', edgecolor='#2D3748', lw=0.5, label='Soil evaporation $E_s$ (% change)')
    ax_c.bar(x + 0.5 * w, ec_pct, width=w, color=TEAL, edgecolor='#2D3748', lw=0.5, label='Transpiration $E_c$ (% change)')
    ax_c.bar(x + 1.5 * w, ndvi_pct, width=w, color=NAVY, edgecolor='#2D3748', lw=0.5, label='Growing-season NDVI (% change)')

    for bi in range(3):
        ax_c.text(x[bi] - 1.5 * w, gpm_pct[bi] + 0.45, f'+{gpm_pct[bi]:.1f}%', ha='center', va='bottom', fontsize=6.8)
        ax_c.text(x[bi] - 0.5 * w, es_pct[bi] + 0.45, f'+{es_pct[bi]:.1f}%', ha='center', va='bottom', fontsize=6.8)
        star = '**' if ec_p[bi] < 0.01 else ''
        ax_c.text(x[bi] + 0.5 * w, ec_pct[bi] + 0.45, f'+{ec_pct[bi]:.1f}%{star}', ha='center', va='bottom', fontsize=6.9, fontweight='bold', color=TEAL)
        ax_c.text(x[bi] + 1.5 * w, ndvi_pct[bi] + 0.45, f'+{ndvi_pct[bi]:.1f}%', ha='center', va='bottom', fontsize=6.9, fontweight='bold', color=NAVY)

    ax_c2 = ax_c.twinx()
    ax_c2.spines['top'].set_visible(False)
    ax_c2.plot(x, t2m_sl, marker='D', ms=6.8, lw=2.2, color=CRIMSON, label='Air warming $T_{2m}$ (°C dec⁻¹)')
    for xi, tv in zip(x, t2m_sl):
        ax_c2.text(xi + 0.04, tv + 0.055, f'+{tv:.2f} °C/dec', ha='center', va='bottom', fontsize=7.5,
                   fontweight='bold', color=CRIMSON,
                   bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.82))
    ax_c2.set_ylim(0.15, 1.72)
    ax_c2.set_ylabel('Warming Rate (°C decade⁻¹)', color=CRIMSON)
    ax_c2.tick_params(axis='y', colors=CRIMSON)

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(z_labels, fontsize=8.3)
    ax_c.set_ylim(0, 35.0)
    ax_c.set_ylabel('2020–2024 vs 2004–2008 Change (%)')
    ax_c.set_title('(c) Elevation-Dependent Cascade: ≥4,300 m Hotspot', fontweight='bold', loc='left', pad=7)
    h1, l1 = ax_c.get_legend_handles_labels()
    h2, l2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=6.8, frameon=True, facecolor='white', framealpha=0.93, edgecolor='#CBD5E0')

    # (d) GRACE Formal Error & Monte Carlo (zero overlap with (c) twinx or bottom-right legend!)
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis('off')
    sub_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], height_ratios=[1.1, 1.05], hspace=0.46)
    ax_d1 = fig.add_subplot(sub_gs[0])
    ax_d2 = fig.add_subplot(sub_gs[1])

    style_ax(ax_d1, 'y')
    ph_sub = ph.iloc[[0, 1, 2, 4]].copy()
    ph_names = ['Pre-shift\n(03–17)', 'Pluvial Peak\n(18–20)', 'Drawdown\n(21–24)', 'Latest Yr\n(2024)']
    means = ph_sub['Mean_JPL_cm'].values
    errs_2se = 2.0 * ph_sub['Phase_Mean_Formal_SE_cm'].values
    snrs = ph_sub['SNR_vs_Pre'].values
    ph_cols = ['#718096', TEAL, PURPLE, ORANGE]
    ax_d1.bar(range(4), means, yerr=errs_2se, capsize=4, color=ph_cols, width=0.50,
              edgecolor='#2D3748', lw=0.6, error_kw=dict(ecolor='#1A202C', lw=1.2))
    for i, (mval, e2, sn) in enumerate(zip(means, errs_2se, snrs)):
        lbl_s = f'{mval:.2f}\n(Ref)' if i == 0 else f'+{mval:.2f}\nSNR={sn:.1f}'
        ax_d1.text(i, mval + e2 + 0.16, lbl_s, ha='center', va='bottom', fontsize=7.1, fontweight='bold')
    ax_d1.set_xticks(range(4))
    ax_d1.set_xticklabels(ph_names, fontsize=7.5)
    ax_d1.set_ylim(0, 7.5)
    ax_d1.set_ylabel('Mean JPL (cm)')
    ax_d1.set_title('(d1) GRACE Phase Means ±2×SE & Signal-to-Noise Ratio (SNR)', fontweight='bold', loc='left', pad=5, fontsize=9.6)

    style_ax(ax_d2, 'x')
    # Compact y labels so zero collision with panel (c) right axis!
    win_names = ['Post (18–24)', 'Pre (03–17)', 'Full (03–24)']
    mc_ord = mc.iloc[[2, 1, 0]].copy()
    mc_m = mc_ord['MC_mean_slope_cma'].values
    wls_m = mc_ord['WLS_HAC_slope_cma'].values
    lo_vals = [float(s.split(',')[0].strip('[')) for s in mc_ord['MC_95pct_unc_band']]
    hi_vals = [float(s.split(',')[1].strip('] ')) for s in mc_ord['MC_95pct_unc_band']]
    for idx in range(3):
        ax_d2.plot([lo_vals[idx], hi_vals[idx]], [idx, idx], color=NAVY, lw=4.5, alpha=0.35)
        ax_d2.scatter([mc_m[idx]], [idx], color=NAVY, s=44, zorder=4, label='MC 2,000-draw mean (±95% band)' if idx == 0 else '')
        ax_d2.scatter([wls_m[idx]], [idx], color=CRIMSON, marker='D', s=40, zorder=5, label='WLS-HAC weighted slope' if idx == 0 else '')
        frac = mc_ord['MC_frac_same_sign'].iloc[idx] * 100
        ax_d2.text(hi_vals[idx] + 0.03, idx, f'MC {mc_m[idx]:+.3f} | WLS {wls_m[idx]:+.3f} ({frac:.0f}%)',
                   va='center', ha='left', fontsize=7.2, fontweight='bold')
    ax_d2.axvline(0, color='#718096', linestyle='--', lw=0.9)
    ax_d2.set_yticks(range(3))
    ax_d2.set_yticklabels(win_names, fontsize=7.8)
    ax_d2.set_ylim(-0.6, 2.7)
    ax_d2.set_xlim(-1.08, 1.18)
    ax_d2.set_xlabel('Slope (cm a⁻¹)')
    ax_d2.legend(loc='upper left', fontsize=6.7, frameon=True, facecolor='white', framealpha=0.93, edgecolor='#CBD5E0')
    ax_d2.set_title('(d2) 2,000-Draw Monte Carlo Error Propagation & WLS Check', fontweight='bold', loc='left', pad=5, fontsize=9.6)

    out = os.path.join(OUT_DIR, 'Fig3_Subsurface_EDW_and_Uncertainty.png')
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print('Saved:', out)


# =============================================================================
# FIGURE 4: PCMCI Causality, ET Partitioning & Memory Modulation
# =============================================================================
def plot_fig4():
    pcorr = pd.read_csv(os.path.join(RES_DIR, 'partial_corr_et_partition.csv'))
    mem = pd.read_csv(os.path.join(RES_DIR, 'memory_sensitivity_repaired.csv'))

    fig = plt.figure(figsize=(14.6, 5.5), facecolor='white')
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.28, 1.0, 1.06], wspace=0.28,
                           left=0.03, right=0.955, top=0.87, bottom=0.13)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 10)
    ax_a.axis('off')
    ax_a.set_title('(a) Directed Hydro-Ecological Causality (PCMCI)', fontweight='bold', loc='left', pad=10)

    nodes = {
        'P': (1.45, 8.35, 'Precipitation\n(GPM V07)', '#EBF8FF', BLUE),
        'T2m': (8.45, 8.35, 'Air Temp.\n(ERA5 T2m)', '#FFF5F5', CRIMSON),
        'Ec': (5.15, 6.75, 'Transpiration\n(PML Ec)', '#FEFCBF', '#975A16'),
        'SM': (1.55, 3.85, 'Soil Moisture\n(GLDAS-2.1)', '#E6FFFA', TEAL),
        'NDVI': (8.45, 4.15, 'Greenness\n(MODIS NDVI)', '#F0FFF4', '#2F855A'),
        'TWSA': (5.15, 1.25, 'Total Water Storage\n(GRACE TWSA)', '#EDF2F7', NAVY),
    }
    for k, (nx_p, ny_p, label, fc, ec) in nodes.items():
        box = FancyBboxPatch((nx_p - 1.24, ny_p - 0.64), 2.48, 1.28,
                             boxstyle='round,pad=0.12,rounding_size=0.24',
                             facecolor=fc, edgecolor=ec, linewidth=1.6, zorder=4)
        ax_a.add_patch(box)
        ax_a.text(nx_p, ny_p, label, ha='center', va='center', fontsize=7.5, fontweight='bold', color='#1A202C', zorder=5)

    def draw_edge(u, v, rad, col, ls, lbl, lpos, lw=1.9, shrink=26):
        x1, y1 = nodes[u][0], nodes[u][1]
        x2, y2 = nodes[v][0], nodes[v][1]
        arr = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f'arc3,rad={rad}',
                              arrowstyle='-|>,head_length=6.0,head_width=3.5',
                              color=col, linestyle=ls, linewidth=lw, shrinkA=shrink, shrinkB=shrink, zorder=3)
        ax_a.add_patch(arr)
        ax_a.text(lpos[0], lpos[1], lbl, fontsize=6.75, fontweight='bold', color=col, ha='center', va='center',
                  bbox=dict(boxstyle='round,pad=0.14', fc='white', ec='none', alpha=0.92), zorder=6)

    draw_edge('P', 'SM', 0.14, BLUE, '-', 'lag 1: +0.237\n($q$ = 0.0012)', (0.62, 6.35), lw=2.2)
    draw_edge('SM', 'Ec', -0.22, TEAL, '-', 'lag 1: +0.234 ($q$ = 0.0012)\nlag 2: +0.160 ($q$ = 0.039)', (3.65, 5.45), lw=2.2)
    draw_edge('Ec', 'SM', 0.25, ORANGE, '--', 'feedback lag 1:\n−0.152 ($p$ = 0.015)', (4.35, 3.85), lw=1.6)
    draw_edge('P', 'Ec', -0.16, BLUE, '-', 'lag 1: +0.166 ($q$ = 0.034)', (4.35, 8.85), lw=1.8)
    draw_edge('T2m', 'NDVI', 0.10, CRIMSON, '-', 'bidirectional lag 2:\n−0.210 ($q$ < 0.007)', (8.45, 6.25), lw=1.8)
    draw_edge('TWSA', 'Ec', 0.22, PURPLE, '-', 'lag 3: +0.274\n($q$ = 0.0085)', (6.75, 2.85), lw=2.1)
    draw_edge('TWSA', 'SM', -0.14, NAVY, '--', 'lag 1: +0.208 ($p$ = 0.009)', (3.35, 2.28), lw=1.6)
    draw_edge('P', 'TWSA', -0.36, NAVY, ':', 'lag 0: $r$ = +0.231\n($p$ = 0.0038)', (0.58, 2.08), lw=1.5)

    ax_a.text(5.0, 0.16, 'Solid lines: FDR $q$ < 0.05 (continuous differences); dashed: nominal $p$ < 0.05',
              ha='center', va='center', fontsize=7.1, color='#4A5568', style='italic')

    # (b)
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, 'y')
    lags = np.arange(4)
    w = 0.20
    comps = [
        ('anom_PML_Ei', 'Interception ($E_i$, 8.6% ET)', PURPLE),
        ('anom_PML_Ec', 'Transpiration ($E_c$, 29.4% ET)', TEAL),
        ('anom_PML_Es', 'Soil Evap. ($E_s$, 62.0% ET)', ORANGE),
        ('anom_PML_ET_tot', 'Total ET (PML-V2.2a)', SLATE)
    ]
    for ci, (col_t, lbl_c, c_hex) in enumerate(comps):
        vals = [float(pcorr[(pcorr['Predictor'] == 'anom_NDVI') & (pcorr['Target'] == col_t) & (pcorr['Lag'] == lg)]['Partial_r'].iloc[0]) for lg in lags]
        offs = (ci - 1.5) * w
        ax_b.bar(lags + offs, vals, width=w, color=c_hex, edgecolor='#2D3748', lw=0.45, label=lbl_c)
        for lg, v in zip(lags, vals):
            if abs(v) >= 0.15:
                ax_b.text(lg + offs, v + 0.008, f'{v:.2f}', ha='center', va='bottom', fontsize=6.8, fontweight='bold')
    ax_b.axhline(0, color='#718096', lw=0.8)
    ax_b.set_xticks(lags)
    ax_b.set_xticklabels([f'Lag {lg}' for lg in lags], fontsize=8.5)
    ax_b.set_ylim(-0.08, 0.39)
    ax_b.set_xlabel('Lag After NDVI Anomaly (months)')
    ax_b.set_ylabel('Partial Correlation $r$ (ctrl: $P$, $T_{2m}$)')
    ax_b.set_title('(b) ET Partitioning Response to Greening', fontweight='bold', loc='left', pad=10)
    ax_b.legend(loc='upper right', fontsize=7.1, frameon=True, facecolor='white', framealpha=0.93, edgecolor='#CBD5E0')

    # (c)
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c, 'y')
    ks = mem['Window_k_months'].values
    r_p = mem['r_P_TWSA'].values
    pr_ndvi = mem['partial_r_NDVIk_TWSA_ctrl_Pk_T'].values

    ax_c.plot(ks, r_p, marker='o', ms=5.5, color=BLUE, lw=2.1, label='Precipitation $P$ vs TWSA ($r$ climbs to 0.735)')
    ax_c.plot(ks, pr_ndvi, marker='s', ms=5.5, color=CRIMSON, lw=2.1, label='Partial $r$(NDVI, TWSA | $P$, $T_{2m}$) → −0.161*')
    ax_c.axhline(0, color='#718096', linestyle='--', lw=0.85)
    ax_c.scatter([12], [-0.1605], s=90, color=CRIMSON, edgecolor='white', lw=1.0, zorder=5)
    ax_c.annotate('$k$ = 12 mo: $r$ = −0.161\n($p$ = 0.0206)', xy=(12, -0.1605), xytext=(7.8, -0.26),
                  fontsize=7.5, fontweight='bold', color=CRIMSON,
                  arrowprops=dict(arrowstyle='->', color=CRIMSON, lw=1.0))
    ax_c.text(6.8, 0.25,
              'Soil Moisture Regime Split:\n'
              'Dry half: NDVI → $E_c$ $r$ = +0.210 ($p$ = 0.038)\n'
              'Wet half: NDVI → $E_c$ $r$ = +0.142 ($p$ = 0.163)',
              fontsize=7.3, color='#1A202C', va='center', ha='center',
              bbox=dict(boxstyle='round,pad=0.30', fc='#F7FAFC', ec='#CBD5E0', alpha=0.96))
    ax_c.set_xticks(ks)
    ax_c.set_xlim(0.5, 12.6)
    ax_c.set_ylim(-0.33, 0.90)
    ax_c.set_xlabel('Cumulative Anomaly Memory Window $k$ (months)')
    ax_c.set_ylabel('Correlation / Partial Correlation')
    ax_c.set_title('(c) Memory Dominance & Multi-Month Suppression', fontweight='bold', loc='left', pad=10)
    ax_c.legend(loc='upper left', fontsize=7.0, frameon=True, facecolor='white', framealpha=0.93, edgecolor='#CBD5E0')

    out = os.path.join(OUT_DIR, 'Fig4_Causality_and_ET_Partitioning.png')
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print('Saved:', out)


# =============================================================================
# FIGURE 5: Leakage-Free Walk-Forward Attribution & Exogenous SHAP
# =============================================================================
def plot_fig5():
    imp = pd.read_csv(os.path.join(RES_DIR, 'shap_no_leakage_importance.csv'))
    mat = pd.read_csv(os.path.join(RES_DIR, 'shap_no_leakage_matrix.csv'))

    from sklearn.linear_model import LassoCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    df_m = pd.read_csv(os.path.join(RES_DIR, 'basin_merged_monthly_2003_2024.csv'))
    df_m['ym_dt'] = pd.to_datetime(df_m['ym_dt'])
    df_m = df_m.sort_values('ym_dt').reset_index(drop=True)
    df_m['P_cum3'] = df_m['anom_GPM_P'].rolling(3).sum()
    df_m['P_cum6'] = df_m['anom_GPM_P'].rolling(6).sum()
    df_m['T2m_cum3'] = df_m['anom_ERA5_T2m'].rolling(3).mean()
    df_m['ET_cum3'] = df_m['anom_PML_ET_tot'].rolling(3).sum()
    df_m['Ec_cum3'] = df_m['anom_PML_Ec'].rolling(3).sum()
    df_m['NDVI_lag1'] = df_m['anom_NDVI'].shift(1)
    df_m['NDVI_lag2'] = df_m['anom_NDVI'].shift(2)
    df_m['NDVI_lag3'] = df_m['anom_NDVI'].shift(3)
    df_m['lwe_lag1'] = df_m['anom_lwe_basin'].shift(1)

    exog_cols = list(imp['feature'])
    df_v = df_m[['ym_dt', 'year', 'anom_lwe_basin', 'lwe_lag1'] + exog_cols].dropna().reset_index(drop=True)
    preds_rf, preds_lasso_ar, dates_oos, y_oos = [], [], [], []
    for yr in range(2014, 2025):
        tr = df_v['year'] < yr
        te = df_v['year'] == yr
        if te.sum() == 0:
            continue
        rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
        rf.fit(df_v.loc[tr, exog_cols], df_v.loc[tr, 'anom_lwe_basin'])
        p_rf = rf.predict(df_v.loc[te, exog_cols])

        cols_ar = exog_cols + ['lwe_lag1']
        sc = StandardScaler()
        Xtr = sc.fit_transform(df_v.loc[tr, cols_ar])
        Xte = sc.transform(df_v.loc[te, cols_ar])
        las = LassoCV(cv=5, random_state=42, max_iter=5000).fit(Xtr, df_v.loc[tr, 'anom_lwe_basin'])
        p_las = las.predict(Xte)

        dates_oos.extend(df_v.loc[te, 'ym_dt'].tolist())
        y_oos.extend(df_v.loc[te, 'anom_lwe_basin'].tolist())
        preds_rf.extend(p_rf.tolist())
        preds_lasso_ar.extend(p_las.tolist())

    fig = plt.figure(figsize=(14.0, 5.5), facecolor='white')
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.22, 1.0], wspace=0.38,
                           left=0.06, right=0.96, top=0.88, bottom=0.13)

    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, 'y')
    ax_a.plot(dates_oos, y_oos, color=NAVY, lw=2.0, label='Observed GRACE TWSA (test years 2014–2024)')
    ax_a.plot(dates_oos, preds_lasso_ar, color=TEAL, lw=1.45, linestyle='--',
              label='Exogenous + AR(1) Lasso ($r$ = 0.952, $R^2$ = 0.903, RMSE = 0.62 cm)')
    ax_a.plot(dates_oos, preds_rf, color=ORANGE, lw=1.65,
              label='Exogenous-only RF ($r$ = 0.472, $R^2$ = 0.123, RMSE = 2.00 cm)')
    ax_a.axvspan(pd.Timestamp('2017-07-01'), pd.Timestamp('2018-05-31'), color='#E2E8F0', alpha=0.7)
    ax_a.axhline(0, color='#A0AEC0', linestyle=':', lw=0.8)
    ax_a.text(0.02, 0.05,
              'Persistence baseline ($t-1$): $R^2$ = 0.913, RMSE = 0.585 cm\n'
              'Zero look-ahead leakage: AR(1) dominates month-to-month\n'
              'continuity; exogenous drivers govern regime trajectory',
              transform=ax_a.transAxes, fontsize=7.3, va='bottom',
              bbox=dict(boxstyle='round,pad=0.28', fc='#F7FAFC', ec='#CBD5E0', alpha=0.94))
    ax_a.set_ylim(-6.2, 7.6)
    ax_a.set_ylabel('Deseasonalized TWSA Anomaly (cm)')
    ax_a.set_xlabel('Walk-Forward Out-of-Sample Year')
    ax_a.set_title('(a) Leakage-Free Out-of-Sample Predictive Decomposition', fontweight='bold', loc='left', pad=8)
    ax_a.legend(loc='upper left', fontsize=7.2, frameon=True, facecolor='white', framealpha=0.94, edgecolor='#CBD5E0')
    ax_a.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, 'x')
    pretty_names = {
        'P_cum6': '6-mo Cum. Precipitation',
        'T2m_cum3': '3-mo Mean Air Temp.',
        'P_cum3': '3-mo Cum. Precipitation',
        'Ec_cum3': '3-mo Cum. Transpiration',
        'anom_GPM_P': 'Monthly Precipitation',
        'NDVI_lag2': 'NDVI Anomaly (lag 2 mo)',
        'anom_PML_ET_tot': 'Monthly Total ET (PML)',
        'NDVI_lag3': 'NDVI Anomaly (lag 3 mo)',
        'NDVI_lag1': 'NDVI Anomaly (lag 1 mo)',
        'ET_cum3': '3-mo Cum. Total ET',
    }
    top_imp = imp.iloc[:10].copy().iloc[::-1].reset_index(drop=True)
    y_idx = np.arange(len(top_imp))
    for idx_r, row in top_imp.iterrows():
        feat = row['feature']
        share = float(row['share_pct'])
        r_dir = np.corrcoef(mat[feat], mat[f'shap_{feat}'])[0, 1]
        bar_col = BLUE if r_dir >= 0 else CRIMSON
        ax_b.barh(idx_r, share, color=bar_col, height=0.58, alpha=0.86, edgecolor='#2D3748', lw=0.45, zorder=2)
        sign_str = '+' if r_dir >= 0 else '−'
        ax_b.text(share + 0.7, idx_r, f'{share:.1f}% ({sign_str})', va='center', ha='left',
                  fontsize=7.6, fontweight='bold', color='#1A202C')

    ax_b.set_yticks(y_idx)
    ax_b.set_yticklabels([pretty_names.get(f, f) for f in top_imp['feature']], fontsize=8.0)
    ax_b.set_xlim(0, 56)
    ax_b.set_xlabel('Exogenous-Only Mean |SHAP| Share (%)')
    ax_b.set_title('(b) Climate & Canopy Driver Importance (Exogenous RF)', fontweight='bold', loc='left', pad=8)

    leg_p = [
        mpatches.Patch(color=BLUE, alpha=0.86, label='Positive TWSA response (+ slope)'),
        mpatches.Patch(color=CRIMSON, alpha=0.86, label='Negative / suppressive response (− slope)'),
    ]
    ax_b.legend(handles=leg_p, loc='lower right', fontsize=7.4, frameon=True, facecolor='white', framealpha=0.93, edgecolor='#CBD5E0')

    out = os.path.join(OUT_DIR, 'Fig5_WalkForward_and_SHAP.png')
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print('Saved:', out)


# =============================================================================
# FIGURE 6: Water Balance Closure, ET Benchmarking & Budyko Trajectory
# =============================================================================
def plot_fig6():
    wb = pd.read_csv(os.path.join(RES_DIR, 'table4_water_balance_budyko.csv'))
    bud = pd.read_csv(os.path.join(RES_DIR, 'budyko_long_term_2004_2024.csv'))

    fig = plt.figure(figsize=(14.0, 5.6), facecolor='white')
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.16, 1.0], wspace=0.25,
                           left=0.06, right=0.94, top=0.88, bottom=0.13)

    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, 'y')
    yrs = wb['Year'].values
    ax_a.bar(yrs, wb['P_GPM_mm'], width=0.58, color='#BEE3F8', edgecolor='#2B6CB0', lw=0.7, alpha=0.65, label=r'GPM Precipitation ($P_{\mathrm{GPM}}$)')
    ax_a.bar(yrs, wb['R_obs_mm'], width=0.34, color=BLUE, edgecolor='#1A365D', lw=0.6, alpha=0.85, label=r'Tangnaihai Runoff ($R_{\mathrm{obs}}$)')
    ax_a.plot(yrs, wb['E_wb_inferred_mm'], marker='o', ms=6.2, color='#1A202C', lw=2.3, zorder=6,
              label=r'Closure $E_{\mathrm{wb}} = P - R - \Delta S_{\mathrm{G22}}$ (423.5 mm a⁻¹)')
    ax_a.plot(yrs, wb['E_TC_mm'], marker='^', ms=4.8, color=TEAL, lw=1.45, linestyle='--',
              label='TerraClimate ET (427.0, +0.8%)')
    ax_a.plot(yrs, wb['E_PML_mm'], marker='s', ms=4.5, color='#2F855A', lw=1.55,
              label='PML-V2.2a ET (377.4, −10.9%)')
    ax_a.plot(yrs, wb['E_ERA5_mm'], marker='d', ms=4.5, color=ORANGE, lw=1.35, linestyle='-.',
              label='ERA5-Land ET (455.8, +7.6%)')
    ax_a.plot(yrs, wb['E_MOD16_mm'], marker='x', ms=4.8, color=CRIMSON, lw=1.35, linestyle=':',
              label='MOD16A2GF ET (605.6, +43.0%)')

    ax_a.set_xticks(yrs)
    ax_a.set_ylim(0, 975)
    ax_a.set_xlabel('Year (Complete Gauge Record 2016–2024)')
    ax_a.set_ylabel('Annual Water Flux (mm a⁻¹)')
    ax_a.set_title('(a) Catchment Water-Balance Closure & Multi-Product ET', fontweight='bold', loc='left', pad=8)
    ax_a.legend(loc='upper left', fontsize=6.9, ncol=2, frameon=True, facecolor='white', framealpha=0.94, edgecolor='#CBD5E0')

    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, 'both')
    phi_grid = np.linspace(0.4, 2.3, 200)
    for n_val, ls_n in [(0.9, ':'), (1.15, '--'), (1.45, '-.')]:
        ep_curve = phi_grid / ((1.0 + phi_grid ** n_val) ** (1.0 / n_val))
        ax_b.plot(phi_grid, ep_curve, color='#718096', linestyle=ls_n, lw=1.2, label=f'Choudhury–Yang $n$ = {n_val}')
    ax_b.plot([0.4, 1.0, 2.3], [0.4, 1.0, 1.0], color='#2D3748', lw=1.1, alpha=0.65)

    early_b = bud[bud['year'] <= 2017]
    ax_b.scatter(early_b['pet_p'], early_b['ep_pml'], s=34, color='#A0AEC0', alpha=0.75, zorder=3, label='Annual points (2004–2017)')

    wb_b = bud[bud['year'] >= 2018].copy()
    wb_b = wb_b.merge(wb[['Year', 'NDVI_GS']], left_on='year', right_on='Year', how='left')
    sc = ax_b.scatter(wb_b['pet_p'], wb_b['ep_pml'], c=wb_b['NDVI_GS'], cmap='YlGn',
                      vmin=0.51, vmax=0.56, s=82, edgecolor='#1A202C', lw=0.9, zorder=5)
    cbar = fig.colorbar(sc, ax=ax_b, fraction=0.044, pad=0.03)
    cbar.set_label('Growing-season NDVI', fontsize=8.2)
    cbar.ax.tick_params(labelsize=7.8)

    offsets = {
        2018: (-0.095, 0.015),
        2019: (-0.095, 0.012),
        2020: (0.025, -0.022),
        2021: (0.028, -0.022),
        2022: (-0.085, 0.020),
        2023: (0.028, 0.015),
        2024: (0.028, 0.012),
    }
    for _, row in wb_b.iterrows():
        yr_i = int(row['year'])
        dx, dy = offsets.get(yr_i, (0.025, 0.012))
        ax_b.text(row['pet_p'] + dx, row['ep_pml'] + dy, str(yr_i), fontsize=7.6, fontweight='bold', color='#1A202C', zorder=6)

    p1_x = bud[(bud['year'] >= 2018) & (bud['year'] <= 2020)]['pet_p'].mean()
    p1_y = bud[(bud['year'] >= 2018) & (bud['year'] <= 2020)]['ep_pml'].mean()
    p2_x = bud[(bud['year'] >= 2021) & (bud['year'] <= 2024)]['pet_p'].mean()
    p2_y = bud[(bud['year'] >= 2021) & (bud['year'] <= 2024)]['ep_pml'].mean()
    ax_b.scatter([p1_x, p2_x], [p1_y, p2_y], marker='D', s=95, color=[TEAL, CRIMSON], edgecolor='white', lw=1.0, zorder=7)
    ax_b.annotate('', xy=(p2_x, p2_y), xytext=(p1_x, p1_y),
                  arrowprops=dict(arrowstyle='-|>', color=CRIMSON, lw=2.3, mutation_scale=15), zorder=6)
    ax_b.text(0.98, 0.05,
              'Regime Shift Trajectory:\n'
              '2018–2020: $P$ = 697.3, $R/P$ = 0.361, $E_{\\mathrm{wb}}/P$ = 0.629, $n$ = 0.93\n'
              '2021–2024: $P$ = 587.6 (−15.7%), $R/P$ = 0.288, $E_{\\mathrm{wb}}/P$ = 0.721, $n$ = 1.14',
              transform=ax_b.transAxes, ha='right', va='bottom', fontsize=6.9,
              bbox=dict(boxstyle='round,pad=0.28', fc='white', ec=CRIMSON, lw=0.8, alpha=0.95))

    ax_b.set_xlim(0.92, 2.18)
    ax_b.set_ylim(0.32, 0.92)
    ax_b.set_xlabel(r'Dryness Index ($\mathrm{PET_{PML}} / P_{\mathrm{GPM}}$)')
    ax_b.set_ylabel(r'Evaporative Index ($E_{\mathrm{PML}} / P_{\mathrm{GPM}}$)')
    ax_b.set_title('(b) Budyko Regime Shift & Greening Amplification', fontweight='bold', loc='left', pad=8)
    ax_b.legend(loc='upper left', fontsize=7.0, frameon=True, facecolor='white', framealpha=0.94, edgecolor='#CBD5E0')

    out = os.path.join(OUT_DIR, 'Fig6_WaterBalance_and_Budyko.png')
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print('Saved:', out)


# =============================================================================
# GRAPHICAL ABSTRACT (JMS 3-Panel Visual Summary)
# =============================================================================
def plot_graphical_abstract():
    fig, ax = plt.subplots(figsize=(12.2, 6.2), facecolor='white')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis('off')

    hdr = FancyBboxPatch((0.2, 5.25), 11.6, 0.78, boxstyle='round,pad=0.05,rounding_size=0.12',
                         facecolor=NAVY, edgecolor='none')
    ax.add_patch(hdr)
    ax.text(6.0, 5.74, 'Elevation-Dependent Warming, Alpine Greening & Post-Pluvial Water Storage Relaxation',
            ha='center', va='center', fontsize=12.8, fontweight='bold', color='white')
    ax.text(6.0, 5.42, 'Source Region of the Yellow River (Tangnaihai, 123,800 km², 2,572–6,221 m a.s.l., 2003–2024)',
            ha='center', va='center', fontsize=9.8, color='#BEE3F8')

    cols = [
        (0.25, 3.65, '#EBF8FF', BLUE, '1. Alpine Elevation Cascade',
         'Three MERIT DEM Belts (2004–2024):\n'
         '• ≥4,300 m (41.8% area):\n'
         '  – Air T2m: +1.19 °C dec⁻¹ (+2.32 °C)\n'
         '  – Deep soil ST4: +0.46 °C dec⁻¹\n'
         '  – Growing NDVI: +8.38% (+0.0164 dec⁻¹)\n'
         '  – Transpiration Ec: +16.44% (p = 0.002)\n'
         '• 3,800–4,300 m (30.3% area):\n'
         '  – T2m +1.08 °C/10a | NDVI +3.64% | Ec +5.00%\n'
         '• <3,800 m (28.0% area):\n'
         '  – T2m +0.90 °C/10a | NDVI +2.63% | Ec +3.10%\n'
         '→ NW Permafrost (≥34.5°N) greens\n'
         '   2.8× faster than SE (+7.72% vs +2.77%)'),
        (4.18, 3.65, '#F0FFF4', TEAL, '2. Subsurface Transition & Memory',
         '2017–2018 Hydrological Regime Shift:\n'
         '• Pettitt change-point: May 2017 (p = 4.2×10⁻²⁰)\n'
         '• Post-2018 TWSA relaxation: −0.663 cm a⁻¹\n'
         '  – WLS formal-error weighted: −0.797 cm a⁻¹\n'
         '  – Monte Carlo (n = 2,000): 100% negative\n'
         '• YET 2021–2024 level remains +2.12 cm above\n'
         '  2003–2017 baseline (SNR = 5.93)\n'
         '• Vertical Loss Shares (2018–2024):\n'
         '  – GLDAS-2.2 GWS 73.3% | RZSM 22.6%\n'
         '  – Indep. GLDAS-2.1 40–200 cm SM: 82.2%\n'
         '• Leakage-Free Attribution:\n'
         '  – AR(1) memory dominates (R² = 0.903 vs 0.123)\n'
         '  – Exog. SHAP: 6-mo P 44.5%, 3-mo T2m 10.5%'),
        (8.10, 3.65, '#FFFAF0', ORANGE, '3. ET Partitioning & Water Closure',
         'Canopy Buffering vs Water-Balance Closure:\n'
         '• PCMCI Causal Chain (FDR q < 0.05):\n'
         '  P → SM (lag 1: +0.237) → Ec (+0.234)\n'
         '  TWSA → Ec (lag 3: +0.274, q = 0.0085)\n'
         '• PML-V2.2a ET Shares (363.9 mm a⁻¹):\n'
         '  – Es 62.0% | Ec 29.4% | Ei 8.6%\n'
         '  – Greening shifts flux into Ec & Ei,\n'
         '    suppressing storage at k = 12 mo (r = −0.161*)\n'
         '• Tangnaihai Gauge Closure (2016–2024):\n'
         '  – P = 615.8, R = 187.1, E_wb = 423.5 mm a⁻¹\n'
         '  – Post-2020 drawdown response:\n'
         '    P −15.7%, R/P 0.361 → 0.288,\n'
         '    E_wb/P 0.629 → 0.721, Budyko n 0.93 → 1.14')
    ]
    for x0, width, fc, ec, title, body in cols:
        card = FancyBboxPatch((x0, 0.35), width, 4.65, boxstyle='round,pad=0.06,rounding_size=0.14',
                              facecolor=fc, edgecolor=ec, linewidth=1.6)
        ax.add_patch(card)
        ax.text(x0 + width / 2, 4.74, title, ha='center', va='center', fontsize=10.1, fontweight='bold', color=ec)
        ax.plot([x0 + 0.18, x0 + width - 0.18], [4.50, 4.50], color=ec, lw=1.0, alpha=0.55)
        ax.text(x0 + 0.12, 4.35, body, ha='left', va='top', fontsize=7.45, color='#1A202C', linespacing=1.40)

    for x_arr in [3.95, 7.87]:
        ax.annotate('', xy=(x_arr + 0.25, 2.7), xytext=(x_arr - 0.03, 2.7),
                    arrowprops=dict(arrowstyle='-|>', color='#2D3748', lw=2.2, mutation_scale=15))

    out = os.path.join(OUT_DIR, 'Graphical_Abstract.png')
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print('Saved:', out)


if __name__ == '__main__':
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    plot_fig5()
    plot_fig6()
    plot_graphical_abstract()
    print('All 7 JMS 600-DPI figures regenerated cleanly!')
