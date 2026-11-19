import os, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, ee
import statsmodels.api as sm
from matplotlib.path import Path as MPath
OUT = '/Users/yueliangmanle/Desktop/小论文中文2/results_repaired'
GEO = '/Users/yueliangmanle/Desktop/小论文中文2/高级图设计包/fig1_研究区/yr_source_boundary.geojson'
rng = np.random.default_rng(11)
d = np.load(os.path.join(OUT, '_cache_pixel_panel.npz')); ndvi, T, P = d['ndvi'], d['T'], d['P']
st4 = np.load(os.path.join(OUT, 'jms_st4_clim_grid.npz'))['st4_clim']
cache = os.path.join(OUT, '_cache_pixel_extra.npz')
if os.path.exists(cache):
    e = np.load(cache); SM, SC, LC = e['SM'], e['SC'], e['LC']
else:
    ee.Initialize(project='round-fusion-508306-p4')
    rect = ee.Geometry.Rectangle([95.5, 32.0, 103.5, 36.5])
    era = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
    def sm_img(y): return era.filterDate(f'{y}-05-01', f'{y}-10-01').select('volumetric_soil_water_layer_2').mean().reproject(crs='EPSG:4326', scale=5000).rename(f'y{y}')
    def sc_img(y): return era.filterDate(f'{y-1}-10-01', f'{y}-06-01').select('snow_cover').mean().reproject(crs='EPSG:4326', scale=5000).rename(f'y{y}')
    def fetch(fn):
        img = ee.Image.cat([fn(y) for y in range(2004, 2025)]).toFloat().unmask(-999.0)
        for k in range(6):
            try:
                s = img.sampleRectangle(region=rect, defaultValue=-999.0).getInfo()['properties']
                return np.stack([np.array(s[f'y{y}'], dtype=float) for y in range(2004, 2025)])
            except Exception as ex:
                print('retry', k, str(ex)[:60]); time.sleep(4 * (k + 1)); ee.Initialize(project='round-fusion-508306-p4')
    SM, SC = fetch(sm_img), fetch(sc_img)
    lc = ee.ImageCollection('MODIS/061/MCD12Q1').filterDate('2020-01-01', '2021-01-01').first().select('LC_Type1')
    lcf = lc.toFloat().unmask(-999.0).reproject(crs='EPSG:4326', scale=5000)
    s = lcf.sampleRectangle(region=rect, defaultValue=-999.0).getInfo()['properties']
    LC = np.array(s['LC_Type1'], dtype=float)
    np.savez_compressed(cache, SM=SM, SC=SC, LC=LC)
for a in (SM, SC, LC): a[a <= -900] = np.nan
ny, nx = st4.shape
lons = np.linspace(95.5 + 8 / (2 * nx), 103.5 - 8 / (2 * nx), nx); lats = np.linspace(36.5 - 4.5 / (2 * ny), 32.0 + 4.5 / (2 * ny), ny)
lo, la = np.meshgrid(lons, lats)
gj = json.load(open(GEO)); rings = [np.array(gj['coordinates'][0])] if gj['type'] == 'Polygon' else [np.array(p[0]) for p in gj['coordinates']]
ins = np.zeros(lo.size, bool)
for r in rings: ins |= MPath(r[:, :2]).contains_points(np.column_stack([lo.ravel(), la.ravel()]))
ins = ins.reshape(lo.shape)
for a in (ndvi, T, P): a[a <= -900] = np.nan
valid = ins & np.isfinite(st4) & np.all(np.isfinite(ndvi), 0) & np.all(np.isfinite(T), 0) & np.all(np.isfinite(P), 0) & np.all(np.isfinite(SM), 0) & np.all(np.isfinite(SC), 0) & (ndvi.mean(0) > 0.05)
vals, cnt = np.unique(LC[valid & np.isfinite(LC)], return_counts=True)
print('MCD12Q1 LC_Type1 classes in basin (IGBP: 10=grass,16=barren,11=wetland,9=savanna,8=wood savanna,15=snow/ice,17=water):', dict(zip(vals.astype(int), cnt)))
yrs = np.arange(2004, 2025)
def dt(v): return v - np.polyval(np.polyfit(yrs, v, 1), yrs)
def z(v): return (v - v.mean()) / v.std(ddof=1)
def run(mask, label, use):
    s = {k: v[:, mask].mean(1) for k, v in dict(nd=ndvi, T=T, P=P, SM=SM, SC=SC).items()}
    cols = [dt(s[k]) for k in use]; y = dt(s['nd'])
    f = sm.OLS(y, sm.add_constant(np.column_stack(cols))).fit()
    tr_nd = np.polyfit(yrs, s['nd'], 1)[0] * 10
    expl = sum(f.params[i + 1] * np.polyfit(yrs, s[k], 1)[0] * 10 for i, k in enumerate(use))
    return tr_nd, expl, f.rsquared
grass = valid & np.isin(LC, [10, 9, 8])
ROWS = []
for lbl, base in [('all vegetated pixels', valid), ('grassland/savanna only (IGBP 8-10)', grass)]:
    for pz, m in [('permafrost ST4<0', base & (st4 < 0)), ('seasonally frozen ST4>=0', base & (st4 >= 0))]:
        for uname, use in [('T+P', ['T', 'P']), ('T+P+soil moisture+snow cover', ['T', 'P', 'SM', 'SC'])]:
            if m.sum() < 30: continue
            tr, ex, r2 = run(m, pz, use)
            # block bootstrap of the unexplained fraction via year resampling (block 3)
            fr = []
            for _ in range(1000):
                st = rng.integers(0, len(yrs) - 2, size=7); idx = np.concatenate([np.arange(a, a + 3) for a in st])[:21]
                sd = {k: v[:, m].mean(1) for k, v in dict(nd=ndvi, T=T, P=P, SM=SM, SC=SC).items()}
                yy = dt(sd['nd'])[idx]; XX = np.column_stack([dt(sd[k])[idx] for k in use])
                try:
                    ff = sm.OLS(yy, sm.add_constant(XX)).fit()
                    ex_b = sum(ff.params[i + 1] * np.polyfit(yrs, sd[k], 1)[0] * 10 for i, k in enumerate(use))
                    fr.append(1 - ex_b / tr)
                except Exception: pass
            ROWS.append(dict(Mask=lbl, Zone=pz, Predictors=uname, n_pix=int(m.sum()), NDVI_trend_dec=tr, Explained=ex, R2=r2, Unexpl_pct=100*(1-ex/tr), Unexpl_lo=100*np.percentile(fr,2.5), Unexpl_hi=100*np.percentile(fr,97.5)))
            print(f'{lbl:<36}|{pz:<26}|{uname:<30}| n={int(m.sum()):>4} NDVI trend {tr:+.4f}/dec, explained {ex:+.4f} (R2={r2:.2f}); unexplained share {100*(1-ex/tr):.0f}% [{100*np.percentile(fr,2.5):.0f},{100*np.percentile(fr,97.5):.0f}]')

pd.DataFrame(ROWS).to_csv(os.path.join(OUT, 'probe_A2_greening_residual.csv'), index=False)
print('saved')
