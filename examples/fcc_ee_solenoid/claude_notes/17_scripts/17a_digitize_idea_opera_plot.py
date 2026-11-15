"""Digitize the Opera plot 'Bz (T) vs Z (m)' (IDEA main / screening / both).

Per curve, the centreline is found column by column (row by row where the
curve is steep, |dy/dx| > 1.5 px/px). Occlusion is handled without assuming
a drawing order: the curve's band (thickness t*sqrt(1+slope^2)) must cover
all visible pixels of its colour and may only extend into adjacent
non-background pixels (another curve, a gridline, the logo). The admissible
band positions give the centre and its uncertainty.

The black 'Both energized' curve has the same colour as the gridlines; it is
isolated by thickness (3x3 opening) and gridline rows sticking out of its
band are trimmed. Under the Opera logo it cannot be separated -> gap.
"""
import sys
import json
import numpy as np
from PIL import Image
from scipy import ndimage

IMG = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None

im = np.array(Image.open(IMG).convert('RGB')).astype(int)
H, W, _ = im.shape
R, G, B = im[..., 0], im[..., 1], im[..., 2]
black_px = (R < 90) & (G < 90) & (B < 90)
background = (np.abs(R - 204) <= 14) & (np.abs(G - 236) <= 14) & (np.abs(B - 255) <= 14)
BG_RGB = np.array([204, 236, 255])

def blend_with_bg(rgb, tol=30):
    """Pixels on the segment between the background and colour rgb (anti-aliasing)."""
    c = np.array(rgb, float)
    d = c - BG_RGB
    p = im - BG_RGB
    alpha = np.clip((p @ d) / (d @ d), 0, 1)
    resid = np.linalg.norm(p - alpha[..., None] * d, axis=-1)
    return resid < tol

edge_ok = {'red': background | blend_with_bg((248, 0, 0)),
           'green': background | blend_with_bg((29, 248, 29)),
           'black': background | blend_with_bg((0, 0, 0))}

# ---------------------------------------------------------------- calibration
def refine_line(counts, guess, half=4):
    lo = guess - half
    k = lo + int(np.argmax(counts[lo:guess + half + 1]))
    w = counts[k - 1:k + 2].astype(float)
    return float(np.sum(w * np.arange(k - 1, k + 2)) / np.sum(w))

col_counts = black_px[140:880, :].sum(axis=0)
z_lines = np.arange(0, 6.01, 0.5)
x_lines = np.array([refine_line(col_counts, int(round(201 + 293.5 * z))) for z in z_lines])
px_per_m, x0 = np.polyfit(z_lines, x_lines, 1)
res_x = x_lines - (px_per_m * z_lines + x0)

bz_lines = np.arange(0, 2.01, 0.5)
y_lines = []
for bz in bz_lines:
    cols = (np.r_[210:560] if bz == 0 else np.r_[700:1170] if bz == 2.0
            else np.r_[210:560, 700:1170])
    y_lines.append(refine_line(black_px[:, cols].sum(axis=1), int(round(983.5 - 459.75 * bz))))
y_lines = np.array(y_lines)
px_per_T, y0 = np.polyfit(bz_lines, y_lines, 1)       # px_per_T < 0 (rows grow downwards)
res_y = y_lines - (px_per_T * bz_lines + y0)
to_z = lambda x: (np.asarray(x, float) - x0) / px_per_m
to_bz = lambda y: (np.asarray(y, float) - y0) / px_per_T
print(f'calibration x: {px_per_m:.3f} px/m, x(z=0)={x0:.2f}, max|res|={np.abs(res_x).max():.2f} px')
print(f'calibration y: {-px_per_T:.3f} px/T, y(Bz=0)={y0:.2f}, max|res|={np.abs(res_y).max():.2f} px')

# ------------------------------------------------------------- region masks
x_first, x_last = int(np.ceil(x0)), int(np.floor(x0 + 6 * px_per_m))
plot_area = np.zeros((H, W), bool)
plot_area[int(round(y0 + 2.05 * px_per_T)):1000, x_first:x_last + 1] = True
legend = np.zeros((H, W), bool)
legend[100:460, 1180:1910] = True
# Opera logo: letters are the dark pixels right of x=1870 above the curves
cols_logo = np.where(black_px[945:972, 1870:1990].any(axis=0))[0] + 1870
logo_x0, logo_x1 = cols_logo.min() - 2, cols_logo.max() + 2
logo = np.zeros((H, W), bool)
logo[945:1000, logo_x0:logo_x1 + 1] = True
valid = plot_area & ~legend

red_m = (R > 180) & (G < 150) & (B < 150) & valid
grn_m = (G > 180) & (R < 170) & (B < 170) & valid
blk_m = ndimage.binary_opening(black_px & valid & ~logo, structure=np.ones((3, 3)))
lab, n = ndimage.label(blk_m, structure=np.ones((3, 3)))
sizes = ndimage.sum(blk_m, lab, range(1, n + 1))
blk_m = np.isin(lab, 1 + np.where(sizes > 150)[0])

row_counts = black_px[:, 210:1170].sum(axis=1)
grid_rows = set(np.where(row_counts > 60)[0])
iso = black_px & ~ndimage.binary_opening(black_px, structure=np.ones((3, 1)))
grid_cols = set(np.where(iso[140:880, :].sum(axis=0) > 60)[0])

xs = np.arange(x_first, x_last + 1)

def runs(idx):
    idx = np.asarray(idx)
    if len(idx) == 0:
        return []
    cut = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[cut + 1]]
    ends = np.r_[idx[cut], idx[-1]]
    return list(zip(starts, ends))

def thickness(mask, edge):
    xm, ym, hm = [], [], []
    for x in xs:
        rr = [r for r in runs(np.where(mask[:, x])[0]) if r[1] - r[0] >= 1]
        if len(rr) == 1:
            a, b = rr[0]
            if edge[a - 1, x] and edge[b + 1, x]:
                xm.append(x); ym.append(0.5 * (a + b)); hm.append(b - a + 1)
    xm, ym, hm = map(np.array, (xm, ym, hm))
    sl = np.gradient(ym, xm)
    sel = (np.abs(sl) < 0.2) & (np.r_[np.diff(xm), 9] == 1) & (np.r_[9, np.diff(xm)] == 1)
    return float(np.mean(hm[sel] / np.sqrt(1 + sl[sel] ** 2)))

def band_centre(line_mask, line_bg, idx_pred, h_exp, half_win):
    """1D: line_mask/line_bg are boolean arrays along the scan direction."""
    lo_w = max(int(idx_pred - half_win), 1)
    hi_w = min(int(idx_pred + half_win), len(line_mask) - 2)
    vis = np.where(line_mask[lo_w:hi_w + 1])[0] + lo_w
    if len(vis) == 0:
        return None
    # merge visible runs separated only by non-background pixels
    groups = []
    for a, b in runs(vis):
        if groups and not line_bg[groups[-1][1] + 1:a].any():
            groups[-1][1] = b
        else:
            groups.append([a, b])
    a, b = min(groups, key=lambda g: abs(0.5 * (g[0] + g[1]) - idx_pred))
    n_up = 0
    while a - n_up - 1 >= 0 and not line_bg[a - n_up - 1] and not line_mask[a - n_up - 1] and n_up < 40:
        n_up += 1
    n_dn = 0
    while b + n_dn + 1 < len(line_mask) and not line_bg[b + n_dn + 1] and not line_mask[b + n_dn + 1] and n_dn < 40:
        n_dn += 1
    h = max(h_exp, b - a + 1)
    s_lo = max(a - n_up, b - h + 1)
    s_hi = min(a, b + n_dn - h + 1)
    if s_lo > s_hi:          # band does not fit: thickness estimate too large here
        return 0.5 * (a + b), 0.5 * (h - (b - a + 1))
    return 0.5 * (s_lo + s_hi) + 0.5 * (h - 1), 0.5 * (s_hi - s_lo)

def prediction(mask, guide=None):
    """Centre per column from clean (fully bounded by background) bands."""
    yc = np.full(len(xs), np.nan)
    for i, x in enumerate(xs):
        rr = [r for r in runs(np.where(mask[:, x])[0]) if r[1] - r[0] >= 1]
        if len(rr) == 1:
            yc[i] = 0.5 * (rr[0][0] + rr[0][1])
    ok = ~np.isnan(yc)
    if guide is not None:     # bridge long gaps with the guide curve's shape
        g = np.interp(xs, *guide)
        off = np.interp(xs, xs[ok], yc[ok] - g[ok])
        yfill = g + off
        yfill[ok] = yc[ok]
    else:
        yfill = np.interp(xs, xs[ok], yc[ok])
    return yfill

def extract_coloured(mask, t, yfill, edge):
    slope = np.gradient(ndimage.uniform_filter1d(yfill, 9), xs)
    pts = []
    for i, x in enumerate(xs):
        if abs(slope[i]) > 1.5:
            continue
        h_exp = t * np.sqrt(1 + slope[i] ** 2)
        r = band_centre(mask[:, x], edge[:, x], yfill[i], h_exp, h_exp / 2 + 10)
        if r is None:
            pts.append((float(x), np.nan, np.nan, 'hidden'))
        else:
            pts.append((float(x), r[0], r[1], 'visible' if r[1] < 0.5 else 'partial'))
    pts += row_scan(mask, t, yfill, slope, coloured=True, edge=edge)
    return pts

def row_scan(mask, t, yfill, slope, coloured, edge=None):
    pts = []
    for i0, i1 in runs(np.where(np.abs(slope) > 1.5)[0]):
        seg_y, seg_x = yfill[i0:i1 + 1], xs[i0:i1 + 1]
        order = np.argsort(seg_y)
        for y in range(int(np.ceil(seg_y.min())), int(np.floor(seg_y.max())) + 1):
            if not coloured and y in grid_rows:
                continue
            x_pred = np.interp(y, seg_y[order], seg_x[order])
            j = i0 + int(np.argmin(np.abs(seg_x - x_pred)))
            w_exp = t * np.sqrt(1 + 1 / slope[j] ** 2)
            if coloured:
                r = band_centre(mask[y, :], edge[y, :], x_pred, w_exp, w_exp / 2 + 10)
            else:
                r = band_centre(mask[y, :], ~mask[y, :], x_pred, w_exp, w_exp / 2 + 10)
            if r is not None:
                pts.append((r[0], float(y), r[1] / max(abs(slope[j]), 1e-9),
                            'visible' if r[1] < 0.5 else 'partial'))
    return pts

def extract_black(t, yfill):
    slope = np.gradient(ndimage.uniform_filter1d(yfill, 9), xs)
    pts = []
    for i, x in enumerate(xs):
        if x in grid_cols or abs(slope[i]) > 1.5:
            continue
        if logo_x0 <= x <= logo_x1:
            pts.append((float(x), np.nan, np.nan, 'hidden_under_logo'))
            continue
        h_exp = t * np.sqrt(1 + slope[i] ** 2)
        col = np.where(blk_m[:, x])[0]
        col = col[np.abs(col - yfill[i]) < h_exp / 2 + 10]
        if len(col) == 0:
            pts.append((float(x), np.nan, np.nan, 'hidden'))
            continue
        a, b = col.min(), col.max()
        while b - a + 1 > h_exp + 0.5 and (a in grid_rows or b in grid_rows):
            if a in grid_rows:
                a += 1
            else:
                b -= 1
        pts.append((float(x), 0.5 * (a + b), 0.0, 'visible'))
    pts += row_scan(blk_m, t, yfill, slope, coloured=False)
    return pts

t_blk, t_red, t_grn = (thickness(blk_m, edge_ok['black']), thickness(red_m, edge_ok['red']),
                       thickness(grn_m, edge_ok['green']))
y_blk = prediction(blk_m)
y_red = prediction(red_m)
pts_blk = extract_black(t_blk, y_blk)
pts_red = extract_coloured(red_m, t_red, y_red, edge_ok['red'])
# green hides under red from ~1.9 to ~4.1 m: bridge with the red curve's shape
red_vis = np.array(sorted((p[0], p[1]) for p in pts_red if np.isfinite(p[1])))
y_grn = prediction(grn_m, guide=(red_vis[:, 0], red_vis[:, 1]))
pts_grn = extract_coloured(grn_m, t_grn, y_grn, edge_ok['green'])

curves = {'main_only': (pts_grn, t_grn), 'screening_only': (pts_red, t_red),
          'both': (pts_blk, t_blk)}

def centre_at(pts, x):
    p = [q for q in pts if np.isfinite(q[1]) and abs(q[0] - x) < 0.6]
    return (np.mean([q[1] for q in p]), np.mean([q[2] for q in p])) if p else (np.nan, np.nan)

# Fully hidden coloured points: take the curve on top if the two overlap,
# otherwise interpolate between visible neighbours.
for name, others in [('screening_only', ['both']), ('main_only', ['screening_only', 'both'])]:
    pts, t = curves[name]
    vis = np.array(sorted((p[0], p[1]) for p in pts if np.isfinite(p[1])))
    new = []
    for (x, y, s, f) in pts:
        if np.isfinite(y):
            new.append((x, y, s, f))
            continue
        y_int = np.interp(x, vis[:, 0], vis[:, 1])
        # A fully hidden line lies inside the band of the line on top of it:
        # its centre is within delta = |t_top - t|/2 + 1 px (anti-aliasing) of
        # that line's centre. Take the covering curve nearest to the
        # interpolated estimate, and keep the estimate clipped to +-delta.
        cands = []
        for oc in others:
            yo, so = centre_at(curves[oc][0], x)
            if np.isfinite(yo):
                delta = 0.5 * abs(curves[oc][1] - t) + 1.0
                cands.append((abs(yo - y_int) - delta, oc, yo, delta))
        cands = [c for c in cands if c[0] <= 3.0]   # interpolation itself is good to a few px
        if cands:
            _, oc, yo, delta = min(cands)
            y_h = yo + np.clip(y_int - yo, -delta, delta)
            new.append((x, y_h, delta, f'hidden_under_{oc}'))
        else:
            if logo_x0 <= x <= logo_x1:
                continue   # under the logo: leave a gap
            new.append((x, y_int, 2.0, 'interpolated'))
    curves[name] = (new, t)

out = {}
for name, (pts, t) in curves.items():
    pts = sorted((p for p in pts if np.isfinite(p[1])), key=lambda p: p[0])
    fl = [p[3] for p in pts]
    print(f'{name:15s} t={t:.1f}px  n={len(pts)}  ' + ', '.join(
        f'{k}={fl.count(k)}' for k in sorted(set(fl))))
    sig_px = np.hypot(np.array([p[2] for p in pts]), 0.5)
    out[name] = dict(x_px=[p[0] for p in pts], y_px=[p[1] for p in pts],
                     z=to_z([p[0] for p in pts]).tolist(), Bz=to_bz([p[1] for p in pts]).tolist(),
                     sigma_Bz=(sig_px / abs(px_per_T)).tolist(), flag=fl, thickness_px=t)
out['_calibration'] = dict(x0_px=x0, px_per_m=px_per_m, y0_px=y0, px_per_T=-px_per_T,
                           x_lines_px=x_lines.tolist(), z_lines=z_lines.tolist(),
                           y_lines_px=y_lines.tolist(), bz_lines=bz_lines.tolist(),
                           max_abs_residual_px=[float(np.abs(res_x).max()), float(np.abs(res_y).max())],
                           logo_x_px=[int(logo_x0), int(logo_x1)], image_shape=[H, W])
if OUT:
    json.dump(out, open(OUT, 'w'))
    print('wrote', OUT)
