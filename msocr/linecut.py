"""Our own line cutter: a row of text cut out of the page and straightened along its baseline, in place of Kraken's
`extract_polygons`, so that nothing of Kraken remains in the reading path, and in a form simple enough to be the same
in JavaScript (Manuscribe) and in Python.

The row is a baseline (a polyline, simplified to the few points it needs) and a band around it: `above` pixels over
it and `below` pixels under it (the row finder's band: 0.9 and 0.3 of the line pitch). The line image has one column
for every pixel of the baseline's length and one row for every pixel of the band. Column u is the point at distance u
along the baseline; its direction there is taken from the baseline's course over 8 px either side, so the strip turns
smoothly where the baseline bends; row v is that point moved v - above pixels along the normal (downwards on the page).
Each pixel is read from the page by bilinear interpolation; outside the page it is white.

    cut_line(page, baseline, above, below) -> uint8 array (above + below, length[, channels])

Measured on LDII_85885's 19 held-out pages (recogniser run3, trained on Kraken's crops), 2026-11-24: 2.96 % of
characters wrong on Kraken's lines (3.38 % with Kraken's cutter), 2.60 % on the U-Net's (3.28 %). The default in
predict since then (MSOCR_CUTTER=kraken brings Kraken's back). Recognisers are still trained on Kraken's crops
(recognizer.build_cache); the next training should rebuild its cache with this cutter.
"""
import numpy as np

from .params import P as _P
SMOOTH = _P["cutter"]["smooth"]   # px: the baseline's direction at a point is taken over this distance either side of it


def _along(pts, u):
    """Points at arc lengths u along the polyline pts (clamped to its ends)."""
    seg = np.diff(pts, axis=0)
    lens = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    u = np.clip(u, 0.0, cum[-1])
    k = np.clip(np.searchsorted(cum, u, side="right") - 1, 0, len(lens) - 1)
    t = (u - cum[k]) / np.maximum(lens[k], 1e-9)
    return pts[k] + seg[k] * t[:, None]


def _bilinear(img, x, y):
    """img sampled at float coordinates (x, y), white outside."""
    h, w = img.shape[:2]
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    out = 0.0
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            xi, yi = x0 + dx, y0 + dy
            inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            v = np.where(inside[..., None] if img.ndim == 3 else inside,
                         img[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)].astype(np.float64), 255.0)
            wgt = (wx * wy)[..., None] if img.ndim == 3 else wx * wy
            out = out + v * wgt
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def cut_line(page, baseline, above, below):
    """page: H x W (x C) uint8 array; baseline: [(x, y), ...] in page pixels; above, below: the band in pixels."""
    pts = np.asarray(baseline, dtype=np.float64)
    keep = np.concatenate([[True], np.hypot(*np.diff(pts, axis=0).T) > 1e-6])
    pts = pts[keep]
    if len(pts) < 2:
        pts = np.array([pts[0], pts[0] + [1.0, 0.0]])
    length = float(np.hypot(*np.diff(pts, axis=0).T).sum())
    u = np.arange(int(round(length)) + 1, dtype=np.float64)
    p = _along(pts, u)
    d = _along(pts, u + SMOOTH) - _along(pts, u - SMOOTH)
    d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)                 # the normal, pointing down the page for a level line
    above, below = int(round(above)), int(round(below))
    v = np.arange(above + below, dtype=np.float64) - above + 0.5
    x = p[None, :, 0] + v[:, None] * n[None, :, 0]
    y = p[None, :, 1] + v[:, None] * n[None, :, 1]
    return _bilinear(page, x, y)


def band_of(baseline, outline):
    """(above, below) in pixels from a row's merged baseline and outline as align._merge builds them (the outline is
    the baseline moved up by `above` and, reversed, down by `below`; the top may be clamped at the page's edge, so the
    largest offset is taken)."""
    n = len(baseline)
    top, bottom = outline[:n], outline[n:][::-1]
    above = max(b[1] - t[1] for b, t in zip(baseline, top))
    below = max(t[1] - b[1] for b, t in zip(baseline, bottom))
    return above, below
