"""The blocks of a tripāṭha or pañcapāṭha page, and the text rows in each.

A page of these layouts holds the main text in the middle and a commentary in blocks around
it: above and below (tripāṭha), or above, right, left and below (pañcapāṭha); some pages lack
a block. The blocks are named as Manuscribe names its boxes: main, top, right, left, bottom.

How the blocks are found, from the pieces Kraken detected and the page image:
  1. columns: the stretches of the page width that pieces cover, parted at the rules drawn
     down the page (thin red or black lines) and at blank vertical strips with writing on both
     sides, both found in the image; a piece that crosses one is cut there. The column holding the page's centre, or the
     widest, is the centre column; narrower columns to its left and right that are long enough
     are the side blocks. Two columns of similar width that together straddle the centre are
     the two halves of the main text (verses written in two columns) and count as one.
  2. bands: the rows of the centre column (pieces at one height) are cut into bands where
     the page says a block ends: at a blank horizontal run of the column's ink profile clearly
     longer than the gaps between lines; where the line spacing changes (a change point in the
     sequence of row gaps, the rows at the boundary going to the side whose spacing they fit,
     the body height of the lines breaking ties); and where the body height of the lines, the
     hand size measured on the page, changes sharply on its own. A cut on a blank run alone
     is kept only if the hands or the line pitches on its two sides differ, so a paragraph
     space inside one block does not split it.
  3. names: with one band it is the main text. With more, the band whose hand and pitch are
     clearly the largest is the main text; when they are alike, the middle band of three, or
     the band nearer the page's middle of two, and the page is noted as doubtful. Bands above
     the main text are the top block, below it the bottom block.
  4. rows: each block is handed to the row finder of align.py as if it were a small page of
     its own, with the pitch of its own rows.
Pieces outside every block (marginal notes, specks) are left out.
"""
import csv
import functools
import json
import statistics

import numpy as np

from .align import _info, _ink, _merge, fill_edge_rows, find_rows, image_pitch, uneven
from .config import DATA, MANUSCRIPTS, SEG, ms_dir

BOXES = ("main", "top", "right", "left", "bottom")                       # Manuscribe's names and commentary order
COLOR = {"main": (217, 72, 15), "top": (43, 138, 62), "bottom": (25, 113, 194), "left": (156, 54, 181), "right": (12, 133, 153)}
EXPECTED = {"linear": ("main",), "tripatha": ("main", "top", "bottom"), "panchapatha": BOXES}


@functools.lru_cache(maxsize=None)
def layout_of(ms_id):
    """The manuscript's layout from data/layouts.csv: linear, tripatha, panchapatha or special."""
    f = DATA / "layouts.csv"
    if f.exists():
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["ms_id"] == ms_id:
                return r["layout"]
    return "linear"


def _runs(mask, min_gap):
    """Stretches of True in mask, with gaps shorter than min_gap closed. -> [(start, end)]"""
    runs, start = [], None
    for i, on in enumerate(list(mask) + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append([start, i])
            start = None
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [tuple(r) for r in merged]


def _pieces(seg):
    out = []
    for l in seg["lines"]:
        if len(l["boundary"]) < 3:
            continue
        d = _info(l)
        ys = [y for _, y in l["boundary"]]
        d["y0"], d["y1"] = min(ys), max(ys)
        d["h"] = d["y1"] - d["y0"]
        out.append(d)
    return out


def separators(gray, width, cov):
    """x positions that part the columns, from the image and from where Kraken's pieces are:
      - a rule drawn down the page (a thin red or black line: ink down most of the page's middle
        height, few pieces on it),
      - a corridor no piece covers, between pieces on both sides,
      - a blank vertical strip of the ink profile with writing on both sides."""
    h = gray.shape[0]
    area = gray[int(0.15 * h):int(0.85 * h)].astype(np.float32)
    p5, p50 = np.percentile(area, [5, 50])
    frac = (area < (p5 + p50) / 2).mean(axis=0)                   # ink down the middle height, per x
    rule = (area < p50 - 0.25 * (p50 - p5)).mean(axis=0)         # the same with red ink counted
    peak = np.percentile(frac, 90)
    seps = set()
    for x0, x1 in _runs(rule > 0.45, 2):
        if x1 - x0 <= 10 and cov[x0:x1 + 1].mean() <= 1.5:
            seps.add((x0 + x1) // 2)
    inked = frac >= 0.15 * max(peak, 1e-6)
    for x0, x1 in _runs(cov[:width] == 0, 1):
        if x1 - x0 >= 6 and 0.03 * width < x0 and x1 < 0.97 * width and cov[max(0, x0 - 60):x0].any() and cov[x1:x1 + 60].any():
            seps.add((x0 + x1) // 2)
    for x0, x1 in _runs(~inked, 1):
        if x1 - x0 >= 4 and 0.03 * width < x0 and x1 < 0.97 * width and inked[max(0, x0 - 60):x0].any() and inked[x1:x1 + 60].any():
            seps.add((x0 + x1) // 2)
    return sorted(seps)


def rules(gray):
    """x positions of the rules drawn down the page (kept for the pictures)."""
    width = gray.shape[1]
    return separators(gray, width, np.zeros(width + 1))


def _cut_piece(d, x):
    """A piece crossing a rule at x, cut into the part left of it and the part right of it."""
    parts = []
    for side in ("left", "right"):
        pts = [p for p in d["line"]["baseline"] if (p[0] < x) == (side == "left")]
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        bnd = [[min(max(px, x0 := min(xs)), x1 := max(xs)), py] for px, py in d["line"]["boundary"]]
        line = {"baseline": pts, "boundary": bnd}
        parts.append({"x0": x0, "x1": x1, "y": d["y"], "y0": d["y0"], "y1": d["y1"], "h": d["h"], "line": line})
    return parts


def _columns(pieces, width, gray=None):
    cov = np.zeros(width + 1)
    for d in pieces:
        if d["x1"] - d["x0"] > 0.01 * width:
            cov[int(max(0, d["x0"])):int(min(width, d["x1"])) + 1] += 1
    cuts = separators(gray, width, cov) if gray is not None else []
    if cuts:
        split = []
        for d in pieces:
            inner = [x for x in cuts if d["x0"] + 8 < x < d["x1"] - 8]
            if inner:
                parts = [d]
                for x in inner:
                    parts = [q for d2 in parts for q in (_cut_piece(d2, x) if d2["x0"] + 8 < x < d2["x1"] - 8 else [d2])]
                split.extend(parts)
            else:
                split.append(d)
        pieces = split
    cov = np.zeros(width + 1)
    for d in pieces:
        if d["x1"] - d["x0"] > 0.01 * width:
            cov[int(max(0, d["x0"])):int(min(width, d["x1"])) + 1] += 1
    for x in cuts:
        cov[max(0, x - 4):x + 5] = 0
    runs = [r for r in _runs(cov > 0, 0.008 * width) if r[1] - r[0] > 0.03 * width]
    cols = []
    for x0, x1 in runs:
        inside = []
        for d in pieces:
            overlap = min(d["x1"], x1) - max(d["x0"], x0)
            if overlap >= 0.6 * max(1, d["x1"] - d["x0"]):
                if d["x0"] < x0 or d["x1"] > x1:          # overhangs the column a little: clipped to it
                    d = dict(d, x0=max(d["x0"], x0), x1=min(d["x1"], x1))
                inside.append(d)
        if len(inside) >= 2:
            cols.append({"x0": x0, "x1": x1, "pieces": inside})
    return cols


def _centre(cols, width):
    """Index of the centre column; two half-columns of the main text are merged first."""
    if not cols:
        return None, cols
    cols = sorted(cols, key=lambda c: c["x0"])
    for i in range(len(cols) - 1):
        a, b = cols[i], cols[i + 1]
        wa, wb = a["x1"] - a["x0"], b["x1"] - b["x0"]
        if min(wa, wb) > 0.25 * width and min(wa, wb) / max(wa, wb) > 0.6 and a["x0"] < width / 2 < b["x1"]:
            cols[i:i + 2] = [{"x0": a["x0"], "x1": b["x1"], "pieces": a["pieces"] + b["pieces"], "halves": True}]
            break
    mid = [i for i, c in enumerate(cols) if c["x0"] <= width / 2 <= c["x1"]]
    k = mid[0] if mid else max(range(len(cols)), key=lambda i: cols[i]["x1"] - cols[i]["x0"])
    return k, cols


def _rows(pieces, cw):
    """Pieces grouped by height into rows, top to bottom; rows of specks (under a tenth of the column) dropped."""
    rows, cur = [], [pieces[0]]
    for d in pieces[1:]:
        if d["y"] - cur[-1]["y"] < 0.45 * max(d["h"], cur[-1]["h"]):
            cur.append(d)
        else:
            rows.append(cur)
            cur = [d]
    rows.append(cur)
    out = []
    for r in rows:
        if sum(d["x1"] - d["x0"] for d in r) < 0.1 * cw:
            continue
        wide = [d for d in r if d["x1"] - d["x0"] > 0.2 * cw] or r
        out.append({"pieces": r, "y": statistics.mean(d["y"] for d in r), "h": statistics.median(d["h"] for d in wide),
                    "y0": min(d["y0"] for d in r), "y1": max(d["y1"] for d in r)})
    return out


def _blank_runs(gray, x0, x1):
    """Stretches of the page height with no ink across the column, from its ink profile. -> [(y0, y1)]"""
    area = gray[:, int(x0) + 10:int(x1) - 10].astype(np.float32)
    if area.shape[1] < 10:
        return []
    p5, p50 = np.percentile(area, [5, 50])
    prof = (area < (p5 + p50) / 2).mean(axis=1)
    peak = np.percentile(prof, 90)
    return _runs(prof < 0.08 * max(peak, 1e-6), 1)


def _pitch_of(gray, x0, x1, band):
    """Line pitch of a band of rows: the typical spacing of its rows, the larger gaps (a line Kraken
    missed) left out; a single row's height when there is nothing else."""
    rows = band["rows"]
    gaps = sorted(b["y"] - a["y"] for a, b in zip(rows, rows[1:]))
    if not gaps:
        return band["h"]
    usual = [g for g in gaps if g <= 1.7 * gaps[0]] or gaps[:1]     # a gap of two pitches is a missed line
    return statistics.median(usual)


def _band(rows):
    return {"rows": rows, "pieces": [d for r in rows for d in r["pieces"]], "y0": rows[0]["y0"], "y1": rows[-1]["y1"],
            "h": statistics.median(r["h"] for r in rows)}


def _fits(gap, pitch):
    """Does a gap look like one or two line pitches (a line in between that Kraken missed)?"""
    return abs(gap - pitch) <= 0.35 * pitch or abs(gap - 2 * pitch) <= 0.25 * 2 * pitch


def _bodies(gray, x0, x1, rows):
    """Body height of each row from the page: the dense band of its ink profile, measured between its
    neighbours over the row's widest piece. None where the measure is unreliable."""
    out = []
    ys = [r["y"] for r in rows]
    for i, r in enumerate(rows):
        d = max(r["pieces"], key=lambda d: d["x1"] - d["x0"])
        xa, xb = int(max(x0, d["x0"])), int(min(x1, d["x1"]))
        if xb - xa < 30:
            out.append(None)
            continue
        col = gray[:, xa:xb].astype(np.float32)
        p5, p50 = np.percentile(col, [5, 50])
        prof = (col < (p5 + p50) / 2).mean(axis=1)
        y = ys[i]
        lo = (ys[i - 1] + y) / 2 if i > 0 else (y - (ys[1] - y) / 2 if len(ys) > 1 else y - 20)
        hi = (y + ys[i + 1]) / 2 if i + 1 < len(ys) else (y + (y - ys[i - 1]) / 2 if len(ys) > 1 else y + 20)
        lo, hi = int(max(0, lo)), int(min(len(prof), hi))
        w = prof[lo:hi]
        if len(w) < 4 or w.max() == 0:
            out.append(None)
            continue
        k = int(np.argmax(w))
        a = b = k
        while a > 0 and w[a - 1] >= 0.4 * w[k]:
            a -= 1
        while b + 1 < len(w) and w[b + 1] >= 0.4 * w[k]:
            b += 1
        out.append(None if a == 0 or b == len(w) - 1 or b - a < 2 else b - a + 1)
    return out


def _normalised(gaps):
    """Row gaps with a gap of two pitches (a line Kraken missed) halved, so that it does not look like a change of spacing."""
    out = list(gaps)
    for i, g in enumerate(gaps):
        near = gaps[max(0, i - 2):i] + gaps[i + 1:i + 3]
        if near and abs(g - 2 * statistics.median(near)) <= 0.5 * statistics.median(near):
            out[i] = g / 2
    return out


def _line_rows(pieces, cw):
    """The centre column's pieces as rows, one per written line: a row within two thirds of the going pitch under
    the previous one is a piece of the same line (vowel signs or a split line Kraken drew separately)."""
    rows = _rows(sorted(pieces, key=lambda d: d["y"]), cw)
    if not rows:
        return []
    kept = []
    gaps0 = _normalised([b["y"] - a["y"] for a, b in zip(rows, rows[1:])])
    for i, r in enumerate(rows):
        before, after = gaps0[max(0, i - 3):i - 1], gaps0[i:i + 2]   # the gaps before this row's own gap, and after the next
        own = r["y"] - kept[-1]["y"] if kept else None
        if own is not None and before and after and own < 0.65 * statistics.median(before) and own < 0.65 * statistics.median(after):
            a = kept[-1]
            wide = max((a, r), key=lambda q: max(d["x1"] - d["x0"] for d in q["pieces"]))
            kept[-1] = {"pieces": a["pieces"] + r["pieces"], "y": wide["y"], "h": max(a["h"], r["h"]),
                        "y0": min(a["y0"], r["y0"]), "y1": max(a["y1"], r["y1"])}
        else:
            kept.append(r)
    return kept


def _letter_height(gray, row, x0, x1, pitch, thr):
    """A row's letter height: from its baseline up to its headline (the shirorekhā, the strongest horizontal line of ink
    between a fifth and four fifths of a pitch above the baseline). None where the row is too narrow to measure."""
    y = float(np.median([p[1] for d in row["pieces"] for p in d["line"]["baseline"]]))
    xs = [(max(x0, d["x0"]), min(x1, d["x1"])) for d in row["pieces"] if d["x1"] - d["x0"] > 0.05 * (x1 - x0)]
    lo, hi = int(max(0, y - 0.8 * pitch)), int(min(gray.shape[0], y - 0.2 * pitch))
    if not xs or hi - lo < 3:
        return None
    prof = np.zeros(hi - lo)
    for a, b in xs:
        prof += (gray[lo:hi, int(a):int(b)] < thr).sum(axis=1)
    return y - (lo + int(np.argmax(prof)))


def _clearance(ink, cap):
    """For every pixel, the vertical distance to the nearest ink above or below it in its column (capped)."""
    h, w = ink.shape
    up = np.full((h, w), cap, dtype=np.float32)
    dn = np.full((h, w), cap, dtype=np.float32)
    run = np.full(w, cap, dtype=np.float32)
    for y in range(h):
        run = np.where(ink[y], 0, np.minimum(run + 1, cap))
        up[y] = run
    run = np.full(w, cap, dtype=np.float32)
    for y in range(h - 1, -1, -1):
        run = np.where(ink[y], 0, np.minimum(run + 1, cap))
        dn[y] = run
    return np.minimum(up, dn)


def _seam(cost, start, vpen):
    """The cheapest path from the left edge (starting between rows start[0] and start[1]) to the right edge, one step
    to the right per column, moving up or down within a column as far as it likes at cost + vpen a pixel (so it can step
    round the end of a half line through blank paper). -> y of the path in every column."""
    h, w = cost.shape
    D = np.empty((h, w), dtype=np.float64)
    big = 1e12
    col = np.full(h, big)
    col[start[0]:start[1]] = cost[start[0]:start[1], 0]
    D[:, 0] = col
    for x in range(1, w):
        c = D[:, x - 1] + cost[:, x]
        step = cost[:, x] + vpen
        S = np.cumsum(step)
        c = np.minimum(c, np.minimum.accumulate(c - S) + S)                    # moving down the column
        Sr = np.cumsum(step[::-1])
        cr = c[::-1]
        c = np.minimum(c, (np.minimum.accumulate(cr - Sr) + Sr)[::-1])        # moving up the column
        D[:, x] = c
    ys = np.empty(w, dtype=np.int64)
    y = int(np.argmin(D[:, w - 1]))
    for x in range(w - 1, 0, -1):
        ys[x] = y
        step = cost[:, x] + vpen
        S = np.concatenate(([0.0], np.cumsum(step)))
        lo, hi = np.minimum(np.arange(h), y), np.maximum(np.arange(h), y)
        move = S[hi + 1] - S[lo] - step[y]                                    # walking from y' to y in this column
        y = int(np.argmin(D[:, x - 1] + cost[y, x] + move))
    ys[0] = y
    return ys


def _seam_bands(pieces, gray, x0, x1):
    """Top, main text and bottom of a tripāṭha or pañcapāṭha centre column, cut along the blank space between them.
    Where a block ends in a half line, the next block begins beside it, a little lower: no straight blank band runs
    across, but the white space between the blocks still does, stepping round the end of the half line (a sigmoid, or a
    reversed one). From every gap between two lines a path is found across the column through the clearest white space
    within a line and a fifth of it (_seam, on the distance of every pixel from the nearest ink above or below it);
    those whose white space is clearly wider than that of most such paths (the gaps between lines) are the cuts, and
    each piece goes to the block on its side of them.
    -> {"top": band, "main": band, "bottom": band} (absent ones left out), or None when no cut stands out."""
    rows = _line_rows(pieces, x1 - x0)
    if len(rows) < 3:
        return None
    norm = _normalised([b["y"] - a["y"] for a, b in zip(rows, rows[1:])])
    pitch = statistics.median(norm)
    f = 2                                                                   # work at half resolution
    Y0 = int(max(0, min(d["y0"] for d in pieces)))
    Y1 = int(min(gray.shape[0], max(d["y1"] for d in pieces)))
    X0 = int(max(0, max(x0, min(d["x0"] for d in pieces))))                # the writing's own extent, not the margins
    X1 = int(min(gray.shape[1], min(x1, max(d["x1"] for d in pieces))))
    crop = gray[Y0:Y1:f, X0:X1:f].astype(np.float32)
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    p5, p50 = np.percentile(crop, [5, 50])
    ink = crop < (p5 + p50) / 2
    ink[0, :] = ink[-1, :] = True                                           # walls: no cut along the edges, and
    ink[:, :2] = ink[:, -2:] = True                                         # no running up and down the margins
    cap = max(4.0, pitch / f)
    clear = _clearance(ink, cap)
    cost = 1.0 / (1.0 + clear) + 4.0 * ink
    # one candidate path from every gap between two rows at the left edge
    left = sorted((r for r in rows), key=lambda r: r["y"])
    cands = []
    for a, b in zip(left, left[1:]):
        s0, s1 = int((a["y"] - Y0) / f), int((b["y"] - Y0 - 0.5 * pitch) / f)
        s0, s1 = max(0, s0), min(cost.shape[0], max(s0 + 1, s1))
        mid = (s0 + s1) / 2                 # each path keeps near its own gap: room for one step round a half line
        lo, hi = int(max(0, mid - 1.2 * pitch / f)), int(min(cost.shape[0], mid + 1.2 * pitch / f))
        ys = _seam(cost[lo:hi], (s0 - lo, s1 - lo), 1.0) + lo
        cl = clear[ys, np.arange(len(ys))]
        cands.append((float(np.percentile(cl, 25)), ys))
    # the same white space reached from several starts counts once; a cut must have a line of writing on each side
    first, last = (rows[0]["y"] - Y0) / f, (rows[-1]["y"] - Y0) / f
    distinct = []
    for score, ys in sorted(cands, key=lambda c: -c[0]):
        if not first + 0.3 * pitch / f < float(np.mean(ys)) < last - 0.3 * pitch / f:
            continue
        if all(np.mean(np.abs(ys - o)) > 0.5 * pitch / f for _, o in distinct):
            distinct.append((score, ys))
    if len(distinct) < 2:
        return None
    usual = statistics.median(c for c, _ in distinct)                     # most white space lies between lines
    cuts = [ys for score, ys in distinct if score >= max(1.6 * usual, 0.25 * pitch / f)][:2]
    if not cuts:
        return None
    cuts.sort(key=lambda ys: float(np.mean(ys)))

    def side(d):
        xm = int(((d["x0"] + d["x1"]) / 2 - X0) / f)
        xm = min(max(xm, 0), cost.shape[1] - 1)
        return sum(1 for ys in cuts if (d["y"] - Y0) / f > ys[xm])
    groups = [[], [], []]
    for d in pieces:
        groups[side(d)].append(d)
    if len(cuts) == 1:                      # one cut: the side written larger, or the longer side, is the main text
        a, b = groups[0], groups[1]
        ha = statistics.median(d["h"] for d in a) if a else 0
        hb = statistics.median(d["h"] for d in b) if b else 0
        names = ("top", "main") if (hb, len(b)) > (ha, len(a)) else ("main", "bottom")
        parts = {names[0]: a, names[1]: b}
    else:
        parts = {"top": groups[0], "main": groups[1], "bottom": groups[2]}
    named = {}
    for name, ps in parts.items():
        prow = _line_rows(ps, x1 - x0) if ps else []
        if prow:
            band = _band(prow)
            own = _normalised([q["y"] - p["y"] for p, q in zip(prow, prow[1:])])
            band["pitch"] = statistics.median(own) if own else pitch
            named[name] = band
    return named if "main" in named else None


def _hand_bands(pieces, gray, x0, x1):
    """Top, main text and bottom of a tripāṭha or pañcapāṭha centre column, from the size of the hand: the main text is
    written larger than the commentary around it. Three measures of each line, each compared with the page's own lines:
    the letter height (baseline to headline), the line spacing, and the height of Kraken's pieces. The main text is the
    one run of lines where they stand out together; its edges are then set at a blank gap of the column next to them,
    and a line at an edge goes to the side whose hand two of its three measures match.
    -> {"top": band, "main": band, "bottom": band} (the absent ones left out), or None when no run stands out clearly
    (at least two measures 12% larger inside than outside), and the cuts are left to _bands."""
    import math
    rows = _line_rows(pieces, x1 - x0)
    n = len(rows)
    if n < 3:
        return None
    ys = [r["y"] for r in rows]
    gaps = [b - a for a, b in zip(ys, ys[1:])]
    norm = _normalised(gaps)
    pitch = statistics.median(norm)
    area = gray[:, int(x0):int(x1)]
    p5, p50 = np.percentile(area, [5, 50])
    thr = (p5 + p50) / 2
    letter = [_letter_height(gray, r, x0, x1, pitch, thr) for r in rows]
    spacing = []
    for i in range(n):                                  # the gaps on either side that are one line apart
        g = [norm[j] for j in (i - 1, i) if 0 <= j < n - 1 and norm[j] < 1.6 * pitch]
        spacing.append(statistics.mean(g) if g else None)
    piece = [r["h"] for r in rows]
    feats = [letter, spacing, piece]

    def z(vals):
        xs = [math.log(v) for v in vals if v and v > 0]
        if len(xs) < 3:
            return [None] * len(vals)
        m, sd = statistics.mean(xs), statistics.pstdev(xs) or 1.0
        return [(math.log(v) - m) / sd if v and v > 0 else None for v in vals]
    zs = [z(f) for f in feats]
    comb = [statistics.mean([f[i] for f in zs if f[i] is not None] or [0.0]) for i in range(n)]
    best = None
    for a in range(n):
        for b in range(a, n):
            inside, out = comb[a:b + 1], comb[:a] + comb[b + 1:]
            if not out:
                continue
            sc = (statistics.mean(inside) - statistics.mean(out)) * math.sqrt(len(inside) * len(out) / n)
            if best is None or sc > best[0]:
                best = (sc, a, b)
    if best is None:
        return None
    _, a, b = best
    big = {i for i in range(n - 1) if gaps[i] >= 1.6 * pitch}     # blank gaps between two lines

    def size(f, i, a, b):                                # is line i's measure f nearer the main text's or the commentary's?
        if not f[i]:
            return None
        ins = [f[k] for k in range(a, b + 1) if k != i and f[k]]
        out = [f[k] for k in list(range(0, a)) + list(range(b + 1, n)) if k != i and f[k]]
        if not ins or not out:
            return None
        return abs(math.log(f[i] / statistics.median(ins))) < abs(math.log(f[i] / statistics.median(out)))

    def votes(i, a, b):
        v = [size(f, i, a, b) for f in feats]
        return sum(1 for x in v if x is True), sum(1 for x in v if x is False)
    # edges at a blank gap within a line of them
    for i in (a - 1, a - 2, a):
        if i in big and 0 <= i < n - 1 and i + 1 <= b:
            a = i + 1
            break
    for i in (b, b - 1, b + 1):
        if i in big and a <= i < n - 1:
            b = i
            break
    # lines at the edges go to the side whose hand they match
    for _ in range(n):
        moved = False
        if b > a and votes(a, a, b)[1] >= 2 and (a - 1) not in big and a not in big:
            a += 1; moved = True
        if b > a and votes(b, a, b)[1] >= 2 and b not in big and (b - 1) not in big:
            b -= 1; moved = True
        if a > 0 and (a - 1) not in big and votes(a - 1, a, b)[0] >= 2:
            a -= 1; moved = True
        if b < n - 1 and b not in big and votes(b + 1, a, b)[0] >= 2:
            b += 1; moved = True
        if not moved:
            break
    # clear enough?
    clear = 0
    for f in feats:
        ins = [f[k] for k in range(a, b + 1) if f[k]]
        out = [f[k] for k in list(range(0, a)) + list(range(b + 1, n)) if f[k]]
        if ins and out and statistics.median(ins) / statistics.median(out) >= 1.12:
            clear += 1
    if clear < 2:
        return None
    named = {}
    for name, part in (("top", rows[:a]), ("main", rows[a:b + 1]), ("bottom", rows[b + 1:])):
        if part:
            band = _band(part)
            own = [g for g in norm[rows.index(part[0]):rows.index(part[-1])]]
            band["pitch"] = statistics.median(own) if own else _pitch_of(gray, x0, x1, band)
            named[name] = band
    return named


def _bands(pieces, gray, x0, x1):
    """The centre column's rows cut into bands.
    Cuts come from: a blank run of the column's ink (hard), a change of line spacing found as a
    change point in the sequence of row gaps, the rows at the boundary going to the side whose
    spacing they fit (one or two pitches), with the body height of the lines breaking ties; and a
    sharp change of body height on its own. A lone row left by a body cut joins the neighbour it
    is spaced like; a blank cut between two alike bands is undone."""
    import math
    cw = x1 - x0
    rows = _line_rows(pieces, cw)
    if not rows:
        return []
    n = len(rows)
    if n == 1:
        b = _band(rows)
        b["pitch"] = _pitch_of(gray, x0, x1, b)
        return [b]
    ys = [r["y"] for r in rows]
    gaps = [b - a for a, b in zip(ys, ys[1:])]
    norm = _normalised(gaps)
    body = _bodies(gray, x0, x1, rows)
    nearest = lambda ks: med([body[k] for k in ks if 0 <= k < n and body[k]][:3])
    med_h = statistics.median(r["h"] for r in rows)
    med = lambda xs: statistics.median(xs) if xs else None
    blanks = [(a, b) for a, b in _blank_runs(gray, x0, x1) if b - a >= max(8, 0.35 * med_h)]
    cuts = {}                                               # i: cut between rows i and i+1 -> kind
    share = [sum(d["x1"] - d["x0"] for d in r["pieces"]) / cw for r in rows]
    for ya, yb in blanks:
        # rows whose baseline lies in the run: short lines, too little ink across the width to break it
        inside = [i for i in range(n) if ya <= ys[i] <= yb]
        if any(share[i] >= 0.5 for i in inside):
            continue
        above = [i for i in range(n) if ys[i] < ya]
        below = [i for i in range(n) if ys[i] > yb]
        if not above or not below:
            continue
        # a short line in the gap is the last line of the block above
        cut = max(inside) if inside else above[-1]
        if cut < below[0] and cut < n - 1:
            cuts[cut] = "blank"
    # change points of the spacing: the gaps before and after differ (the gap itself left out)
    score = {}
    for i in range(n - 1):
        before, after = norm[max(0, i - 2):i], norm[i + 1:i + 3]
        if before and after:
            sc = abs(math.log(med(after) / med(before)))
            if sc >= math.log(1.3):
                score[i] = sc
    runs, cur = [], []
    for i in sorted(score):
        if cur and i == cur[-1] + 1:
            cur.append(i)
        else:
            cur = [i]
            runs.append(cur)
    for run in runs:
        i0, i1 = run[0], run[-1]
        if any(i0 - 1 <= c <= i1 + 1 for c in cuts):
            continue                                        # a blank run already settles this boundary
        pa = med(norm[max(0, i0 - 3):i0]) or norm[i0]
        pb = med(norm[i1 + 1:i1 + 4]) or norm[i1]
        ba = nearest(range(i0, i0 - 6, -1))                   # the nearest measured bodies on either side
        bb = nearest(range(i1 + 1, i1 + 7))
        cut, prev = i1, i0
        for r in range(i0 + 1, i1 + 1):                     # the rows between the first and the last candidate gap
            ga, gb = ys[r] - ys[prev], (ys[r + 1] - ys[r] if r + 1 < n else None)
            fit_a = _fits(ga, pa)
            fit_b = gb is not None and _fits(gb, pb)
            if fit_b and not fit_a:
                cut = r - 1
                break
            if fit_a and not fit_b:
                prev = r
                continue
            if body[r] and ba and bb:
                to_b = abs(body[r] - bb) < abs(body[r] - ba)
            elif body[r] and ba:                             # a hand unlike the band above belongs below
                to_b = abs(body[r] - ba) / ba >= 0.15
            elif body[r] and bb:
                to_b = abs(body[r] - bb) / bb < 0.15
            else:                                            # which spacing does it fit more closely?
                to_b = gb is not None and abs(gb - pb) / pb < abs(ga - pa) / pa
            if to_b:
                cut = r - 1
                break
            prev = r
        cuts[cut] = "pitch"
    # a wide gap, blank between the lines, with the hand changing across it: a block boundary even
    # when the spacing on both sides is alike (the commentary's last lines under the main text)
    for i in range(n - 1):
        if any(abs(i - c) <= 1 for c in cuts):
            continue
        near = norm[max(0, i - 3):i] + norm[i + 1:i + 4]
        if not near or gaps[i] < 1.35 * med(near):
            continue
        before = [body[k] for k in range(max(0, i - 2), i + 1) if body[k]]
        after = [body[k] for k in range(i + 1, min(n, i + 4)) if body[k]]
        if before and after and max(med(before), med(after)) / min(med(before), med(after)) >= 1.1 \
                and any(ys[i] - 0.35 * rows[i]["h"] < ya and yb < ys[i + 1] + 0.2 * rows[i + 1]["h"] for ya, yb in _blank_runs(gray, x0, x1) if yb - ya >= 6):
            cuts[i] = "gap"
    # a sharp change of body height with no cut near it
    for i in range(n - 1):
        if any(abs(i - c) <= 1 for c in cuts) or not body[i] or not body[i + 1]:
            continue
        before = [body[k] for k in range(max(0, i - 2), i + 1) if body[k]]
        after = [body[k] for k in range(i + 1, min(n, i + 4)) if body[k]]
        if len(before) >= 2 and len(after) >= 2 and max(med(after), med(before)) / min(med(after), med(before)) >= 1.5 \
                and max(body[i], body[i + 1]) / min(body[i], body[i + 1]) >= 1.3:
            cuts[i] = "body"
    bands, kinds, start = [], [None], 0
    for i in sorted(cuts) + [n - 1]:
        b = _band(rows[start:i + 1])
        own = norm[start:i]
        b["pitch"] = med(own) or _pitch_of(gray, x0, x1, b)
        bs = [body[k] for k in range(start, i + 1) if body[k]]
        b["hand"] = med(bs) if len(bs) >= 2 else 0.45 * b["h"]
        bands.append(b)
        if i in cuts:
            kinds.append(cuts[i])
        start = i + 1
    # a lone row left by a body cut joins the neighbour it is spaced like
    changed = True
    while changed and len(bands) > 1:
        changed = False
        for i, b in enumerate(bands):
            if len(b["rows"]) > 1:
                continue
            y = b["rows"][0]["y"]
            options = []
            if i > 0 and kinds[i] == "body":
                a = bands[i - 1]
                options.append((abs(a["h"] - b["h"]), i - 1, _fits(y - a["rows"][-1]["y"], a["pitch"])))
            if i + 1 < len(bands) and kinds[i + 1] == "body":
                c = bands[i + 1]
                options.append((abs(c["h"] - b["h"]), i + 1, _fits(c["rows"][0]["y"] - y, c["pitch"])))
            fitting = [o for o in options if o[2]]
            if not fitting:
                continue
            _, j, _ = min(fitting)
            lo, hi = min(i, j), max(i, j)
            m = _join(bands[lo], bands[hi], gray, x0, x1)
            bands[lo:hi + 1] = [m]
            del kinds[hi]
            changed = True
            break
    # a cut on a blank run alone is kept only where the two sides differ in hand or pitch, or the run is wide
    merged = [bands[0]]
    for b, kind in zip(bands[1:], kinds[1:]):
        prev = merged[-1]
        alike = max(prev["hand"], b["hand"]) / max(1, min(prev["hand"], b["hand"])) < 1.1 and max(prev["pitch"], b["pitch"]) / max(1, min(prev["pitch"], b["pitch"])) < 1.2
        run = max((yb - ya for ya, yb in blanks if prev["rows"][-1]["y"] < ya < b["rows"][0]["y"]), default=0)
        if kind == "blank" and alike and run < 0.5 * med_h:
            merged[-1] = _join(prev, b, gray, x0, x1)
        else:
            merged.append(b)
    return _rebalance(merged, gray, x0, x1)


def _join(a, b, gray, x0, x1):
    """Two neighbouring bands as one, its pitch and hand from both."""
    m = _band(a["rows"] + b["rows"])
    m["pitch"] = _pitch_of(gray, x0, x1, m)
    wa, wb = len(a["rows"]), len(b["rows"])
    m["hand"] = (a.get("hand", 0.45 * a["h"]) * wa + b.get("hand", 0.45 * b["h"]) * wb) / (wa + wb)
    return m


def _rebalance(bands, gray, x0, x1):
    """A row at the edge of a band that is spaced like the neighbouring band (Kraken drew its outline too
    small or too large) is moved across: the first main line under a tiny commentary strip, for instance."""
    def gap_fits(g, pitch):
        return abs(g - pitch) <= 0.35 * pitch
    changed = True
    while changed:
        changed = False
        for i in range(len(bands) - 1):
            a, b = bands[i], bands[i + 1]
            if len(a["rows"]) >= 2:
                last, prev = a["rows"][-1], a["rows"][-2]
                if last["y"] - prev["y"] > 1.8 * a["pitch"] and gap_fits(b["rows"][0]["y"] - last["y"], b["pitch"]):
                    bands[i], bands[i + 1] = _band(a["rows"][:-1]), _band([last] + b["rows"])
                    changed = True
            if not changed and len(b["rows"]) >= 2:
                first, nxt = b["rows"][0], b["rows"][1]
                if nxt["y"] - first["y"] > 1.8 * b["pitch"] and gap_fits(first["y"] - a["rows"][-1]["y"], a["pitch"]):
                    bands[i], bands[i + 1] = _band(a["rows"] + [first]), _band(b["rows"][1:])
                    changed = True
            if changed:
                for bb, src in ((bands[i], a), (bands[i + 1], b)):
                    bb["pitch"] = _pitch_of(gray, x0, x1, bb)
                    bb["hand"] = src.get("hand", 0.45 * src["h"])
                break
    return bands


def _name_bands(bands, height):
    """-> ({name: band}, note). The main text is the band with the clearly largest hand and pitch."""
    if len(bands) == 1:
        return {"main": bands[0]}, ""
    score = [b.get("hand", 0.45 * b["h"]) * b["pitch"] * (1 if len(b["rows"]) >= 2 or all(len(x["rows"]) < 2 for x in bands) else 0) for b in bands]
    order = sorted(range(len(bands)), key=lambda i: score[i], reverse=True)
    note = ""
    if score[order[0]] / max(1, score[order[1]]) >= 1.2:
        k = order[0]
    elif len(bands) == 3:
        k, note = 1, "hands of like size: the middle band taken as the main text"
    else:
        k = min(range(len(bands)), key=lambda i: abs((bands[i]["y0"] + bands[i]["y1"]) / 2 - height / 2))
        note = "hands of like size: the band nearer the middle taken as the main text"
    named = {"main": bands[k]}
    above = [b for i, b in enumerate(bands) if i < k]
    below = [b for i, b in enumerate(bands) if i > k]
    if above:
        named["top"] = _band([r for b in above for r in b["rows"]])
        named["top"]["pitch"] = statistics.median(b["pitch"] for b in above)
    if below:
        named["bottom"] = _band([r for b in below for r in b["rows"]])
        named["bottom"]["pitch"] = statistics.median(b["pitch"] for b in below)
    return named, note


def find_blocks(seg, size, gray, expect=None):
    """-> ({box: {"rect": (x0, y0, x1, y1), "pieces": [...]}}, note). Boxes absent from the page are absent."""
    width, height = size
    pieces = [d for d in _pieces(seg) if d["x1"] - d["x0"] >= 0.01 * width and d["h"] >= 4]
    if not pieces:
        return {}, "no pieces"
    k, cols = _centre(_columns(pieces, width, gray), width)
    if k is None:
        return {}, "no columns"
    centre = cols[k]
    notes = []
    if centre.get("halves"):
        notes.append("main text in two half-columns")
    named = None
    if expect and ("top" in expect or "bottom" in expect):
        named = _seam_bands(centre["pieces"], gray, centre["x0"], centre["x1"])
        if named:
            notes.append("blocks cut along the blank space between them")
    if not named and expect and ("top" in expect or "bottom" in expect):
        named = _hand_bands(centre["pieces"], gray, centre["x0"], centre["x1"])
        if named:
            notes.append("main text found by its larger hand")
    if not named:
        bands = _bands(centre["pieces"], gray, centre["x0"], centre["x1"])
        if not bands:
            return {}, "only specks in the centre column"
        named, note = _name_bands(bands, height)
        if note:
            notes.append(note)
    blocks = {}
    for name, b in named.items():
        xs0 = min(d["x0"] for d in b["pieces"])
        xs1 = max(d["x1"] for d in b["pieces"])
        blocks[name] = {"rect": (int(xs0), int(b["y0"]), int(xs1), int(b["y1"])), "pieces": b["pieces"],
                        "pitch": b.get("pitch") if len(b.get("rows", [])) >= 2 else None}
    ch = centre_h = max(1, max(d["y1"] for d in centre["pieces"]) - min(d["y0"] for d in centre["pieces"]))
    for side, cands in (("left", [c for i, c in enumerate(cols) if i < k]), ("right", [c for i, c in enumerate(cols) if i > k])):
        best = None
        for c in cands:
            span = max(d["y1"] for d in c["pieces"]) - min(d["y0"] for d in c["pieces"])
            if len(c["pieces"]) >= 3 and span >= 0.3 * ch and (best is None or len(c["pieces"]) > len(best["pieces"])):
                best = c
        if best:
            srows = _rows(sorted(best["pieces"], key=lambda d: d["y"]), best["x1"] - best["x0"])
            sgaps = _normalised([b["y"] - a["y"] for a, b in zip(srows, srows[1:])])
            blocks[side] = {"rect": (int(best["x0"]), int(min(d["y0"] for d in best["pieces"])), int(best["x1"]), int(max(d["y1"] for d in best["pieces"]))),
                            "pieces": best["pieces"], "pitch": statistics.median(sgaps) if sgaps else None}
    dropped = len(pieces) - sum(len(b["pieces"]) for b in blocks.values())
    if dropped:
        notes.append(f"{dropped} piece{'s' if dropped > 1 else ''} outside the blocks left out")
    if expect:
        missing = [b for b in expect if b not in blocks]
        if missing:
            notes.append("no " + ", ".join(missing) + " block on this page")
    return blocks, "; ".join(notes)


def rows_in_block(block, gray, others=()):
    """The row finder run on one block as if it were a page. `others`: the rectangles of the page's other
    blocks; an edge row filled in from ink that belongs to one of them is dropped.
    -> {rows (page coordinates, merged), raw, pitch, note, warn, offset, block}"""
    x0, y0, x1, y1 = block["rect"]
    pad = 4
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(gray.shape[1], x1 + pad), min(gray.shape[0], y1 + pad)
    sub = {"lines": [{"baseline": [[x - x0, y - y0] for x, y in d["line"]["baseline"]],
                      "boundary": [[x - x0, y - y0] for x, y in d["line"]["boundary"]]} for d in block["pieces"]]}
    crop = gray[y0:y1, x0:x1]
    rows, blk, pitch, note = find_rows(sub, x1 - x0, crop)
    p_rows = block.get("pitch")
    n_rows = max(1, len(_rows(sorted(block["pieces"], key=lambda d: d["y"]), x1 - x0)))
    # the pitch the row finder read off the ink must fit the rows Kraken found in this small block;
    # a period of the block's edges, or a harmonic, is replaced by the block's own row spacing
    if p_rows and pitch and (pitch > 1.5 * (y1 - y0) / n_rows or pitch < 0.4 * p_rows):
        rows, blk, pitch, note = find_rows(sub, x1 - x0, crop, pitch_hint=p_rows)
        note += f"; pitch {p_rows:.0f}px from the block's own rows"
    if not rows:
        return {"rows": [], "raw": [], "pitch": pitch, "note": note, "warn": "no rows", "offset": (x0, y0), "block": blk}
    # a short line (the last of a commentary block) that seeded no row: a piece of some width at a
    # height no row has, inside the block, is a row of its own
    used = {id(d["line"]) for r in rows for d in r}
    bw = x1 - x0
    typical = statistics.median(_ink(crop, blk, r[0]["y"], pitch) for r in rows)
    for d in sorted((_info(l) for l in sub["lines"]), key=lambda d: d["y"]):
        if id(d["line"]) in used or d["x1"] - d["x0"] < max(60, 0.15 * bw):
            continue
        if _ink(crop, (d["x0"], d["x1"]), d["y"], pitch) < 0.5 * typical:      # a smear or a stamp, not writing
            continue
        if all(abs(d["y"] - r[0]["y"]) >= 0.6 * pitch for r in rows):
            rows.append([d])
            rows.sort(key=lambda r: r[0]["y"])
            note += f"; a short line at y {int(d['y'] + y0)} added as a row"
    # two rows closer than six tenths of a pitch are one line (vowel signs or a split line drawn apart)
    close = []
    for r in rows:
        if close and r[0]["y"] - close[-1][0]["y"] < 0.6 * pitch and not r[0].get("filled") and not close[-1][0].get("filled"):
            close[-1] = sorted(close[-1] + r, key=lambda d: d["x0"])
        else:
            close.append(r)
    rows = close
    if len(rows) >= 3:
        # the edge check needs to see one pitch beyond the block: a taller crop, the rows shifted into it
        ext = int(1.5 * pitch)
        ty0, ty1 = max(0, y0 - ext), min(gray.shape[0], y1 + ext)
        dy = y0 - ty0
        for r in rows:
            for d in r:
                d["y"] += dy
                d["line"]["baseline"] = [[x, y + dy] for x, y in d["line"]["baseline"]]
        n_before = len(rows)
        rows, edge = fill_edge_rows(rows, blk, pitch, gray[ty0:ty1, x0:x1])
        n_added = len(rows) - n_before

        def in_other(r):                                   # an added edge row whose ink belongs to another block
            yb = r[0]["y"] + ty0                           # on the page
            band = (yb - 1.2 * pitch, yb + 0.65 * pitch)   # half a pitch of tolerance each way
            return any(band[0] < oy1 and band[1] > oy0 and x0 < ox1 and x1 > ox0 for ox0, oy0, ox1, oy1 in others)
        while rows and rows[0][0].get("filled") and in_other(rows[0]):
            rows = rows[1:]
        while rows and rows[-1][0].get("filled") and in_other(rows[-1]):
            rows = rows[:-1]
        kept = []
        for r in rows:
            for d in r:
                d["y"] -= dy
                d["line"]["baseline"] = [[x, y - dy] for x, y in d["line"]["baseline"]]
            kept.append(r)
        if edge and len(kept) - n_before == n_added:          # every added row survived the check
            note += "; " + edge
        elif edge and len(kept) > n_before:
            note += "; Kraken missed a row at the block's edge: added from the ink on the page"
        rows = kept
    warn = uneven(rows, pitch)
    missed = [b for b in note.split("; ") if b.startswith("Kraken missed") or "short line may be missing" in b
              or "may have been missed" in b]
    if missed:
        warn = (warn + "; " if warn else "") + "; ".join(missed)
    merged = []
    H, W = gray.shape
    clamp = lambda x, y: (min(max(x + x0, 0), W - 1), min(max(y + y0, 0), H - 1))   # a band may reach past the page's edge
    for r in rows:
        base, poly = _merge(r, blk, pitch)
        merged.append(([clamp(x, y) for x, y in base], [clamp(x, y) for x, y in poly]))
    return {"rows": merged, "raw": rows, "pitch": pitch, "note": note, "warn": warn, "offset": (x0, y0), "block": blk}


def page_blocks(seg, size, gray, expect=None):
    """-> ({box: {rect, rows, raw, pitch, note, warn, offset, block}}, page note)"""
    blocks, note = find_blocks(seg, size, gray, expect)
    out = {}
    for box in BOXES:
        if box in blocks:
            others = [b["rect"] for name, b in blocks.items() if name != box]
            found = rows_in_block(blocks[box], gray, others)
            if found["rows"]:
                out[box] = dict(rect=blocks[box]["rect"], **found)
    return out, note


def blocks_for_page(ms_id, image):
    """The blocks and rows of a segmented page of a tripāṭha or pañcapāṭha manuscript.
    -> ({box: {rect, rows, raw, pitch, note, warn, ...}}, size, page note)"""
    from PIL import Image
    d = ms_dir(ms_id)
    seg = json.load(open(d / SEG / (image.rsplit(".", 1)[0] + ".json")))
    with Image.open(MANUSCRIPTS[ms_id]["images"] / image) as im:
        size = im.size
        gray = np.asarray(im.convert("L"))
    blocks, note = page_blocks(seg, size, gray, EXPECTED.get(layout_of(ms_id)))
    return blocks, size, note
