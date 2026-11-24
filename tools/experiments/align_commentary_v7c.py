"""Step 3: pair the lines Kraken found with the transcribed lines, and write training files.

Kraken returns main text lines, marginal notes, interlinear additions and stray specks,
and sometimes splits one line into fragments. This step:

  1. takes "long" detections (over half the page width) as the main text rows,
  2. glues fragments that sit at the same height into their row,
     and fills in a row Kraken missed altogether (a gap of two pitches between neighbours),
     On a hand Kraken breaks into words (few or no long detections) the page image steps in:
     the line pitch is read off the periodicity of its ink, the text block off where the
     pieces cluster, rows are built from pieces that together cover half the block, and a
     filled-in row is only kept where the page has ink,
  3. discards everything else (margins, interlinear notes, specks),
  4. accepts the page only if the number of rows equals the number of transcribed lines,
  5. pairs row k with transcribed line k and writes a PAGE-XML file for Kraken.

Pages that fail the count check are left out of training and listed in the report.
Pages are then divided into train / validation / test sets (whole pages, never single lines,
so no page contributes to both training and scoring).
"""
import collections
import json
import random
import statistics
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image, ImageDraw

from .config import MANUSCRIPTS, ms_dir
from . import gt

TEST_PAGES = 20
VAL_FRACTION = 0.10
SEED = 85885


def _info(line):
    xs = [p[0] for p in line["baseline"]]
    ys = [p[1] for p in line["baseline"]]
    return {"x0": min(xs), "x1": max(xs), "y": sum(ys) / len(ys), "line": line}


def image_pitch(gray, block, lo=12, hi=200):
    """Line pitch of a page from the periodicity of its ink, independent of Kraken.

    The horizontal ink profile inside the text block is autocorrelated; writing repeats at
    every line, so the strongest peak is one pitch (the headline and the letter bodies give
    a weaker peak at half a pitch, a multiple of the pitch a weaker one still). None when
    the page has no periodic ink (blank, or a picture)."""
    x0, x1 = int(max(0, block[0])), int(min(gray.shape[1], block[1]))
    if x1 - x0 < 10:
        return None
    prof = (255 - gray[:, x0:x1].astype(np.float32)).mean(axis=1)
    prof -= prof.mean()
    ac = np.correlate(prof, prof, mode="full")[len(prof) - 1:]
    if ac[0] <= 0:
        return None
    ac /= ac[0]
    hi = min(hi, len(ac) - 2)
    peaks = [i for i in range(lo, hi) if ac[i] >= ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.05]
    if not peaks:
        return None
    return max(peaks, key=lambda i: ac[i])


def _block_from_pieces(dets, width):
    """The text block when no detection is long enough to take it from: the widest stretch of
    the page covered by many pieces (the main text broken into words), margins excluded."""
    cov = np.zeros(width + 1)
    for d in dets:
        if d["x1"] - d["x0"] > 0.02 * width:
            cov[int(max(0, d["x0"])):int(min(width, d["x1"])) + 1] += 1
    if cov.max() == 0:
        return None
    win = max(1, int(0.03 * width))
    cov = np.convolve(cov, np.ones(win) / win, mode="same")
    on = cov >= 0.3 * cov.max()
    best, start = None, None
    for x in range(len(on) + 1):
        if x < len(on) and on[x]:
            start = x if start is None else start
        elif start is not None:
            if best is None or x - start > best[1] - best[0]:
                best = (start, x)
            start = None
    return best


def _ink(gray, block, yb, pitch):
    """Share of dark pixels in the band of a row whose baseline would be at yb."""
    x0, x1 = int(max(0, block[0])), int(min(gray.shape[1], block[1]))
    area = gray[:, x0:x1]
    p5, p50 = (float(v) for v in np.percentile(area, [5, 50]))
    thr = (p5 + p50) / 2                  # halfway between the ink and the parchment of this page
    y0, y1 = max(0, int(yb - 0.7 * pitch)), min(gray.shape[0], int(yb + 0.15 * pitch))
    return float((gray[y0:y1, x0:x1] < thr).mean()) if y1 > y0 else 0.0


def _across_puspika(dets, width, pitch, bx0, bx1, notes):
    """The text block (bx0, bx1) widened over the puṣpikā to the other halves of the lines it cuts
    (see find_rows). Only pieces at least a fifth of the width count, so marginal notes stay out."""
    def inside(d):
        return min(d["x1"], bx1) - max(d["x0"], bx0) >= 0.5 * (d["x1"] - d["x0"])
    wide = [d for d in dets if d["x1"] - d["x0"] >= 0.2 * width]
    halves = [d for d in wide if not inside(d)
              and any(inside(o) and abs(o["y"] - d["y"]) < 0.3 * pitch for o in wide)]
    mid = (bx0 + bx1) / 2
    right = [d["x1"] for d in halves if d["x0"] >= mid]
    left = [d["x0"] for d in halves if d["x1"] <= mid]
    n0, n1 = (min(bx0, statistics.median(left)) if left else bx0), (max(bx1, statistics.median(right)) if right else bx1)
    if (n0, n1) != (bx0, bx1):
        notes.append(f"lines cut by the puṣpikā: block widened to {len(halves)} other half-line{'s' if len(halves) != 1 else ''}")
    return n0, n1


def _pitch_from_pieces(dets, width):
    """The line pitch from the heights of Kraken's pieces, for a page with too few lines for the page's ink to show
    one (a last page of two lines, a single colophon line): the pieces that are more than a sliver are grouped by
    height, the groups that carry a line's worth of writing kept, and the smallest distance between them is the
    pitch; with a single line, the pitch is taken as 1.3 times the height of its pieces' outlines."""
    big = sorted((d for d in dets if d["x1"] - d["x0"] > 0.03 * width), key=lambda d: d["y"])
    if not big:
        return None
    hs = [max(p[1] for p in d["line"]["boundary"]) - min(p[1] for p in d["line"]["boundary"]) for d in big]
    h = statistics.median(hs)
    groups = []
    for d in big:
        if groups and abs(d["y"] - statistics.mean(x["y"] for x in groups[-1])) < 0.5 * h:
            groups[-1].append(d)
        else:
            groups.append([d])
    lines = [statistics.mean(x["y"] for x in g) for g in groups if sum(x["x1"] - x["x0"] for x in g) > 0.15 * width]
    gaps = [b - a for a, b in zip(lines, lines[1:]) if b - a > 0.5 * h]
    if gaps:
        return min(gaps)
    return 1.3 * h if lines else None


def _rule_score(band, x0, x1):
    """For each x in x0..x1 (at the band's mid-height): the largest share of the band's rows in which a thin line darker
    than the paper on both sides of it passes within a pixel of a path through x leaning up to 4 % either way. A rule
    drawn down the page scores 0.8 to 1 even when faint on dark paper or drawn aslant; writing, whose strokes are
    short, stays under 0.7."""
    H, W = band.shape
    a, b = max(0, x0 - 30), min(W, x1 + 30)
    if b - a < 3 or H < 4:
        return np.zeros(max(0, x1 - x0))
    win = band[:, a:b].astype(np.float32)
    p5, p50 = np.percentile(band, [5, 50])
    delta = max(8.0, 0.12 * (p50 - p5))
    w = win.shape[1]
    ridge = np.zeros(win.shape, dtype=bool)
    for k in (3, 5, 8):
        left = np.pad(win, ((0, 0), (k, 0)), mode="edge")[:, :w]
        right = np.pad(win, ((0, 0), (0, k)), mode="edge")[:, k:]
        ridge |= (np.minimum(left, right) - win) > delta
    rp = np.pad(ridge, ((0, 0), (1, 1)))
    ridge = rp[:, :-2] | rp[:, 1:-1] | rp[:, 2:]
    ys, xs = np.arange(H), np.arange(w)
    best = np.zeros(w)
    for slant in np.linspace(-0.04, 0.04, 17):
        off = np.round(slant * (ys - H / 2)).astype(int)
        best = np.maximum(best, ridge[ys[:, None], np.clip(xs[None, :] + off[:, None], 0, w - 1)].mean(axis=0))
    return best[x0 - a:x0 - a + (x1 - x0)]


def _to_the_ink(gray, dets, bx0, bx1, pitch, width, notes):
    """The text block widened to where the page's own ink ends. Kraken's line pieces often stop short of the writing
    (a hand broken into word-sized pieces most of all), and the block, taken from their median ends, then cuts the
    first and last letters off every line. Across the band of the text lines, the block is pushed out, column by
    column, while there is writing; it stops at a rule drawn down the page or at a blank gap wider than half a line
    pitch (what parts the text from a marginal note), and goes at most a sixth of the width beyond Kraken's lines.
    -> (bx0, bx1, left rule or None, right rule or None)"""
    inside = [d for d in dets if bx0 <= (d["x0"] + d["x1"]) / 2 <= bx1]
    if not inside or gray is None:
        return bx0, bx1, None, None
    y0 = int(max(0, min(d["y"] for d in inside) - 0.9 * pitch))
    y1 = int(min(gray.shape[0], max(d["y"] for d in inside) + 0.3 * pitch))
    band = gray[y0:y1].astype(np.float32)
    if band.shape[0] < 4:
        return bx0, bx1, None, None
    p5, p50 = np.percentile(band, [5, 50])
    ink = (band < (p5 + p50) / 2).mean(axis=0)                       # writing in each column
    # a rule: dark (red ink counted) down nearly the whole band, the gaps between the lines too, where writing never
    # is; a few pixels' play either side, as a rule drawn by hand leans
    pad = np.pad(band, ((0, 0), (3, 3)), mode="edge")
    lean = np.lib.stride_tricks.sliding_window_view(pad, 7, axis=1).min(axis=-1)
    dark = (lean < p50 - 0.25 * (p50 - p5)).mean(axis=0)
    k = max(1, int(0.3 * pitch))
    sm = np.convolve(ink, np.ones(k) / k, mode="same")
    a, b = int(max(0, bx0)), int(min(len(ink) - 1, bx1))
    ref = float(np.median(sm[a:b + 1])) if b > a else 0.0
    if ref <= 0:
        return bx0, bx1, None, None
    is_rule = dark >= 0.8                                              # writing reaches 0.6 at most
    gap_max, reach = int(0.5 * pitch), int(width / 6)
    # a thin rule, faint on dark paper or drawn aslant, is found by following it down the band
    for lo, hi in ((max(0, a - reach - 2), a + 1), (b, min(len(ink), b + reach + 2))):
        if hi > lo:
            is_rule[lo:hi] |= _rule_score(band, lo, hi) >= 0.75

    def push(x, step, limit):
        last, quiet, rule = x, 0, None
        while 0 < x + step < len(ink) and abs(x + step - limit) > 0 and abs(x - limit) < reach:
            x += step
            if is_rule[x]:
                rule = x
                break
            if sm[x] >= 0.2 * ref:
                last, quiet = x, 0
            else:
                quiet += 1
                if quiet > gap_max:
                    break
        return last, rule
    nx0, rule_l = push(a, -1, a)
    nx1, rule_r = push(b, 1, b)
    if nx0 < bx0 - 2 or nx1 > bx1 + 2:
        notes.append(f"block widened to the ink: x {min(bx0, nx0):.0f}-{max(bx1, nx1):.0f} (Kraken's lines {bx0:.0f}-{bx1:.0f})")
    return min(bx0, nx0), max(bx1, nx1), rule_l, rule_r


def _hand(gray, row, x0, x1, pitch, paper, ink):
    """(density, stem, pen) of a row's writing, or None where there is next to no writing to measure.
      density: how much of the band just over the baseline the ink fills, in the columns that have any ink (so a
               short line is not penalised);
      stem:    the length of the long vertical strokes (the mean of the longest fifth of the ink's vertical runs over
               the line): in Devanagari the stem of a letter runs from the headline down to the baseline, so this is the
               height of the letters, whatever the ink's colour or darkness;
      pen:     the mean of the shorter vertical runs, where a headline or a stroke is cut across: the pen's width.
    The ink is taken from the row's own darkness (its darkest 3 %), the threshold set 30 % of the way from the paper
    to it, so a line in red ink, or the red words of a black line, is measured as fully as black. A commentary in a smaller hand has shorter stems, a finer pen, and fills less of the band."""
    pts = [p[1] for d in row for p in d["line"]["baseline"]]
    y = float(np.median(pts))
    lo, hi = int(max(0, y - 0.45 * pitch)), int(min(gray.shape[0], y + 0.05 * pitch))
    a, b = int(max(0, x0)), int(min(gray.shape[1], x1))
    if hi - lo < 4 or b - a < 10:
        return None
    band = gray[lo:hi, a:b]
    own = float(np.percentile(band, 3))           # the row's own ink: black, or red, which is lighter in grey
    if paper - own < 0.2 * (paper - ink):
        return None                               # next to no writing to measure
    thr = paper - 0.3 * (paper - own)             # nearer the paper than the ink: red strokes in a black line count
    m = band < thr
    cols = m.any(axis=0)
    if cols.sum() < 10:
        return None
    density = float(m[:, cols].mean())
    tall = gray[int(max(0, y - 0.95 * pitch)):int(min(gray.shape[0], y + 0.3 * pitch)), a:b] < thr
    d = np.diff(np.pad(tall.astype(np.int8), ((1, 1), (0, 0))), axis=0)
    runs = np.nonzero(d.T == -1)[1] - np.nonzero(d.T == 1)[1]
    runs = runs[runs >= 2]
    if len(runs) < 20:
        return None
    short = runs[runs <= np.percentile(runs, 50)]
    long_ = runs[runs >= np.percentile(runs, 80)]   # a mean, not a percentile: whole pixels are too coarse for stems
    return density, float(long_.mean()), float(short.mean())


def _small_hand_edges(rows, gray, x0, x1, pitch):
    """The rows at the top and the bottom that are a commentary written above or below the text, not lines of it.
    Each row is measured against the page's own rows (medians of _hand's density, stem and pen). A row is clearly
    commentary when two of these hold: smaller letters (stem at most 0.82), a finer pen (pen at most 0.90), thinner
    ink (density under 0.8); or thinner ink and a blank band of 1.3 pitches or more parting it from the text; or it was
    filled in from the ink with next to no writing in it. It looks a little like commentary when any one of the three
    holds. From each edge, the rows up to the innermost clear commentary row among the outer three go, provided every
    row from the edge to it looks at least a little like commentary (a plain line of the text is never dropped for
    the sake of an odd line inside it). Two or more rows of thin ink packed closer than three quarters of a pitch at
    an edge go too (a block of notes). At least four rows are always kept. -> (rows kept, how many dropped above,
    below)"""
    if gray is None or len(rows) < 5:
        return rows, 0, 0
    ys = [p[1] for r in rows for d in r for p in d["line"]["baseline"]]
    area = gray[int(max(0, min(ys) - pitch)):int(min(gray.shape[0], max(ys) + pitch)), int(max(0, x0)):int(x1)]
    p5, p50 = np.percentile(area, [5, 50])
    hands = [_hand(gray, r, x0, x1, pitch, p50, p5) for r in rows]
    known = [h for h in hands if h]
    if len(known) < 5:
        return rows, 0, 0
    dens, stem, pen = (statistics.median(h[k] for h in known) for k in range(3))
    ys = [float(np.median([p[1] for d in r for p in d["line"]["baseline"]])) for r in rows]

    def signs(i):
        h = hands[i]
        return (h[1] <= 0.82 * stem, h[2] <= 0.90 * pen, h[0] < 0.8 * dens)     # smaller, finer, thinner

    def clear(i, j):
        """row i, whose neighbour towards the text is row j, is clearly a commentary"""
        if hands[i] is None:
            return bool(rows[i][0].get("filled"))
        smaller, finer, thinner = signs(i)
        apart = abs(ys[j] - ys[i]) >= 1.3 * pitch
        return smaller + finer + thinner >= 2 or (thinner and apart)

    def somewhat(i):
        return hands[i] is None or any(signs(i))

    def outer(order):
        order = list(order)
        cut = 0
        for k in range(min(3, len(order) - 1)):
            if not somewhat(order[k]):
                break
            if clear(order[k], order[k + 1]):
                cut = k + 1
        return cut

    def notes(order):
        run = []
        for i in order:
            if hands[i] is None or hands[i][0] >= 0.7 * dens:
                break
            run.append(i)
        if len(run) >= 2 and all(abs(ys[a] - ys[b]) < 0.75 * pitch for a, b in zip(run, run[1:])):
            return len(run)
        return 0
    n = len(rows)
    top = max(outer(range(n)), notes(range(n)))
    bottom = max(outer(range(n - 1, top - 1, -1)), notes(range(n - 1, top - 1, -1))) if n - top > 4 else 0
    if n - top - bottom < 4:                       # never leave fewer than four rows
        return rows, 0, 0
    return rows[top:len(rows) - bottom], top, bottom


def _fits_better(long_, p_new, p_old):
    """True when the gaps between Kraken's long lines are whole multiples of p_new clearly more nearly than of p_old
    (Kraken's pitch, a median of few gaps, can be a mixture of one and two pitches)."""
    gaps = [b["y"] - a["y"] for a, b in zip(long_, long_[1:])]
    gaps = [g for g in gaps if g >= 0.6 * min(p_new, p_old)]
    if len(gaps) < 2:
        return False
    misfit = lambda p: sum((g / p - round(g / p)) ** 2 for g in gaps) / len(gaps)
    return misfit(p_new) < 0.5 * misfit(p_old)


def find_rows(seg, width, gray=None, pitch_hint=None):
    """Return (rows, block, pitch, note). Each row is a list of detections, left to right; rows
    run top to bottom. `gray`, the page as a 2-D array of grey levels, lets the page itself
    be consulted when Kraken's lines are unreliable: for the line pitch when Kraken has broken
    the lines into words (image_pitch), for the text block when no detection spans half the
    page, and to check that a row filled in between two neighbours really holds writing.
    `pitch_hint`, a pitch already known (a block's own spacing), is used instead of either.

    Puṣpikā: many leaves keep a blank hole in the middle for the binding string (a diamond, a
    circle or a band), and the lines that pass it are cut in two with no letter lost. When the
    block found from the long lines (or from the pieces) holds only one side of the hole, the
    wide pieces on the other side, each level with a wide piece inside the block, are the other
    halves of those lines: the block is widened to them, and they join their rows."""
    dets = [_info(l) for l in seg["lines"] if len(l["boundary"]) >= 3]
    long_ = sorted((d for d in dets if d["x1"] - d["x0"] > 0.5 * width), key=lambda d: d["y"])
    notes = []
    pitch = None
    if len(long_) >= 2:
        pitch = statistics.median(b["y"] - a["y"] for a, b in zip(long_, long_[1:]))
        bx0 = statistics.median(d["x0"] for d in long_)
        bx1 = statistics.median(d["x1"] for d in long_)
    else:
        blk = _block_from_pieces(dets, width) if gray is not None else None
        if blk is None:
            return [], None, 0, "fewer than 2 long lines detected"
        bx0, bx1 = blk
        notes.append(f"block from the pieces Kraken found ({len(long_)} long line{'s' if len(long_) != 1 else ''})")
        if bx1 - bx0 < 0.25 * width:          # a stamp, a number, a smear: no text block is so narrow
            notes.append(f"text block only {100 * (bx1 - bx0) / width:.0f}% of the page wide: the text may have been missed")
    if pitch_hint:
        pitch = pitch_hint
    elif gray is not None:
        pimg = image_pitch(gray, (bx0, bx1))
        if pimg and (pitch is None or abs(pitch - pimg) > 0.3 * pimg):
            notes.append(f"pitch {pimg}px from the page's ink" + (f", not {pitch:.0f}px from Kraken's long lines" if pitch else ""))
            pitch = pimg
        elif pimg and pitch and abs(pitch - pimg) > 0.1 * pimg and _fits_better(long_, pimg, pitch):
            notes.append(f"pitch {pimg}px from the page's ink, not {pitch:.0f}px from Kraken's long lines (the gaps "
                         "between them are whole multiples of it)")
            pitch = pimg
    if not pitch:
        pitch = _pitch_from_pieces(dets, width)
        if pitch:
            notes.append(f"pitch {pitch:.0f}px from the heights of Kraken's pieces (too few lines for the page's ink to show it)")
    if not pitch:
        return [], None, 0, "fewer than 2 long lines detected"
    bx0, bx1 = _across_puspika(dets, width, pitch, bx0, bx1, notes)

    rows = []
    for d in long_:                       # two long detections at the same height = same row
        if rows and d["y"] - rows[-1][0]["y"] < 0.4 * pitch:
            rows[-1].append(d)
        else:
            rows.append([d])

    def overlaps(row, d):
        return any(min(d["x1"], r["x1"]) - max(d["x0"], r["x0"]) > 0.3 * (d["x1"] - d["x0"]) for r in row)

    used = {id(d) for row in rows for d in row}
    rest = [d for d in dets if id(d) not in used
            and bx0 < (d["x0"] + d["x1"]) / 2 < bx1
            and d["x1"] - d["x0"] > 0.03 * width]
    orphans = []
    for d in rest:                        # fragments that continue an existing row
        row = min(rows, key=lambda r: abs(r[0]["y"] - d["y"])) if rows else None
        if row and abs(row[0]["y"] - d["y"]) < 0.25 * pitch and not overlaps(row, d):
            row.append(d)
        else:
            orphans.append(d)
    # a row broken into pieces that are each under half the width
    orphans.sort(key=lambda d: d["y"])
    groups = []
    for d in orphans:
        if groups and abs(d["y"] - statistics.mean(g["y"] for g in groups[-1])) < 0.3 * pitch:
            groups[-1].append(d)
        else:
            groups.append([d])
    for g in groups:
        covered = sum(d["x1"] - d["x0"] for d in g)
        gy = statistics.mean(d["y"] for d in g)
        far = all(abs(gy - statistics.mean(d["y"] for d in r)) > 0.6 * pitch for r in rows)
        if covered > 0.5 * (bx1 - bx0) and far:
            rows.append(g)
    rows.sort(key=lambda r: statistics.mean(d["y"] for d in r))
    for r in rows:
        r.sort(key=lambda d: d["x0"])
    if not rows:
        return [], None, 0, "; ".join(notes + ["no row covers half the text block"])
    rows, above, below = _small_hand_edges(rows, gray, bx0, bx1, pitch)
    for n, where in ((above, "above"), (below, "below")):
        if n:
            notes.append(f"{n} row{'s' if n > 1 else ''} in a smaller hand {where} the text left out (a commentary)")
    if above:
        rows[0][0]["commentary_above"] = True        # fill_edge_rows adds no row where the commentary was
    if below:
        rows[-1][0]["commentary_below"] = True
    # Now and then Kraken misses a whole line, leaving a gap of two (or three) pitches between
    # neighbours. Fill it with a straight baseline interpolated between the rows above and below.
    # With the page at hand each filled row is checked for ink first (a wide gap can also be
    # blank, or a heading in larger letters), and up to four missing rows are bridged.
    most = 5 if gray is not None else 3

    def ends(row):
        pts = [p for d in row for p in d["line"]["baseline"]]
        return min(pts)[1], max(pts)[1]          # y at the leftmost and the rightmost point
    block = (bx0 - 0.015 * width, bx1 + 0.015 * width)
    typical = statistics.median(_ink(gray, block, statistics.mean(d["y"] for d in r), pitch) for r in rows) if gray is not None else None
    filled, blank, doubtful, out = [], [], [], []
    for i, row in enumerate(rows):
        if out:
            ya, yb = statistics.mean(d["y"] for d in out[-1]), statistics.mean(d["y"] for d in row)
            k = round((yb - ya) / pitch)
            if 2 <= k <= most:
                (al, ar), (bl, br) = ends(out[-1]), ends(row)
                for j in range(1, k):
                    t = j / k
                    yl, yr = al + t * (bl - al), ar + t * (br - ar)
                    if typical is not None:
                        ink = _ink(gray, block, (yl + yr) / 2, pitch) / max(typical, 1e-6)
                        if ink < 0.25:            # blank parchment: a paragraph break, not a missed line
                            out[-1][0]["blank_after"] = True
                            blank.append(len(out))
                            continue
                        if ink < 0.5:             # some ink, too little for a full line: left for the eye
                            doubtful.append(len(out))
                            continue
                    out.append([{"x0": bx0, "x1": bx1, "y": (yl + yr) / 2, "filled": True,
                                 "line": {"baseline": [[bx0, yl], [bx1, yr]]}}])
                    filled.append(len(out))
        out.append(row)
    if filled and len(filled) >= len(rows):     # a row filled in between every pair: the pitch was halved, not rows missed
        out, filled, blank, doubtful = rows, [], [], []
        for r in rows:
            r[0].pop("blank_after", None)
        notes.append("as many rows filled in as found, so the pitch is doubtful: nothing filled in")
    rows = out
    # every decision above is taken on the block Kraken's lines give; the rows are then cut out as wide as the writing
    # goes, with a little room beyond the ends, never across a rule (a rule in a line image could be read as a daṇḍa)
    bx0, bx1, rule_l, rule_r = _to_the_ink(gray, dets, bx0, bx1, pitch, width, notes)
    block = (max(min(block[0], bx0 - 0.015 * width), rule_l + 2 if rule_l is not None else -1e9),
             min(max(block[1], bx1 + 0.015 * width), rule_r - 2 if rule_r is not None else 1e9))
    note = f"pitch {pitch:.0f}px, block x {bx0:.0f}-{bx1:.0f}"
    if notes:
        note += "; " + "; ".join(notes)
    if filled:
        note += f"; Kraken missed row{'s' if len(filled) > 1 else ''} {', '.join(map(str, filled))}: filled in between neighbours"
    if blank:
        note += f"; blank gap after row{'s' if len(blank) > 1 else ''} {', '.join(map(str, dict.fromkeys(blank)))}: a paragraph break, nothing filled in"
    if doubtful:
        note += f"; gap after row{'s' if len(doubtful) > 1 else ''} {', '.join(map(str, dict.fromkeys(doubtful)))} holds some ink but no row: a short line may be missing"
    return rows, block, pitch, note


def uneven(rows, pitch):
    """The spacing problem of a page, or None: a gap of more than 1.5 pitches between neighbours
    that is not a verified blank (a line was missed), or a step under 0.6 (a note squeezed in)."""
    ys = [statistics.mean(dd["y"] for dd in r) for r in rows]
    bad = [b - a for r, a, b in zip(rows, ys, ys[1:])
           if (b - a > 1.5 * pitch and not r[0].get("blank_after")) or b - a < 0.6 * pitch]
    return f"uneven row spacing ({min(bad):.0f}-{max(bad):.0f}px, pitch {pitch:.0f}px)" if bad else None


def fill_edge_rows(rows, block, pitch, gray):
    """Kraken sometimes misses the first or the last line of a page, which the interpolation
    in find_rows cannot reach (no neighbour on one side). `gray` is the page as a 2-D array of
    grey levels. One pitch above the first row and below the last, the band is measured for
    ink: if it holds at least half as much as a typical text row, a row is added there, with
    the slope of its neighbour. Returns the rows and a note ('' when nothing was added).
    The ink threshold is taken from the page itself (halfway between its darkest 5% and its
    median), so a darker parchment is not mistaken for writing."""
    import statistics as st

    def ends(row):
        pts = [p for d in row for p in d["line"]["baseline"]]
        return min(pts)[1], max(pts)[1]

    typical = st.median(_ink(gray, block, st.mean(d["y"] for d in r), pitch) for r in rows)
    added = []
    for side in ("above", "below"):
        if (rows[0] if side == "above" else rows[-1])[0].get(f"commentary_{side}"):
            continue                                    # the ink beyond is a commentary the row finder left out
        for _ in range(2):
            row = rows[0] if side == "above" else rows[-1]
            yl, yr = ends(row)
            shift = -pitch if side == "above" else pitch
            yb = st.mean(d["y"] for d in row) + shift
            if yb < 0.5 * pitch or yb > gray.shape[0] - 0.2 * pitch or _ink(gray, block, yb, pitch) < 0.5 * typical:
                break
            synth = [{"x0": block[0], "x1": block[1], "y": yb, "filled": True,
                      "line": {"baseline": [[block[0], yl + shift], [block[1], yr + shift]]}}]
            trial = [synth] + rows if side == "above" else rows + [synth]
            _, top, bottom = _small_hand_edges(trial, gray, block[0], block[1], pitch)
            if (top if side == "above" else bottom):
                break                                   # the ink there is a commentary in a smaller hand
            rows = trial
            added.append(side)
    note = ""
    if added:
        note = "Kraken missed " + " and ".join(f"a row {s} the {'first' if s == 'above' else 'last'} one" for s in dict.fromkeys(added)) + ": added from the ink on the page"
    return rows, note


ABOVE, BELOW = 0.90, 0.30     # band around the baseline, as fractions of the line pitch


def _merge(row, block, pitch):
    """One baseline and one outline for a row made of 1+ detections.

    Kraken's own outlines hug the letter bodies and slice off what sits above the headline
    (anusvāra, repha, the tops of e/ai/o/au signs). So the outline used here is a band of
    fixed proportions around the baseline instead: 0.90 of the line pitch above it, which
    reaches the marks over the headline, and 0.30 below it, for u/ū signs and stacked
    conjuncts. Neighbouring bands overlap slightly; a sliver of the next line is harmless,
    a missing anusvāra is not. The band runs the full width of the text block, so the
    first and last characters are complete and marginal notes stay out.
    """
    x0, x1 = block
    pts = sorted({(min(max(x, x0), x1), y) for d in row for x, y in map(tuple, d["line"]["baseline"])})
    xs = sorted({x for x, _ in pts})
    base = [(x, statistics.mean(y for px, y in pts if px == x)) for x in xs]     # one height per x
    if base[0][0] > x0:
        base.insert(0, (x0, base[0][1]))
    if base[-1][0] < x1:
        base.append((x1, base[-1][1]))
    poly = [(x, y - ABOVE * pitch) for x, y in base] + [(x, y + BELOW * pitch) for x, y in reversed(base)]
    return [(int(x), int(y)) for x, y in base], [(int(x), max(0, int(y))) for x, y in poly]


def _pts(points):
    return " ".join(f"{int(x)},{int(y)}" for x, y in points)


def _page_xml(image_path, size, lines):
    w, h = size
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">',
           f'<Page imageFilename="{escape(str(image_path))}" imageWidth="{w}" imageHeight="{h}">',
           f'<TextRegion id="r1"><Coords points="0,0 {w},0 {w},{h} 0,{h}"/>']
    for i, (baseline, poly, text) in enumerate(lines, 1):
        out.append(f'<TextLine id="l{i}"><Coords points="{_pts(poly)}"/><Baseline points="{_pts(baseline)}"/>'
                   f'<TextEquiv><Unicode>{escape(text)}</Unicode></TextEquiv></TextLine>')
    out += ['</TextRegion>', '</Page>', '</PcGts>']
    return "\n".join(out)


def align(ms_id: str, preview: int = 3):
    ms = MANUSCRIPTS[ms_id]
    d = ms_dir(ms_id)
    xml_dir, prev_dir = d / "pagexml", d / "preview"
    xml_dir.mkdir(exist_ok=True)
    prev_dir.mkdir(exist_ok=True)

    by_image = collections.defaultdict(list)
    for r in gt.load(ms_id):
        by_image[r["image"]].append(r)

    accepted, rejected, fragments_joined, line_heights = [], [], 0, []
    print(f"[align] {ms_id}: pairing detected lines with transcribed lines")
    for image in sorted(by_image):
        gt_lines = by_image[image]        # order in the TEI file; its n= numbers have typos
        seg_path = d / "seg" / (image.rsplit(".", 1)[0] + ".json")
        if not seg_path.exists():
            rejected.append((image, "not segmented yet"))
            continue
        seg = json.load(open(seg_path))
        img_path = ms["images"] / image
        with Image.open(img_path) as im:
            size = im.size
            gray = np.asarray(im.convert("L"))
        rows, block, pitch, note = find_rows(seg, size[0], gray)
        if len(rows) != len(gt_lines):
            rejected.append((image, f"found {len(rows)} rows, transcription has {len(gt_lines)} ({note})"))
            continue
        # Right count is not enough: a missed line plus a stray note also adds up. Rows of a
        # page are evenly spaced, so a gap of two line-heights or a row squeezed in betrays it.
        problem = uneven(rows, pitch)
        if problem:
            rejected.append((image, f"{problem}: a line was probably missed and a note counted instead"))
            continue
        merged = [_merge(r, block, pitch) for r in rows]
        fragments_joined += sum(len(r) - 1 for r in rows)
        for _, poly in merged:
            ys = [p[1] for p in poly]
            line_heights.append(max(ys) - min(ys))
        xml_path = xml_dir / (image.rsplit(".", 1)[0] + ".xml")
        xml_path.write_text(_page_xml(img_path, size, [(b, p, g["text"]) for (b, p), g in zip(merged, gt_lines)]),
                            encoding="utf-8")
        accepted.append(xml_path)

        if len(accepted) <= preview:      # picture of what the recogniser will be shown
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                dr = ImageDraw.Draw(im)
                for i, (b, p) in enumerate(merged, 1):
                    dr.polygon(p, outline=(0, 90, 255) if i % 2 else (255, 60, 0))
                    dr.text((p[0][0] - 28, b[0][1] - 12), str(i), fill=(0, 0, 0))
                im.save(prev_dir / image)

    total = len(by_image)
    print(f"  pages accepted : {len(accepted)}/{total}  ({sum(len(v) for k, v in by_image.items() if (xml_dir / (k.rsplit('.',1)[0]+'.xml')) in accepted)} lines)")
    print(f"  pages rejected : {len(rejected)}")
    for image, why in rejected[:40]:
        print(f"     {image}: {why}")
    if len(rejected) > 40:
        print(f"     ... and {len(rejected) - 40} more")
    print(f"  line fragments rejoined : {fragments_joined}")
    if line_heights:
        print(f"  line outline height (px): median {statistics.median(line_heights):.0f}, "
              f"min {min(line_heights)}, max {max(line_heights)}")
    print(f"  preview images : {prev_dir}")

    if all((d / f"{n}.txt").exists() for n in ("train", "val", "test")):
        # keep every page in the set it was first given, so scores stay comparable between runs
        old = {n: set((d / f"{n}.txt").read_text().split()) for n in ("train", "val", "test")}
        ok = {str(p) for p in accepted}
        test, val = sorted(old["test"] & ok), sorted(old["val"] & ok)
        train = sorted(ok - old["test"] - old["val"])
        print("  split: kept the existing assignment of pages (newly rejected pages removed)")
    else:
        rng = random.Random(SEED)
        pages = sorted(str(p) for p in accepted)
        rng.shuffle(pages)
        test = sorted(pages[:TEST_PAGES])
        n_val = max(1, round(VAL_FRACTION * (len(pages) - TEST_PAGES)))
        val = sorted(pages[TEST_PAGES:TEST_PAGES + n_val])
        train = sorted(pages[TEST_PAGES + n_val:])
    for name, part in (("train", train), ("val", val), ("test", test)):
        (d / f"{name}.txt").write_text("\n".join(str(p) for p in part) + "\n")
    print(f"  split (whole pages) : train {len(train)}, validation {len(val)}, test {len(test)}")
    print("    train      = pages the model learns from")
    print("    validation = pages scored after every pass to track progress and pick the best model")
    print("    test       = pages never touched until the final score")
    return accepted, rejected


if __name__ == "__main__":
    import sys
    align(sys.argv[1])
