"""Edge-row commentary classifier, fitted to the user's review verdicts (2026-11-24 batch, drawn with the current row finder).

Labels per reviewed linear page:
  - rows the row finder already dropped as commentary on a page the user passed (ok) -> commentary (1)
  - rows the user named as commentary on a wrong page                                -> commentary (1)
  - every other row of an ok page, and the other rows of a pure-commentary wrong page -> main text (0)
  - rows on pages with other faults (wavy, half lines, missed rows) or named as doubtful -> left out
Features (ratios to the page's own medians): density, stem (letter height), pen width; gap to the inner and outer
neighbours in pitches; position from the edge; filled-in flag. Only the outer four rows at each edge are used.
Evaluation: leave one manuscript out. The rows and their labels are kept in data/commentary/clf_data.npz.
    PYTHONPATH=. .venv/bin/python tools/commentary_fit.py <feedback tsv> ...   (the batch reviewed on the current pictures)
"""
import csv, json, statistics, sys, warnings
from collections import defaultdict
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
from msocr.config import MANUSCRIPTS, DATA
from msocr.blocks import layout_of
import msocr.align as A


def hand(gray, y, x0, x1, pitch, paper, ink):
    """density, stem, pen of the writing on a baseline at y (the experimental _hand, v7c)"""
    lo, hi = int(max(0, y - 0.45 * pitch)), int(min(gray.shape[0], y + 0.05 * pitch))
    a, b = int(max(0, x0)), int(min(gray.shape[1], x1))
    if hi - lo < 4 or b - a < 10:
        return None
    band = gray[lo:hi, a:b]
    own = float(np.percentile(band, 3))
    if paper - own < 0.2 * (paper - ink):
        return None
    thr = paper - 0.3 * (paper - own)
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
    long_ = runs[runs >= np.percentile(runs, 80)]
    return density, float(long_.mean()), float(short.mean())


def page_rows_full(ms, stem, segdir):
    """(all rows top to bottom including those the current code drops at the edges, how many dropped above/below,
    block, pitch, gray): the rows as drawn in the review (find_rows + fill_edge_rows), with the dropped ones put back"""
    seg = json.load(open(DATA / ms / segdir / f"{stem}.json"))
    with Image.open(MANUSCRIPTS[ms]["images"] / f"{stem}.jpg") as im:
        gray = np.asarray(im.convert("L"))
    width = gray.shape[1]
    orig = A._small_hand_edges
    got = {}

    def spy(rows, g, x0, x1, pitch):
        out = orig(rows, g, x0, x1, pitch)
        if "first" not in got:                       # the call from find_rows (later calls come from fill_edge_rows)
            got["first"] = (list(rows), out[1], out[2], (x0, x1))
        return out
    A._small_hand_edges = spy
    try:
        rows, block, pitch, note = A.find_rows(seg, width, gray)
        if rows and len(rows) >= 3:
            rows, _ = A.fill_edge_rows(rows, block, pitch, gray)
    finally:
        A._small_hand_edges = orig
    if not rows or "first" not in got:
        return None
    raw, top, bottom, kb = got["first"]
    full = raw[:top] + rows + (raw[len(raw) - bottom:] if bottom else [])
    return full, top, bottom, kb, pitch, gray


def features(full, kb, pitch, gray):
    x0, x1 = kb
    ys = [float(np.median([p[1] for d in r for p in d["line"]["baseline"]])) for r in full]
    area = gray[int(max(0, min(ys) - pitch)):int(min(gray.shape[0], max(ys) + pitch)), int(max(0, x0)):int(x1)]
    p5, p50 = np.percentile(area, [5, 50])
    hs = [hand(gray, y, x0, x1, pitch, p50, p5) for y in ys]
    known = [h for h in hs if h]
    if len(known) < 5:
        return None
    med = [statistics.median(h[k] for h in known) for k in range(3)]
    n, out = len(full), []
    for i in range(n):
        pos = min(i, n - 1 - i)
        if pos > 3:
            out.append(None)
            continue
        top_side = i < n / 2
        inner = i + 1 if top_side else i - 1
        outer = i - 1 if top_side else i + 1
        gin = abs(ys[inner] - ys[i]) / pitch if 0 <= inner < n else 1.0
        gout = abs(ys[outer] - ys[i]) / pitch if 0 <= outer < n else 3.0
        h = hs[i]
        f = [h[0] / med[0], h[1] / med[1], h[2] / med[2]] if h else [0.0, 0.0, 0.0]
        out.append(f + [min(gin, 4.0), min(gout, 4.0), float(pos), float(bool(full[i][0].get("filled"))), float(h is None)])
    return out


# rows the user named as commentary on wrong pages (numbering of the review picture), and rows to leave out
COMM = {"009851_00004_2": [1, 2, 15, 16], "015152_00006_1": [1, 2, 3, 22], "015152_00006_2": [1, 2, 22],
        "015152_00007_1": [1, 2, 3, 22], "015152_00007_2": [1, 2, 21], "015152_00008_1": [1, 2], "015152_00004_1": [1],
        "022995_00012_1": [1], "022995_00019_2": [1], "022995_00009_2": [1], "022995_00006_1": [1], "022995_00004_1": [1],
        "045071_00044_1": [14], "045071_00053_1": [1], "045071_00054_1": [1, 2], "045071_00035_2": [1],
        "045364_00011_1": [1], "045364_00010_2": [1, 17], "045364_00032_2": [1, 17], "082320_00016_1": [1, 2],
        "082320_00019_1": [1, 2, 18, 19], "082320_00021_2": [1, 2], "045071_00001_2": [1, 2], "045071_00002_1": [1, 17]}
DOUBT = {"022995_00004_1": [2], "045364_00032_2": [16], "045071_00002_1": [14, 15, 16], "015152_00006_2": [21]}
MIXED = {"045071_00001_2"}          # only the named commentary rows are labelled; the rest of the page is a mess


def gather(paths):
    verdicts = {}
    for p in paths:
        for r in csv.DictReader(open(p, encoding="utf-8", newline=""), delimiter="\t"):
            verdicts[(r["manuscript"], r["page"])] = r          # later files win (the rechecks come last)
    X, y, groups, where = [], [], [], []
    for (ms, stem), r in sorted(verdicts.items()):
        if r["verdict"] not in ("ok", "wrong") or layout_of(ms) != "linear":
            continue
        if r["verdict"] == "wrong" and stem not in COMM:
            continue                                           # faults of other kinds: labels unreliable
        segdir = "seg_repaired" if r["group"] == "REPAIRED" and (DATA / ms / "seg_repaired" / f"{stem}.json").exists() else "seg"
        got = page_rows_full(ms, stem, segdir)
        if not got:
            continue
        full, top, bottom, kb, pitch, gray = got
        fs = features(full, kb, pitch, gray)
        if not fs:
            continue
        n = len(full)
        named = {top + k - 1 for k in COMM.get(stem, [])}
        doubt = {top + k - 1 for k in DOUBT.get(stem, [])}
        for i, f in enumerate(fs):
            if f is None or i in doubt:
                continue
            dropped = i < top or i >= n - bottom
            if dropped or i in named:
                label = 1
            elif stem in MIXED:
                continue
            else:
                label = 0
            X.append(f); y.append(label); groups.append(ms); where.append(f"{ms} {stem[-7:]} row {i - top + 1}")
    return np.array(X), np.array(y), np.array(groups), where


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    X, y, groups, where = gather(sys.argv[1:])
    (DATA / "commentary").mkdir(exist_ok=True)
    np.savez(DATA / "commentary" / "clf_data.npz", X=X, y=y, groups=groups, where=np.array(where))
    names = ["dens", "stem", "pen", "gap_in", "gap_out", "pos", "filled", "no_hand"]
    print(f"rows: {len(y)} ({int(y.sum())} commentary, {int((1 - y).sum())} main) from {len(set(groups))} manuscripts")
    prob = np.zeros(len(y))
    for g in sorted(set(groups)):
        tr, te = groups != g, groups == g
        if y[tr].sum() == 0:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)).fit(X[tr], y[tr])
        prob[te] = m.predict_proba(X[te])[:, 1]
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        fp = int(((prob >= t) & (y == 0)).sum()); tp = int(((prob >= t) & (y == 1)).sum())
        print(f"  threshold {t:.1f}: commentary found {tp}/{int(y.sum())}, main rows taken for commentary {fp}/{int((1 - y).sum())}")
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)).fit(X, y)
    sc, lr = m.named_steps["standardscaler"], m.named_steps["logisticregression"]
    w = lr.coef_[0] / sc.scale_
    b = lr.intercept_[0] - float((lr.coef_[0] * sc.mean_ / sc.scale_).sum())
    print("  weights on the raw features:", ", ".join(f"{n} {v:+.2f}" for n, v in zip(names, w)), f"; bias {b:+.2f}")
    print("  held-out errors at 0.7:")
    for i in np.argsort(-np.abs(prob - y)):
        if (prob[i] >= 0.7) != bool(y[i]):
            print(f"    {'MISSED commentary' if y[i] else 'MAIN taken'} {where[i]}  p {prob[i]:.2f}  features {np.round(X[i], 2).tolist()}")
