"""Errors at the two ends of a line, and whether the segmentation causes them.

Every corrected line of the pool's eight manuscripts is read twice by one recogniser (greedily, no language model), once
from the rows found on the U-Net's lines (seg_torch) and once from the rows found on Kraken's (seg), both cut by our own
cutter as predict cuts them. For each line it records where along the truth the errors fall, and for each end of the
row what the cut did there: how far the row runs beyond the segmenter's own lines (the row finder makes every row as
wide as the text block), whether ink goes on past the cut (a clipped letter), and how far the band has drifted off the
writing near the end compared with its middle.

    .venv/bin/python tools/experiments/margins/edges.py [--model pool1] [--out tools/experiments/margins/results/edges.jsonl]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from msocr import pagetext                                   # noqa: E402
from msocr.align import _merge                               # noqa: E402
from msocr.config import MANUSCRIPTS, ms_dir                 # noqa: E402
from msocr.linecut import band_of, cut_line                  # noqa: E402
from msocr.pagediff import align, model_path, pair           # noqa: E402
from msocr.predict import page_rows, read_line               # noqa: E402
from msocr.recognizer import load_model                      # noqa: E402
from msocr.segment import reading_from                       # noqa: E402

MSS = ["LDII_1806", "LDII_4278", "LDII_45364", "LDII_6487", "LDII_80136", "LDII_82320", "LDII_82668", "LDII_9851"]
SEGS = ["seg_torch", "seg"]


def end_geometry(gray, base, above, below, pitch, det_x0, det_x1):
    """what the cut does at each end of a row -> {left: {...}, right: {...}}"""
    ext = int(round(0.6 * pitch))
    b = [(base[0][0] - ext, base[0][1])] + [tuple(p) for p in base] + [(base[-1][0] + ext, base[-1][1])]
    strip = cut_line(gray, b, above, below).astype(np.float32)
    H, W = strip.shape
    inner = strip[:, ext:W - ext]
    p5, p50 = np.percentile(inner, [5, 50])
    ink = strip < (p5 + p50) / 2
    col = ink.mean(axis=0)
    col[col > 0.8] = 0.0                                   # a rule drawn down the page, not writing
    k = max(1, int(0.15 * pitch))
    sm = np.convolve(col, np.ones(k) / k, mode="same")
    typical = float(np.median(sm[ext:W - ext])) or 1e-6
    rows_y = np.arange(H)[:, None]

    def centroid(c0, c1):
        m = ink[:, c0:c1].astype(np.float32)
        m[:, col[c0:c1] == 0] = 0
        s = m.sum()
        return float((m * rows_y).sum() / s) if s > 0 else None
    mid = centroid(ext + (W - 2 * ext) // 3, W - ext - (W - 2 * ext) // 3)
    span = int(0.25 * pitch)
    out = {}
    for side in ("left", "right"):
        if side == "left":
            edge_in = sm[ext:ext + span].mean()
            beyond = sm[max(0, ext - int(0.4 * pitch)):ext].mean()
            c = centroid(ext, ext + int(1.5 * pitch))
            extra = (det_x0 - base[0][0]) / pitch
        else:
            edge_in = sm[W - ext - span:W - ext].mean()
            beyond = sm[W - ext:W - ext + int(0.4 * pitch)].mean()
            c = centroid(W - ext - int(1.5 * pitch), W - ext)
            extra = (base[-1][0] - det_x1) / pitch
        out[side] = {"edge_in": round(float(edge_in / typical), 3), "beyond": round(float(beyond / typical), 3),
                     "clipped": bool(edge_in >= 0.3 * typical and beyond >= 0.3 * typical),
                     "drift": None if c is None or mid is None else round((c - mid) / pitch, 3),
                     "extra": round(float(extra), 3)}
    return out


def errors_by_place(truth, read):
    """-> (wrong truth positions, akṣaras of the truth, {left, right, mid: akṣara edits}, {left, right: kind})"""
    a, b, ops = align(truth, read)
    n = len(a)
    wrong = set()
    cnt = {"left": 0, "right": 0, "mid": 0}
    kinds = {"left": None, "right": None}
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        e = max(i2 - i1, j2 - j1)
        wrong.update(range(i1, i2) if i2 > i1 else [min(i1, n - 1)])
        kd = {"delete": "missing", "insert": "extra", "replace": "misread"}[tag]
        if i1 == 0 and j1 == 0:
            cnt["left"] += e
            kinds["left"] = kd
        elif i2 == n and j2 == len(b):
            cnt["right"] += e
            kinds["right"] = kd
        else:
            cnt["mid"] += e
    return sorted(wrong), n, cnt, kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pool1")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "edges.jsonl"))
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    net, chars, _ = load_model(model_path(a.model))
    out = open(a.out, "w", encoding="utf-8")
    for ms in MSS:
        for gt in sorted((ms_dir(ms) / "corrected").glob("*.txt")):
            stem = gt.stem
            image = stem + ".jpg"
            truth = [x.strip().replace(" ", "") for x in pagetext.read(gt).get("main", [])]
            while truth and not truth[-1]:
                truth.pop()
            with Image.open(MANUSCRIPTS[ms]["images"] / image) as im:
                rgb = np.asarray(im.convert("RGB"))
                gray = np.asarray(im.convert("L"))
            for seg in SEGS:
                with reading_from(seg):
                    rows, block, pitch, size, note, warn = page_rows(ms, image)
                merged = [_merge(r, block, pitch) for r in rows]
                reads, geos = [], []
                for r, (base, poly) in zip(rows, merged):
                    above, below = band_of(base, poly)
                    crop = Image.fromarray(cut_line(rgb, base, above, below))
                    reads.append(read_line(net, chars, crop)[0])
                    xs = [p[0] for d in r for p in d["line"]["baseline"]]
                    geos.append(end_geometry(gray, base, above, below, pitch, min(xs), max(xs)))
                for i, j in pair(reads, truth):
                    if i is None or j is None or not truth[j]:
                        continue
                    wrong, n, cnt, kinds = errors_by_place(truth[j], reads[i])
                    rec = {"ms": ms, "page": stem, "seg": seg, "row": i + 1, "line": j + 1, "rows": len(rows),
                           "lines": len(truth), "pitch": round(pitch, 1), "block": [round(block[0]), round(block[1])],
                           "n": n, "wrong": wrong, "cnt": cnt, "kinds": kinds, "geo": geos[i],
                           "truth": truth[j], "read": reads[i], "filled": bool(rows[i][0].get("filled"))}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  {ms} {stem}: done", flush=True)
    out.close()


if __name__ == "__main__":
    main()
