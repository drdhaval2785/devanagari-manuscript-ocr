"""The ends of a row cut three ways, each line read again by one recogniser (greedily):
  now     as predict cuts it: the row as wide as the text block, stopping 2 px short of a rule
  pad     the same, with a quarter pitch of blank paper (the line's own paper colour) added at both ends
  wide    the cut itself a quarter pitch longer at both ends (across a rule, if one is there)
  erase   as wide, the columns of a rule in it (dark or red down nearly the whole line image) painted paper
For each end the record also says whether the row finder stopped it at a rule drawn down the page.

    .venv/bin/python tools/experiments/margins/variants.py [--model pool1] [--out results/variants.jsonl]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from msocr import align as A                                 # noqa: E402
from msocr import pagetext                                   # noqa: E402
from msocr.config import MANUSCRIPTS, ms_dir                 # noqa: E402
from msocr.linecut import band_of, cut_line                  # noqa: E402
from msocr.pagediff import model_path, pair                  # noqa: E402
from msocr.predict import page_rows, read_line               # noqa: E402
from msocr.recognizer import load_model                      # noqa: E402
from msocr.segment import reading_from                       # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from edges import MSS, errors_by_place                       # noqa: E402

RULES = {}
_orig = A._to_the_ink


def _spy(*a, **k):
    r = _orig(*a, **k)
    RULES["last"] = (r[2], r[3])
    return r


A._to_the_ink = _spy


def erase_rules(im, edge, paper):
    """the columns within `edge` px of either end that are dark (or red) down nearly the whole height, painted paper"""
    f = im.astype(np.int32)
    lum = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    p5, p50 = np.percentile(lum, [5, 50])
    ink = (lum < p50 - 0.35 * (p50 - p5)) | (f[..., 0] - np.maximum(f[..., 1], f[..., 2]) > 45)
    col = ink.mean(axis=0)
    out = im.copy()
    W = im.shape[1]
    for x in list(range(min(edge, W))) + list(range(max(0, W - edge), W)):
        if col[x] >= 0.7:
            lo, hi = max(0, x - 1), min(W, x + 2)
            out[:, lo:hi] = paper
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pool1")
    ap.add_argument("--seg", default="seg_torch")
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "variants.jsonl"))
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    net, chars, _ = load_model(model_path(a.model))
    out = open(a.out, "w", encoding="utf-8")
    for ms in MSS:
        for gt in sorted((ms_dir(ms) / "corrected").glob("*.txt")):
            stem, image = gt.stem, gt.stem + ".jpg"
            truth = [x.strip().replace(" ", "") for x in pagetext.read(gt).get("main", [])]
            while truth and not truth[-1]:
                truth.pop()
            rgb = np.asarray(Image.open(MANUSCRIPTS[ms]["images"] / image).convert("RGB"))
            with reading_from(a.seg):
                rows, block, pitch, size, note, warn = page_rows(ms, image)
            rule_l, rule_r = RULES.get("last", (None, None))
            p = int(round(a.frac * pitch))
            reads = {"now": [], "pad": [], "wide": [], "erase": []}
            for r in rows:
                base, poly = A._merge(r, block, pitch)
                above, below = band_of(base, poly)
                now = cut_line(rgb, base, above, below)
                paper = np.median(now.reshape(-1, 3), axis=0).astype(np.uint8)
                side = np.broadcast_to(paper, (now.shape[0], p, 3))
                pad = np.concatenate([side, now, side], axis=1)
                wb = [(base[0][0] - p, base[0][1])] + [tuple(q) for q in base] + [(base[-1][0] + p, base[-1][1])]
                wide = cut_line(rgb, wb, above, below)
                erase = erase_rules(wide, p + 6, paper)
                for k, im in (("now", now), ("pad", pad), ("wide", wide), ("erase", erase)):
                    reads[k].append(read_line(net, chars, Image.fromarray(np.ascontiguousarray(im)))[0])
            for i, j in pair(reads["now"], truth):
                if i is None or j is None or not truth[j]:
                    continue
                rec = {"ms": ms, "page": stem, "row": i + 1, "line": j + 1, "truth": truth[j],
                       "rule": {"left": rule_l is not None and abs(block[0] - (rule_l + 2)) < 1.5,
                                "right": rule_r is not None and abs(block[1] - (rule_r - 2)) < 1.5}}
                for k in reads:
                    wrong, n, cnt, kinds = errors_by_place(truth[j], reads[k][i])
                    rec[k] = {"read": reads[k][i], "wrong": wrong, "cnt": cnt, "kinds": kinds}
                rec["n"] = n
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  {ms} {stem}: rules {rule_l is not None}/{rule_r is not None}", flush=True)
    out.close()


if __name__ == "__main__":
    main()
