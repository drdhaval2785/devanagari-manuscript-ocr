"""Which segmentation reads better? The recogniser is run on the same pages twice: once on the rows the row finder
builds from Kraken's lines (data/<ms>/seg), once on the rows it builds from our U-Net's lines (data/<ms>/seg_torch),
and each page's reading is scored against its corrected transcription. The pages are LDII_85885's held-out test pages
(data/LDII_85885/test.txt), which neither the recogniser (run3) nor the U-Net (tseg1) was trained on.

Scoring is per page on the lines joined end to end, spaces and line breaks removed, so that a line split differently
is not counted as an error, but a line missed, a line read twice or a line cut short is.

    PYTHONPATH=. .venv/bin/python tools/segocr_test.py
"""
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
from rapidfuzz.distance import Levenshtein

from msocr import blocks, gt, predict
from msocr.config import DATA
from msocr.recognizer import load_model

MS = "LDII_85885"
MODEL = DATA / MS / "models" / "run3" / "best.pt"
SEG2 = next((a.split("=", 1)[1] for a in __import__("sys").argv[1:] if a.startswith("--seg=")), "seg_torch")   # the segmentation to set against Kraken's


def squash(s):
    return "".join(s.split())


def main():
    net, chars, ck = load_model(MODEL)
    truth = defaultdict(list)
    for r in gt.load(MS):
        truth[r["image"]].append((r["n"], r["text"]))
    tests = [Path(p).stem + ".jpg" for p in (DATA / MS / "test.txt").read_text().split()]
    missing = [t for t in tests if not (DATA / MS / SEG2 / t.replace(".jpg", ".json")).exists()]
    if missing:
        raise SystemExit(f"U-Net segmentation missing for {len(missing)} test pages: run "
                         f"python -m msocr.tseg segment {MS} --pages {' '.join(m[:-4] for m in missing)}")
    results = {}
    for seg in ("seg", SEG2):
        predict.SEG = blocks.SEG = seg
        per, errs, total, t0 = [], 0, 0, time.time()
        for img in tests:
            boxes, note, warn = predict.read_page(MS, img, net, chars)
            lines = [l["text"] for b in boxes.values() for l in b]
            ref = squash("".join(t for n, t in sorted(truth[img])))
            e = Levenshtein.distance(squash("".join(lines)), ref)
            errs += e
            total += len(ref)
            per.append((img, len(lines), len(truth[img]), e, len(ref), warn))
        results[seg] = (errs, total, per, time.time() - t0)
    names = {"seg": "Kraken lines", SEG2: "U-Net lines" if SEG2 == "seg_torch" else SEG2}
    print(f"{MS}: {len(tests)} held-out test pages, recogniser {MODEL.parent.name}")
    for seg, (errs, total, per, secs) in results.items():
        print(f"  {names[seg]:13s} CER {100 * errs / total:5.2f}%  ({errs} errors in {total} characters; read in {secs:.0f} s)")
    print(f"\n  {'page':16s} {'true lines':>10s} {'Kraken rows':>11s} {'CER':>7s} {'U-Net rows':>11s} {'CER':>7s}")
    for (img, nk, nt, ek, n, wk), (_, nu, _, eu, _, wu) in zip(results["seg"][2], results[SEG2][2]):
        flag = "  <- differs" if abs(ek - eu) > 0.01 * n else ""
        print(f"  {img[:-4]:16s} {nt:10d} {nk:11d} {100 * ek / n:6.2f}% {nu:11d} {100 * eu / n:6.2f}%{flag}")


if __name__ == "__main__":
    main()
