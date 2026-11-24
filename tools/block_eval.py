"""How well the block finder splits tripāṭha and pañcapāṭha pages, against the true rows per block from the review
(data/kraken_review/feedback/block_counts.tsv), from Kraken's lines (seg) and from the U-Net's (seg_torch).

    PYTHONPATH=. .venv/bin/python tools/block_eval.py [--seg=seg,seg_torch] [-v]

A page counts as right when every block has the true number of rows. Also given: pages whose main block is right, and
the mean error in rows per block. -v lists every page.
"""
import csv, sys, warnings
from collections import defaultdict
warnings.filterwarnings("ignore")
from msocr import blocks, predict, show
from msocr.config import DATA

BOXES = ("top", "main", "bottom", "left", "right")


def truth():
    with open(DATA / "kraken_review" / "feedback" / "block_counts.tsv", encoding="utf-8") as f:
        lines = [l for l in f if not l.startswith("#")]
    return [(r["manuscript"], r["page"], {b: int(r[b]) for b in BOXES}) for r in csv.DictReader(lines, delimiter="\t")]


def counts(ms, stem, seg):
    old = (show.SEG, blocks.SEG, predict.SEG)
    show.SEG = blocks.SEG = predict.SEG = seg
    try:
        found, size, note = blocks.blocks_for_page(ms, stem + ".jpg")
    finally:
        show.SEG, blocks.SEG, predict.SEG = old
    return {b: len(found[b]["rows"]) if b in found else 0 for b in BOXES}


def main():
    segs = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--seg=")), ["seg", "seg_torch"])
    verbose = "-v" in sys.argv
    pages = truth()
    for seg in segs:
        right = main_right = 0
        err = defaultdict(list)
        per_ms = defaultdict(lambda: [0, 0])
        for ms, stem, t in pages:
            if not (DATA / ms / seg / f"{stem}.json").exists():
                continue
            c = counts(ms, stem, seg)
            ok = all(c[b] == t[b] for b in BOXES)
            right += ok
            main_right += c["main"] == t["main"]
            per_ms[ms][0] += ok
            per_ms[ms][1] += 1
            for b in BOXES:
                if t[b] or c[b]:
                    err[b].append(abs(c[b] - t[b]))
            if verbose:
                print(f"  {seg:9s} {ms} {stem}: {'RIGHT' if ok else 'wrong'}  found " +
                      ", ".join(f"{b} {c[b]}" for b in BOXES if c[b] or t[b]) + "  | true " + ", ".join(f"{b} {t[b]}" for b in BOXES if c[b] or t[b]))
        n = sum(v[1] for v in per_ms.values())
        print(f"{seg}: {right} of {n} pages right in every block; main block right on {main_right}; mean error in rows: " +
              ", ".join(f"{b} {sum(v) / len(v):.2f}" for b, v in err.items()) + "  (" +
              ", ".join(f"{ms} {a}/{m}" for ms, (a, m) in sorted(per_ms.items())) + ")")


if __name__ == "__main__":
    main()
