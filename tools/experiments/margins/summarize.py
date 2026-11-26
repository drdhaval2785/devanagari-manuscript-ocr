"""Tables from edges.py's records: where along a line the errors fall, under each segmentation, and what the cut did
at the ends where they fall.

    .venv/bin/python tools/experiments/margins/summarize.py results/edges.jsonl [--model pool1]
"""
import argparse
import collections
import json

# pool1 trained on every corrected page of these but the last
SEEN = {"LDII_1806": "001806_00068_1", "LDII_45364": "045364_00059_1", "LDII_82668": "082668_00056_1",
        "LDII_9851": "009851_00066_1"}


def unseen(r, model):
    return model != "pool1" or r["ms"] not in SEEN or r["page"] == SEEN[r["ms"]]


def pct(a, b):
    return f"{100 * a / b:5.1f}" if b else "    -"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--model", default="pool1")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.file, encoding="utf-8")]
    segs = ["seg_torch", "seg"]
    groups = [("unseen pages", lambda r: unseen(r, a.model)), ("all pages", lambda r: True)]
    for gname, gf in groups:
        print(f"\n=== {gname} ({a.model}, greedy)")
        print("seg        lines aks   err  CER-like | first wrong% last wrong% middle wrong% | share of edits at: left right")
        for s in segs:
            R = [r for r in recs if r["seg"] == s and gf(r)]
            n = sum(r["n"] for r in R)
            e = sum(sum(r["cnt"].values()) for r in R)
            first = sum(1 for r in R if 0 in r["wrong"])
            last = sum(1 for r in R if r["n"] - 1 in r["wrong"])
            midw = sum(sum(1 for k in r["wrong"] if 0 < k < r["n"] - 1) for r in R)
            midn = sum(max(0, r["n"] - 2) for r in R)
            el, er = sum(r["cnt"]["left"] for r in R), sum(r["cnt"]["right"] for r in R)
            print(f"{s:10s} {len(R):5d} {n:5d} {e:5d} {pct(e, n)} | {pct(first, len(R)):>11s} {pct(last, len(R)):>10s} "
                  f"{pct(midw, midn):>12s}  | {pct(el, e)} {pct(er, e)}")
        # the profile along the line
        print("  wrong % by place (lines of 12+ akṣaras):   from the left 1..6        |        from the right 6..1")
        for s in segs:
            R = [r for r in recs if r["seg"] == s and gf(r) and r["n"] >= 12]
            L = [pct(sum(1 for r in R if k in r["wrong"]), len(R)) for k in range(6)]
            Rt = [pct(sum(1 for r in R if r["n"] - 1 - k in r["wrong"]), len(R)) for k in reversed(range(6))]
            print(f"  {s:10s} " + " ".join(L) + "  | " + " ".join(Rt))
        # kinds at the ends
        print("  at the ends, the kind of error (missing = the reading lacks akṣaras the truth has; extra = has more)")
        for s in segs:
            R = [r for r in recs if r["seg"] == s and gf(r)]
            for side in ("left", "right"):
                c = collections.Counter(r["kinds"][side] for r in R if r["kinds"][side])
                print(f"  {s:10s} {side:5s} " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
        # geometry
        print("  the cut at each end: share of ends, and how often the end akṣara is then wrong")
        print("  seg        ends | clipped: share  wrong% | drift>0.15 pitch: share wrong% | row runs >1 pitch past the segmenter: share wrong% | otherwise wrong%")
        for s in segs:
            R = [r for r in recs if r["seg"] == s and gf(r)]
            ends = [(r, side) for r in R for side in ("left", "right")]

            def bad(r, side):
                return (0 if side == "left" else r["n"] - 1) in r["wrong"]
            clip = [(r, sd) for r, sd in ends if r["geo"][sd]["clipped"]]
            drift = [(r, sd) for r, sd in ends if r["geo"][sd]["drift"] is not None and abs(r["geo"][sd]["drift"]) > 0.15]
            extra = [(r, sd) for r, sd in ends if r["geo"][sd]["extra"] > 1.0]
            other = [(r, sd) for r, sd in ends if (r, sd) not in clip and (r, sd) not in drift and (r, sd) not in extra]
            f = lambda xs: f"{pct(len(xs), len(ends))} {pct(sum(bad(r, sd) for r, sd in xs), len(xs))}"   # noqa: E731
            print(f"  {s:10s} {len(ends):4d} |      {f(clip)}    |            {f(drift)}     |                            {f(extra)}         | {pct(sum(bad(r, sd) for r, sd in other), len(other))}")
    # the same lines under both segmentations
    print("\n=== the same line under both (paired to the same truth line)")
    by = collections.defaultdict(dict)
    for r in recs:
        by[(r["ms"], r["page"], r["line"])][r["seg"]] = r
    both = [v for v in by.values() if len(v) == 2]
    for side in ("left", "right"):
        def bad(r):
            return (0 if side == "left" else r["n"] - 1) in r["wrong"]
        c = collections.Counter((bad(v["seg_torch"]), bad(v["seg"])) for v in both)
        print(f"  {side:5s} end of {len(both)} lines: wrong in both {c[(True, True)]}, only from U-Net rows {c[(True, False)]}, "
              f"only from Kraken rows {c[(False, True)]}, right in both {c[(False, False)]}")
    print("\n=== per manuscript: first and last akṣara wrong % (U-Net rows / Kraken rows), middle wrong %, rows vs lines")
    for ms in sorted({r["ms"] for r in recs}):
        row = []
        for s in segs:
            R = [r for r in recs if r["seg"] == s and r["ms"] == ms]
            first = pct(sum(1 for r in R if 0 in r["wrong"]), len(R))
            last = pct(sum(1 for r in R if r["n"] - 1 in r["wrong"]), len(R))
            midw = sum(sum(1 for k in r["wrong"] if 0 < k < r["n"] - 1) for r in R)
            midn = sum(max(0, r["n"] - 2) for r in R)
            pages = {r["page"]: (r["rows"], r["lines"]) for r in R}
            mism = sum(1 for v in pages.values() if v[0] != v[1])
            row.append(f"{first}/{last}/{pct(midw, midn)} ({len(R)} lines, {mism} pages miscounted)")
        print(f"  {ms:10s}  U-Net {row[0]}   Kraken {row[1]}")


if __name__ == "__main__":
    main()
