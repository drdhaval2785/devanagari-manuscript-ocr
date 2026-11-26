"""Ends where the truth may have inherited an old cut: today's reading runs further at an end of the line than the
truth does, and the draft the line was corrected from stopped where the truth stops.

    .venv/bin/python tools/experiments/margins/inherited.py results/edges.jsonl [--seg seg_torch] [--list]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from msocr import pagetext                      # noqa: E402
from msocr.config import ms_dir                 # noqa: E402
from msocr.pagediff import align                # noqa: E402


def overhang(truth, other):
    """akṣaras `other` has beyond the truth at the (left, right) end: the akṣaras before the first match and after the
    last, the other's minus the truth's"""
    a, b, ops = align(truth, other)
    eq = [o for o in ops if o[0] == "equal"]
    if not eq:
        return 0, 0
    left = eq[0][3] - eq[0][1]
    right = (len(b) - eq[-1][4]) - (len(a) - eq[-1][2])
    return left, right


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--seg", default="seg_torch")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.file, encoding="utf-8") if json.loads(l)["seg"] == a.seg]
    drafts = {}
    stats = collections.defaultdict(collections.Counter)
    rows = []
    for r in recs:
        key = (r["ms"], r["page"])
        if key not in drafts:
            f = ms_dir(r["ms"]) / "drafts" / f"{r['page']}.txt"
            drafts[key] = [x.strip().replace(" ", "") for x in pagetext.read(f).get("main", [])] if f.exists() else []
        d = drafts[key]
        draft = d[r["line"] - 1] if r["line"] - 1 < len(d) else None
        rl, rr = overhang(r["truth"], r["read"])
        dl, dr = overhang(r["truth"], draft) if draft else (None, None)
        s = stats[r["ms"]]
        s["lines"] += 1
        for side, ro, do, k in (("left", rl, dl, 0), ("right", rr, dr, r["n"] - 1)):
            s[f"{side} wrong"] += k in r["wrong"]
            if ro >= 1:
                s[f"{side} reading longer"] += 1
                if do is not None and do <= 0:
                    s[f"{side} inherited"] += 1
                    rows.append((r["ms"], r["page"], r["line"], side, ro, r["truth"], r["read"], draft))
    print(f"{'manuscript':10s} lines | left: wrong  reading longer  of which the draft stopped there too | right: same")
    tot = collections.Counter()
    for ms in sorted(stats):
        s = stats[ms]
        tot.update(s)
        print(f"{ms:10s} {s['lines']:5d} | {s['left wrong']:5d} {s['left reading longer']:6d} {s['left inherited']:6d}"
              f"   | {s['right wrong']:5d} {s['right reading longer']:6d} {s['right inherited']:6d}")
    print(f"{'all':10s} {tot['lines']:5d} | {tot['left wrong']:5d} {tot['left reading longer']:6d} {tot['left inherited']:6d}"
          f"   | {tot['right wrong']:5d} {tot['right reading longer']:6d} {tot['right inherited']:6d}")
    if a.list:
        for ms, page, line, side, ro, t, rd, dr in rows:
            cut = (lambda x: x[:14]) if side == "left" else (lambda x: x[-14:])
            print(f"  {page} L{line:<2d} {side:5s} +{ro}  truth {cut(t):16s} read {cut(rd):16s} draft {cut(dr or ''):16s}")


if __name__ == "__main__":
    main()
