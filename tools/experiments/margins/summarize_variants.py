"""Tables from variants.py: the ends of a line wrong under each cut, at ruled and unruled ends, leaving out the ends
where the truth inherited an old cut (inherited.py).

    .venv/bin/python tools/experiments/margins/summarize_variants.py results/variants.jsonl [--model pool1]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).parent))
from msocr import pagetext                      # noqa: E402
from msocr.config import ms_dir                 # noqa: E402
from inherited import overhang                  # noqa: E402
from summarize import unseen, pct               # noqa: E402

V = ["now", "pad", "wide", "erase"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--model", default="pool1")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.file, encoding="utf-8")]
    drafts = {}
    for r in recs:                                   # the ends whose truth inherited an old cut
        key = (r["ms"], r["page"])
        if key not in drafts:
            f = ms_dir(r["ms"]) / "drafts" / f"{r['page']}.txt"
            drafts[key] = [x.strip().replace(" ", "") for x in pagetext.read(f).get("main", [])] if f.exists() else []
        d = drafts[key]
        draft = d[r["line"] - 1] if r["line"] - 1 < len(d) else None
        rl, rr = overhang(r["truth"], r["now"]["read"])
        dl, dr = overhang(r["truth"], draft) if draft else (1, 1)
        r["inherited"] = {"left": rl >= 1 and dl <= 0, "right": rr >= 1 and dr <= 0}
    for gname, gf in (("unseen pages", lambda r: unseen(r, a.model)), ("all pages", lambda r: True)):
        R = [r for r in recs if gf(r)]
        print(f"\n=== {gname} ({a.model}, greedy): {len(R)} lines")
        print("cut    akṣara edits/100 | middle wrong% | left end wrong%: ruled  unruled | right end wrong%: ruled  unruled")
        for v in V:
            n = sum(r["n"] for r in R)
            e = sum(sum(r[v]["cnt"].values()) for r in R)
            midw = sum(sum(1 for k in r[v]["wrong"] if 0 < k < r["n"] - 1) for r in R)
            midn = sum(max(0, r["n"] - 2) for r in R)
            cells = []
            for side, k_of in (("left", lambda r: 0), ("right", lambda r: r["n"] - 1)):
                for ruled in (True, False):
                    E = [r for r in R if r["rule"][side] == ruled and not r["inherited"][side]]
                    cells.append(f"{pct(sum(k_of(r) in r[v]['wrong'] for r in E), len(E))} ({len(E):3d})")
            print(f"{v:5s}  {pct(e, n):>17s}    | {pct(midw, midn):>12s}  |    {cells[0]} {cells[1]} |    {cells[2]} {cells[3]}")
        c = collections.Counter()
        for r in R:
            for side, k in (("left", 0), ("right", r["n"] - 1)):
                if r["inherited"][side]:
                    continue
                c[(side, k in r["now"]["wrong"], k in r["pad"]["wrong"])] += 1
        print("  now -> pad, ends that changed:", ", ".join(f"{s} {'wrong' if x else 'right'}->{'wrong' if y else 'right'} {c[(s, x, y)]}"
                                                     for s in ("left", "right") for x, y in ((True, False), (False, True))))
    print("\n=== per manuscript, right end wrong % now / pad / wide / erase (inherited ends left out), ruled ends of its pages")
    for ms in sorted({r["ms"] for r in recs}):
        R = [r for r in recs if r["ms"] == ms and not r["inherited"]["right"]]
        L = [r for r in recs if r["ms"] == ms and not r["inherited"]["left"]]
        rr = " / ".join(pct(sum(r["n"] - 1 in r[v]["wrong"] for r in R), len(R)) for v in V)
        ll = " / ".join(pct(sum(0 in r[v]["wrong"] for r in L), len(L)) for v in V)
        ru = sum(r["rule"]["right"] for r in R), sum(r["rule"]["left"] for r in L)
        print(f"  {ms:10s} right {rr}   left {ll}   ruled ends right {ru[0]}/{len(R)}, left {ru[1]}/{len(L)}")


if __name__ == "__main__":
    main()
