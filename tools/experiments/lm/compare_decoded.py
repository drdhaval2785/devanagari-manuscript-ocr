"""What the LM decoding changed: the akṣara mistakes it mended and the ones it made, by pattern.

    .venv/bin/python tools/experiments/lm/compare_decoded.py <decoded.jsonl> [--top 25] [--show 12]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from msocr.aksara import aksharas, _encode   # noqa: E402


def edits(read, truth):
    at, ar = aksharas(truth), aksharas(read)
    et, er = _encode(at, ar)
    out = collections.Counter()
    for op, i, j in Levenshtein.editops(et, er):
        out[(at[i] if op != "insert" else "∅", ar[j] if op != "delete" else "∅")] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decoded")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--show", type=int, default=12)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.decoded, encoding="utf-8")]
    mended, made, kept = collections.Counter(), collections.Counter(), collections.Counter()
    examples = []
    for r in rows:
        g, d = edits(r["greedy"], r["truth"]), edits(r["decoded"], r["truth"])
        for k in set(g) | set(d):
            n_g, n_d = g.get(k, 0), d.get(k, 0)
            mended[k] += max(0, n_g - n_d)
            made[k] += max(0, n_d - n_g)
            kept[k] += min(n_g, n_d)
        if r["greedy"] != r["decoded"] and len(examples) < 400:
            examples.append(r)
    tm, tk, tmade = sum(mended.values()), sum(kept.values()), sum(made.values())
    print(f"{Path(a.decoded).name}: akṣara mistakes of the greedy reading: {tm + tk}; mended {tm}, kept {tk}; new ones made {tmade}")
    print(f"  mended (truth → greedy read), commonest:")
    for k, v in mended.most_common(a.top):
        print(f"    {v:>4}  {k[0]} → {k[1]}")
    print(f"  made (truth → decoded), commonest:")
    for k, v in made.most_common(a.top):
        print(f"    {v:>4}  {k[0]} → {k[1]}")
    print(f"  kept (not mended), commonest:")
    for k, v in kept.most_common(a.top):
        print(f"    {v:>4}  {k[0]} → {k[1]}")
    print("\n  some lines that changed (truth / greedy / decoded):")
    shown = 0
    for r in examples:
        dg, dd = Levenshtein.distance(r["greedy"], r["truth"]), Levenshtein.distance(r["decoded"], r["truth"])
        if dg == dd:
            continue
        print(f"  {r['name']} ({r['set']}): {dg} -> {dd} edits\n    T: {r['truth']}\n    G: {r['greedy']}\n    D: {r['decoded']}")
        shown += 1
        if shown >= a.show:
            break


if __name__ == "__main__":
    main()
