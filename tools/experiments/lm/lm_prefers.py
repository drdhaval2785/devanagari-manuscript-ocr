"""Would the corpus tell a misread akṣara from the right one? For every akṣara the recogniser read wrongly in place
(a substitution), the LM's probability of the line as written against the line as read, the rest of the line kept.
Also the same for every akṣara read RIGHT, swapped for the model's second guess... no: for a fair view of false alarms,
the whole-line comparison: how often the LM prefers the truth to the greedy reading.

    .venv/bin/python tools/experiments/lm/lm_prefers.py <dump.jsonl> --lm <lm.npz>
"""
import argparse
import collections
import json
import math
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from charlm import symbol                                     # noqa: E402
from decode import load_lm                                    # noqa: E402
from msocr.aksara import aksharas, _encode                    # noqa: E402
from mine_errors import held_out                              # noqa: E402


def ids_of(text):
    out = []
    for ch in text:
        s = symbol(ch)
        if s or (out and out[-1] != 0):
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--lm", required=True)
    a = ap.parse_args()
    lm = load_lm(a.lm)
    model = Path(a.dump).stem
    rows = [json.loads(l) for l in open(a.dump, encoding="utf-8") if held_out(model, json.loads(l))]
    by = collections.defaultdict(lambda: [0, 0, 0, 0, 0.0])    # set -> [subs, lm prefers truth, by >1 nat, lines, lines truth preferred]
    margins = []
    patterns = collections.Counter()
    pattern_wins = collections.Counter()
    for r in rows:
        key = f"{r['ms']} {r['set']}" if r["ms"] == "LDII_85885" else r["ms"]
        at, ap_ = aksharas(r["truth"]), aksharas(r["greedy"])
        et, ep = _encode(at, ap_)
        lt = lm.logprob(ids_of(r["truth"]))
        lg = lm.logprob(ids_of(r["greedy"]))
        by[key][3] += 1
        by[key][4] += lt > lg
        for op, i, j in Levenshtein.editops(et, ep):
            if op != "replace":
                continue
            wrong = at[:i] + [ap_[j]] + at[i + 1:]
            lw = lm.logprob(ids_of("".join(wrong)))
            m = lt - lw
            by[key][0] += 1
            by[key][1] += m > 0
            by[key][2] += m > 1
            margins.append(m)
            patterns[(at[i], ap_[j])] += 1
            pattern_wins[(at[i], ap_[j])] += m > 0
    print(f"[{model}] lm {a.lm}")
    print(f"  {'set':<18} {'lines':>5} {'truth more probable than reading':>34} {'akṣara subs':>11} {'LM prefers truth':>16} {'by > 1 nat':>10}")
    T = [0] * 5
    for key, v in by.items():
        print(f"  {key:<18} {v[3]:>5} {100 * v[4] / v[3]:>33.0f}% {v[0]:>11} {100 * v[1] / max(1, v[0]):>15.0f}% {100 * v[2] / max(1, v[0]):>9.0f}%")
        T = [x + y for x, y in zip(T, v)]
    print(f"  {'all':<18} {T[3]:>5} {100 * T[4] / T[3]:>33.0f}% {T[0]:>11} {100 * T[1] / max(1, T[0]):>15.0f}% {100 * T[2] / max(1, T[0]):>9.0f}%")
    margins.sort()
    print(f"  margin (log P truth - log P as read) over substitutions: median {margins[len(margins) // 2]:.2f}, "
          f"quartiles {margins[len(margins) // 4]:.2f} / {margins[3 * len(margins) // 4]:.2f}")
    print("  commonest substitutions and how often the LM prefers the truth:")
    for (t, p), n in patterns.most_common(25):
        print(f"    {n:>4}  {t} → {p}   {100 * pattern_wins[(t, p)] / n:.0f}%")


if __name__ == "__main__":
    main()
