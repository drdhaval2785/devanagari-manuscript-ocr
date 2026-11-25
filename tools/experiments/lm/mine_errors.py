"""The recurring mistakes of a recogniser on its held-out lines: by character, by akṣara, and by kind.

    .venv/bin/python tools/experiments/lm/mine_errors.py <dump.jsonl> [--top 40]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from msocr.aksara import aksharas, _encode, MARK, VIRAMA, CONS   # noqa: E402

# which lines a model never trained on
LAST = {"LDII_1806": "001806_00068_1", "LDII_19634": "019634_00030_2", "LDII_45364": "045364_00059_1",
        "LDII_82668": "082668_00056_1", "LDII_9851": "009851_00066_1"}


def held_out(model, r):
    if r["ms"] == "LDII_85885":
        return True
    if model == "pool1":
        return r["page"] == LAST[r["ms"]]
    return r["ms"] != "LDII_19634" or r["page"] == LAST["LDII_19634"]


def kind(t, p):
    """what sort of mistake a character edit is"""
    t = t or ""
    p = p or ""
    vowel_signs = set("ािीुूृॄेैोौॢॣ")
    if t and p:
        if t in vowel_signs and p in vowel_signs:
            return "vowel sign for vowel sign"
        if t in CONS and p in CONS:
            return "consonant for consonant"
        if (t in vowel_signs) != (p in vowel_signs):
            return "vowel sign vs other"
        return "other substitution"
    x = t or p
    what = "missed" if t else "invented"
    if x == VIRAMA:
        return f"virama {what}"
    if x in "ंँ":
        return f"anusvāra {what}"
    if x == "ः":
        return f"visarga {what}"
    if x in vowel_signs:
        return f"vowel sign {what}"
    if x in CONS:
        return f"consonant {what}"
    if x in "।॥":
        return f"daṇḍa {what}"
    if x in "०१२३४५६७८९":
        return f"digit {what}"
    return f"other {what}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    model = Path(a.dump).stem
    rows = [json.loads(l) for l in open(a.dump, encoding="utf-8")]
    rows = [r for r in rows if held_out(model, r)]
    print(f"[{model}] {len(rows)} held-out lines")
    # per set
    sets = collections.OrderedDict()
    for r in rows:
        key = f"{r['ms']} {r['set']}" if r["ms"] == "LDII_85885" else r["ms"]
        s = sets.setdefault(key, [0, 0, 0, 0, 0])
        s[0] += 1
        s[1] += Levenshtein.distance(r["greedy"], r["truth"])
        s[2] += len(r["truth"])
        ea, eb = _encode(aksharas(r["greedy"]), aksharas(r["truth"]))
        s[3] += Levenshtein.distance(ea, eb)
        s[4] += len(eb)
    print(f"  {'set':<22} {'lines':>5} {'chars':>6} {'CER':>6} {'GER':>6}")
    tot = [0] * 5
    for k, s in sets.items():
        print(f"  {k:<22} {s[0]:>5} {s[2]:>6} {100 * s[1] / s[2]:>6.2f} {100 * s[3] / s[4]:>6.2f}")
        tot = [x + y for x, y in zip(tot, s)]
    print(f"  {'all':<22} {tot[0]:>5} {tot[2]:>6} {100 * tot[1] / tot[2]:>6.2f} {100 * tot[3] / tot[4]:>6.2f}")

    # character edits
    cc, kinds = collections.Counter(), collections.Counter()
    for r in rows:
        t, p = r["truth"], r["greedy"]
        for op, i, j in Levenshtein.editops(t, p):
            tc = t[i] if op != "insert" else None
            pc = p[j] if op != "delete" else None
            cc[(tc, pc)] += 1
            kinds[kind(tc, pc)] += 1
    n = sum(cc.values())
    print(f"\n  character edits: {n}; by kind:")
    for k, v in kinds.most_common():
        print(f"    {v:>5} {100 * v / n:5.1f}%  {k}")
    print(f"\n  commonest character edits (truth → read; share of all edits):")
    for (tc, pc), v in cc.most_common(a.top):
        print(f"    {v:>5} {100 * v / n:5.1f}%  {tc or '∅'} → {pc or '∅'}")

    # akṣara edits with context
    ac, examples = collections.Counter(), collections.defaultdict(list)
    for r in rows:
        at, ap_ = aksharas(r["truth"]), aksharas(r["greedy"])
        et, ep = _encode(at, ap_)
        for op, i, j in Levenshtein.editops(et, ep):
            tk = at[i] if op != "insert" else "∅"
            pk = ap_[j] if op != "delete" else "∅"
            ac[(tk, pk)] += 1
            if len(examples[(tk, pk)]) < 3:
                left = "".join(at[max(0, i - 2):i])
                right = "".join(at[i + (op != "insert"):i + (op != "insert") + 2])
                examples[(tk, pk)].append(f"{left}[{tk}]{right}")
    na = sum(ac.values())
    print(f"\n  akṣara edits: {na}; commonest (truth → read), with the truth's neighbours:")
    for (tk, pk), v in ac.most_common(a.top):
        print(f"    {v:>5} {100 * v / na:5.1f}%  {tk} → {pk}    {'  '.join(examples[(tk, pk)])}")
    # how concentrated: share of akṣara edits in the top-k patterns
    cum = 0
    for k, (_, v) in enumerate(ac.most_common(), 1):
        cum += v
        if k in (10, 25, 50, 100, 200):
            print(f"  top {k} akṣara patterns cover {100 * cum / na:.0f}% of akṣara edits")
    print(f"  distinct akṣara patterns: {len(ac)}, seen once: {sum(1 for v in ac.values() if v == 1)}")


if __name__ == "__main__":
    main()
