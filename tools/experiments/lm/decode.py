"""CTC prefix beam search over a recogniser's log-probabilities with a character LM, scored against the truth.

    .venv/bin/python tools/experiments/lm/decode.py <dump prefix> --lm <lm.npz>[,<lm.npz>:<weight>...] --alpha 0.5 --beta 0.5
        [--beam 16] [--sets ...] [--prev-line] [--limit N] [--out decoded.jsonl] [--jobs 6]

The score of a hypothesis is log P_ctc + alpha * log P_lm + beta * (its length): alpha weighs the LM against the
recogniser, beta pays for each symbol (an LM alone would rather read nothing). --prev-line reads a page's lines in
order and gives each line the previous line's reading as its context (a line starts mid-sentence)."""
import argparse
import collections
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from charlm import CharLM, MixLM, SYM, A, symbol              # noqa: E402
from msocr.aksara import aksharas, _encode                    # noqa: E402
from mine_errors import held_out                              # noqa: E402

NEG = -1e30
LOG_HALF = math.log(0.5)


def lse(a, b):
    if a < b:
        a, b = b, a
    return a if b <= NEG else a + math.log1p(math.exp(b - a))


def decode_line(lp, chars, lm, alpha, beta, beam=16, prune=-7.0, context=(), punct=False):
    """lp: (T, C) log-probabilities, class 0 blank; chars[c-1] for class c. -> (text, lm ids of the text)"""
    T, C = lp.shape
    sym = [0] + [symbol(ch, punct) for ch in chars]           # class -> LM symbol (0 = boundary for anything else)
    n_ctx = lm.order - 1
    # prefix -> [log p ending in blank, log p ending in non-blank]; prefixes as tuples of classes
    beams = {(): [0.0, NEG]}
    state = {(): (0.0, tuple(context))}                       # prefix -> (lm log prob total, lm context)
    for t in range(T):
        row = lp[t]
        cands = np.nonzero(row > prune)[0]
        nxt = collections.defaultdict(lambda: [NEG, NEG])
        for prefix, (pb, pnb) in beams.items():
            ptot = lse(pb, pnb)
            # blank: the prefix stays
            e = nxt[prefix]
            e[0] = lse(e[0], ptot + row[0])
            last = prefix[-1] if prefix else -1
            for c in cands:
                c = int(c)
                if c == 0:
                    continue
                if c == last:
                    e[1] = lse(e[1], pnb + row[c])            # the same letter continuing
                    new = prefix + (c,)
                    nxt[new][1] = lse(nxt[new][1], pb + row[c])    # the same letter again, after a blank
                else:
                    new = prefix + (c,)
                    nxt[new][1] = lse(nxt[new][1], ptot + row[c])
                if new not in state:
                    total, ctx = state[prefix]
                    s = sym[c]
                    if s == 0:
                        lpl = LOG_HALF if ctx and ctx[-1] == 0 else float(lm.dist(ctx)[0])
                        nctx = (0,)
                    else:
                        lpl = float(lm.dist(ctx)[s])
                        nctx = (ctx + (s,))[-n_ctx:]
                    state[new] = (total + lpl, nctx)
        # keep the best
        scored = sorted(nxt.items(), key=lambda kv: -(lse(*kv[1]) + alpha * state[kv[0]][0] + beta * len(kv[0])))
        beams = dict(scored[:beam])
        keep = set(beams)
        state = {p: v for p, v in state.items() if p in keep}
    best = max(beams.items(), key=lambda kv: lse(*kv[1]) + alpha * state[kv[0]][0] + beta * len(kv[0]))[0]
    return "".join(chars[c - 1] for c in best), [sym[c] for c in best]


def load_lm(spec):
    parts = []
    for p in spec.split(","):
        path, _, w = p.partition(":")
        if path.endswith(".pt"):
            from lstmlm import LSTMDist
            lm = LSTMDist(path)
        else:
            lm = CharLM.load(path)
        parts.append((lm, float(w) if w else 1.0))
    if len(parts) == 1:
        return parts[0][0]
    tot = sum(w for _, w in parts)
    return MixLM([l for l, _ in parts], [w / tot for _, w in parts])


_G = {}


def _init(spec, alpha, beta, beam, prev, punct):
    _G["lm"] = load_lm(spec)
    _G.update(alpha=alpha, beta=beta, beam=beam, prev=prev, punct=punct)


def _page(job):
    """decode a page's lines in order"""
    names, mats, chars = job
    lm = _G["lm"]
    out, ctx = [], ()
    for name, m in zip(names, mats):
        for l in getattr(lm, "lms", [lm]):
            if hasattr(l, "reset"):
                l.reset()
        if hasattr(lm, "cache"):
            lm.cache = {}
        text, ids = decode_line(m.astype(np.float32), chars, lm, _G["alpha"], _G["beta"], _G["beam"], context=ctx, punct=_G["punct"])
        out.append(text)
        if _G["prev"]:
            ctx = tuple(ids[-(lm.order - 1):])
    return names, out


def rates(pairs):
    e = n = ae = an = 0
    for p, t in pairs:
        e += Levenshtein.distance(p, t)
        n += len(t)
        a, b = _encode(aksharas(p), aksharas(t))
        ae += Levenshtein.distance(a, b)
        an += len(b)
    return 100.0 * e / max(1, n), 100.0 * ae / max(1, an)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--lm", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--sets", nargs="*", help="e.g. 'LDII_85885 val' LDII_1806")
    ap.add_argument("--prev-line", action="store_true")
    ap.add_argument("--punct", action="store_true", help="daṇḍas and digits are symbols the LM knows (an LM built with --punct)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    model = Path(a.dump).stem
    rows = [json.loads(l) for l in open(a.dump + ".jsonl", encoding="utf-8")]
    rows = [r for r in rows if held_out(model, r)]
    for r in rows:
        r["key"] = f"{r['ms']} {r['set']}" if r["ms"] == "LDII_85885" else r["ms"]
    if a.sets:
        rows = [r for r in rows if r["key"] in a.sets]
    if a.limit:
        rows = rows[:a.limit]
    z = np.load(a.dump + ".npz")
    chars = list(z["chars"])
    pages = collections.OrderedDict()
    for r in rows:
        pages.setdefault(r["page"], []).append(r)
    jobs = [([r["name"] for r in rs], [z[r["name"]] for r in rs], chars) for rs in pages.values()]
    with Pool(a.jobs, _init, (a.lm, a.alpha, a.beta, a.beam, a.prev_line, a.punct)) as pool:
        decoded = {}
        for names, texts in pool.imap_unordered(_page, jobs):
            decoded.update(zip(names, texts))
    by = collections.OrderedDict()
    for r in rows:
        by.setdefault(r["key"], []).append(r)
    print(f"[{model}] lm {a.lm} alpha {a.alpha} beta {a.beta} beam {a.beam}{' prev-line' if a.prev_line else ''}{' punct' if a.punct else ''}")
    print(f"  {'set':<18} {'lines':>5} {'greedy CER':>10} {'decoded':>8} {'greedy GER':>10} {'decoded':>8}   lines better/worse")
    allg, alld = [], []
    for key, rs in by.items():
        g = [(r["greedy"], r["truth"]) for r in rs]
        d = [(decoded[r["name"]], r["truth"]) for r in rs]
        allg += g
        alld += d
        gc, gg = rates(g)
        dc, dg = rates(d)
        better = sum(Levenshtein.distance(*y) < Levenshtein.distance(*x) for x, y in zip(g, d))
        worse = sum(Levenshtein.distance(*y) > Levenshtein.distance(*x) for x, y in zip(g, d))
        print(f"  {key:<18} {len(rs):>5} {gc:>10.2f} {dc:>8.2f} {gg:>10.2f} {dg:>8.2f}   {better}/{worse}")
    gc, gg = rates(allg)
    dc, dg = rates(alld)
    print(f"  {'all':<18} {len(rows):>5} {gc:>10.2f} {dc:>8.2f} {gg:>10.2f} {dg:>8.2f}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"name": r["name"], "set": r["key"], "truth": r["truth"], "greedy": r["greedy"], "decoded": decoded[r["name"]]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
