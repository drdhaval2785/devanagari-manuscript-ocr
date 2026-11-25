"""A character n-gram language model (interpolated absolute discounting) over Devanagari letters, to weigh a recogniser's readings
by how Sanskrit text runs. Symbols: every letter and sign of the Devanagari block; everything else (daṇḍa, digit, mark,
space) is one boundary symbol '|'. Counts are taken with numpy (n-grams packed into 64-bit keys), probabilities stored in
sorted arrays; `dist(context)` gives log P(next symbol | context) for every symbol at once.

    python tools/experiments/lm/charlm.py build <order> <out.npz> <text.txt> [<text.txt> ...]   [--min-count 2] [--punct]
    python tools/experiments/lm/charlm.py perplexity <lm.npz> <text.txt>
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

BOUND = "|"
DIGIT = "#"                                # every digit one symbol
PUNCT = ["।", "॥", DIGIT]                  # kept as symbols of their own by an LM built with --punct (else boundary)
ALPHA = [BOUND] + [chr(c) for c in range(0x900, 0x980) if chr(c) not in "०१२३४५६७८९।॥॰ॐ"] + PUNCT
A = len(ALPHA)
SYM = {c: i for i, c in enumerate(ALPHA)}
PUNCT_IDS = {SYM[c] for c in PUNCT}


def symbol(ch, punct=False):
    """the LM symbol of a character: a letter itself; daṇḍas and digits themselves if punct, else the boundary"""
    if ch in "०१२३४५६७८९0123456789":
        return SYM[DIGIT] if punct else 0
    s = SYM.get(ch, 0)
    return s if punct or s not in PUNCT_IDS else 0
BITS = 7                                   # symbols per packed key: 9 in 63 bits
assert A < (1 << BITS)
LOG_A = math.log(A)


def encode(lines, punct=False):
    """text lines -> one array of symbol ids, every line bounded, boundaries never doubled"""
    out = bytearray()
    for line in lines:
        if out and out[-1] != 0 or not out:
            out.append(0)
        for ch in line:
            s = symbol(ch, punct)
            if s or out[-1] != 0:
                out.append(s)
    if out[-1] != 0:
        out.append(0)
    return np.frombuffer(bytes(out), dtype=np.uint8)


def _mask(k):
    return np.uint64((1 << (BITS * k)) - 1) if k else np.uint64(0)


def ngram_counts(ids, k, chunk=30_000_000):
    """(sorted packed keys of every k-gram, their counts)"""
    n = len(ids) - k + 1
    ks, cs = [], []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        key = np.zeros(e - s, dtype=np.uint64)
        for j in range(k):
            key = (key << np.uint64(BITS)) | ids[s + j:e + j].astype(np.uint64)
        u, c = np.unique(key, return_counts=True)
        ks.append(u)
        cs.append(c)
    if len(ks) == 1:
        return ks[0], cs[0].astype(np.int64)
    keys, cnts = np.concatenate(ks), np.concatenate(cs)
    u, inv = np.unique(keys, return_inverse=True)
    return u, np.bincount(inv, weights=cnts).astype(np.int64)


class CharLM:
    def __init__(self, order, tabs):
        self.order = order
        self.tabs = tabs            # k -> (keys, logp, ctx_keys, ctx_logbow)
        self.cache = {}

    # ---- building
    @classmethod
    def build(cls, ids, order, min_count=2, verbose=True):
        t = time.time()
        raw = {k: ngram_counts(ids, k) for k in range(1, order + 1)}
        if verbose:
            print(f"  counted {order}-grams of {len(ids):,} symbols in {time.time() - t:.0f}s: " + ", ".join(f"{k}: {len(raw[k][0]):,}" for k in raw))
        tabs = {}
        for k in range(1, order + 1):
            # raw counts at every order (interpolated absolute discounting): a line starts with no history, and its
            # first symbols are then scored by the lower orders themselves, which Kneser-Ney's continuation counts
            # would flatten
            keys, c = raw[k]
            n1, n2 = int((c == 1).sum()), int((c == 2).sum())
            D = n1 / (n1 + 2 * n2) if n1 and n2 else 0.5
            ctx = keys >> np.uint64(BITS)
            uctx, start, nw = np.unique(ctx, return_index=True, return_counts=True)
            ctot = np.add.reduceat(c, start).astype(np.float64)
            keep = c >= (min_count if k >= 4 else 1)
            pruned = np.add.reduceat(np.where(keep, 0.0, c - D), start)
            bow = (D * nw + pruned) / ctot
            # probability of the kept entries: discounted count share + bow * lower-order probability
            share = (c[keep] - D) / ctot[np.searchsorted(uctx, ctx[keep])]
            lower = cls._logp(tabs, k - 1, keys[keep] & _mask(k - 1)) if k > 1 else np.full(int(keep.sum()), -LOG_A)
            p = share + bow[np.searchsorted(uctx, ctx[keep])] * np.exp(lower)
            tabs[k] = (keys[keep], np.log(p).astype(np.float32), uctx, np.log(bow).astype(np.float32))
            if verbose:
                print(f"  order {k}: {int(keep.sum()):,} entries kept of {len(keys):,}, {len(uctx):,} contexts, D={D:.2f}")
        return cls(order, tabs)

    @staticmethod
    def _logp(tabs, k, keys):
        """log P of packed (context, symbol) keys of k symbols, by backing off"""
        if k == 0:
            return np.full(len(keys), -LOG_A, dtype=np.float64)
        tk, tlp, ck, cbow = tabs[k]
        i = np.searchsorted(tk, keys)
        i[i >= len(tk)] = 0
        found = tk[i] == keys
        out = np.empty(len(keys), dtype=np.float64)
        out[found] = tlp[i[found]]
        if not found.all():
            rest = keys[~found]
            ctx = rest >> np.uint64(BITS)
            j = np.searchsorted(ck, ctx)
            j[j >= len(ck)] = 0
            has = ck[j] == ctx
            bow = np.where(has, cbow[j], 0.0)
            out[~found] = bow + CharLM._logp(tabs, k - 1, rest & _mask(k - 1))
        return out

    # ---- using
    def dist(self, context):
        """log P(symbol | context) for every symbol, context = tuple of symbol ids (most recent last)"""
        ctx = tuple(context[-(self.order - 1):]) if self.order > 1 else ()
        d = self.cache.get(ctx)
        if d is None:
            k = len(ctx) + 1
            h = np.uint64(0)
            for s in ctx:
                h = (h << np.uint64(BITS)) | np.uint64(s)
            keys = (h << np.uint64(BITS)) | np.arange(A, dtype=np.uint64)
            d = self._logp(self.tabs, k, keys).astype(np.float32)
            if len(self.cache) > 200_000:
                self.cache.clear()
            self.cache[ctx] = d
        return d

    def logprob(self, ids, context=()):
        """log P of a symbol sequence given what came before (default: nothing known)"""
        ctx, total = tuple(context), 0.0
        for s in ids:
            total += float(self.dist(ctx)[s])
            ctx = (ctx + (s,))[-(self.order - 1):] if self.order > 1 else ()
        return total

    def save(self, path):
        np.savez(path, order=self.order, **{f"{n}{k}": v for k, tab in self.tabs.items() for n, v in zip(("k", "p", "c", "b"), tab)})

    @classmethod
    def load(cls, path):
        z = np.load(path)
        order = int(z["order"])
        return cls(order, {k: tuple(z[f"{n}{k}"] for n in ("k", "p", "c", "b")) for k in range(1, order + 1)})


class MixLM:
    """a mixture of LMs: P = sum_i w_i P_i"""
    def __init__(self, lms, weights):
        self.lms, self.lw = lms, [math.log(w) for w in weights]
        self.order = max(l.order for l in lms)
        self.cache = {}

    def dist(self, context):
        ctx = tuple(context[-(self.order - 1):])
        d = self.cache.get(ctx)
        if d is None:
            d = np.logaddexp.reduce([lw + lm.dist(ctx) for lw, lm in zip(self.lw, self.lms)], axis=0).astype(np.float32)
            if len(self.cache) > 200_000:
                self.cache.clear()
            self.cache[ctx] = d
        return d

    logprob = CharLM.logprob


def text_ids(path, punct=False):
    return encode((l.rstrip("\n") for l in open(path, encoding="utf-8")), punct)


def perplexity(lm, lines, punct=False):
    """per-symbol perplexity of text lines, each line scored from nothing known (as a manuscript line is read)"""
    tot = n = 0
    for line in lines:
        ids = [symbol(ch, punct) for ch in line.replace(" ", "")]
        ids = [s for i, s in enumerate(ids) if s or (i and ids[i - 1])]
        if ids:
            tot += lm.logprob(ids)
            n += len(ids)
    return math.exp(-tot / max(1, n)), n


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        argv = sys.argv[2:]
        mc, punct = 2, False
        if "--punct" in argv:
            punct = True
            argv.remove("--punct")
        if "--min-count" in argv:
            i = argv.index("--min-count")
            mc = int(argv[i + 1])
            del argv[i:i + 2]
        args = argv
        order, out, texts = int(args[0]), args[1], args[2:]
        ids = np.concatenate([text_ids(t, punct) for t in texts])
        lm = CharLM.build(ids, order, mc)
        lm.save(out)
        print(f"  saved {out} ({Path(out).stat().st_size / 1e6:.0f} MB)")
    elif cmd == "perplexity":
        lm = CharLM.load(sys.argv[2])
        pp, n = perplexity(lm, (l.rstrip("\n") for l in open(sys.argv[3], encoding="utf-8")))
        print(f"  perplexity {pp:.2f} per symbol over {n:,} symbols")
