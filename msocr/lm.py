"""The language model: how Sanskrit runs, letter by letter, weighed against the recogniser's doubt when a line is read.

A character n-gram (6-gram, interpolated absolute discounting) over the letters of the Devanagari block; daṇḍas,
digits, marks and spaces are all one boundary symbol, which the model neither reads nor judges. Two models are built
and mixed when a line is decoded: `lexica`, from the lexica texts (the IAST files in ../sanskrit-lexica-for-annotation:
Abhidhānacintāmaṇi and its commentaries, Amarakośa commentaries, Anekārthasaṃgraha...), and `own`, from the manuscripts'
own transcriptions (every training line the recogniser learns from: the transcribed manuscript's training split, the
corrected pages but the last of each manuscript, the contributed lines). The reading is then a prefix beam search over
the recogniser's probabilities: log P(recogniser) + alpha * log P(LM) + beta * letters (pipeline.json, "lm"). Measured
2026-11-25 (tools/experiments/lm): CER 2.81 -> 2.40 on pool1's held-out lines, 5.28 -> 4.34 on ft6's; the lexica texts
weigh far more than general literature, the own text a third again; a neural LM did no better.

The same model and the same search run in Manuscribe (app/js/recoglm.js): the tables are sorted integer keys, read by a
binary search, probabilities in thousandths of a nat, in a file both read (lexica.bin.gz, own.bin.gz: msocr.export_onnx
puts them in the pack). Lines are decoded one by one, from nothing known: a manuscript line starts mid-sentence.

    python -m msocr.lm build [--lexica DIR] [--order 6] [--prune 3] [--out data/lm/<today>]
    python -m msocr.lm show [DIR]                         what a build holds
    python -m msocr.lm odds अमृत असृत                      which reading the model prefers, and by how much

The newest build under data/lm/ is what predict, pagediff, recognizer test and export_onnx use (--lm DIR for another,
--no-lm for none).
"""
import argparse
import datetime
import gzip
import json
import math
import os
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

from .config import CORPUS, DATA, SEG
from .params import P

BOUND = "|"
SYMBOLS = [BOUND] + [chr(c) for c in range(0x900, 0x980) if chr(c) not in "०१२३४५६७८९।॥॰ॐ"]
A = len(SYMBOLS)
SYM = {c: i for i, c in enumerate(SYMBOLS)}
BITS = 7                                   # bits per symbol in a packed key (A < 128); 7 symbols fit a double exactly
LOG_A = math.log(A)
NEG = -1e30
MAGIC = b"MSLM"
LEXICA = Path(os.environ.get("MSOCR_LEXICA") or CORPUS.parent / "sanskrit-lexica-for-annotation")
LM_DIR = DATA / "lm"


def symbol(ch):
    """the LM symbol of a character: a letter itself, anything else (daṇḍa, digit, mark, space) the boundary"""
    return SYM.get(ch, 0)


# ---------------------------------------------------------------- text
def encode(lines):
    """text lines -> one array of symbol ids, every line bounded, boundaries never doubled"""
    out = bytearray()
    for line in lines:
        if not out or out[-1] != 0:
            out.append(0)
        for ch in line:
            s = SYM.get(ch, 0)
            if s or out[-1] != 0:
                out.append(s)
    if out and out[-1] != 0:
        out.append(0)
    return np.frombuffer(bytes(out), dtype=np.uint8)


_V = dict(zip("aAiIuUfFxXeEoO", "अआइईउऊऋॠऌॡएऐओऔ"))
_MATRA = dict(zip("aAiIuUfFxXeEoO", ["", "ा", "ि", "ी", "ु", "ू", "ृ", "ॄ", "ॢ", "ॣ", "े", "ै", "ो", "ौ"]))
_C = dict(zip("kKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzshL", "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ"))
_OTHER = {"M": "ं", "H": "ः", "'": "ऽ", "~": "ँ"}
_TWO = {"ai": "E", "au": "O", "kh": "K", "gh": "G", "ch": "C", "jh": "J", "ṭh": "W", "ḍh": "Q", "th": "T", "dh": "D", "ph": "P", "bh": "B"}
_ONE = {"a": "a", "ā": "A", "i": "i", "ī": "I", "u": "u", "ū": "U", "ṛ": "f", "ṝ": "F", "ḷ": "x", "ḹ": "X", "e": "e", "o": "o",
        "ṃ": "M", "ṁ": "M", "ḥ": "H", "k": "k", "g": "g", "ṅ": "N", "c": "c", "j": "j", "ñ": "Y", "ṭ": "w", "ḍ": "q", "ṇ": "R",
        "t": "t", "d": "d", "n": "n", "p": "p", "b": "b", "m": "m", "y": "y", "r": "r", "l": "l", "v": "v", "ś": "S", "ṣ": "z",
        "s": "s", "h": "h", "ḻ": "L", "'": "'", "~": "~", "’": "'"}
_LETTERS = "".join(_ONE) + "".join(set("".join(_TWO)))


def slp_to_deva(s):
    out, i, n = [], 0, len(s)
    while i < n:
        ch = s[i]
        if ch in _C:
            out.append(_C[ch])
            if i + 1 < n and s[i + 1] in _MATRA:
                out.append(_MATRA[s[i + 1]])
                i += 2
            else:
                out.append("्")
                i += 1
        elif ch in _V:
            out.append(_V[ch])
            i += 1
        elif ch in _OTHER:
            out.append(_OTHER[ch])
            i += 1
        else:
            i += 1
    return "".join(out)


def iast_to_slp(s):
    out, i, n = [], 0, len(s)
    while i < n:
        two = s[i:i + 2]
        if two in _TWO:
            out.append(_TWO[two])
            i += 2
        elif s[i] in _ONE:
            out.append(_ONE[s[i]])
            i += 1
        else:
            return None
    return "".join(out)


def lexica_lines(folder=LEXICA):
    """the lexica texts (IAST, one work a file) as Devanagari sentences of letters only, spaces removed -> (lines, files)

    Each word is transliterated on its own and the words joined afterwards, in SLP1, where every letter is one
    character: 'hāra iti' gives हारइति, as a manuscript writes it, and 'vāg iti' gives वागिति. Taking the spaces out of
    the IAST first made the vowels of a hiatus one of IAST's two-letter vowels ('hāraiti': हारैति; 'sa uvāca': सौवाच), so
    that until 28 November 2026 the model never saw a word in -a before one in i- or u- (हारइति read as हारिति)."""
    lines, files = [], []
    for p in sorted(Path(folder).glob("*.txt")):
        files.append(p.name)
        text = p.read_text(encoding="utf-8").lower()
        for run in re.split(r"[^" + re.escape(_LETTERS) + r" ]+", text):
            slp = "".join(w for w in (iast_to_slp(w) for w in run.split()) if w)
            t = slp_to_deva(slp)
            if len(t) >= 4:
                lines.append(t)
    return lines, files


def own_lines(holdout=1, seg=SEG):
    """every training line's text, as recognizer.pool draws them: the transcribed manuscript's training split, the
    corrected pages but the last `holdout` of each manuscript (those the pool scores on), the contributed lines
    -> (lines, [{manuscript, kind, lines}])"""
    from . import pagetext
    lines, sources = [], []
    for d in sorted(DATA.iterdir()):
        if not d.is_dir():
            continue
        ldir = d / ("lines" if seg == "seg" else f"lines_{seg}")
        train = sorted((ldir / "train").glob("*.gt.txt"))
        if train:
            ls = [p.read_text(encoding="utf-8").strip() for p in train]
            ls = [l for l in ls if l]
            lines += ls
            sources.append({"manuscript": d.name, "kind": "transcribed", "lines": len(ls)})
            continue
        pages = sorted((d / "corrected").glob("*.txt"))
        if not pages:
            continue
        taken = ldir / "corrected" / "taken.txt"
        if seg != "seg" and taken.exists():
            only = set(taken.read_text().split())
            pages = [p for p in pages if p.stem in only]
        ls = []
        for p in pages[:-holdout] if holdout else pages:
            ls += [t.strip().replace(" ", "") for t in pagetext.flat(pagetext.read(p)) if t.strip()]
        if ls:
            lines += ls
            sources.append({"manuscript": d.name, "kind": "corrected", "lines": len(ls)})
    try:
        from .contributions import allowed
        by = {}
        for l in allowed():
            by.setdefault(l["manuscript"], []).append(l["text"].replace(" ", ""))
        for ms, ls in sorted(by.items()):
            lines += ls
            sources.append({"manuscript": ms, "kind": "contributed", "lines": len(ls)})
    except Exception:
        pass
    return lines, sources


# ---------------------------------------------------------------- counting and estimation
def _mask(k):
    return np.uint64((1 << (BITS * k)) - 1) if k else np.uint64(0)


def ngram_counts(ids, k, chunk=30_000_000):
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


def _logp_vec(tabs, k, keys):
    """log P of packed (context, symbol) keys of k symbols, by backing off (float tables, at build time)"""
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
        out[~found] = np.where(has, cbow[j], 0.0) + _logp_vec(tabs, k - 1, rest & _mask(k - 1))
    return out


def build(ids, order, prune, verbose=True):
    """-> {k: (keys, log p, context keys, log backoff)}: raw counts at every order (a line starts with no history, and
    its first letters are scored by the lower orders themselves), absolute discounting with the discount of each order
    from its counts of counts, interpolated with the order below; at orders 4 and up the n-grams seen fewer than `prune`
    times are dropped, their share given to the backoff, so that every distribution still sums to one; a context
    without surviving n-grams has nothing to say (backoff 1) and is dropped too."""
    t = time.time()
    raw = {k: ngram_counts(ids, k) for k in range(1, order + 1)}
    if verbose:
        print(f"  counted {order}-grams of {len(ids):,} symbols in {time.time() - t:.0f}s: " + ", ".join(f"{k}: {len(raw[k][0]):,}" for k in raw))
    tabs = {}
    for k in range(1, order + 1):
        keys, c = raw[k]
        n1, n2 = int((c == 1).sum()), int((c == 2).sum())
        D = n1 / (n1 + 2 * n2) if n1 and n2 else 0.5
        ctx = keys >> np.uint64(BITS)
        uctx, start, nw = np.unique(ctx, return_index=True, return_counts=True)
        ctot = np.add.reduceat(c, start).astype(np.float64)
        keep = c >= (prune if k >= 4 else 1)
        pruned = np.add.reduceat(np.where(keep, 0.0, c - D), start)
        bow = (D * nw + pruned) / ctot
        at = np.searchsorted(uctx, ctx[keep])
        share = (c[keep] - D) / ctot[at]
        lower = _logp_vec(tabs, k - 1, keys[keep] & _mask(k - 1)) if k > 1 else np.full(int(keep.sum()), -LOG_A)
        p = share + bow[at] * np.exp(lower)
        alive = np.zeros(len(uctx), dtype=bool)
        alive[at] = True
        tabs[k] = (keys[keep], np.log(p), uctx[alive], np.log(bow[alive]))
        if verbose:
            print(f"  order {k}: {int(keep.sum()):,} n-grams kept of {len(keys):,}, {int(alive.sum()):,} contexts, discount {D:.2f}")
    return tabs


# ---------------------------------------------------------------- the file
def _to_bytes(keys, nbytes):
    b = np.zeros((len(keys), nbytes), dtype=np.uint8)
    for i in range(nbytes):
        b[:, i] = ((keys >> np.uint64(8 * i)) & np.uint64(0xFF)).astype(np.uint8)
    return b.tobytes()


def _from_bytes(buf, n, nbytes):
    b = np.frombuffer(buf, dtype=np.uint8, count=n * nbytes).reshape(n, nbytes)
    out = np.zeros(n, dtype=np.uint64)
    for i in range(nbytes):
        out |= b[:, i].astype(np.uint64) << np.uint64(8 * i)
    return out


def _milli(x):
    """-log p in thousandths of a nat, 16 bits (0 ... 65.535 nats)"""
    return np.minimum(65535, np.round(-x * 1000.0)).astype("<u2")


def save(path, order, tabs, meta):
    """lexica.bin.gz: MSLM, version, a JSON header, then for every order its sorted keys (little-endian, as few bytes
    as its symbols need), -log p in thousandths of a nat (uint16), the context keys and their backoff weights alike"""
    orders, body = [], []
    for k in range(1, order + 1):
        keys, lp, ck, bow = tabs[k]
        kb, cb = (BITS * k + 7) // 8, max(1, (BITS * (k - 1) + 7) // 8)
        orders.append({"k": k, "entries": int(len(keys)), "contexts": int(len(ck)), "key_bytes": kb, "context_bytes": cb})
        body += [_to_bytes(keys, kb), _milli(lp).tobytes(), _to_bytes(ck, cb), _milli(bow).tobytes()]
    header = json.dumps(dict({"format": "manuscribe-lm", "version": 1, "order": order, "bits": BITS, "symbols": SYMBOLS,
                              "unit": "thousandths of a nat, negated", "orders": orders}, **meta), ensure_ascii=False).encode("utf-8")
    raw = MAGIC + struct.pack("<II", 1, len(header)) + header + b"".join(body)
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(raw)
    return len(raw)


class LM:
    """one model read from its file: logp(context, symbol)"""

    def __init__(self, raw, name=""):
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        if raw[:4] != MAGIC:
            raise ValueError("not a language model file")
        _, hlen = struct.unpack("<II", raw[4:12])
        self.header = h = json.loads(raw[12:12 + hlen].decode("utf-8"))
        if h["symbols"] != SYMBOLS:
            raise ValueError("the model's symbols are not this code's")
        self.order, self.name = h["order"], name or h.get("name", "")
        self.tabs, pos = {}, 12 + hlen
        for o in h["orders"]:
            n, m, kb, cb = o["entries"], o["contexts"], o["key_bytes"], o["context_bytes"]
            keys = _from_bytes(raw[pos:pos + n * kb], n, kb)
            pos += n * kb
            nlp = np.frombuffer(raw[pos:pos + 2 * n], dtype="<u2").astype(np.int64)
            pos += 2 * n
            ck = _from_bytes(raw[pos:pos + m * cb], m, cb)
            pos += m * cb
            nbow = np.frombuffer(raw[pos:pos + 2 * m], dtype="<u2").astype(np.int64)
            pos += 2 * m
            self.tabs[o["k"]] = (keys, nlp, ck, nbow)
        self.memo = {}

    @classmethod
    def load(cls, path):
        return cls(Path(path).read_bytes(), Path(path).name.split(".")[0])

    def logp(self, ctx, s):
        """log P(symbol s | the symbols before it), ctx a tuple of symbol ids, most recent last, any length"""
        n = min(self.order - 1, len(ctx))
        h = 0
        for x in ctx[len(ctx) - n:]:
            h = (h << BITS) | x
        key = ((h << BITS) | s) * 8 + n            # n too: (0, s1) and (s1,) pack to the same integer
        v = self.memo.get(key)
        if v is not None:
            return v
        acc = 0.0
        for k in range(n + 1, 0, -1):
            hk = h & ((1 << (BITS * (k - 1))) - 1)
            keys, nlp, ck, nbow = self.tabs[k]
            kk = (hk << BITS) | s
            i = int(np.searchsorted(keys, np.uint64(kk)))
            if i < len(keys) and int(keys[i]) == kk:
                v = acc - int(nlp[i]) / 1000
                break
            j = int(np.searchsorted(ck, np.uint64(hk)))
            if j < len(ck) and int(ck[j]) == hk:
                acc -= int(nbow[j]) / 1000
        else:
            v = acc - LOG_A
        if len(self.memo) > 500_000:
            self.memo.clear()
        self.memo[key] = v
        return v


class Mix:
    """the models mixed: P = sum of weight_i * P_i"""

    def __init__(self, parts):
        self.parts = [(lm, float(w)) for lm, w in parts]
        tot = sum(w for _, w in self.parts)
        self.parts = [(lm, w / tot) for lm, w in self.parts]
        self.order = max(lm.order for lm, _ in self.parts)
        self.memo = {}

    def logp(self, ctx, s):
        if len(self.parts) == 1:
            return self.parts[0][0].logp(ctx, s)
        n = min(self.order - 1, len(ctx))
        h = 0
        for x in ctx[len(ctx) - n:]:
            h = (h << BITS) | x
        key = ((h << BITS) | s) * 8 + n
        v = self.memo.get(key)
        if v is None:
            v = math.log(sum(w * math.exp(lm.logp(ctx, s)) for lm, w in self.parts))
            if len(self.memo) > 500_000:
                self.memo.clear()
            self.memo[key] = v
        return v


def latest(root=LM_DIR):
    """the newest build's folder, or None"""
    builds = sorted(p for p in Path(root).glob("*") if (p / "lm.json").exists()) if Path(root).exists() else []
    return builds[-1] if builds else None


def load_dir(folder):
    """a build's models mixed as its lm.json weighs them -> Mix"""
    spec = json.loads((Path(folder) / "lm.json").read_text(encoding="utf-8"))
    return Mix([(LM.load(Path(folder) / c["file"]), c["weight"]) for c in spec["components"]])


def load_manifest(pack_dir, manifest):
    """a pack's language model, from its manifest's lm.components -> Mix, or None where it has none"""
    comps = (manifest.get("lm") or {}).get("components")
    return Mix([(LM.load(Path(pack_dir) / c["file"]), c["weight"]) for c in comps]) if comps else None


def load_default(choice=None):
    """the mixture predict and friends read with: --lm DIR, or the newest build; None for 'none' or no build"""
    if choice in ("none", "no", "off"):
        return None
    folder = Path(choice) if choice else latest()
    return load_dir(folder) if folder else None


# ---------------------------------------------------------------- decoding
def _lse(a, b):
    if a < b:
        a, b = b, a
    return a if b <= NEG else a + math.log1p(math.exp(b - a))


def log_softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def beam_search(lp, chars, mix, prm):
    """prefix beam search over log-softmax rows (T x C, class 0 blank) -> the best prefix, a tuple of classes.
    A hypothesis scores log P(recogniser) + alpha * log P(LM) + beta * its length; at every column only the classes
    with log probability over `prune` are tried; a boundary after a boundary costs the LM log(boundary_repeat), the
    model having no view of a run of daṇḍas and digits."""
    alpha, beta, width, prune = prm["alpha"], prm["beta"], prm["beam"], prm["prune"]
    rep = math.log(prm["boundary_repeat"])
    T, C = lp.shape
    sym = [0] + [symbol(ch) for ch in chars]
    n_ctx = mix.order - 1
    beams = {(): [0.0, NEG]}
    state = {(): (0.0, ())}
    rows = lp.tolist()
    for t in range(T):
        row = rows[t]
        cands = [c for c in range(1, C) if row[c] > prune]
        nxt = {}
        for prefix, (pb, pnb) in beams.items():
            ptot = _lse(pb, pnb)
            e = nxt.get(prefix)
            if e is None:
                e = nxt[prefix] = [NEG, NEG]
            e[0] = _lse(e[0], ptot + row[0])
            last = prefix[-1] if prefix else -1
            for c in cands:
                new = prefix + (c,)
                nn = nxt.get(new)
                if nn is None:
                    nn = nxt[new] = [NEG, NEG]
                if c == last:
                    e[1] = _lse(e[1], pnb + row[c])
                    nn[1] = _lse(nn[1], pb + row[c])
                else:
                    nn[1] = _lse(nn[1], ptot + row[c])
                if new not in state:
                    total, ctx = state[prefix]
                    s = sym[c]
                    if s == 0:
                        lpl = rep if ctx and ctx[-1] == 0 else mix.logp(ctx, 0)
                        nctx = (0,)
                    else:
                        lpl = mix.logp(ctx, s)
                        nctx = (ctx + (s,))[-n_ctx:]
                    state[new] = (total + lpl, nctx)
        scored = sorted(nxt.items(), key=lambda kv: -(_lse(kv[1][0], kv[1][1]) + alpha * state[kv[0]][0] + beta * len(kv[0])))
        beams = dict(scored[:width])
        state = {p: state[p] for p in beams}
    best = max(beams.items(), key=lambda kv: _lse(kv[1][0], kv[1][1]) + alpha * state[kv[0]][0] + beta * len(kv[0]))[0]
    return best


def align(lp, classes):
    """the best CTC alignment of a reading on the log-softmax rows -> each letter's probability (the highest over the
    columns it is aligned to); the recogniser's own view of a letter the search chose"""
    L = len(classes)
    if not L:
        return []
    T = lp.shape[0]
    ext = [0]
    for c in classes:
        ext += [c, 0]
    S = len(ext)
    dp = np.full((T, S), NEG)
    back = np.zeros((T, S), dtype=np.int32)
    dp[0, 0], dp[0, 1] = lp[0, 0], lp[0, ext[1]]
    for t in range(1, T):
        prev = dp[t - 1]
        for s in range(S):
            best, arg = prev[s], s
            if s >= 1 and prev[s - 1] > best:
                best, arg = prev[s - 1], s - 1
            if s >= 2 and ext[s] != 0 and ext[s] != ext[s - 2] and prev[s - 2] > best:
                best, arg = prev[s - 2], s - 2
            dp[t, s] = best + lp[t, ext[s]]
            back[t, s] = arg
    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    probs = [0.0] * L
    for t in range(T - 1, -1, -1):
        if ext[s] != 0:
            i = (s - 1) // 2
            probs[i] = max(probs[i], math.exp(lp[t, ext[s]]))
        s = back[t, s]
    return probs


def decode(logits, chars, mix, prm=None):
    """a line's logits (T x classes) read with the LM -> (text, confidence in %, each letter's probability)"""
    prm = prm or P["lm"]
    lp = log_softmax(logits)
    best = beam_search(lp, chars, mix, prm)
    probs = align(lp, best)
    text = "".join(chars[c - 1] for c in best)
    return text, 100 * (sum(probs) / len(probs) if probs else 0.0), probs


# ---------------------------------------------------------------- building a pack's models
def build_pack(out=None, lexica=LEXICA, order=None, prune=None, verbose=True):
    """-> the folder data/lm/<today>/ with lexica.bin.gz, own.bin.gz and lm.json"""
    prm = P["lm"]
    order, prune = order or prm["order"], prune or prm["min_count"]
    if out is None:
        day = datetime.date.today().strftime("%Y%m%d")
        out, k = LM_DIR / day, 0
        while out.exists():
            k += 1
            out = LM_DIR / f"{day}{chr(ord('a') + k)}"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    components = []
    lex, files = lexica_lines(lexica)
    own, sources = own_lines()
    for name, lines, about in (("lexica", lex, {"folder": str(lexica), "files": len(files)}), ("own", own, {"sources": sources})):
        ids = encode(lines)
        if verbose:
            print(f"[lm] {name}: {len(lines):,} lines, {len(ids):,} symbols")
        tabs = build(ids, order, prune if name == "lexica" else 1, verbose)
        meta = dict({"name": name, "built": datetime.datetime.now().isoformat(timespec="seconds"), "lines": len(lines), "symbols_counted": int(len(ids)),
                     "pruned_below": prune if name == "lexica" else 1}, **about)
        raw = save(out / f"{name}.bin.gz", order, tabs, meta)
        components.append({"name": name, "file": f"{name}.bin.gz", "weight": prm["weights"][name], "order": order, "pruned_below": meta["pruned_below"],
                           "entries": int(sum(len(tabs[k][0]) for k in tabs)), "lines": len(lines), "letters": int(len(ids)), "bytes": os.path.getsize(out / f"{name}.bin.gz"), "raw_bytes": raw,
                           **about})
        if verbose:
            print(f"  {name}.bin.gz: {components[-1]['entries']:,} entries, {os.path.getsize(out / f'{name}.bin.gz') / 1e6:.1f} MB ({raw / 1e6:.0f} MB read)")
    spec = {"format": "manuscribe-lm-pack", "version": 1, "built": datetime.datetime.now().isoformat(timespec="seconds"), "pipeline_version": P["version"],
            "order": order, "components": components, "decoding": {k: prm[k] for k in ("alpha", "beta", "beam", "prune", "boundary_repeat")}}
    (out / "lm.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    if verbose:
        print(f"  -> {out}")
    return out


def show(folder):
    spec = json.loads((Path(folder) / "lm.json").read_text(encoding="utf-8"))
    print(f"[lm] {folder}: built {spec['built']}, {spec['order']}-gram, decoding {spec['decoding']}")
    for c in spec["components"]:
        print(f"  {c['name']:<7} weight {c['weight']}  {c['entries']:>10,} entries  {c['lines']:>9,} lines  {c['letters']:>11,} letters  {c['bytes'] / 1e6:5.1f} MB")
        for s in c.get("sources", []):
            print(f"          {s['manuscript']:<14} {s['kind']:<12} {s['lines']:>6} lines")


def odds(mix, a, b):
    ids = lambda t: [symbol(ch) for ch in t]   # noqa: E731
    def score(t):
        ctx, tot = (), 0.0
        for s in ids(t):
            tot += mix.logp(ctx, s)
            ctx = (ctx + (s,))[-(mix.order - 1):]
        return tot
    la, lb = score(a), score(b)
    x, y = (a, b) if la >= lb else (b, a)
    print(f"  {a}: {la:.2f}   {b}: {lb:.2f}   -> {x} is {math.exp(abs(la - lb)):,.0f} times as probable as {y}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "show", "odds"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--lexica", default=str(LEXICA))
    ap.add_argument("--order", type=int)
    ap.add_argument("--prune", type=int)
    ap.add_argument("--out")
    ap.add_argument("--lm", help="a build's folder (default: the newest under data/lm)")
    a = ap.parse_args()
    if a.command == "build":
        build_pack(a.out, Path(a.lexica), a.order, a.prune)
    elif a.command == "show":
        show(a.args[0] if a.args else latest())
    elif a.command == "odds":
        odds(load_default(a.lm), a.args[0], a.args[1])
