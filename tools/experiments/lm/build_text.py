"""Training text for a character language model, in Devanagari without spaces (what a manuscript line looks like):
one sentence per line, letters only (a sentence ends at anything else).

    python tools/experiments/lm/build_text.py pramana <sentences.jsonl> <out.txt>     the spacer's Pramāṇa sentences (SLP1)
    python tools/experiments/lm/build_text.py lexica <folder of IAST .txt> <out.txt>   the lexica texts (IAST)
"""
import json
import re
import sys
from pathlib import Path

V = dict(zip("aAiIuUfFxXeEoO", "अआइईउऊऋॠऌॡएऐओऔ"))
MATRA = dict(zip("aAiIuUfFxXeEoO", ["", "ा", "ि", "ी", "ु", "ू", "ृ", "ॄ", "ॢ", "ॣ", "े", "ै", "ो", "ौ"]))
C = dict(zip("kKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzshL", "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ"))
OTHER = {"M": "ं", "H": "ः", "'": "ऽ", "~": "ँ"}


def slp_to_deva(s):
    out, i, n = [], 0, len(s)
    while i < n:
        ch = s[i]
        if ch in C:
            out.append(C[ch])
            if i + 1 < n and s[i + 1] in MATRA:
                out.append(MATRA[s[i + 1]])
                i += 2
            else:
                out.append("्")
                i += 1
        elif ch in V:
            out.append(V[ch])
            i += 1
        elif ch in OTHER:
            out.append(OTHER[ch])
            i += 1
        else:
            i += 1          # space or stray
    return "".join(out)


_TWO = {"ai": "E", "au": "O", "kh": "K", "gh": "G", "ch": "C", "jh": "J", "ṭh": "W", "ḍh": "Q", "th": "T", "dh": "D", "ph": "P", "bh": "B"}
_ONE = {"a": "a", "ā": "A", "i": "i", "ī": "I", "u": "u", "ū": "U", "ṛ": "f", "ṝ": "F", "ḷ": "x", "ḹ": "X", "e": "e", "o": "o",
        "ṃ": "M", "ṁ": "M", "ḥ": "H", "k": "k", "g": "g", "ṅ": "N", "c": "c", "j": "j", "ñ": "Y", "ṭ": "w", "ḍ": "q", "ṇ": "R",
        "t": "t", "d": "d", "n": "n", "p": "p", "b": "b", "m": "m", "y": "y", "r": "r", "l": "l", "v": "v", "ś": "S", "ṣ": "z",
        "s": "s", "h": "h", "ḻ": "L", "ḻ": "L", "'": "'", "~": "~", "’": "'"}
LETTERS = "".join(_ONE) + "".join(set("".join(_TWO)))


def iast_to_slp(s):
    out, i, n = [], 0, len(s)
    while i < n:
        two = s[i:i + 2]
        if two in _TWO:
            out.append(_TWO[two]); i += 2
        elif s[i] in _ONE:
            out.append(_ONE[s[i]]); i += 1
        else:
            return None
    return "".join(out)


def main():
    kind, src, out = sys.argv[1:4]
    n = chars = 0
    with open(out, "w", encoding="utf-8") as f:
        if kind == "pramana":
            for line in open(src, encoding="utf-8"):
                d = json.loads(line)
                t = slp_to_deva(d["slp"].replace(" ", ""))
                if len(t) >= 4:
                    f.write(t + "\n"); n += 1; chars += len(t)
        elif kind == "lexica-punct":
            # daṇḍas and verse numbers kept: '/' -> ।, '//' -> ॥, a run of digits -> #; a sentence runs to a line break
            for p in sorted(Path(src).glob("*.txt")):
                for raw in p.read_text(encoding="utf-8").lower().splitlines():
                    raw = raw.replace("//", " ॥ ").replace("/", " । ").replace("|", " । ")
                    raw = re.sub(r"[0-9०-९]+", " # ", raw)
                    out_line = []
                    for run in re.split(r"[^" + re.escape(LETTERS) + r" ।॥#]+", raw):
                        for w in run.split():
                            if w in ("।", "॥", "#"):
                                out_line.append(w)
                            else:
                                slp = iast_to_slp(w)
                                if slp:
                                    out_line.append(slp_to_deva(slp))
                    t = "".join(out_line)
                    if sum(ch not in "।॥#" for ch in t) >= 4:
                        f.write(t + "\n"); n += 1; chars += len(t)
        else:
            for p in sorted(Path(src).glob("*.txt")):
                text = p.read_text(encoding="utf-8").lower()
                for run in re.split(r"[^" + re.escape(LETTERS) + r" ]+", text):
                    slp = iast_to_slp(run.replace(" ", ""))
                    if slp is None:
                        # a run with a letter outside IAST: keep its clean pieces
                        pieces = [iast_to_slp(w) for w in run.split()]
                        slp = "".join(w for w in pieces if w)
                    t = slp_to_deva(slp)
                    if len(t) >= 4:
                        f.write(t + "\n"); n += 1; chars += len(t)
    print(f"{out}: {n} sentences, {chars} characters")


if __name__ == "__main__":
    main()
