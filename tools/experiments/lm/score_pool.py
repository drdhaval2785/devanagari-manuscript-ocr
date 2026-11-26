"""Pool runs scored on the pool's held-out lines (the last corrected page of each manuscript, the first 120 validation
lines of a transcribed one), each line read on its own, greedily and with a language model.

    .venv/bin/python tools/experiments/lm/score_pool.py pool1 pool2 [--lm data/lm/20261126] [--seg seg_torch]
"""
import argparse
import sys
import warnings
from pathlib import Path

import torch
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from msocr import lm as L                                    # noqa: E402
from msocr.aksara import ger                                 # noqa: E402
from msocr.pagediff import model_path                        # noqa: E402
from msocr.recognizer import load_model, pool_sources        # noqa: E402


def greedy(logits, chars):
    out, prev = [], 0
    for v in logits.argmax(-1).tolist():
        if v != prev and v != 0:
            out.append(chars[v - 1])
        prev = v
    return "".join(out)


def cer(pairs):
    return 100.0 * sum(Levenshtein.distance(p, t) for p, t in pairs) / max(1, sum(len(t) for _, t in pairs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--lm")
    ap.add_argument("--seg", default="seg_torch")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    lm = L.load_default(a.lm)
    sources = [s for s in pool_sources(seg=a.seg, contributions=False) if s[3]]
    res = {}
    for m in a.models:
        net, chars, _ = load_model(model_path(m))
        for ms, kind, _, sc in sources:
            g, w = [], []
            with torch.no_grad():
                for x, truth, _ in sc:
                    lg = net(x[None, None])[0]
                    g.append((greedy(lg, chars), truth))
                    w.append((L.decode(lg.numpy(), chars, lm)[0], truth))
            res[(m, ms)] = (cer(g), ger(g), cer(w), ger(w), len(sc), sum(len(t) for _, t in g))
    print(f"language model: {lm.parts[0][0].path.parent if hasattr(lm.parts[0][0], 'path') else a.lm or 'newest'}")
    head = "manuscript    lines chars | " + " | ".join(f"{m:^29s}" for m in a.models)
    print(head)
    print(" " * 26 + "| " + " | ".join("CER greedy  +LM | GER greedy  +LM" for _ in a.models))
    means = {m: [0, 0, 0, 0] for m in a.models}
    for ms, *_ in sources:
        n, c = res[(a.models[0], ms)][4:6]
        cells = []
        for m in a.models:
            r = res[(m, ms)]
            cells.append(f"{r[0]:6.2f} {r[2]:6.2f} | {r[1]:6.2f} {r[3]:6.2f}")
            for k in range(4):
                means[m][k] += r[[0, 2, 1, 3][k]] / len(sources)
        print(f"{ms:12s} {n:5d} {c:5d} | " + " | ".join(cells))
    print(f"{'mean':12s} {'':11s} | " + " | ".join(f"{means[m][0]:6.2f} {means[m][1]:6.2f} | {means[m][2]:6.2f} {means[m][3]:6.2f}" for m in a.models))


if __name__ == "__main__":
    main()
