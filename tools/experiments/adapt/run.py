"""Quick adaptation to a new hand, page by page: can the pages corrected so far mend the reading of the next one in
seconds, without training the recogniser as a whole?

The setting imitated is Manuscribe's: the user reads a page with the shipped models, corrects it, and the next page
should come out better. The manuscript here is one the model has never seen (pool3b never trained on RTJ_5520; pool4
did, after hours of pooled training, and is the yardstick). Pages are taken in order; page k is read with what pages
1..k-1 taught, then counted as corrected. Two cheap kinds of adaptation, each of which Manuscribe could do in the
browser:

  A  page LM     a third character 6-gram, from the corrected lines so far, mixed into the language model's two
                 (lexica, own) with weight w; the recogniser is not touched
  C  head        the recogniser's last layer only (384 -> classes, 34,650 weights) trained for a few steps on the
                 corrected lines with the CTC loss, the 384 features of every column kept fixed (computed once, as the
                 page was read), pulled back toward its shipped weights by lam * |W - W0|^2; begun afresh from the
                 shipped weights after every page, on all lines corrected so far

    tune   --split val  [--pages 15]      the settings of C (lr, lam, steps) and of A (w) on the validation pages
    run    --split test --lr .. --lam .. --steps .. --w ..   every arm on the test pages, pool4 as yardstick

Only numbers are written (results/), never the manuscript's text.
"""
import argparse
import collections
import itertools
import json
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from msocr import lm as L                      # noqa: E402
from msocr import recognizer as R              # noqa: E402
from msocr.params import P                     # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
TMP = Path(tempfile.mkdtemp(prefix="adapt_"))


# ---------------------------------------------------------------- data and features
def pages_of(ms, split, limit=None):
    """[(page, [(image tensor, text, stem)])] in page order"""
    by = collections.OrderedDict()
    for x, t, stem in R.load_split(ms, split, seg="seg_torch"):
        by.setdefault(stem.rsplit("_", 1)[0], []).append((x, t.strip(), stem))
    return list(by.items())[:limit]


@torch.no_grad()
def features(net, x):
    """the 384 numbers of every column, before the last layer (T x 384)"""
    f = net.cnn(x[None, None])
    b, c, h, w = f.shape
    y, _ = net.rnn(f.permute(0, 3, 1, 2).reshape(b, w, c * h))
    return y[0]


def prepare(net, chars, pages):
    """every line's features and its target classes"""
    idx = {c: i + 1 for i, c in enumerate(chars)}
    out = []
    for pid, lines in pages:
        rows = []
        for x, t, stem in lines:
            h = features(net, x)
            rows.append({"h": h, "text": t, "target": [idx[c] for c in t if c in idx]})
        out.append((pid, rows))
    # the features give the model's own output through its last layer
    h = out[0][1][0]["h"]
    with torch.no_grad():
        full = net(pages[0][1][0][0][None, None])[0]
        assert torch.allclose(net.out(h), full, atol=1e-4)
    return out


# ---------------------------------------------------------------- C: the last layer
def adapt(W0, b0, rows, lr, lam, steps, at):
    """Adam on the last layer, full batch over `rows`; yields (step, W, b) at the steps in `at`"""
    rows = [r for r in rows if r["target"]]
    N, Tm = len(rows), max(r["h"].shape[0] for r in rows)
    H = torch.zeros(N, Tm, W0.shape[1])
    for k, r in enumerate(rows):
        H[k, :r["h"].shape[0]] = r["h"]
    in_len = torch.tensor([r["h"].shape[0] for r in rows])
    tgt_len = torch.tensor([len(r["target"]) for r in rows])
    tgt = torch.tensor([c for r in rows for c in r["target"]])
    n_chars = float(tgt_len.sum())
    W, b = W0.clone().requires_grad_(), b0.clone().requires_grad_()
    opt = torch.optim.Adam([W, b], lr=lr)
    for s in range(1, steps + 1):
        lp = (H @ W.T + b).log_softmax(-1).transpose(0, 1)
        loss = F.ctc_loss(lp, tgt, in_len, tgt_len, blank=0, reduction="sum", zero_infinity=True) / n_chars
        loss = loss + lam * ((W - W0).pow(2).sum() + (b - b0).pow(2).sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s in at:
            yield s, W.detach().clone(), b.detach().clone()


# ---------------------------------------------------------------- A: the page LM
def page_lm(texts):
    tabs = L.build(L.encode(texts), 6, 1, verbose=False)
    path = TMP / "page.bin.gz"
    L.save(path, 6, tabs, {"name": "page"})
    return L.LM.load(path)


def with_page(base, page, w):
    return L.Mix([(lm, wt * (1 - w)) for lm, wt in base.parts] + [(page, w)])


# ---------------------------------------------------------------- reading and scoring
def greedy(logits, chars):
    return R.decode(logits, chars)


def beam(logits, chars, mix):
    best = L.beam_search(L.log_softmax(logits.numpy()), chars, mix, P["lm"])
    return "".join(chars[c - 1] for c in best)


class Score:
    """edits and characters per arm and page"""

    def __init__(self):
        self.e = collections.defaultdict(lambda: collections.defaultdict(int))
        self.n = collections.defaultdict(lambda: collections.defaultdict(int))

    def add(self, arm, page, pred, truth):
        self.e[arm][page] += Levenshtein.distance(pred, truth)
        self.n[arm][page] += len(truth)

    def cer(self, arm, pages=None):
        ps = [p for p in self.e[arm] if pages is None or p in pages]
        return 100.0 * sum(self.e[arm][p] for p in ps) / max(1, sum(self.n[arm][p] for p in ps))


def load(path):
    torch.set_num_threads(2)
    net, chars, _ = R.load_model(path)
    return net, chars


# ---------------------------------------------------------------- tune
def tune(a):
    net, chars = load(a.model)
    data = prepare(net, chars, pages_of(a.ms, a.split, a.pages))
    W0, b0 = net.out.weight.detach().clone(), net.out.bias.detach().clone()
    later = {pid for pid, _ in data[1:]}
    at = set(a.at)
    sc, secs = Score(), collections.defaultdict(float)
    base_mix = L.load_default() if a.lm_weights else None
    for k, (pid, rows) in enumerate(data):
        seen = [r for _, rs in data[:k] for r in rs]
        for r in rows:
            sc.add("base", pid, greedy(net.out(r["h"]).detach(), chars), r["text"])
        if seen:
            for lr, lam in itertools.product(a.lrs, a.lams):
                t = time.time()
                for s, W, b in adapt(W0, b0, seen, lr, lam, max(at), at):
                    secs[(lr, lam)] += time.time() - t
                    for r in rows:
                        sc.add(f"C lr={lr} lam={lam} steps={s}", pid, greedy(r["h"] @ W.T + b, chars), r["text"])
                    t = time.time()
        if base_mix is not None:
            for r in rows:
                sc.add("base+LM", pid, beam(net.out(r["h"]).detach(), chars, base_mix), r["text"])
            if seen:
                plm = page_lm([r["text"] for r in seen])
                for w in a.lm_weights:
                    mix = with_page(base_mix, plm, w)
                    for r in rows:
                        sc.add(f"A w={w}", pid, beam(net.out(r["h"]).detach(), chars, mix), r["text"])
        print(f"  page {k + 1}/{len(data)} {pid}: base {sc.cer('base', {pid}):.2f}", flush=True)
    arms = sorted(sc.e, key=lambda arm: sc.cer(arm, later))
    print(f"\n{a.ms} {a.split}, {len(data)} pages, scored on pages 2-{len(data)} ({sum(len(r) for _, r in data[1:])} lines)")
    for arm in arms:
        print(f"  {sc.cer(arm, later):6.2f}  {arm}")
    OUT.mkdir(exist_ok=True)
    (OUT / f"tune_{a.ms}_{a.split}{a.tag}.json").write_text(json.dumps(
        {"model": a.model, "pages": len(data), "cer_pages_2_on": {arm: round(sc.cer(arm, later), 3) for arm in arms},
         "seconds_adapting_total": {f"lr={k[0]} lam={k[1]}": round(v, 1) for k, v in secs.items()},
         "per_page": {arm: {p: round(sc.cer(arm, {p}), 2) for p in sc.e[arm]} for arm in arms}}, indent=1))


# ---------------------------------------------------------------- run
def run(a):
    net, chars = load(a.model)
    pages = pages_of(a.ms, a.split, a.pages)
    data = prepare(net, chars, pages)
    W0, b0 = net.out.weight.detach().clone(), net.out.bias.detach().clone()
    ref, ref_chars = load(a.yardstick)
    base_mix = L.load_default()
    sc, log = Score(), []
    for k, (pid, rows) in enumerate(data):
        seen = [r for _, rs in data[:k] for r in rs]
        t = time.time()
        if seen:
            _, W, b = next(adapt(W0, b0, seen, a.lr, a.lam, a.steps, {a.steps}))
        else:
            W, b = W0, b0
        t_head = time.time() - t
        t = time.time()
        mix = with_page(base_mix, page_lm([r["text"] for r in seen]), a.w) if seen else base_mix
        t_lm = time.time() - t
        for (x, _, _), r in zip(pages[k][1], rows):
            lb, la = net.out(r["h"]).detach(), (r["h"] @ W.T + b).detach()
            with torch.no_grad():
                lref = ref(x[None, None])[0]
            for arm, pred in (("base", greedy(lb, chars)), ("base+LM", beam(lb, chars, base_mix)),
                              ("A: page LM", beam(lb, chars, mix)),
                              ("C: head", greedy(la, chars)), ("C: head +LM", beam(la, chars, base_mix)),
                              ("A+C", beam(la, chars, mix)),
                              ("pool4", greedy(lref, ref_chars)), ("pool4+LM", beam(lref, ref_chars, base_mix))):
                sc.add(arm, pid, pred, r["text"])
        log.append({"page": pid, "lines": len(rows), "corrected_lines_before": len(seen),
                    "seconds_head": round(t_head, 2), "seconds_page_lm": round(t_lm, 2),
                    **{arm: round(sc.cer(arm, {pid}), 2) for arm in sc.e}})
        print(f"  page {k + 1}/{len(data)} {pid}: " + "  ".join(f"{arm} {sc.cer(arm, {pid}):.2f}" for arm in sc.e), flush=True)
    later = {pid for pid, _ in data[1:]}
    arms = list(sc.e)
    summary = {"model": a.model, "yardstick": a.yardstick, "ms": a.ms, "split": a.split, "pages": len(data),
               "settings": {"lr": a.lr, "lam": a.lam, "steps": a.steps, "w": a.w},
               "cer_pages_2_on": {arm: round(sc.cer(arm, later), 3) for arm in arms},
               "cer_all_pages": {arm: round(sc.cer(arm), 3) for arm in arms}, "per_page": log}
    print(f"\n{a.ms} {a.split}: pages 2-{len(data)}, {sum(len(r) for _, r in data[1:])} lines")
    for arm in arms:
        print(f"  {sc.cer(arm, later):6.2f}  {arm}")
    OUT.mkdir(exist_ok=True)
    (OUT / f"run_{a.ms}_{a.split}.json").write_text(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["tune", "run"])
    ap.add_argument("--ms", default="RTJ_5520")
    ap.add_argument("--split", default="val")
    ap.add_argument("--pages", type=int)
    ap.add_argument("--model", default="data/pool/models/pool3b/best.pt")
    ap.add_argument("--yardstick", default="data/pool/models/pool4/best.pt")
    ap.add_argument("--lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    ap.add_argument("--lams", type=float, nargs="+", default=[0, 1e-3, 1e-2, 1e-1])
    ap.add_argument("--at", type=int, nargs="+", default=[10, 25, 50, 100], help="the steps of C scored")
    ap.add_argument("--tag", default="", help="added to the results file name")
    ap.add_argument("--lm-weights", type=float, nargs="*", default=[])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--w", type=float, default=0.2)
    ap.add_argument("--lm", default="data/lm/20261127",
                    help="the language model build: 20261127, pool3b's, has no line of the RTJ manuscripts pool4 added "
                         "(the builds from 20261128 on have RTJ_5520's training split)")
    a = ap.parse_args()
    L.load_default = lambda choice=None, _f=a.lm: L.load_dir(_f)
    (tune if a.mode == "tune" else run)(a)


if __name__ == "__main__":
    main()
