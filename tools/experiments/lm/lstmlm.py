"""A small character LSTM language model over the same symbols as charlm.py, trained on the Mac GPU (MPS), to compare
with the n-gram LM: perplexity, and decoding.

    .venv/bin/python tools/experiments/lm/lstmlm.py train <out.pt> <text.txt> [<text.txt>...] [--steps 20000] [--hidden 256]
    .venv/bin/python tools/experiments/lm/lstmlm.py perplexity <model.pt> <text.txt>
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from charlm import A, SYM, symbol, text_ids   # noqa: E402

SEQ = 256


class LSTMLM(nn.Module):
    def __init__(self, hidden=256, layers=2, emb=64, n_sym=A):
        super().__init__()
        self.emb = nn.Embedding(n_sym, emb)
        self.rnn = nn.LSTM(emb, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, n_sym)
        self.hidden, self.layers = hidden, layers

    def forward(self, x, state=None):
        y, state = self.rnn(self.emb(x), state)
        return self.out(y), state


def train(out, texts, steps=20000, hidden=256, batch=128, lr=2e-3):
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ids = np.concatenate([text_ids(t) for t in texts])
    n_val = min(500_000, len(ids) // 100)
    tr, va = ids[:-n_val], ids[-n_val:]
    print(f"  {len(tr):,} training symbols, {n_val:,} validation; device {dev}")
    net = LSTMLM(hidden).to(dev)
    print(f"  parameters: {sum(p.numel() for p in net.parameters()):,}")
    opt = torch.optim.Adam(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(0)
    tr_t = torch.from_numpy(tr.astype(np.int64))
    best, t0 = float("inf"), time.time()
    for step in range(1, steps + 1):
        net.train()
        starts = rng.integers(0, len(tr) - SEQ - 1, batch)
        x = torch.stack([tr_t[s:s + SEQ] for s in starts]).to(dev)
        y = torch.stack([tr_t[s + 1:s + SEQ + 1] for s in starts]).to(dev)
        logits, _ = net(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, A), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps:
            vl = evaluate(net, va, dev)
            flag = ""
            if vl < best:
                best = vl
                torch.save({"state": net.state_dict(), "hidden": hidden, "layers": net.layers, "step": step, "val_loss": vl}, out)
                flag = " saved"
            print(f"  step {step:>6}  train loss {loss.item():.3f}  val loss {vl:.3f}  perplexity {math.exp(vl):.2f}  {(time.time() - t0) / 60:.0f} min{flag}", flush=True)


def evaluate(net, va, dev, batch=64):
    net.eval()
    tot = n = 0
    with torch.no_grad():
        windows = [(s, s + SEQ) for s in range(0, len(va) - SEQ - 1, SEQ)]
        for i in range(0, len(windows), batch):
            w = windows[i:i + batch]
            x = torch.stack([torch.from_numpy(va[s:e].astype(np.int64)) for s, e in w]).to(dev)
            y = torch.stack([torch.from_numpy(va[s + 1:e + 1].astype(np.int64)) for s, e in w]).to(dev)
            logits, _ = net(x)
            tot += nn.functional.cross_entropy(logits.reshape(-1, A), y.reshape(-1), reduction="sum").item()
            n += y.numel()
    return tot / n


def load(path, dev="cpu"):
    ck = torch.load(path, map_location="cpu")
    net = LSTMLM(ck["hidden"], ck["layers"], n_sym=ck["state"]["out.weight"].shape[0])   # symbols known when it was trained
    net.load_state_dict(ck["state"])
    net.eval()
    return net.to(dev)


def perplexity(net, lines, dev="cpu"):
    """each line from nothing known, as charlm.perplexity"""
    tot = n = 0
    with torch.no_grad():
        for line in lines:
            ids = [symbol(ch) for ch in line.replace(" ", "")]
            ids = [s for i, s in enumerate(ids) if s or (i and ids[i - 1])]
            if len(ids) < 2:
                continue
            x = torch.tensor([ids[:-1]], device=dev)
            y = torch.tensor([ids[1:]], device=dev)
            logits, _ = net(x)
            # the first symbol: from the unigram of the model's output with an empty history is not defined; score
            # it with the output after a boundary symbol
            first, _ = net(torch.tensor([[0]], device=dev))
            tot += nn.functional.cross_entropy(logits[0], y[0], reduction="sum").item()
            tot += nn.functional.cross_entropy(first[0], torch.tensor([ids[0]], device=dev), reduction="sum").item()
            n += len(ids)
    return math.exp(tot / max(1, n)), n


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        argv = sys.argv[2:]
        kw = {}
        for flag, key, typ in (("--steps", "steps", int), ("--hidden", "hidden", int)):
            if flag in argv:
                i = argv.index(flag)
                kw[key] = typ(argv[i + 1])
                del argv[i:i + 2]
        train(argv[0], argv[1:], **kw)
    elif cmd == "perplexity":
        net = load(sys.argv[2])
        pp, n = perplexity(net, (l.rstrip("\n") for l in open(sys.argv[3], encoding="utf-8")))
        print(f"  perplexity {pp:.2f} per symbol over {n:,} symbols")


class LSTMDist:
    """the LSTM as decode.py sees an LM: dist(context) over the whole history, hidden states cached per prefix"""
    order = 10 ** 9

    def __init__(self, path):
        torch.set_num_threads(1)
        self.net = load(path)
        self.cache = {}

    def reset(self):
        self.cache = {}

    def dist(self, context):
        ctx = tuple(context)
        hit = self.cache.get(ctx)
        if hit is not None:
            return hit[0]
        with torch.no_grad():
            if not ctx:
                logits, state = self.net(torch.tensor([[0]]))       # nothing known: as after a boundary
            else:
                parent = self.cache.get(ctx[:-1])
                if parent is None:
                    self.dist(ctx[:-1])
                    parent = self.cache[ctx[:-1]]
                logits, state = self.net(torch.tensor([[ctx[-1]]]), parent[1])
            d = torch.log_softmax(logits[0, -1], -1).numpy().astype(np.float32)
            if len(d) < A:                                       # symbols added since it was trained: impossible to it
                d = np.concatenate([d, np.full(A - len(d), -1e30, dtype=np.float32)])
        if len(self.cache) > 50_000:
            self.cache = {}
        self.cache[ctx] = (d, state)
        return d
