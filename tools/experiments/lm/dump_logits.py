"""Every held-out line read by a recogniser, with the model's log-probabilities kept (for decoding experiments).

    .venv/bin/python tools/experiments/lm/dump_logits.py --model data/pool/models/pool1/best.pt --out <dir>/pool1

Writes <out>.jsonl (one line per line image: name, ms, page, set, truth, greedy reading) and <out>.npz (name ->
log-softmax matrix, frames x classes, float16). Sets: LDII_85885 val (first 120 lines, as pool scores) and test
(19 pages), the corrected pages of the pool manuscripts as recut from the U-Net lines (every page; which of them a
model trained on is decided when scoring, from the model's own record)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from msocr.recognizer import load_model, load_split, load_corrected, collate, batches, decode   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

net, chars, ck = load_model(a.model)
sets = []
for split in ("val", "test"):
    ls = load_split("LDII_85885", split, seg="seg_torch")
    if split == "val":
        ls = ls[:120]
    sets += [("LDII_85885", split, s) for s in ls]
for ms in ("LDII_1806", "LDII_19634", "LDII_45364", "LDII_82668", "LDII_9851"):
    pages, problems = load_corrected(ms, "seg_torch")
    for pr in problems:
        print("  !", ms, pr)
    for page in sorted(pages):
        sets += [(ms, "corrected", s) for s in pages[page]]
print(f"{len(sets)} lines; model {a.model} ({len(chars)} chars)")

samples = [s for _, _, s in sets]
meta = [(ms, st, s[2], s[1]) for ms, st, s in sets]
rows, mats = [], {}
with torch.no_grad():
    for idx in batches(samples, 16):
        x, lens = collate(samples, idx)
        lp = torch.log_softmax(net(x), -1)
        for k, i in enumerate(idx):
            m = lp[k, :lens[k]]
            ms, st, name, truth = meta[i]
            rows.append({"name": name, "ms": ms, "page": name.rsplit("_", 1)[0], "set": st, "truth": truth, "greedy": decode(m, chars)})
            mats[name] = m.numpy().astype(np.float16)
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
with open(str(out) + ".jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
np.savez_compressed(str(out) + ".npz", chars=np.array(chars), **mats)
print(f"wrote {out}.jsonl / .npz")
