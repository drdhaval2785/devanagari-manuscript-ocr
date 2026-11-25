"""Step 4: train our own line recogniser (plain PyTorch), reporting every pass.

How the model reads a line
  1. the line image is scaled to 48 pixels high (grey, ink = high values)
  2. a small convolutional network turns it into a row of feature columns, one per 4 pixels
  3. a two-layer bidirectional LSTM reads the columns left-to-right and right-to-left
  4. for every column the model outputs a probability for each character, or "blank"
  5. repeated characters and blanks are collapsed to give the text (CTC decoding)

How it is trained, in two stages
  stage 1 "pieces": each line is cut into 3-6 short pieces (see chunks.py). Short pieces
                    let a blank model discover which image columns go with which character.
  stage 2 "lines" : whole lines with their exact text. Starts automatically once stage 1
                    has brought the validation error under PIECES_UNTIL_CER.

Words used in the report
  pass : the model has seen every training line once
  loss : what training minimises (average per character); should fall
  CER  : character error rate on the validation pages = edits needed to turn the model's
         reading into the transcription, per 100 characters. Always measured on WHOLE lines
         of pages the model never trains on. Lower is better.

Files written to data/<ms>/models/<run>/ : train.log, history.csv, best.pt, last.pt

Which lines: --seg seg (the default) trains on the lines align paired from Kraken's segmentation, cut by Kraken's
extract_polygons (lines/<split>/, as run1-run3 were); --seg seg_torch on those it paired from our U-Net's, cut by our
own cutter exactly as predict and Manuscribe cut them (lines_seg_torch/<split>/, from align --seg=seg_torch), so that
the recogniser learns from the line images it will read. finetune and pool take --seg too: a corrected manuscript's
pages are then those `recut` cut again from that segmentation (lines_<seg>/corrected/, only pages whose rows match their
corrected lines in number and in reading; corrected/ and lines/pred/ untouched), and a manuscript without lines of that
cut is left out rather than mixed in.

    recut LDII_19634 --seg seg_torch     the corrected pages cut again from the U-Net's lines, for finetune/pool --seg

Three ways to train
  train     LDII_85885 --run run1      a blank model on one transcribed manuscript (stages 1 and 2)
  finetune  LDII_19634 --run ft1       a trained model adapted to one new hand from its corrected pages
  pool      --run pool1                a trained model continued on EVERY manuscript that has training
                                       text at once, each manuscript given the same number of lines
                                       per pass however many it has (see pool below): the road to a
                                       model that reads hands it has never seen; with the lines
                                       contributed from Manuscribe that passed their checks
                                       (msocr.contributions, msocr.linequality), unless --no-contributions
"""
import argparse
import collections
import csv
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from rapidfuzz.distance import Levenshtein

from .aksara import ger
from .chunks import split_text
from .config import ms_dir

from .params import P as _P
HEIGHT = _P["recogniser"]["height"]
STRIDE = _P["recogniser"]["stride"]         # image columns per output column
PIECES_UNTIL_CER = 30.0     # leave stage 1 once validation CER is under this ...
PIECES_MAX_PASSES = 25      # ... or after this many passes regardless


# ---------------------------------------------------------------- data
def lines_dir(ms_id, seg="seg"):
    return ms_dir(ms_id) / ("lines" if seg == "seg" else f"lines_{seg}")


def build_cache(ms_id, verbose=True, seg="seg"):
    """Cut every paired line out of its page once and store it, so training starts fast."""
    if seg != "seg":
        return _build_cache_own(ms_id, seg, verbose)
    from kraken.lib.segmentation import extract_polygons
    from kraken.lib.xml import XMLPage
    warnings.filterwarnings("ignore")
    d = ms_dir(ms_id)
    for split in ("train", "val", "test"):
        out = d / "lines" / split
        pages = (d / f"{split}.txt").read_text().split()
        if out.exists() and len(list(out.glob("*.png"))) > 0:
            if verbose:
                print(f"[lines] {split}: already cut ({len(list(out.glob('*.png')))} line images)")
            continue
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for xml in pages:
            seg = XMLPage(xml, filetype="page").to_container()
            im = Image.open(seg.imagename)
            stem = xml.rsplit("/", 1)[-1][:-4]
            for li, (crop, line) in enumerate(extract_polygons(im, seg), 1):
                p = out / f"{stem}_{li:02d}.png"
                crop.save(p)
                p.with_suffix(".gt.txt").write_text(line.text, encoding="utf-8")
                n += 1
        if verbose:
            print(f"[lines] {split}: cut {n} line images from {len(pages)} pages", flush=True)


def _build_cache_own(ms_id, seg, verbose=True):
    """The lines align paired from another segmentation (pagexml_<seg>/<page>.json: the rows as predict makes them),
    cut by our own cutter (linecut), as predict and Manuscribe cut them."""
    import json
    from .config import MANUSCRIPTS
    from .linecut import band_of, cut_line
    d = ms_dir(ms_id)
    for split in ("train", "val", "test"):
        out = lines_dir(ms_id, seg) / split
        pages = (d / f"{split}_{seg}.txt").read_text().split()
        if out.exists() and len(list(out.glob("*.png"))) > 0:
            if verbose:
                print(f"[lines] {split}: already cut ({len(list(out.glob('*.png')))} line images, {seg}, own cutter)")
            continue
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for xml in pages:
            stem = Path(xml).stem
            rows = json.loads(Path(xml).with_suffix(".json").read_text(encoding="utf-8"))
            page = np.asarray(Image.open(MANUSCRIPTS[ms_id]["images"] / f"{stem}.jpg").convert("RGB"))
            for li, r in enumerate(rows, 1):
                above, below = band_of(r["baseline"], r["outline"])
                p = out / f"{stem}_{li:02d}.png"
                Image.fromarray(cut_line(page, r["baseline"], above, below)).save(p)
                p.with_suffix(".gt.txt").write_text(r["text"], encoding="utf-8")
                n += 1
        if verbose:
            print(f"[lines] {split}: cut {n} line images from {len(pages)} pages ({seg}, own cutter)", flush=True)


def to_tensor(im):
    im = im.convert("L")
    w = max(STRIDE * 2, round(im.size[0] * HEIGHT / im.size[1]))
    im = im.resize((w, HEIGHT), Image.BILINEAR)
    return torch.from_numpy(1.0 - np.asarray(im, dtype=np.float32) / 255.0)


def load_split(ms_id, split, limit=None, seg="seg"):
    files = sorted((lines_dir(ms_id, seg) / split).glob("*.png"))
    if limit:
        files = files[:limit]
    return [(to_tensor(Image.open(f)), f.with_suffix(".gt.txt").read_text(encoding="utf-8"), f.stem) for f in files]


def make_pieces(lines, rng):
    out = []
    for x, text, name in lines:
        texts, cuts = split_text(text, rng.choice((3, 4, 5, 6)))
        xs = [0] + [round(c * x.shape[1]) for c in cuts] + [x.shape[1]]
        for k, t in enumerate(texts):
            if t and xs[k + 1] - xs[k] >= STRIDE * (len(t) + 2):
                out.append((x[:, xs[k]:xs[k + 1]], t, f"{name}_{k}"))
    return out


def batches(samples, size, rng=None):
    idx = list(range(len(samples)))
    if rng:
        rng.shuffle(idx)
        groups = [sorted(idx[i:i + size * 20], key=lambda j: samples[j][0].shape[1]) for i in range(0, len(idx), size * 20)]
        out = [g[i:i + size] for g in groups for i in range(0, len(g), size)]
        rng.shuffle(out)
    else:
        idx.sort(key=lambda j: samples[j][0].shape[1])
        out = [idx[i:i + size] for i in range(0, len(idx), size)]
    return out


def augment(x):
    """Return a slightly altered copy of a line image. A different alteration every pass
    means the model never sees exactly the same picture twice, so it has to learn the
    letter shapes instead of memorising the training lines."""
    r = torch.rand(8)
    w = max(STRIDE * 2, int(x.shape[1] * (0.93 + 0.14 * r[0])))            # stretch or squeeze sideways
    x = nn.functional.interpolate(x[None, None], size=(HEIGHT, w), mode="bilinear", align_corners=False)[0, 0]
    shift = int(r[1] * 5) - 2                                               # move up or down by up to 2px
    if shift:
        x = torch.roll(x, shift, 0)
        if shift > 0:
            x[:shift] = 0
        else:
            x[shift:] = 0
    x = x.clamp(0, 1) ** (0.7 + 0.7 * r[2])                                 # fainter or heavier ink
    x = x * (0.85 + 0.25 * r[3])                                            # overall contrast
    if r[4] < 0.3:                                                          # slight blur
        x = 0.5 * x + 0.5 * nn.functional.avg_pool2d(x[None, None], 3, 1, 1)[0, 0]
    x = x + torch.randn_like(x) * 0.04 * r[5]                               # scanner noise
    return x.clamp(0, 1)


def collate(samples, idx, aug=False):
    xs = [augment(samples[i][0]) if aug else samples[i][0] for i in idx]
    w = max(x.shape[1] for x in xs)
    x = torch.zeros(len(idx), 1, HEIGHT, w)
    for k, t in enumerate(xs):
        x[k, 0, :, :t.shape[1]] = t
    return x, torch.tensor([t.shape[1] // STRIDE for t in xs])


# ---------------------------------------------------------------- model
class Recognizer(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        def block(i, o, pool):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(pool)]

        self.cnn = nn.Sequential(*block(1, 32, (2, 2)), *block(32, 64, (2, 2)),
                                 *block(64, 96, (2, 1)), *block(96, 128, (2, 1)))
        self.rnn = nn.LSTM(128 * (HEIGHT // 16), 192, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.out = nn.Linear(384, n_classes)

    def forward(self, x):
        f = self.cnn(x)
        b, c, h, w = f.shape
        y, _ = self.rnn(f.permute(0, 3, 1, 2).reshape(b, w, c * h))
        return self.out(y)


def decode(logits, chars):
    out, prev = [], 0
    for v in logits.argmax(-1).tolist():
        if v != prev and v != 0:
            out.append(chars[v - 1])
        prev = v
    return "".join(out)


def evaluate(net, samples, chars, size=16):
    net.eval()
    pairs = []
    with torch.no_grad():
        for idx in batches(samples, size):
            x, lens = collate(samples, idx)
            logits = net(x)
            for k, i in enumerate(idx):
                pairs.append((decode(logits[k, :lens[k]], chars), samples[i][1], samples[i][2]))
    errors = sum(Levenshtein.distance(p, t) for p, t, _ in pairs)
    return 100.0 * errors / max(1, sum(len(t) for _, t, _ in pairs)), pairs


def confusions(pairs, top=6):
    c = collections.Counter()
    for pred, truth, _ in pairs:
        for op, i, j in Levenshtein.editops(truth, pred):
            c[f"{truth[i]}→{pred[j]}" if op == "replace" else f"{truth[i]} missed" if op == "delete" else f"{pred[j]} invented"] += 1
    return ", ".join(f"{k} ×{v}" for k, v in c.most_common(top))


# ---------------------------------------------------------------- training
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        pass


def train(ms_id, run="run1", max_passes=80, lag=8, lrate=1e-3, batch_pieces=16, batch_lines=8,
          threads=6, limit=None, seed=0, augment_on=True, min_gain=0.1, seg="seg"):
    d = ms_dir(ms_id)
    run_dir = d / "models" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(sys.__stdout__, open(run_dir / "train.log", "a", encoding="utf-8"))
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    rng = random.Random(seed)

    print(f"\n[train] run '{run}' on {ms_id}   ({time.strftime('%Y-%m-%d %H:%M')})")
    build_cache(ms_id, seg=seg)
    print(f"  lines            : {'from Kraken segmentation, cut by Kraken (lines/)' if seg == 'seg' else f'from {seg}, cut by our own cutter as predict and Manuscribe cut them (lines_{seg}/)'}")
    t = time.time()
    train_lines, val_lines = load_split(ms_id, "train", limit, seg), load_split(ms_id, "val", limit and max(40, limit // 8), seg)
    chars = sorted(set("".join(s[1] for s in train_lines)))
    known = set(chars)
    c2i = {c: i + 1 for i, c in enumerate(chars)}
    unseen = sorted(set("".join(s[1] for s in val_lines)) - known)
    net = Recognizer(len(chars) + 1)
    opt = torch.optim.AdamW(net.parameters(), lrate, weight_decay=1e-4)
    ctc = nn.CTCLoss(zero_infinity=True)
    n_chars = sum(len(s[1]) for s in train_lines)

    print(f"  training lines   : {len(train_lines):,} ({n_chars:,} characters), loaded in {time.time() - t:.0f}s")
    print(f"  validation lines : {len(val_lines):,} (whole lines from pages never trained on)")
    print(f"  alphabet         : {len(chars)} distinct characters" + (f"; in validation only: {' '.join(unseen)}" if unseen else ""))
    print(f"  model            : {sum(p.numel() for p in net.parameters()) / 1e6:.1f} million weights, starts blank (random)")
    print(f"  line image       : {HEIGHT}px high, typically {int(np.median([s[0].shape[1] for s in train_lines]))}px wide "
          f"-> {int(np.median([s[0].shape[1] for s in train_lines])) // STRIDE} output columns for ~{n_chars // len(train_lines)} characters")
    print(f"  stage 1 (pieces) until validation CER < {PIECES_UNTIL_CER:.0f}% or {PIECES_MAX_PASSES} passes; then stage 2 (whole lines)")
    print(f"  stops when CER has not improved by at least {min_gain} for {lag} passes in stage 2, or after {max_passes} passes")
    print(f"  augmentation     : {'on (each pass sees slightly altered copies of the line images)' if augment_on else 'off'}")
    print(f"  learning rate {lrate}, halved after 3 passes without improvement\n")
    print("  pass | stage  | loss/char | val CER % | change  | best CER % (pass) | min  | verdict")
    print("  -----+--------+-----------+-----------+---------+-------------------+------+---------------------------")

    history, stage, best, best_ep, flat, t_start = [], "pieces", None, 0, 0, time.time()
    mark = None          # the CER a pass must beat by min_gain to count as real progress
    for ep in range(1, max_passes + 1):
        t_ep = time.time()
        samples = make_pieces(train_lines, rng) if stage == "pieces" else train_lines
        plan = batches(samples, batch_pieces if stage == "pieces" else batch_lines, rng)
        net.train()
        tot_loss = tot_chars = 0
        for bi, idx in enumerate(plan, 1):
            x, lens = collate(samples, idx, aug=augment_on)
            logp = net(x).log_softmax(-1)
            tg = torch.tensor([c2i[c] for i in idx for c in samples[i][1]])
            tl = torch.tensor([len(samples[i][1]) for i in idx])
            loss = nn.functional.ctc_loss(logp.permute(1, 0, 2), tg, lens, tl, reduction="sum", zero_infinity=True)
            opt.zero_grad()
            (loss / len(idx)).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot_loss += loss.item()
            tot_chars += int(tl.sum())
            if bi % max(1, len(plan) // 4) == 0 and bi < len(plan):
                print(f"         pass {ep}: {100 * bi // len(plan):3d}% of {len(samples):,} {stage}, loss/char so far "
                      f"{tot_loss / tot_chars:.3f}, {tot_chars / (time.time() - t_ep):,.0f} chars/s", flush=True)

        cer, pairs = evaluate(net, val_lines, chars)
        prev = history[-1]["val_cer"] if history else None
        improved = best is None or cer < best
        if mark is None or cer < mark - min_gain:
            mark, flat = cer, 0
        elif stage == "lines":           # stage 1 normally sits near 100% for a while; that is not a plateau
            flat += 1
        if improved:
            best, best_ep = cer, ep
            torch.save({"state": net.state_dict(), "chars": chars, "height": HEIGHT, "pass": ep, "val_cer": cer, "lines": seg}, run_dir / "best.pt")
        torch.save({"state": net.state_dict(), "chars": chars, "height": HEIGHT, "pass": ep, "val_cer": cer, "lines": seg}, run_dir / "last.pt")

        note = ""
        if stage == "pieces" and (cer < PIECES_UNTIL_CER or ep >= PIECES_MAX_PASSES):
            stage, flat, note = "lines", 0, "  -> switching to whole lines next pass"
        if prev is None:
            verdict = "first measurement"
        elif flat == 0:
            verdict = "improving"
        elif flat < 3:
            verdict = f"no real gain for {flat} pass(es)" + (" (new best, but by under %.1f)" % min_gain if improved else "")
        else:
            verdict = f"flat: no real gain for {flat} passes (stops at {lag})"
        if flat and flat % 3 == 0:
            for g in opt.param_groups:
                g["lr"] *= 0.5
            note += f"  -> learning rate halved to {opt.param_groups[0]['lr']:.1e}"
        stop = stage == "lines" and flat >= lag
        if stop:
            verdict = "PLATEAU: stopping"

        row = {"pass": ep, "stage": "pieces" if note.startswith("  -> switching") else stage, "loss_per_char": round(tot_loss / tot_chars, 4),
               "val_cer": round(cer, 3), "best_cer": round(best, 3), "minutes": round((time.time() - t_ep) / 60, 2),
               "total_minutes": round((time.time() - t_start) / 60, 1), "lr": opt.param_groups[0]["lr"], "verdict": verdict}
        history.append(row)
        change = "   -   " if prev is None else f"{cer - prev:+7.2f}"
        print(f"  {ep:4d} | {row['stage']:6s} | {row['loss_per_char']:9.3f} | {cer:9.2f} | {change} | "
              f"{best:9.2f} ({best_ep:3d})   | {row['minutes']:4.1f} | {verdict}{note}")
        for pred, truth, _ in pairs[:2]:
            print(f"         truth: {truth[:64]}")
            print(f"         model: {pred[:64] or '(nothing yet)'}")
        if cer < 60:
            print(f"         commonest mistakes: {confusions(pairs)}")
        with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerows(history)
        if stop:
            break

    print(f"\n[train] finished: {len(history)} passes in {history[-1]['total_minutes']} min. "
          f"Best validation CER {best:.2f}% at pass {best_ep}.")
    print(f"  best model : {run_dir / 'best.pt'}")
    print(f"  history    : {run_dir / 'history.csv'}")
    return history


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    net = Recognizer(len(ck["chars"]) + 1)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["chars"], ck


JUDGE = Path(__file__).resolve().parent.parent / "data" / "LDII_19634" / "models" / "ft4" / "best.pt"


def recut_corrected(ms_id, seg="seg_torch", judge=JUDGE, verbose=True):
    """A manuscript's corrected pages cut again from another segmentation's lines, into a set of their own
    (lines_<seg>/corrected/, the pages taken listed in taken.txt), as predict would cut them now; corrected/ and
    lines/pred/ are not touched. A page is taken only where every block has as many rows as its corrected text has
    lines, and the judge (a recogniser) reads every line as its text, within 60 % of its characters and within 35 % at
    the page's median (msocr.linequality's limits): a row that is not the line whose text it is paired with reads far
    from it. -> (pages taken, [(page, why not)])"""
    import shutil
    import statistics
    import tempfile
    from . import pagetext
    from .linequality import LIMITS, cer
    from .predict import read_page
    from .segment import ensure_segmented, reading_from
    d = ms_dir(ms_id)
    out = lines_dir(ms_id, seg) / "corrected"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    net, chars, _ = load_model(judge)
    pages = sorted((d / "corrected").glob("*.txt"))
    ensure_segmented(ms_id, [t.stem + ".jpg" for t in pages], seg)
    taken, refused = [], []
    for txt in pages:
        want = {b: [t.strip().replace(" ", "") for t in ls] for b, ls in pagetext.read(txt).items()}
        for b in want:
            while want[b] and not want[b][-1]:
                want[b].pop()
        with tempfile.TemporaryDirectory() as t:
            with reading_from(seg):
                got, note, warn = read_page(ms_id, txt.stem + ".jpg", net, chars, Path(t))
            counts = {b: len(got.get(b, [])) for b in set(want) | set(got)}
            why = [f"{b}: {counts[b]} rows, {len(want.get(b, []))} corrected lines" for b in sorted(counts) if counts[b] != len(want.get(b, []))]
            if not why:
                diffs = [cer(r["text"], w) for b in want for r, w in zip(got[b], want[b]) if w]
                far = sum(x > LIMITS["reading_max"] for x in diffs)
                if far or (diffs and statistics.median(diffs) > LIMITS["package_reading_median_max"]):
                    why.append(f"rows not the lines of the text: {far} read more than 60 % away, median {100 * statistics.median(diffs):.0f} %")
            if why:
                refused.append((txt.stem, "; ".join(why)))
                continue
            for f in Path(t).glob("*.png"):
                shutil.copy(f, out / f.name)
            taken.append(txt.stem)
    (out / "taken.txt").write_text("".join(f"{x}\n" for x in taken), encoding="utf-8")
    if verbose:
        print(f"[recut] {ms_id}: {len(taken)} of {len(pages)} corrected pages cut from {seg} into {out}")
        for stem, why in refused:
            print(f"  - {stem}: {why}")
    return taken, refused


def load_corrected(ms_id, seg="seg"):
    """Corrected pages of a manuscript: data/<ms>/corrected/<page>.txt, one text line per
    manuscript line (blocks after ;box{...} markers, see pagetext.py), paired with the line
    images that `predict` cut for that page and block; with another seg, with those recut_corrected cut from its
    lines (only the pages it took)."""
    from . import pagetext
    d = ms_dir(ms_id)
    pred = d / "lines" / "pred" if seg == "seg" else lines_dir(ms_id, seg) / "corrected"
    only = None if seg == "seg" else set((pred / "taken.txt").read_text().split()) if (pred / "taken.txt").exists() else set()
    pages, problems = {}, []
    for txt in sorted((d / "corrected").glob("*.txt")):
        if only is not None and txt.stem not in only:
            continue
        boxes = pagetext.read(txt)
        names = sorted(f.name for f in pred.glob(f"{txt.stem}_*.png"))
        samples = []
        for box, texts in boxes.items():
            pat = pagetext.image_pattern(txt.stem, box)
            imgs = [n for n in names if pat.match(n)]
            texts = [t.strip().replace(" ", "") for t in texts]
            while len(texts) > len(imgs) and texts and not texts[-1]:     # a stray blank line at the end
                texts.pop()
            if len(texts) != len(imgs):
                problems.append(f"{txt.name}: {len(texts)} text lines but {len(imgs)} line images" + (f" in the {box} block" if box != "main" or len(boxes) > 1 else "") + " - skipped")
                samples = None
                break
            # a line left empty (nothing written there, an ornament) teaches nothing and is left out
            samples += [(to_tensor(Image.open(pred / n)), t, n[:-4]) for n, t in zip(imgs, texts) if t]
        if samples is not None:
            pages[txt.stem] = samples
    return pages, problems


def _extend_alphabet(net, chars, texts):
    """Give characters the model has never seen their own output slots. Returns (chars, added)."""
    added = sorted(set("".join(texts)) - set(chars))
    if added:
        old_out = net.out
        net.out = nn.Linear(old_out.in_features, len(chars) + len(added) + 1)
        with torch.no_grad():
            net.out.weight[:old_out.out_features] = old_out.weight
            net.out.bias[:old_out.out_features] = old_out.bias
            net.out.bias[old_out.out_features:] = old_out.bias.min()
        chars = chars + added
    return chars, added


def finetune(ms_id, base, run="ft1", passes=30, lrate=3e-4, holdout=1, mix=3, base_ms="LDII_85885", threads=6, seed=0, seg="seg"):
    """Adapt a trained model to a new hand from a few corrected pages.

    - starts from the weights of `base`; no warm-up on pieces is needed
    - the last `holdout` corrected page(s) are NOT trained on: they give the honest before/after score
    - every pass also includes `mix` old lines per new line, so the model keeps what it knew
    - the learning rate is halved after every 3 passes without gain, and training stops after 8
    - characters the base model has never seen get new output slots
    """
    d = ms_dir(ms_id)
    run_dir = d / "models" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(sys.__stdout__, open(run_dir / "train.log", "a", encoding="utf-8"))
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    rng = random.Random(seed)

    print(f"\n[finetune] run '{run}' on {ms_id}, starting from {base}   ({time.strftime('%Y-%m-%d %H:%M')})")
    print(f"  lines            : {'cut from Kraken segmentation' if seg == 'seg' else f'cut from {seg} by our own cutter (recut_corrected; lines_{seg}/)'}")
    pages, problems = load_corrected(ms_id, seg)
    for pr in problems:
        print(f"  ! {pr}")
    names = sorted(pages)
    if len(names) <= holdout:
        print(f"  need more than {holdout} corrected page(s) in {d / 'corrected'}; found {len(names)}")
        return None
    new_train = [s for n in names[:-holdout] for s in pages[n]]
    new_test = [s for n in names[-holdout:] for s in pages[n]]
    old_train = load_split(base_ms, "train", seg=seg)
    old_val = load_split(base_ms, "val", seg=seg)[:120]

    net, chars, ck = load_model(base)
    chars, added = _extend_alphabet(net, chars, [s[1] for s in new_train + new_test])
    c2i = {c: i + 1 for i, c in enumerate(chars)}
    opt = torch.optim.AdamW(net.parameters(), lrate, weight_decay=1e-4)

    print(f"  corrected pages  : {len(names)} -> training on {len(names) - holdout} ({len(new_train)} lines), "
          f"scoring on {', '.join(names[-holdout:])} ({len(new_test)} lines)")
    print(f"  old lines mixed in each pass : {mix} per new line, drawn from {len(old_train):,} lines of {base_ms}")
    print(f"  new characters   : {' '.join(added) if added else 'none'}")
    cer_new, pairs = evaluate(net, new_test, chars)
    ger_new = ger(pairs)
    cer_old, _ = evaluate(net, old_val, chars)
    print(f"  learning rate {lrate}, halved after 3 passes without gain; stops after 8 passes without gain or {passes} passes\n")
    print(f"  pass | loss/char | CER % on new hand | change  | CER % on old hand | verdict")
    print(f"  -----+-----------+-------------------+---------+-------------------+----------------")
    print(f"     0 |     -     | {cer_new:17.2f} |    -    | {cer_old:17.2f} | before fine-tuning")
    for pred, truth, _ in pairs[:2]:
        print(f"         truth: {truth[:64]}\n         model: {pred[:64]}")
    history = [{"pass": 0, "loss_per_char": "", "cer_new": round(cer_new, 3), "ger_new": round(ger_new, 3), "cer_old": round(cer_old, 3)}]
    best, best_ep, start, start_ger, best_ger = cer_new, 0, cer_new, ger_new, ger_new
    for ep in range(1, passes + 1):
        samples = new_train + rng.sample(old_train, min(len(old_train), mix * len(new_train)))
        net.train()
        tot_loss = tot_chars = 0
        for idx in batches(samples, 8, rng):
            x, lens = collate(samples, idx, aug=True)
            logp = net(x).log_softmax(-1)
            tg = torch.tensor([c2i[c] for i in idx for c in samples[i][1]])
            tl = torch.tensor([len(samples[i][1]) for i in idx])
            loss = nn.functional.ctc_loss(logp.permute(1, 0, 2), tg, lens, tl, reduction="sum", zero_infinity=True)
            opt.zero_grad()
            (loss / len(idx)).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot_loss += loss.item()
            tot_chars += int(tl.sum())
        prev = cer_new
        cer_new, pairs = evaluate(net, new_test, chars)
        ger_new = ger(pairs)
        cer_old, _ = evaluate(net, old_val, chars)
        if cer_new < best:
            best, best_ep, best_ger = cer_new, ep, ger_new
            torch.save({"state": net.state_dict(), "chars": chars, "height": HEIGHT, "pass": ep, "val_cer": cer_new, "val_ger": ger_new, "lines": seg}, run_dir / "best.pt")
        verdict = "new best" if best_ep == ep else f"no gain for {ep - best_ep} pass(es)"
        if best_ep != ep and (ep - best_ep) % 3 == 0:
            for g in opt.param_groups:
                g["lr"] *= 0.5
            verdict += f", learning rate halved to {opt.param_groups[0]['lr']:.1e}"
        print(f"  {ep:4d} | {tot_loss / tot_chars:9.3f} | {cer_new:17.2f} | {cer_new - prev:+7.2f} | {cer_old:17.2f} | {verdict}")
        if ep % 5 == 0 or ep == passes:
            for pred, truth, _ in pairs[:2]:
                print(f"         truth: {truth[:64]}\n         model: {pred[:64]}")
            print(f"         commonest mistakes: {confusions(pairs)}")
        history.append({"pass": ep, "loss_per_char": round(tot_loss / tot_chars, 4), "cer_new": round(cer_new, 3), "ger_new": round(ger_new, 3), "cer_old": round(cer_old, 3)})
        with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)
        if ep - best_ep >= 8:
            print("  no gain for 8 passes: stopping")
            break
    print(f"\n[finetune] CER on the held-out page of {ms_id}: {start:.2f}% before -> {best:.2f}% after (pass {best_ep}), "
          f"from {len(new_train)} corrected lines; GER (akṣaras) {start_ger:.2f}% -> {best_ger:.2f}%.")
    print(f"  model: {run_dir / 'best.pt'}" if best_ep else "  no improvement; no model saved")
    return history


CONTRIB_REPEAT = 3          # a contributed manuscript's share of a pass: at most this many draws of each of its lines
CONTRIB_HOLDOUT = 0.1       # ... and the share of its lines (whole folio sides, from the end) kept back for scoring,
CONTRIB_SCORED_FROM = 30    # once it has this many lines on two sides or more
OWN_GUARD = 0.5             # a pass that reads the developer's own manuscripts this much worse (CER points) than the
                            # model started with cannot become best.pt, whatever the contributed lines gain


def contributed_sources(exclude=()):
    """The lines contributed from Manuscribe that training may use (msocr.contributions.allowed: consent, approval and
    the line checks passed), a source per manuscript as its editor named it: (label, kind, train, score, packages)."""
    from . import contributions
    by_ms = {}
    for l in contributions.allowed():
        by_ms.setdefault(l["manuscript"], []).append(l)
    out = []
    for name, ls in sorted(by_ms.items()):
        label = f"contributed: {name}"
        sides = list(dict.fromkeys((l["id"], l["folio"]) for l in ls))
        held = set()
        if label in exclude or name in exclude:
            held = set(sides)
        elif len(ls) >= CONTRIB_SCORED_FROM and len(sides) >= 2:
            for side in reversed(sides[1:]):
                held.add(side)
                if sum(1 for l in ls if (l["id"], l["folio"]) in held) >= CONTRIB_HOLDOUT * len(ls):
                    break
        sample = lambda l: (to_tensor(Image.open(l["png"])), l["text"], f"{l['id']}/{Path(l['png']).stem}")
        tr = [sample(l) for l in ls if (l["id"], l["folio"]) not in held]
        sc = [sample(l) for l in ls if (l["id"], l["folio"]) in held]
        packages = {}
        for l in ls:
            p = packages.setdefault(l["id"], {"id": l["id"], "manuscript": name, "release": l["release"], "lines": 0, "trained": 0})
            p["lines"] += 1
            p["trained"] += (l["id"], l["folio"]) not in held
        out.append((label, "unseen" if held == set(sides) and (label in exclude or name in exclude) else "contributed", tr, sc, list(packages.values())))
    return out


def pool_sources(ms_ids=None, holdout=1, exclude=(), contributions=True, seg="seg"):
    """Every manuscript with training text, as (ms_id, kind, train lines, score lines).
    kind 'transcribed': a manuscript aligned from its TEI (lines/train, scored on lines/val);
    kind 'corrected'  : a manuscript with corrected/ pages (the last `holdout` pages are scored, the rest trained on);
    kind 'contributed': lines contributed from Manuscribe, by manuscript (contributed_sources);
    kind 'unseen'     : an excluded manuscript, scored on everything and trained on nothing."""
    from .config import DATA
    found = []
    for d in sorted(DATA.iterdir()):
        ms = d.name
        if not d.is_dir() or (ms_ids and ms not in ms_ids and ms not in exclude):
            continue
        if list((d / "lines" / "train").glob("*.png")):
            if not list((lines_dir(ms, seg) / "train").glob("*.png")):     # lines of one cut only: no mixing of the two
                print(f"  ! {ms}: no lines from {seg} (python -m msocr.align {ms} --seg={seg}); left out")
                continue
            tr, sc = load_split(ms, "train", seg=seg), load_split(ms, "val", seg=seg)[:120]
            found.append((ms, "transcribed", tr, sc))
        elif list((d / "corrected").glob("*.txt")):
            if seg != "seg" and not (lines_dir(ms, seg) / "corrected" / "taken.txt").exists():
                print(f"  ! {ms}: its corrected pages not cut from {seg} (python -m msocr.recognizer recut {ms} --seg {seg}); left out")
                continue
            pages, problems = load_corrected(ms, seg)
            for pr in problems:
                print(f"  ! {ms} {pr}")
            names = sorted(pages)
            if ms in exclude:
                found.append((ms, "unseen", [], [s for n in names for s in pages[n]]))
            elif len(names) > holdout:
                found.append((ms, "corrected", [s for n in names[:-holdout] for s in pages[n]], [s for n in names[-holdout:] for s in pages[n]]))
            else:
                print(f"  ! {ms}: only {len(names)} corrected page(s), need more than {holdout} to hold one out; skipped")
    if contributions:
        found += [f[:4] for f in contributed_sources(exclude)]
    for ms in exclude:
        if not any(f[0] == ms or f[0] == f"contributed: {ms}" for f in found):
            print(f"  ! --exclude {ms}: nothing to score it on")
    return found


def pool(run="pool1", base=None, passes=40, lrate=3e-4, per_pass=3000, holdout=1, ms_ids=None, exclude=(), threads=6, seed=0, lag=8,
         contributions=True, seg="seg"):
    """Continue a trained model on every manuscript that has training text, all at once.

    - every pass draws the same number of lines (per_pass / number of manuscripts) from each
      manuscript: a manuscript with fewer lines has them repeated, under different augmentation,
      a manuscript with more contributes a different subset each pass
    - whole pages are held out per manuscript (the last `holdout` corrected pages; the val split
      of a transcribed one) and scored after every pass; the mean over manuscripts picks best.pt
    - `exclude` names manuscripts left out of training entirely and scored as unseen hands: the
      number that says how generic the model is
    - characters the base model has never seen get new output slots
    - contributed lines (unless contributions=False) come only as msocr.contributions.allowed() gives them, each
      contributed manuscript a source drawn at most CONTRIB_REPEAT times per line a pass, scored on its last folio
      sides where it has enough; a pass whose mean CER over the developer's own manuscripts is more than OWN_GUARD
      points worse than before training cannot become best.pt, so that contributed lines cannot spoil what the
      model reads already; best.pt lists the packages it learned from ("contributions")
    Files written to data/pool/models/<run>/ : train.log, history.csv, best.pt, last.pt
    """
    run_dir = ms_dir("pool") / "models" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(sys.__stdout__, open(run_dir / "train.log", "a", encoding="utf-8"))
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    rng = random.Random(seed)
    base = Path(base or ms_dir("LDII_85885") / "models" / "run3" / "best.pt")

    print(f"\n[pool] run '{run}', starting from {base}   ({time.strftime('%Y-%m-%d %H:%M')})")
    t = time.time()
    print(f"  lines            : {'cut from Kraken segmentation' if seg == 'seg' else f'cut from {seg} by our own cutter'} (contributed lines: as Manuscribe cut them)")
    sources = pool_sources(ms_ids, holdout, exclude, contributions=False, seg=seg)
    packages = []
    if contributions:
        contrib = contributed_sources(exclude)
        sources += [f[:4] for f in contrib]
        packages = [dict(p, release=p["release"]) for f in contrib if f[1] == "contributed" for p in f[4]]
    training = [f for f in sources if f[2]]
    if len(training) < 2:
        print(f"  need at least two manuscripts with training text; found {len(training)}")
        return None
    if not any(f[3] for f in training):
        print("  no manuscript trained on has lines to score it on")
        return None
    quota = max(1, per_pass // len(training))
    quotas = {f[0]: min(quota, CONTRIB_REPEAT * len(f[2])) if f[1] == "contributed" else quota for f in training}
    net, chars, ck = load_model(base)
    chars, added = _extend_alphabet(net, chars, [s[1] for f in sources for s in f[2] + f[3]])
    c2i = {c: i + 1 for i, c in enumerate(chars)}
    opt = torch.optim.AdamW(net.parameters(), lrate, weight_decay=1e-4)

    short, k = [], 0
    for ms, kind, _, _ in sources:
        if ms.startswith("contributed: "):
            k += 1
            short.append(f"c{k}")
        else:
            short.append(ms.split("_")[1] if "_" in ms else ms[:6])
    print(f"  manuscripts      : {len(training)} trained on, {len(sources) - len(training)} scored as unseen; loaded in {time.time() - t:.0f}s")
    print(f"  {'manuscript':<12} {'kind':<11} {'train':>6} {'score':>6}  share of each pass")
    for (ms, kind, tr, sc), sh in zip(sources, short):
        q = quotas.get(ms, 0)
        how = ("-" if not tr else f"{q} lines" + (f", each line about {q / len(tr):.1f} times" if q > len(tr) else f" of {len(tr):,}, a different draw each pass"))
        print(f"  {(sh if ms.startswith('contributed: ') else ms):<12} {kind:<11} {len(tr):>6} {len(sc):>6}  {how}" + ("" if sc or not tr else "  (not scored: too few lines or sides)"))
    for (ms, kind, tr, sc), sh in zip(sources, short):
        if ms.startswith("contributed: "):
            print(f"  {sh} = {ms}")
    if packages:
        print(f"  contributed      : {len(packages)} package(s), {sum(p['trained'] for p in packages)} lines trained on "
              f"(data release{'s' if len({p['release'] for p in packages}) > 1 else ''} {', '.join(sorted({p['release'] or '-' for p in packages}))})")
    elif contributions:
        print("  contributed      : none usable (python -m msocr.contributions sync, then quality)")
    print(f"  lines per pass   : {sum(quotas.values()):,} ({quota} from each manuscript, a contributed one at most {CONTRIB_REPEAT} draws of each of its lines)")
    print(f"  new characters   : {' '.join(added) if added else 'none'}")
    print(f"  learning rate {lrate}, halved after 3 passes without gain; stops after {lag} passes without gain or {passes} passes\n")
    print(f"  CER % per manuscript on pages never trained on; mean over the manuscripts trained on picks the best model, "
          f"own = the mean over the developer's own (a pass more than {OWN_GUARD} worse there than before cannot be best)")
    print(f"  pass | loss/char | mean  | own   | " + " | ".join(f"{n:>6}" for n in short) + " | min  | verdict")
    print(f"  -----+-----------+-------+-------+-" + "-+-".join("-" * 6 for _ in short) + "-+------+---------------")

    gers = []          # GER per manuscript of the last score(), beside its CER

    def score():
        ev = [evaluate(net, sc, chars) if sc else (float("nan"), []) for _, _, _, sc in sources]
        cers = [c for c, _ in ev]
        gers[:] = [ger(pr) if pr else float("nan") for _, pr in ev]
        scored = [c for c, f in zip(cers, sources) if f[2] and f[3]]
        own = [c for c, f in zip(cers, sources) if f[2] and f[3] and f[1] != "contributed"]
        mean = sum(scored) / len(scored)
        return mean, (sum(own) / len(own) if own else mean), cers

    def mean_ger():
        g = [x for x, f in zip(gers, sources) if f[2] and f[3]]
        return sum(g) / len(g)

    def line(ep, loss, mean, own, cers, minutes, verdict):
        print(f"  {ep:4d} | {loss:>9} | {mean:5.2f} | {own:5.2f} | " + " | ".join(f"{c:6.2f}" for c in cers) + f" | {minutes:4.1f} | {verdict}")

    history = []
    mean, own, cers = score()
    own0 = own
    line(0, "-", mean, own, cers, (time.time() - t) / 60, "before")
    history.append({"pass": 0, "loss_per_char": "", "mean_cer": round(mean, 3), "own_cer": round(own, 3), **{f[0]: round(c, 3) for f, c in zip(sources, cers)},
                    "mean_ger": round(mean_ger(), 3), **{f"{f[0]} GER": round(g, 3) for f, g in zip(sources, gers)}, "minutes": 0})
    best, best_ep, flat = mean, 0, 0
    for ep in range(1, passes + 1):
        t_ep = time.time()
        samples = []
        for ms, _, tr, _ in training:
            q = quotas[ms]
            samples += rng.sample(tr, q) if len(tr) >= q else rng.choices(tr, k=q)
        net.train()
        tot_loss = tot_chars = 0
        for idx in batches(samples, 8, rng):
            x, lens = collate(samples, idx, aug=True)
            logp = net(x).log_softmax(-1)
            tg = torch.tensor([c2i[c] for i in idx for c in samples[i][1]])
            tl = torch.tensor([len(samples[i][1]) for i in idx])
            loss = nn.functional.ctc_loss(logp.permute(1, 0, 2), tg, lens, tl, reduction="sum", zero_infinity=True)
            opt.zero_grad()
            (loss / len(idx)).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot_loss += loss.item()
            tot_chars += int(tl.sum())
        mean, own, cers = score()
        state = {"state": net.state_dict(), "chars": chars, "height": HEIGHT, "pass": ep, "val_cer": mean, "val_ger": mean_ger(), "own_cer": own, "lines": seg,
                 "sources": [f[0] for f in training], "contributions": packages}
        if mean < best and own <= own0 + OWN_GUARD:
            best, best_ep, flat = mean, ep, 0
            torch.save(state, run_dir / "best.pt")
            verdict = "new best"
        else:
            flat += 1
            verdict = (f"held back: own manuscripts {own:.2f} against {own0:.2f} before" if mean < best else f"no gain for {flat} pass(es)")
            if flat % 3 == 0:
                for g in opt.param_groups:
                    g["lr"] *= 0.5
                verdict += f", learning rate now {opt.param_groups[0]['lr']:.1e}"
        torch.save(state, run_dir / "last.pt")
        line(ep, f"{tot_loss / tot_chars:.3f}", mean, own, cers, (time.time() - t_ep) / 60, verdict)
        history.append({"pass": ep, "loss_per_char": round(tot_loss / tot_chars, 4), "mean_cer": round(mean, 3), "own_cer": round(own, 3),
                        **{f[0]: round(c, 3) for f, c in zip(sources, cers)},
                        "mean_ger": round(mean_ger(), 3), **{f"{f[0]} GER": round(g, 3) for f, g in zip(sources, gers)},
                        "minutes": round((time.time() - t_ep) / 60, 2)})
        with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)
        if flat >= lag:
            print(f"  no gain for {lag} passes: stopping")
            break
    print(f"\n[pool] mean CER over the {len([f for f in training if f[3]])} manuscripts scored: {history[0]['mean_cer']:.2f}% before -> {best:.2f}% after (pass {best_ep})"
          + (f"; {len(packages)} contributed package(s) in it" if packages and best_ep else ""))
    print(f"  model: {run_dir / 'best.pt'}" if best_ep else "  no improvement; no model saved")
    if best_ep:         # CER and GER of each manuscript, before and with the model kept
        h0, hb = history[0], history[best_ep]
        print(f"  mean GER (akṣaras): {h0['mean_ger']:.2f}% before -> {hb['mean_ger']:.2f}% after")
        print(f"  {'manuscript':<22} {'CER before':>10} {'after':>7}   {'GER before':>10} {'after':>7}")
        for f in sources:
            print(f"  {f[0]:<22} {h0[f[0]]:>10.2f} {hb[f[0]]:>7.2f}   {h0[f[0] + ' GER']:>10.2f} {hb[f[0] + ' GER']:>7.2f}")
    return history


def test(ms_id, run="run1", show=5, model=None, seg="seg"):
    """Final score on the held-out test pages, which training never touched. `model` scores another model's file; seg
    the test lines of which segmentation (lines/ or lines_<seg>/)."""
    net, chars, ck = load_model(Path(model) if model else ms_dir(ms_id) / "models" / run / "best.pt")
    lines = load_split(ms_id, "test", seg=seg)
    cer, pairs = evaluate(net, lines, chars)
    per_line = sorted(100 * Levenshtein.distance(p, t) / len(t) for p, t, _ in pairs)
    print(f"[test] {ms_id} {'model ' + str(model) if model else 'run ' + repr(run)} (from pass {ck['pass']}, validation CER {ck['val_cer']:.2f}%)")
    print(f"  test lines : {len(lines)} from {len({s[2].rsplit(chr(95), 1)[0] for s in lines})} held-out pages ({seg})")
    print(f"  test CER   : {cer:.2f}%")
    print(f"  test GER   : {ger(pairs):.2f}%  (akṣaras, msocr/aksara.py)")
    print(f"  per line   : median {per_line[len(per_line) // 2]:.1f}%, best {per_line[0]:.1f}%, worst {per_line[-1]:.1f}%")
    print(f"  commonest mistakes: {confusions(pairs, 12)}")
    for pred, truth, name in pairs[:show]:
        print(f"  {name}\n    truth: {truth}\n    model: {pred}")
    return cer


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["train", "test", "finetune", "pool", "recut"])
    ap.add_argument("--base", default=str(ms_dir("LDII_85885") / "models" / "run3" / "best.pt"), help="model to fine-tune from")
    ap.add_argument("--passes", type=int, default=30)
    ap.add_argument("--holdout", type=int, default=1, help="corrected pages kept back for scoring")
    ap.add_argument("ms_id", nargs="*", help="the manuscript; for pool, the manuscripts to use (default: every one with training text)")
    ap.add_argument("--per-pass", type=int, default=3000, help="pool: lines per pass, shared equally between the manuscripts")
    ap.add_argument("--exclude", nargs="*", default=(), help="pool: manuscripts kept out of training and scored as unseen hands")
    ap.add_argument("--model", help="test: score this model file instead of the run's best.pt")
    ap.add_argument("--run", default="run1")
    ap.add_argument("--max-passes", type=int, default=80)
    ap.add_argument("--lag", type=int, default=8)
    ap.add_argument("--lrate", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, help="use only the first N training lines (quick trials)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-contributions", action="store_true", help="pool: leave out the lines contributed from Manuscribe")
    ap.add_argument("--seg", default="seg", help="train/test: lines from Kraken's segmentation (seg) or the U-Net's (seg_torch)")
    a = ap.parse_args()
    if a.command == "pool":
        pool(a.run, a.base, a.passes, per_pass=a.per_pass, holdout=a.holdout, ms_ids=a.ms_id or None, exclude=a.exclude,
             threads=a.threads, lag=a.lag, contributions=not a.no_contributions, seg=a.seg)
        raise SystemExit
    if len(a.ms_id) != 1:
        ap.error(f"{a.command} takes exactly one manuscript")
    if a.command == "train":
        train(a.ms_id[0], a.run, a.max_passes, a.lag, a.lrate, threads=a.threads, limit=a.limit, augment_on=not a.no_augment, seg=a.seg)
    elif a.command == "finetune":
        finetune(a.ms_id[0], a.base, a.run, a.passes, holdout=a.holdout, threads=a.threads, seg=a.seg)
    elif a.command == "recut":
        recut_corrected(a.ms_id[0], a.seg if a.seg != "seg" else "seg_torch")
    else:
        test(a.ms_id[0], a.run, model=a.model, seg=a.seg)
