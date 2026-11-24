"""Our own line segmenter in PyTorch: a fast stand-in for Kraken's baseline segmenter.

Kraken's segmenter runs on the CPU here (its Mac GPU path exhausts memory) at about 45 s a page,
and two of its workers can take some 10 GB each. This one runs on the Mac's GPU in a fraction of
a second a page and writes the same thing Kraken writes, so everything downstream (the row finder,
the block finder, predict, show, rowcheck) works on it unchanged.

What the network sees and predicts
  The page, in grey and scaled to IMG_H pixels high, goes through a small U-Net that predicts three
  maps of the same size: "baseline" (a band a few pixels thick along the baseline of every text
  line or line piece), and, along that band, "up" and "down": how far that line's own outline
  (Kraken's boundary polygon) reaches above and below its baseline. The two distances carry each
  line's hand size, which the block finder needs to tell the main text from the commentary.
How the lines are read off the maps
  Every connected stretch of the baseline map at least MIN_PIECE pixels wide is one line piece: its
  baseline is the middle of the band, column by column, smoothed and sampled every few pixels. Its
  boundary is the baseline moved up and down by the predicted distances (smoothed along the line).
  Pieces are scaled back to the page's own pixels and written in Kraken's JSON form.
What it learns from
  Kraken's own segmentation of every page segmented so far (data/<ms>/seg/), so its pieces are what
  the row finder already expects. Kept out of training for the final test, so that every hand and
  layout is tested: one page in TEST_EVERY_DEFAULT of every manuscript, one in TEST_EVERY[ms] of those
  named there (AKGM 0025383: every fifth page); for LDII_85885 exactly the recogniser's test pages
  (data/LDII_85885/test.txt), which have a transcription to measure the reading on; and every page of a
  manuscript in HELD_OUT (none at present; name one there to test on a hand never seen). Of the rest,
  one page in VAL_EVERY is kept for validation.
Only pages Kraken got right are used, in all three sets
  Kraken's segmentation is wrong on some pages (a line missed, two lines merged, a note squeezed in),
  and the network would learn those mistakes, and be scored against them. So every page must first
  pass rowcheck (msocr/rowcheck.py) clean: text rows found, and no warning (no row filled in where
  Kraken missed one, no uneven spacing, no gap that may hide a short line). Before training or
  comparing, a manuscript's data/<ms>/rowcheck.txt is made afresh if it is missing or older than its
  segmentation or the row finder's code. A page whose only fault is a line Kraken missed is used in its
  repaired form when msocr/repair.py has made one (data/<ms>/seg_repaired/, the missing line added and the
  page checked again). A page marked wrong in the review (msocr/kreview.py, taken in by msocr/kfeedback.py into
  data/kraken_review/feedback/<ms>.tsv) is left out, repaired or not, until a later review marks it otherwise. The pages left out, with their warnings, are listed in data/tseg/flagged_pages.tsv:
  the ones to look at with `python -m msocr.show <ms> --warned`. The pages of each set are chosen by a hash of their names, so they stay the
  same however many pages are added.

    python -m msocr.tseg train --run tseg1 --reviewed [--epochs 24]
        --reviewed says Kraken's segmentation has been checked (rowcheck reports, flagged pages looked at);
        without it nothing is trained: the network learns whatever Kraken did, right or wrong
        trains; data/tseg/models/<run>/ gets train.log, history.csv, best.pt
    python -m msocr.tseg segment AKGM_0025383 [--model data/tseg/models/tseg1/best.pt] [--pages 0001_2 ...] [--redo]
        writes data/<ms>/seg_torch/<image stem>.json for every page image (or the pages named)
    python -m msocr.tseg compare AKGM_0025383 LDII_85885 [--model ...] [--all]
        on the manuscript's test pages (never trained on; --all for every page Kraken has segmented): how
        far the two segmentations agree (baselines), the rows the row finder makes of each, and the time
        per page; writes data/tseg/compare_<ms>.txt

The rest of the tools read Kraken's segmentation unless MSOCR_SEG=seg_torch is set (config.SEG),
e.g. MSOCR_SEG=seg_torch .venv/bin/python -m msocr.show AKGM_0025383 0069_1.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .config import DATA, MANUSCRIPTS, ms_dir

from .params import P as _P
IMG_H = _P["segmenter"]["img_h"]            # the page is scaled to this height for the network
CROP_W = 512                # training crops: full height, this width
BASE_W = 3                  # thickness of the baseline band in the network's pixels
MIN_PIECE = _P["segmenter"]["min_piece"]   # narrowest baseline piece kept, network pixels
NORM = _P["segmenter"]["norm"]             # the up/down distances are predicted divided by this (network pixels)
HELD_OUT = ()                          # whole manuscripts never trained on
TEST_EVERY_DEFAULT = 20                # one page in this many of every manuscript is kept for the final test
TEST_EVERY = {"AKGM_0025383": 5}       # ... or in this many, for the manuscripts named
VAL_EVERY = 25
SEED = 20261123
OUT_DIR = "seg_torch"
MODELS = DATA / "tseg" / "models"
DEFAULT_RUN = "tseg1"


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------- the network

def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Four levels down and up; ~1.2 M parameters. Input (N, 1, H, W) with H and W multiples of 16."""

    def __init__(self, base=24, out=3):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList([_block(1, c[0]), _block(c[0], c[1]), _block(c[1], c[2]), _block(c[2], c[3])])
        self.mid = _block(c[3], c[3])
        self.dec = nn.ModuleList([_block(c[3] + c[3], c[2]), _block(c[2] + c[2], c[1]), _block(c[1] + c[1], c[0]),
                                  _block(c[0] + c[0], c[0])])
        self.head = nn.Conv2d(c[0], out, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.mid(x)
        for d, s in zip(self.dec, reversed(skips)):
            x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = d(torch.cat([x, s], 1))
        return self.head(x)


# ---------------------------------------------------------------- data

def _hash(s):
    return int(hashlib.sha1(f"{SEED}:{s}".encode()).hexdigest(), 16)


def _test_pages():
    f = DATA / "LDII_85885" / "test.txt"
    return {Path(p).stem for p in f.read_text().split()} if f.exists() else set()


def _report_fresh(ms):
    """Is data/<ms>/rowcheck.txt newer than the segmentation and the row finder's code?"""
    rep = DATA / ms / "rowcheck.txt"
    if not rep.exists():
        return False
    code = max((Path(__file__).parent / f).stat().st_mtime for f in ("align.py", "blocks.py", "rowcheck.py"))
    segs = [p.stat().st_mtime for p in (DATA / ms / "seg").glob("*.json")]
    return rep.stat().st_mtime >= max([code] + segs)


def ensure_rowchecks(quiet=False):
    """Make every segmented manuscript's rowcheck report afresh where it is missing or stale."""
    from .rowcheck import check
    for ms in sorted(MANUSCRIPTS):
        if any((DATA / ms / "seg").glob("*.json")) and not _report_fresh(ms):
            if not quiet:
                print(f"[tseg] rowcheck {ms} (report missing or older than its segmentation or the code)", flush=True)
            check(ms, verbose=False)


REVIEWED_WRONG = "marked wrong in the review"


def flagged(ms):
    """{image stem: reason} for the pages of a manuscript that rowcheck does not pass clean."""
    import re
    rep = DATA / ms / "rowcheck.txt"
    out = {}
    if not rep.exists():
        return None                                   # unknown: page_list then trusts nothing of it
    from .blocks import layout_of
    blocky = layout_of(ms) in ("tripatha", "panchapatha")
    for line in rep.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        image, rows, note, warn = f[0], int(f[2] or 0), f[3] if len(f) > 3 else "", f[4] if len(f) > 4 else ""
        found = rows >= 2 or (blocky and re.match(r"(main|top|bottom|left|right) \d", note or ""))
        if warn or not found:
            out[image.rsplit(".", 1)[0]] = warn or note or "no text rows found"
    # a page marked wrong in the review (msocr.kfeedback) stays out until a later review marks it otherwise
    fb = DATA / "kraken_review" / "feedback" / f"{ms}.tsv"
    if fb.exists():
        import csv
        with open(fb, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r.get("verdict") == "wrong":
                    out[r["page"]] = REVIEWED_WRONG + (f": {r['note']}" if r.get("note") else "")
    return out


def page_list(quality=True):
    """-> {"train": [...], "val": [...], "test": [...], "flagged": [...]} of (ms, stem, image path, Kraken json path);
    with quality, the pages rowcheck flags go to "flagged" instead of a set (see the module's notes)."""
    from .config import not_text
    test_stems = _test_pages()
    out = {"train": [], "val": [], "test": [], "flagged": [], "not_text": []}
    for ms in sorted(MANUSCRIPTS):
        bad = flagged(ms) if quality else {}
        skip = not_text(ms)
        for j in sorted((DATA / ms / "seg").glob("*.json")):
            img = MANUSCRIPTS[ms]["images"] / (j.stem + ".jpg")
            if not img.exists():
                continue
            item = (ms, j.stem, img, j)
            if j.stem in skip:                                   # a cover or a blank side
                out["not_text"].append(item)
                continue
            if bad is None or j.stem in bad:
                fixed = DATA / ms / "seg_repaired" / f"{j.stem}.json"
                if (bad is not None and fixed.exists() and fixed.stat().st_mtime >= j.stat().st_mtime
                        and not bad[j.stem].startswith(REVIEWED_WRONG)):
                    item = (ms, j.stem, img, fixed)     # Kraken's lines with the missed ones added, checked again
                else:
                    out["flagged"].append(item)
                    continue
            if ms == "LDII_85885":                    # its test pages are the recogniser's, no others
                is_test = j.stem in test_stems
            else:
                is_test = ms in HELD_OUT or _hash(f"test:{ms}/{j.stem}") % TEST_EVERY.get(ms, TEST_EVERY_DEFAULT) == 0
            if is_test:
                out["test"].append(item)
            elif _hash(f"{ms}/{j.stem}") % VAL_EVERY == 0:
                out["val"].append(item)
            else:
                out["train"].append(item)
    return out


def load_page(img_path, height=IMG_H):
    """The page in grey, ink high (0..1), scaled to `height` -> (array, scale)."""
    with Image.open(img_path) as im:
        g = im.convert("L")
        s = height / g.height
        g = g.resize((max(16, round(g.width * s)), height), Image.BILINEAR)
    return 1.0 - np.asarray(g, dtype=np.float32) / 255.0, s


def targets(seg, size, s):
    """Kraken's lines drawn as the network's targets, at its scale s: (baseline band 0/1, up, down, mask), where
    up/down are, on the baseline band, the distances (network px) from the baseline to the top and the bottom
    of that line's own boundary polygon, and mask marks where they are known."""
    w, h = size
    base = Image.new("L", (w, h), 0)
    db = ImageDraw.Draw(base)
    up = np.zeros((h, w), np.float32)
    down = np.zeros((h, w), np.float32)
    mask = np.zeros((h, w), np.uint8)
    for l in seg["lines"]:
        pts = [(x * s, y * s) for x, y in l["baseline"]]
        if len(pts) < 2:
            continue
        db.line(pts, fill=1, width=BASE_W)
        poly = [(x * s, y * s) for x, y in (l.get("boundary") or [])]
        if len(poly) < 3:
            continue
        px0, py0 = int(min(p[0] for p in poly)), int(min(p[1] for p in poly))
        px1, py1 = int(max(p[0] for p in poly)) + 1, int(max(p[1] for p in poly)) + 1
        local = Image.new("L", (max(1, px1 - px0), max(1, py1 - py0)), 0)
        ImageDraw.Draw(local).polygon([(x - px0, y - py0) for x, y in poly], fill=1)
        m = np.asarray(local, dtype=bool)
        has = m.any(0)
        if not has.any():
            continue
        top = np.where(has, m.argmax(0), 0) + py0
        bot = np.where(has, m.shape[0] - 1 - m[::-1].argmax(0), 0) + py0
        bx = np.array([p[0] for p in pts]); by = np.array([p[1] for p in pts])
        order = np.argsort(bx); bx, by = bx[order], by[order]
        for x in range(max(0, int(bx[0])), min(w, int(bx[-1]) + 1)):
            k = x - px0
            if k < 0 or k >= len(has) or not has[k]:
                continue
            yb = float(np.interp(x, bx, by))
            yi = int(round(yb))
            for yy in range(yi - BASE_W // 2, yi + BASE_W // 2 + 1):
                if 0 <= yy < h:
                    up[yy, x] = max(0.0, yb - top[k]) / NORM
                    down[yy, x] = max(0.0, bot[k] - yb) / NORM
                    mask[yy, x] = 1
    return np.asarray(base, dtype=np.uint8), up, down, mask


class Pages(torch.utils.data.Dataset):
    def __init__(self, items, train=True):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ms, stem, img, js = self.items[i]
        rng = random.Random() if self.train else random.Random(i)
        h = IMG_H
        if self.train:                           # a little scale jitter: the page drawn 85-115% as high
            h = int(IMG_H * rng.uniform(0.85, 1.15)) // 2 * 2
        x, s = load_page(img, h)
        b, up, down, m = targets(json.load(open(js)), (x.shape[1], x.shape[0]), s)
        tgt = np.stack([b.astype(np.float32), up, down, m.astype(np.float32)])
        # to the network's height: crop or pad the jittered page vertically
        if h > IMG_H:
            t0 = rng.randrange(0, h - IMG_H + 1)
            x, tgt = x[t0:t0 + IMG_H], tgt[:, t0:t0 + IMG_H]
        elif h < IMG_H:
            t0 = rng.randrange(0, IMG_H - h + 1) if self.train else 0
            x = np.pad(x, ((t0, IMG_H - h - t0), (0, 0)))
            tgt = np.pad(tgt, ((0, 0), (t0, IMG_H - h - t0), (0, 0)))
        W = CROP_W
        if x.shape[1] < W:
            x = np.pad(x, ((0, 0), (0, W - x.shape[1])))
            tgt = np.pad(tgt, ((0, 0), (0, 0), (0, W - tgt.shape[2])))
        c0 = rng.randrange(0, x.shape[1] - W + 1) if self.train else (x.shape[1] - W) // 2
        x, tgt = x[:, c0:c0 + W], tgt[:, :, c0:c0 + W]
        if self.train:                           # ink, paper and scanner differ from manuscript to manuscript
            x = np.clip((x - x.mean()) * rng.uniform(0.7, 1.3) + x.mean() + rng.uniform(-0.1, 0.1), 0, 1)
            x = x ** rng.uniform(0.7, 1.4)
            x = x + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(0, 0.04), x.shape).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None], torch.from_numpy(np.ascontiguousarray(tgt))


def _loss(logits, tgt):
    """Baseline: weighted cross-entropy and Dice. Up/down: L1 on the baseline band, where they are known."""
    b, t = logits[:, :1], tgt[:, :1]
    bce = F.binary_cross_entropy_with_logits(b, t, pos_weight=torch.tensor(5.0, device=logits.device))
    p = torch.sigmoid(b)
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    m = tgt[:, 3:4]
    l1 = ((logits[:, 1:3] - tgt[:, 1:3]).abs() * m).sum() / (2 * m.sum() + 1)
    return bce + dice + l1


# ---------------------------------------------------------------- from maps to lines

@torch.no_grad()
def predict_maps(net, img_path, dev):
    """-> (baseline probability, up, down (network px), scale) at the network's resolution."""
    x, s = load_page(img_path)
    h, w = x.shape
    W = math.ceil(w / _P["segmenter"]["pad_to"]) * _P["segmenter"]["pad_to"]
    t = torch.from_numpy(np.pad(x, ((0, 0), (0, W - w))))[None, None].to(dev)
    o = net(t)[0, :, :, :w].float().cpu().numpy()
    return 1 / (1 + np.exp(-o[0])), np.clip(o[1], 0, None) * NORM, np.clip(o[2], 0, None) * NORM, s


def _smooth(v, k=9):
    if len(v) < 3:
        return np.asarray(v, dtype=float)
    k = min(k, len(v) if len(v) % 2 else len(v) - 1)
    pad = k // 2
    vp = np.pad(np.asarray(v, dtype=float), pad, mode="edge")
    return np.convolve(vp, np.ones(k) / k, mode="valid")


SIMPLIFY = _P["segmenter"]["simplify"]     # page px: a baseline keeps only the points needed to stay this close to the network's line


def simplify(points, eps=SIMPLIFY):
    """The fewest of `points` that keep the polyline within eps px of them all (Ramer-Douglas-Peucker). A line image is
    cut by straightening the text along its baseline; a baseline with a point every few pixels, each half a pixel off,
    puts a kink in the letters at every point (the recogniser read 4.9 % of characters wrong on such lines, 3.3 % on
    the same lines simplified, 3.4 % on Kraken's, LDII_85885's test pages, 2026-11-24)."""
    if len(points) < 3:
        return [list(p) for p in points]
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        (xa, ya), (xb, yb) = points[a], points[b]
        dx, dy = xb - xa, yb - ya
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-9)
        far, at = 0.0, -1
        for i in range(a + 1, b):
            d = abs(dx * (points[i][1] - ya) - dy * (points[i][0] - xa)) / norm
            if d > far:
                far, at = d, i
        if far > eps:
            keep[at] = True
            stack += [(a, at), (at, b)]
    return [list(p) for p, k in zip(points, keep) if k]


def components(mask):
    """The 8-connected components of a boolean image, in raster order of their first pixel -> [(ys, xs), ...].
    Runs of pixels are found row by row and joined to the runs of the row above that touch them, diagonally too
    (union-find over runs); plain array arithmetic, the same in JavaScript."""
    h = mask.shape[0]
    runs, parent, prev = [], [], []

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for y in range(h):
        d = np.diff(np.concatenate(([0], mask[y].astype(np.int8), [0])))
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1
        cur, j = [], 0
        for a, b in zip(starts.tolist(), ends.tolist()):
            i = len(runs)
            runs.append((y, a, b))
            parent.append(i)
            cur.append(i)
            while j < len(prev) and runs[prev[j]][2] < a - 1:
                j += 1
            k = j
            while k < len(prev) and runs[prev[k]][1] <= b + 1:
                r1, r2 = root(i), root(prev[k])
                if r1 != r2:
                    parent[max(r1, r2)] = min(r1, r2)
                k += 1
        prev = cur
    groups = {}
    for i in range(len(runs)):
        groups.setdefault(root(i), []).append(i)
    out = []
    for r in sorted(groups):
        ys = np.concatenate([np.full(runs[i][2] - runs[i][1] + 1, runs[i][0]) for i in groups[r]])
        xs = np.concatenate([np.arange(runs[i][1], runs[i][2] + 1) for i in groups[r]])
        out.append((ys, xs))
    return out


def vectorize(pbase, up, down, s, thr=_P["segmenter"]["threshold"], step=_P["segmenter"]["step"]):
    """Line pieces from the maps, in the page's own pixels: [{"baseline": [[x, y]...], "boundary": [...]}]."""
    H, W = pbase.shape
    lines = []
    for rows, cols in components(pbase > thr):
        x0, x1 = int(cols.min()), int(cols.max()) + 1
        if x1 - x0 < MIN_PIECE:
            continue
        cnt = np.bincount(cols - x0, minlength=x1 - x0)
        ys = np.bincount(cols - x0, weights=rows, minlength=x1 - x0)
        ok = cnt > 0
        xs = np.arange(x0, x1)[ok]
        yc = _smooth(ys[ok] / cnt[ok], _P["segmenter"]["smooth_baseline"])
        # the distances, read on the band and smoothed along the line
        u = np.array([up[min(H - 1, max(0, int(round(y)))), x] for x, y in zip(xs, yc)])
        d = np.array([down[min(H - 1, max(0, int(round(y)))), x] for x, y in zip(xs, yc)])
        u, d = _smooth(u, _P["segmenter"]["smooth_reach"]), _smooth(d, _P["segmenter"]["smooth_reach"])
        idx = sorted(set(list(range(0, len(xs), step)) + [len(xs) - 1]))
        to_page = lambda x, y: [int(round(x / s)), int(round(y / s))]
        bl = simplify([to_page(xs[i], yc[i]) for i in idx])
        if len(bl) == 1:
            bl.append([bl[0][0] + 1, bl[0][1]])
        top = [to_page(xs[i], yc[i] - max(1.0, u[i])) for i in idx]
        bot = [to_page(xs[i], yc[i] + max(1.0, d[i])) for i in reversed(idx)]
        lines.append({"baseline": bl, "boundary": top + bot})
    lines.sort(key=lambda l: (statistics.mean(p[1] for p in l["baseline"]), l["baseline"][0][0]))
    return lines


def kraken_json(img_path, lines):
    return {"type": "baselines", "imagename": str(img_path), "text_direction": "horizontal-lr", "script_detection": False,
            "lines": [{"id": f"tseg_{i}", "text": None, "base_dir": None, "imagename": None,
                       "tags": {"type": [{"type": "default"}]}, "split": None, "regions": [], "language": None,
                       "type": "baselines", "baseline": l["baseline"], "boundary": l["boundary"]} for i, l in enumerate(lines)],
            "regions": {}, "line_orders": [], "language": None, "segmenter": "msocr.tseg"}


def load_net(path=None):
    path = Path(path or MODELS / DEFAULT_RUN / "best.pt")
    ck = torch.load(path, map_location="cpu")
    net = UNet(**ck.get("arch", {}))
    net.load_state_dict(ck["model"])
    return net.to(device()).eval(), ck


def segment_pages(ms_id, images=None, model=None, redo=False, quiet=False):
    """Segment pages of a manuscript into data/<ms>/seg_torch/. images: file names; default every page image."""
    ms = MANUSCRIPTS[ms_id]
    out = ms_dir(ms_id) / OUT_DIR
    out.mkdir(exist_ok=True)
    images = images if images is not None else sorted(p.name for p in ms["images"].glob("*.jpg"))
    todo = [i for i in images if redo or not (out / (i.rsplit(".", 1)[0] + ".json")).exists()]
    if not quiet:
        print(f"[tseg] {ms_id}: {len(images)} pages requested, {len(images) - len(todo)} already done, {len(todo)} to do")
    if not todo:
        return
    net, ck = load_net(model)
    dev = device()
    t = time.time()
    for k, i in enumerate(todo, 1):
        pb, pu, pd, s = predict_maps(net, ms["images"] / i, dev)
        lines = vectorize(pb, pu, pd, s)
        (out / (i.rsplit(".", 1)[0] + ".json")).write_text(json.dumps(kraken_json(ms["images"] / i, lines)), encoding="utf-8")
        if not quiet and (k % 25 == 0 or k == len(todo)):
            print(f"  {k}/{len(todo)} pages, {(time.time() - t) / k:.2f} s a page", flush=True)


# ---------------------------------------------------------------- agreement with Kraken

def _samples(lines, step=8):
    """(x, y) points every `step` px along each baseline."""
    pts = []
    for l in lines:
        b = l["baseline"]
        for (xa, ya), (xb, yb) in zip(b, b[1:]):
            n = max(1, int(abs(xb - xa) // step))
            for j in range(n):
                t = j / n
                pts.append((xa + t * (xb - xa), ya + t * (yb - ya)))
        pts.append(tuple(b[-1]))
    return np.asarray(pts, dtype=float) if pts else np.zeros((0, 2))


def agreement(pred_lines, ref_lines, tol):
    """Baseline agreement, by length: the share of predicted baseline lying within tol px (vertically, at the
    same x) of a reference baseline (precision), and the reverse (recall)."""
    def covered(a, b):
        if len(a) == 0:
            return 0.0
        if len(b) == 0:
            return 0.0
        order = np.argsort(b[:, 0])
        bx, by = b[order, 0], b[order, 1]
        hit = 0
        for x, y in a:
            lo, hi = np.searchsorted(bx, x - 5), np.searchsorted(bx, x + 5)
            if hi > lo and np.min(np.abs(by[lo:hi] - y)) <= tol:
                hit += 1
        return hit / len(a)
    a, b = _samples(pred_lines), _samples(ref_lines)
    p, r = covered(a, b), covered(b, a)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# ---------------------------------------------------------------- training

def train(run, epochs=24, batch=8, lr=1e-3, workers=4):
    torch.manual_seed(SEED)
    random.seed(SEED)
    dev = device()
    ensure_rowchecks()
    pages = page_list()
    (DATA / "tseg").mkdir(exist_ok=True)
    with open(DATA / "tseg" / "flagged_pages.tsv", "w", encoding="utf-8") as f:
        f.write("manuscript\timage\twhy left out\n")
        reasons = {}
        for ms, stem, img, js in pages["flagged"]:
            if ms not in reasons:
                reasons[ms] = flagged(ms) or {}
            f.write(f"{ms}\t{stem}\t{reasons[ms].get(stem, 'no rowcheck report')}\n")
    d = MODELS / run
    d.mkdir(parents=True, exist_ok=True)
    log = open(d / "train.log", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say(f"[tseg] run {run}: train {len(pages['train'])} pages, val {len(pages['val'])}, test {len(pages['test'])}; "
        f"{len(pages['flagged'])} pages left out because Kraken's segmentation of them is doubtful "
        f"(data/tseg/flagged_pages.tsv); device {dev}")
    dl = torch.utils.data.DataLoader(Pages(pages["train"]), batch_size=batch, shuffle=True, num_workers=workers,
                                     drop_last=True, persistent_workers=workers > 0)
    net = UNet().to(dev)
    say(f"  parameters {sum(p.numel() for p in net.parameters()):,}")
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    total = epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.1)
    hist = open(d / "history.csv", "a", newline="")
    hw = csv.writer(hist)
    if hist.tell() == 0:
        hw.writerow(["epoch", "train_loss", "val_baseline_f1", "val_precision", "val_recall", "seconds"])
    best, t0 = -1.0, time.time()
    for ep in range(1, epochs + 1):
        net.train()
        losses = []
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            loss = _loss(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        net.eval()
        ps, rs, fs = [], [], []
        for ms, stem, img, js in pages["val"]:
            pb, pu, pd, s = predict_maps(net, img, dev)
            ref = json.load(open(js))["lines"]
            tol = max(4.0, 0.012 * Image.open(img).height)
            p, r, f = agreement(vectorize(pb, pu, pd, s), ref, tol)
            ps.append(p), rs.append(r), fs.append(f)
        f1, p, r = statistics.mean(fs), statistics.mean(ps), statistics.mean(rs)
        hw.writerow([ep, f"{statistics.mean(losses):.4f}", f"{f1:.4f}", f"{p:.4f}", f"{r:.4f}", f"{time.time() - t0:.0f}"])
        hist.flush()
        mark = ""
        if f1 > best:
            best = f1
            torch.save({"model": net.state_dict(), "arch": {"out": 3}, "epoch": ep, "val_baseline_f1": f1, "img_h": IMG_H}, d / "best.pt")
            mark = "  (best)"
        say(f"  pass {ep:2d}/{epochs}  loss {statistics.mean(losses):.4f}  val baseline agreement with Kraken: "
            f"P {p:.3f} R {r:.3f} F1 {f1:.3f}  {(time.time() - t0) / 60:5.1f} min{mark}")
    say(f"[tseg] best validation F1 {best:.3f}; model {d / 'best.pt'}")


# ---------------------------------------------------------------- comparison with Kraken

def compare(ms_ids, model=None, limit=None, every=False):
    """On a manuscript's test pages (every page Kraken has segmented with every=True): baseline agreement, the rows
    the row finder makes of each segmentation, time."""
    import warnings
    warnings.filterwarnings("ignore")
    from . import blocks as B
    from .align import find_rows, fill_edge_rows
    net, ck = load_net(model)
    dev = device()
    ensure_rowchecks()
    if ms_ids == ["all"]:
        ms_ids = sorted({m for m, *_ in page_list()["test"]})
    summary = []
    for ms_id in ms_ids:
        ms = MANUSCRIPTS[ms_id]
        layout = B.layout_of(ms_id)
        if every:
            segs = sorted((DATA / ms_id / "seg").glob("*.json"))[:limit]
        else:
            segs = sorted(js for m, stem, img, js in page_list()["test"] if m == ms_id)[:limit]
        lines_out, rows_same, fs, secs = [], 0, [], []
        for j in segs:
            img = ms["images"] / (j.stem + ".jpg")
            t = time.time()
            pb, pu, pd, s = predict_maps(net, img, dev)
            mine = vectorize(pb, pu, pd, s)
            secs.append(time.time() - t)
            ref = json.load(open(j))
            with Image.open(img) as im:
                size = im.size
                gray = np.asarray(im.convert("L"))
            p, r, f = agreement(mine, ref["lines"], max(4.0, 0.012 * size[1]))
            fs.append(f)

            def rows_of(seg):
                if layout in ("tripatha", "panchapatha"):
                    found, note = B.page_blocks(seg, size, gray, B.EXPECTED.get(layout))
                    return {b: len(v["rows"]) for b, v in found.items()}
                rows, block, pitch, note = find_rows(seg, size[0], gray)
                if len(rows) >= 3:
                    rows, _ = fill_edge_rows(rows, block, pitch, gray)
                return {"main": len(rows)}
            rk, rt = rows_of(ref), rows_of({"lines": mine})
            rows_same += rk == rt
            lines_out.append(f"{j.stem}\tbaseline P {p:.3f} R {r:.3f} F1 {f:.3f}\trows Kraken {rk}\trows tseg {rt}"
                             + ("" if rk == rt else "\tDIFFER"))
        n = len(segs)
        head = (f"{ms_id} ({layout}): {n} {'pages' if every else 'test pages (never trained on)'} segmented by both\n"
                f"  baseline agreement with Kraken, mean F1 {statistics.mean(fs):.3f}\n"
                f"  same rows per block as with Kraken's segmentation: {rows_same} of {n} pages\n"
                f"  tseg time {statistics.mean(secs):.2f} s a page (Kraken here: about 45 s a page per worker)\n")
        (DATA / "tseg").mkdir(exist_ok=True)
        (DATA / "tseg" / f"compare_{ms_id}.txt").write_text(head + "\n".join(lines_out) + "\n", encoding="utf-8")
        print(head + f"  page by page: {DATA / 'tseg' / f'compare_{ms_id}.txt'}", flush=True)
        summary.append((ms_id, layout, n, statistics.mean(fs) if fs else 0.0, rows_same, statistics.mean(secs) if secs else 0.0))
    if len(summary) > 1:
        tot = sum(x[2] for x in summary)
        out = ["manuscript\tlayout\tpages\tbaseline F1\tsame rows\ts a page"]
        out += [f"{m}\t{l}\t{n}\t{f:.3f}\t{r}/{n}\t{t:.2f}" for m, l, n, f, r, t in summary]
        out.append(f"ALL\t\t{tot}\t{sum(x[3] * x[2] for x in summary) / tot:.3f}\t{sum(x[4] for x in summary)}/{tot}"
                   f"\t{sum(x[5] * x[2] for x in summary) / tot:.2f}")
        for lay in sorted({x[1] for x in summary}):
            g = [x for x in summary if x[1] == lay]
            k = sum(x[2] for x in g)
            out.append(f"{lay}\t\t{k}\t{sum(x[3] * x[2] for x in g) / k:.3f}\t{sum(x[4] for x in g)}/{k}\t")
        (DATA / "tseg" / "compare_summary.tsv").write_text("\n".join(out) + "\n", encoding="utf-8")
        print("\n".join(out), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("train")
    a.add_argument("--run", default=DEFAULT_RUN)
    a.add_argument("--reviewed", action="store_true",
                   help="Kraken's segmentation has been checked; training refuses to start without it")
    a.add_argument("--epochs", type=int, default=24)
    a.add_argument("--batch", type=int, default=8)
    a.add_argument("--workers", type=int, default=4)
    a = sub.add_parser("segment")
    a.add_argument("ms_id")
    a.add_argument("--model")
    a.add_argument("--pages", nargs="*", help="page ids (0001_2) or image names; default every page")
    a.add_argument("--redo", action="store_true")
    a = sub.add_parser("compare")
    a.add_argument("ms_ids", nargs="+", help='manuscript ids, or "all": every manuscript with test pages')
    a.add_argument("--model")
    a.add_argument("--limit", type=int)
    a.add_argument("--all", action="store_true", help="every page Kraken has segmented, trained on or not")
    args = ap.parse_args()
    if args.cmd == "train":
        if not args.reviewed:
            print("[tseg] not training: Kraken's segmentation has not been marked as reviewed. Check the rowcheck reports\n"
                  "       and the flagged pages first (data/<ms>/rowcheck.txt, msocr.show <ms> --warned), then run\n"
                  "       python -m msocr.tseg train --run tseg1 --reviewed", flush=True)
            sys.exit(0)
        train(args.run, args.epochs, args.batch, workers=args.workers)
    elif args.cmd == "segment":
        imgs = None
        if args.pages:
            ms = MANUSCRIPTS[args.ms_id]
            imgs = [p if p.endswith(".jpg") else ms["image_name"](p) for p in args.pages]
        segment_pages(args.ms_id, imgs, args.model, args.redo)
    else:
        compare(args.ms_ids, args.model, args.limit, args.all)
