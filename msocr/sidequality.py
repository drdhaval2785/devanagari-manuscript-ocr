"""Checks on contributed folio sides before the segmenter may learn from them.

A side comes from Manuscribe (msocr.contributions) as an image of the whole side (sides/<folio>.jpg, drawn from the scan
at the height the segmenter learned from) and its geometry (sides/<folio>.json): its layout, the blocks its editor drew,
each with the number of lines its editor gave it, which Manuscribe checked every block holds as lines of text. Here:

  image     a JPEG that opens, 300-3000 px high, at least 300 px wide
  drawn     the blocks drawn on this very side (geometry "drawn": "here"), not a calibration drawn on another side and
            applied to the whole manuscript, which only roughly fits the others
  blocks    every block a rectangle of some size lying on the image (within 2 % of its edges), with 1-200 lines
  counted   the U-Net (msocr.tseg) finds the lines on the side; in each block, the block finder's own row counter
            (blocks.rows_in_block, on the block's pixels and the lines whose middle lies in it) makes rows of them, and the side is verified
            only where every block has exactly as many rows as its editor said it holds
A verified side is a page the segmenter may learn from: its lines, as the U-Net found them and as the editor's counts
bear out, are written as sides/<folio>.seg.json (Kraken's format, like data/<ms>/seg/), its target. What the counts
confirm is that the U-Net got that side right; so the segmenter learns from contributed sides to read new hands and
scans as it already reads them where it is right, not from its mistakes (to learn from those, a side would need its
lines corrected by hand, which Manuscribe does not yet do). Sides that fail are kept in the store, and not used.

    check(jpg_bytes, geometry) -> (grey image or None, reasons)
    verify(jpg_path, geometry, net, dev) -> (Kraken-format segmentation, {box: {declared, found}}, reasons)
"""
import io

import numpy as np
from PIL import Image

LIMITS = {"height": (300, 3000), "width_min": 300, "edge": 0.02, "lines": (1, 200), "box_min": 12}


def check(data, geom):
    """the image and the blocks -> (grey image or None, reasons)"""
    why = []
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        if im.format != "JPEG":
            return None, [f"image: a {im.format}, not a JPEG"]
        g = im.convert("L")
    except Exception as e:
        return None, [f"image: does not open ({e.__class__.__name__})"]
    w, h = g.size
    lo, hi = LIMITS["height"]
    if not lo <= h <= hi:
        why.append(f"image: {h} px high, not {lo}-{hi}")
    if w < LIMITS["width_min"]:
        why.append(f"image: {w} px wide, under {LIMITS['width_min']}")
    if geom.get("drawn") != "here":
        why.append("blocks: not drawn on this side (a calibration applied to the whole manuscript)")
    if (geom.get("width"), geom.get("height")) != (w, h):
        why.append(f"geometry: says {geom.get('width')}x{geom.get('height')}, the image is {w}x{h}")
    boxes = geom.get("boxes") or {}
    if not boxes:
        why.append("geometry: no blocks")
    e = LIMITS["edge"]
    for b, x in boxes.items():
        try:
            X, Y, W, H = (float(v) for v in x["rect"])
            n = int(x["lines"])
        except Exception:
            why.append(f"block {b}: not a rectangle with a line count")
            continue
        if W < LIMITS["box_min"] or H < LIMITS["box_min"]:
            why.append(f"block {b}: {W:.0f}x{H:.0f} px, too small")
        if X < -e * w or Y < -e * h or X + W > (1 + e) * w or Y + H > (1 + e) * h:
            why.append(f"block {b}: not on the image")
        if not LIMITS["lines"][0] <= n <= LIMITS["lines"][1]:
            why.append(f"block {b}: {n} lines")
    return (g if not why else None), why


def rows_in_block(seg, gray, rect, others=()):
    """the rows the block finder's row counter (blocks.rows_in_block) makes in one block, of the lines whose baseline's
    middle lies in it; others: the other blocks' rectangles"""
    from .blocks import _pieces, rows_in_block as block_rows
    X, Y, W, H = rect
    x0, y0 = max(0, int(round(X))), max(0, int(round(Y)))
    x1, y1 = min(gray.shape[1], int(round(X + W))), min(gray.shape[0], int(round(Y + H)))
    pieces = []
    for d in _pieces(seg):
        b = d["line"]["baseline"]
        mx = (min(p[0] for p in b) + max(p[0] for p in b)) / 2
        my = sum(p[1] for p in b) / len(b)
        if x0 <= mx <= x1 and y0 <= my <= y1:
            pieces.append(d)
    if not pieces:
        return []
    others = [(int(o[0]), int(o[1]), int(o[0] + o[2]), int(o[1] + o[3])) for o in others]
    return block_rows({"rect": (x0, y0, x1, y1), "pieces": pieces}, gray, others)["rows"]


def verify(jpg_path, geom, net, dev):
    """-> (the U-Net's segmentation of the side, {box: {declared, found}}, reasons it is not verified)"""
    from . import tseg
    pb, pu, pd, s = tseg.predict_maps(net, jpg_path, dev)
    seg = tseg.kraken_json(jpg_path, tseg.vectorize(pb, pu, pd, s))
    gray = np.asarray(Image.open(jpg_path).convert("L"))
    counts, why = {}, []
    rects = {b: [float(v) for v in x["rect"]] for b, x in geom["boxes"].items()}
    for b, x in geom["boxes"].items():
        found = len(rows_in_block(seg, gray, rects[b], [r for o, r in rects.items() if o != b]))
        counts[b] = {"declared": int(x["lines"]), "found": found}
        if found != int(x["lines"]):
            why.append(f"counted: block {b} has {found} rows by the segmenter, {x['lines']} by its editor")
    return seg, counts, why
