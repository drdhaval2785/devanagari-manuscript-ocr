"""The reference pipeline: one folio side, from its image to rows of text, as pure functions, one per stage. It is what
the JavaScript port in Manuscribe copies stage by stage and is checked against (stage fixtures); it reads no files and
writes none. Its parameters are in msocr/pipeline.json; the networks are passed in as functions from arrays to arrays,
so the same stages run with PyTorch (torch_segmenter, torch_recogniser) or ONNX Runtime (onnx_function).

  page (RGB uint8, H x W x 3)
    grey(page)                                  8-bit grey, as PIL's convert("L") (ITU-R 601 luma, rounded)
    segmenter_input(grey) -> x, scale, width    the page scaled to img_h (PIL bilinear), ink high, padded to pad_to
    segmenter(x) -> raw (3, h, W)               the network: baseline logit, up and down distances / norm
    maps(raw, width) -> pbase, up, down         sigmoid of the baseline; distances in network px
    lines(pbase, up, down, scale) -> [line]     baselines (simplified) and outlines in page px (tseg.vectorize)
    rows(lines, grey, layout) -> {box: [(baseline, outline)]}   rows and blocks (align, blocks)
    cut(page, baseline, outline) -> crop        the line image, straightened (linecut)
    recogniser_input(crop) -> x                 grey, scaled to height 48 (PIL bilinear), ink high
    recogniser(x) -> logits (T, classes)        the network
    decode(logits, chars) -> text, confidence   greedy CTC
    decode_lm(logits, chars, lm) -> text, confidence   prefix beam search with the language model (msocr/lm.py)
  run_page(page, layout, segmenter, recogniser, chars, lm=None) -> {box: [{"baseline", "outline", "text", "conf"}]}
    (with lm, a pack's language model as lm.load_dir gives it, every line is read with it)
"""
import numpy as np
from PIL import Image

from .params import P, VERSION
from . import align, blocks, linecut, tseg


# ---------------------------------------------------------------- stages

def grey(page):
    return np.asarray(Image.fromarray(page).convert("L"))


def segmenter_input(g):
    h = P["segmenter"]["img_h"]
    s = h / g.shape[0]
    w = max(16, round(g.shape[1] * s))
    x = 1.0 - np.asarray(Image.fromarray(g).resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    pad = P["segmenter"]["pad_to"]
    W = -(-w // pad) * pad
    return np.pad(x, ((0, 0), (0, W - w)))[None, None], s, w


def maps(raw, width):
    o = raw[:, :, :width].astype(np.float32)
    norm = P["segmenter"]["norm"]
    return 1.0 / (1.0 + np.exp(-o[0])), np.clip(o[1], 0, None) * norm, np.clip(o[2], 0, None) * norm


def lines(pbase, up, down, scale):
    return tseg.vectorize(pbase, up, down, scale)


def rows(page_lines, g, layout):
    """{box: [(baseline, outline), ...]} in reading order; "main" alone on a linear page."""
    seg = {"lines": page_lines}
    size = (g.shape[1], g.shape[0])
    if layout in ("tripatha", "panchapatha"):
        found, note = blocks.page_blocks(seg, size, g, blocks.EXPECTED[layout])
        return {box: b["rows"] for box, b in found.items()}
    found, block, pitch, note = align.find_rows(seg, size[0], g)
    if not found:
        return {}
    if len(found) >= 3:
        found, _ = align.fill_edge_rows(found, block, pitch, g)
    return {"main": [align._merge(r, block, pitch) for r in found]}


def cut(page, baseline, outline):
    above, below = linecut.band_of(baseline, outline)
    return linecut.cut_line(page, baseline, above, below)


def recogniser_input(crop):
    h = P["recogniser"]["height"]
    im = Image.fromarray(crop).convert("L")
    w = max(P["recogniser"]["stride"] * 2, round(im.size[0] * h / im.size[1]))
    im = im.resize((w, h), Image.BILINEAR)
    return (1.0 - np.asarray(im, dtype=np.float32) / 255.0)[None, None]


def decode(logits, chars):
    """Greedy CTC: the best class in every column, repeats and blanks (class 0) dropped; confidence = mean probability
    of the classes kept, in percent."""
    z = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
    best, idx = probs.max(-1), probs.argmax(-1)
    out, conf, prev = [], [], 0
    for p, v in zip(best.tolist(), idx.tolist()):
        if v != prev and v != 0:
            out.append(chars[v - 1])
            conf.append(p)
        prev = v
    return "".join(out), 100 * (sum(conf) / len(conf) if conf else 0.0)


def decode_lm(logits, chars, lm):
    """The reading with the language model: a prefix beam search over the columns' probabilities, each hypothesis
    weighed by how its letters run in Sanskrit (lm.py: alpha, beta, beam of pipeline.json); confidence = mean
    probability the recogniser gives the letters read, aligned back to their columns, in percent."""
    from . import lm as L
    text, conf, _ = L.decode(logits, chars, lm)
    return text, conf


# ---------------------------------------------------------------- a page

def run_page(page, layout, segmenter, recogniser, chars, lm=None):
    g = grey(page)
    x, s, w = segmenter_input(g)
    pb, pu, pd = maps(segmenter(x)[0], w)
    out = {}
    for box, rs in rows(lines(pb, pu, pd, s), g, layout).items():
        out[box] = []
        for baseline, outline in rs:
            logits = recogniser(recogniser_input(cut(page, baseline, outline)))[0]
            text, conf = decode_lm(logits, chars, lm) if lm else decode(logits, chars)
            out[box].append({"baseline": baseline, "outline": outline, "text": text, "conf": conf})
    return out


# ---------------------------------------------------------------- the networks as functions

def torch_segmenter(net, device="cpu"):
    import torch
    net = net.to(device).eval()

    def f(x):
        with torch.no_grad():
            return net(torch.from_numpy(x).to(device)).float().cpu().numpy()
    return f


def torch_recogniser(net):
    import torch
    net = net.eval()

    def f(x):
        with torch.no_grad():
            return net(torch.from_numpy(x)).numpy()
    return f


def onnx_function(path):
    """An exported network (tools/browser_bench/export.py) as a function, with ONNX Runtime (in .venv-onnx)."""
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    return lambda x: s.run(None, {name: x})[0]
