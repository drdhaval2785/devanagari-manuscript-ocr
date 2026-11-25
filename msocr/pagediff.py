"""A page read by a model, set against its ground truth, as one HTML page to look at.

    .venv/bin/python -m msocr.pagediff LDII_9851 132 --model pool1
    .venv/bin/python -m msocr.pagediff LDII_9851.pdf 66_1 --gt mine.txt --seg seg_torch \\
        --model data/pool/models/pool1/best.pt --compare LDII_19634/ft6 --open

Input
- the manuscript: its name (LDII_9851), its PDF's name (LDII_9851.pdf) or the PDF's path. The page images extracted
  beside the PDF (LDII_9851/, named <number>_<folio>_<side>.jpg) are what is read.
- one or more pages: a PDF page number (1 is the PDF's first page; the page images in name order are the PDF's pages
  one by one, and the PDF's page count is checked against them when pypdf is installed), a folio and side as 66_1, or
  an image name as 009851_00066_1.
- --gt: the ground truth of each page, in the format of corrected/ (msocr/pagetext.py: one line per manuscript line, a
  ;box{...} marker before each block of a tripāṭha or pañcapāṭha page); given once per page, in the order of the pages.
  Default data/<ms>/corrected/<page>.txt.
- --seg: the segmentation to cut the rows from (default config.SEG, seg_torch: the U-Net's; seg: Kraken's). A page not
  yet segmented so is segmented first.
- --model: the recogniser (a best.pt, its folder, or a run's name: pool1, ft6, or owner/run as LDII_19634/ft6, looked
  for as data/<owner>/models/<run>/best.pt). --compare: a second one, whose reading is shown greyed and scored too.

What is done
- The page is cut into rows as predict cuts it (block by block on a tripāṭha or pañcapāṭha page) and each row is read
  on its own, as predict and Manuscribe read; so a score here can differ by a little from a training log's, which
  reads the same lines in batches.
- The rows are paired with the truth's lines block by block. Where a block has as many rows as the truth has lines,
  they pair in order, as training pairs them (and every score is then comparable with training's); otherwise by the
  closest readings, and the rows and lines left over are shown unpaired and not scored.
- Spaces are taken out of both before comparing, as in training. A line left empty in the truth (an ornament) is shown
  but not scored. CER = edits to turn the reading into the truth / characters of the truth, over the paired lines.
- The differences are marked akṣara by akṣara and sorted into kinds: a different consonant or conjunct, the same
  consonants with other vowel signs or anusvāra, an error at either end of a line, a daṇḍa or numeral.
- Errors per character in red ink against black, where a page has red ink (the colour under each character, from
  where along the line the model read it).
- Hints for the truth: two vowel signs in a row, a virama after a vowel sign, a Latin letter or digit; and a numeral
  that breaks the page's count or verse numbering where the model read the number that would continue it.

Output: data/<ms>/pagediff/<page>_<model>.html (--out to choose), one file with the images inside; --open shows it.
"""
import argparse
import base64
import collections
import html
import io
import re
import tempfile
import warnings
import webbrowser
from pathlib import Path

import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein

from . import pagetext
from .config import DATA, MANUSCRIPTS, ms_dir, not_text
from .config import SEG as DEFAULT_SEG

# ------------------------------------------------------------------ manuscript, pages, models


def manuscript_of(name):
    """LDII_9851, LDII_9851.pdf or a path to the PDF -> the manuscript's id"""
    ms = Path(name).name
    if ms.lower().endswith(".pdf"):
        ms = ms[:-4]
    if ms not in MANUSCRIPTS:
        raise SystemExit(f"{name}: no manuscript of that name in the corpus (its page images in a folder of that name, "
                         f"beside the PDF)")
    return ms


def page_images(ms):
    return sorted(p.name for p in MANUSCRIPTS[ms]["images"].glob("*.jpg"))


def pdf_page_count(ms):
    """the PDF's page count, or None: no PDF beside the images, or pypdf not installed"""
    pdf = MANUSCRIPTS[ms]["images"].parent / f"{ms}.pdf"
    if not pdf.exists():
        return None
    try:
        import pypdf
    except ImportError:
        return None
    return len(pypdf.PdfReader(pdf).pages)


def resolve_page(ms, spec, images):
    """'132' (PDF page), '66_1' (folio, side) or '009851_00066_1' -> (image name, its PDF page number)"""
    s = spec[:-4] if spec.endswith(".jpg") else spec
    if re.fullmatch(r"\d+", s):
        n = int(s)
        if not 1 <= n <= len(images):
            raise SystemExit(f"page {n}: the PDF's pages are 1 to {len(images)}")
        return images[n - 1], n
    m = re.fullmatch(r"(\d+)_(\d+)", s)
    name = MANUSCRIPTS[ms]["image_name"](f"{int(m.group(1)):04d}_{m.group(2)}") if m else s + ".jpg"
    if name not in images:
        raise SystemExit(f"{spec}: no page image {name} in {MANUSCRIPTS[ms]['images']}")
    return name, images.index(name) + 1


def folio_side(stem):
    *_, folio, side = stem.split("_")
    return int(folio), side


def model_path(spec):
    p = Path(spec)
    if p.is_file():
        return p
    if p.is_dir() and (p / "best.pt").is_file():
        return p / "best.pt"
    owner, _, run = spec.rpartition("/")
    hits = sorted(DATA.glob(f"{owner or '*'}/models/{run}/best.pt"))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(f"{spec}: no such model file, and no data/{owner or '*'}/models/{run}/best.pt")
    raise SystemExit(f"{spec}: several runs have that name ({', '.join(str(h.relative_to(DATA)) for h in hits)}); "
                     f"say whose, as {hits[0].parent.parent.parent.name}/{run}")


def short_name(path):
    path = Path(path)
    return path.parent.name if path.name == "best.pt" else path.stem


# ------------------------------------------------------------------ akṣaras and their alignment

from .aksara import MARK, VIRAMA, aksharas, errors as aksara_errors   # noqa: E402  (one akṣara for all of msocr)

R = lambda a, b: {chr(c) for c in range(a, b + 1)}                       # noqa: E731
MATRA = R(0x93E, 0x94C) | R(0x962, 0x963)
DIGIT = {chr(0x966 + k): k for k in range(10)}
PUNCT = set("।॥") | set(DIGIT) | set("0123456789.,-|")


def align(truth, read):
    """-> (truth's akṣaras, reading's akṣaras, opcodes (tag, i1, i2, j1, j2) turning the one into the other)"""
    a, b = aksharas(truth), aksharas(read)
    vocab = {}
    enc = lambda xs: "".join(chr(0xF0000 + vocab.setdefault(x, len(vocab))) for x in xs)     # noqa: E731
    return a, b, [(o.tag, o.src_start, o.src_end, o.dest_start, o.dest_end) for o in Levenshtein.opcodes(enc(a), enc(b))]


KINDS = {"letters": "letters", "marks": "vowel sign / anusvāra", "edge": "first or last akṣara", "punct": "daṇḍa or numeral"}


def kind(tag, a, b, i1, i2, j1, j2):
    ts, rs = "".join(a[i1:i2]), "".join(b[j1:j2])
    if set(ts + rs) <= PUNCT:
        return "punct"
    if (i1 == 0 and j1 == 0) or (i2 == len(a) and j2 == len(b)):
        return "edge"
    skeleton = lambda t: "".join(c for c in t if c not in MARK)            # noqa: E731
    if tag == "replace" and i2 - i1 == j2 - j1 and all(skeleton(x) == skeleton(y) for x, y in zip(a[i1:i2], b[j1:j2])):
        return "marks"
    return "letters"


def pair(got, want, gap=0.75):
    """the rows' readings and the truth's lines of one block -> [(row index or None, line index or None)]: in order
    when the counts agree, as training pairs them; else the alignment of least cost (a pair costs its normalised edit
    distance, a row or line left unpaired `gap`)"""
    n, m = len(got), len(want)
    if n == m:
        return [(i, i) for i in range(n)]
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    back = {}
    for i in range(n + 1):
        for j in range(m + 1):
            if i and j and D[i - 1, j - 1] + Levenshtein.normalized_distance(got[i - 1], want[j - 1]) < D[i, j]:
                D[i, j], back[i, j] = D[i - 1, j - 1] + Levenshtein.normalized_distance(got[i - 1], want[j - 1]), (i - 1, j - 1)
            if i and D[i - 1, j] + gap < D[i, j]:
                D[i, j], back[i, j] = D[i - 1, j] + gap, (i - 1, j)
            if j and D[i, j - 1] + gap < D[i, j]:
                D[i, j], back[i, j] = D[i, j - 1] + gap, (i, j - 1)
    out, (i, j) = [], (n, m)
    while (i, j) != (0, 0):
        pi, pj = back[i, j]
        out.append((i - 1 if pi == i - 1 else None, j - 1 if pj == j - 1 else None))
        i, j = pi, pj
    return out[::-1]


# ------------------------------------------------------------------ hints for the truth

ODD = [(re.compile(f"[{''.join(sorted(MATRA))}]{{2}}"), "two vowel signs in a row"),
       (re.compile(f"[{''.join(sorted(MATRA))}]{VIRAMA}"), "a virama after a vowel sign"),
       (re.compile("[A-Za-z0-9]"), "a Latin letter or digit")]


def oddities(text):
    return [f"{why}: “{text[max(0, m.start() - 3):m.end() + 3]}”" for rx, why in ODD for m in rx.finditer(text)]


def numbers(a):
    """runs of numerals in a line's akṣaras -> [(start, end, value, a verse number?)]; a verse number stands after a
    double daṇḍa (or at the line's start) and before one (or at the line's end), as ॥ ७४ ॥"""
    out, i = [], 0
    while i < len(a):
        if a[i] not in DIGIT:
            i += 1
            continue
        j = i
        while j < len(a) and a[j] in DIGIT:
            j += 1
        before = a[i - 1] == "॥" if i else None
        after = a[j] == "॥" if j < len(a) else None
        verse = (before is True and after is not False) or (before is None and after is True)
        out.append((i, j, int("".join(str(DIGIT[c]) for c in a[i:j])), verse))
        i = j
    return out


def read_at(ops, b, i, j):
    """the number the model read where the truth has akṣaras i..j, or None"""
    got = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "insert":
            if i < i1 < j:
                got += b[j1:j2]
        elif i1 < j and i2 > i:
            got += b[j1 + max(i1, i) - i1:j1 + min(i2, j) - i1] if tag == "equal" else b[j1:j2]
    digits = "".join(str(DIGIT[c]) for t in got for c in t if c in DIGIT)
    return int(digits) if digits else None


def deva(n):
    return "".join(chr(0x966 + int(c)) for c in str(n))


def numbering_hints(lines, model):
    """numerals of the truth that break the page's numbering where the model read the number that continues it:
    a verse number one more than the verse before; a count one more than the count before it, or 1 (a new entry)"""
    seq = []
    for ln in lines:
        if ln.get("truth"):
            for i, j, v, verse in numbers(ln["a"]):
                seq.append((ln, i, j, v, verse))
    hints = collections.defaultdict(list)
    for k, (ln, i, j, v, verse) in enumerate(seq):
        r = read_at(ln["ops"], ln["b"], i, j)
        if r is None or r == v:
            continue
        if verse:
            prev = [x[3] for x in seq[:k] if x[4]]
            nxt = [x[3] for x in seq[k + 1:] if x[4]]
            want = {prev[-1] + 1} if prev else {nxt[0] - 1} if nxt else set()
            why = (f"verse ॥ {deva(prev[-1])} ॥ comes before it" if prev else f"verse ॥ {deva(nxt[0])} ॥ comes after it") if want else ""
        else:
            prev = None
            for x in reversed(seq[:k]):
                if x[4]:
                    break
                prev = x[3]
                break
            want = {prev + 1, 1} if prev is not None else set()
            why = f"the count before it is {deva(prev)}" if want else ""
        if r in want and v not in want:
            hints[id(ln)].append(f"{deva(v)} → {deva(r)}? {why}; {model} read {deva(r)}")
    return hints


# ------------------------------------------------------------------ reading a page


def read_rows(ms, image, seg, net, chars, tmp, lm=None):
    """the page cut into rows as predict cuts it and each row read -> ({box: [(baseline, outline)]}, {box: rect},
    {box: [{file, text, conf}]}, size, note, warning)"""
    from .blocks import blocks_for_page, layout_of
    from .predict import _read_rows, rows_for_page
    from .segment import ensure_segmented, reading_from
    ensure_segmented(ms, [image], seg)
    stem = image.rsplit(".", 1)[0]
    with reading_from(seg):
        if layout_of(ms) in ("tripatha", "panchapatha"):
            blocks, size, note = blocks_for_page(ms, image)
            geo = {b: v["rows"] for b, v in blocks.items() if v["rows"]}
            rects = {b: v.get("rect") for b, v in blocks.items() if v["rows"]}
            warn = "; ".join(f"{b}: {v['warn']}" for b, v in blocks.items() if v.get("warn")) or None
        else:
            merged, size, note, warn = rows_for_page(ms, image)
            geo, rects = ({"main": merged} if merged else {}), {}
        order = lambda b: pagetext.BOXES.index(b) if b in pagetext.BOXES else 99       # noqa: E731
        geo = {b: geo[b] for b in sorted(geo, key=order)}
        reads = {b: _read_rows(ms, image, size, rows, net, chars, tmp, stem, b, lm) for b, rows in geo.items()}
    return geo, rects, reads, size, note, warn


def read_pos(net, chars, crop):
    """the greedy reading with where along the line (crop pixels) each character was read"""
    import torch
    from .recognizer import to_tensor
    with torch.no_grad():
        lg = net(to_tensor(crop)[None, None])[0]
    out, pos, prev = [], [], 0
    for f, v in enumerate(lg.argmax(-1).tolist()):
        if v != prev and v != 0:
            out.append(chars[v - 1])
            pos.append((f + 0.5) * crop.size[0] / lg.shape[0])
        prev = v
    return "".join(out), pos


def ink_errors(net, chars, lines):
    """character errors by the colour of the ink they fall on -> {red|black: [errors, characters read]}"""
    out = {"red": [0, 0], "black": [0, 0]}
    for ln in lines:
        if not ln.get("truth") or ln.get("crop") is None:
            continue
        crop = ln["crop"].convert("RGB")
        arr = np.asarray(crop).astype(int)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        ink = lum < 0.72 * np.median(lum)
        red = ink & (r - np.maximum(g, b) > 45)
        read, pos = read_pos(net, chars, crop)
        if not read:
            continue
        half = crop.size[0] / len(read) * 0.6
        col = []
        for x in pos:
            x0, x1 = max(0, int(x - half)), min(arr.shape[1], int(x + half) + 1)
            n = ink[:, x0:x1].sum()
            col.append("red" if n and red[:, x0:x1].sum() / n > 0.5 else "black")
        for c in col:
            out[c][1] += 1
        for op in Levenshtein.editops(ln["truth"], read):
            out[col[min(op.dest_pos, len(read) - 1)]][0] += 1
    return out


def page_result(ms, image, pdf_page, gt_path, seg, model, other, lm=None):
    """one page read and set against its truth"""
    from .predict import read_line
    net, chars, name = model
    stem = image.rsplit(".", 1)[0]
    truth = pagetext.read(gt_path)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        geo, rects, reads, size, note, warn = read_rows(ms, image, seg, net, chars, tmp, lm)
        lines, problems = [], []
        for box in [b for b in pagetext.BOXES if b in geo or b in truth] + [b for b in truth if b not in pagetext.BOXES]:
            got = reads.get(box, [])
            want = [x.strip().replace(" ", "") for x in truth.get(box, [])]
            while want and not want[-1]:
                want.pop()
            if len(got) != len(want):
                where = "" if box == "main" and len(geo) <= 1 and len(truth) <= 1 else f" in the {box} block"
                problems.append(f"{len(got)} rows found but {len(want)} lines in the truth{where}: paired by the closest "
                                f"readings, the rest shown unpaired and not scored")
            for i, j in pair([g["text"] for g in got], want):
                ln = {"box": box, "row": None if i is None else i + 1, "line": None if j is None else j + 1}
                if i is not None:
                    crop = Image.open(tmp / got[i]["file"])
                    crop.load()
                    base, poly = geo[box][i]
                    ln.update(read=got[i]["text"], conf=got[i]["conf"], crop=crop,
                              base=[list(map(float, p)) for p in base], poly=[list(map(float, p)) for p in poly])
                    if other:
                        ln["other"] = read_line(other[0], other[1], crop, lm)[0]
                if j is not None:
                    ln["truth"] = want[j]
                lines.append(ln)
    errors = collections.Counter()
    for ln in lines:
        if ln.get("truth") and "read" in ln:
            ln["a"], ln["b"], ln["ops"] = align(ln["truth"], ln["read"])
            ln["dist"], ln["len"] = Levenshtein.distance(ln["read"], ln["truth"]), len(ln["truth"])
            ln["adist"], ln["alen"] = aksara_errors(ln["read"], ln["truth"])
            ln["cer"] = 100 * ln["dist"] / ln["len"]
            if other:
                ln["dist_o"] = Levenshtein.distance(ln["other"], ln["truth"])
            ln["changes"] = []
            for k, (tag, i1, i2, j1, j2) in enumerate(ln["ops"]):
                if tag != "equal":
                    kd = kind(tag, ln["a"], ln["b"], i1, i2, j1, j2)
                    errors[kd] += max(i2 - i1, j2 - j1)
                    ln["changes"].append({"k": k, "t": "".join(ln["a"][i1:i2]), "r": "".join(ln["b"][j1:j2]), "kind": kd})
        elif ln.get("truth"):
            ln["a"] = aksharas(ln["truth"])
            ln["ops"], ln["b"] = [], []
    hints = numbering_hints(lines, name)
    for ln in lines:
        ln["hints"] = (oddities(ln["truth"]) if ln.get("truth") else []) + hints.get(id(ln), [])
    scored = [ln for ln in lines if "dist" in ln]
    chars_ = sum(ln["len"] for ln in scored)
    res = {"ms": ms, "image": image, "stem": stem, "pdf_page": pdf_page, "gt": str(gt_path), "size": size, "note": note,
           "warn": warn, "problems": problems, "lines": lines, "rects": rects, "kinds": dict(errors),
           "cer": 100 * sum(ln["dist"] for ln in scored) / max(1, chars_), "chars": chars_, "n_scored": len(scored),
           "ger": 100 * sum(ln["adist"] for ln in scored) / max(1, sum(ln["alen"] for ln in scored)),
           "aksaras": sum(ln["alen"] for ln in scored),
           "unpaired_rows": sum(1 for ln in lines if "read" in ln and not ln.get("truth") and ln.get("line") is None),
           "unpaired_lines": [ln for ln in lines if ln.get("truth") and "read" not in ln],
           "ink": ink_errors(net, chars, lines), "not_text": stem in not_text(ms)}
    if other:
        res["cer_o"] = 100 * sum(ln["dist_o"] for ln in scored) / max(1, chars_)
        res["ger_o"] = 100 * sum(aksara_errors(ln["other"], ln["truth"])[0] for ln in scored) / max(1, res["aksaras"])
        res["ink_o"] = ink_errors(other[0], other[1], lines)
    return res


# ------------------------------------------------------------------ html


def esc(s):
    return html.escape(str(s), quote=True)


def data_uri(im, fmt="PNG", max_w=None):
    im = im.convert("RGB")
    if max_w and im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, fmt, **({"quality": 85} if fmt == "JPEG" else {}))
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def tone(c):
    return "good" if c <= 2 else "fair" if c <= 8 else "poor"


def diff_rows(ln):
    a, b = ln["a"], ln["b"]
    t, r = [], []
    for k, (tag, i1, i2, j1, j2) in enumerate(ln["ops"]):
        ts, rs = "".join(a[i1:i2]), "".join(b[j1:j2])
        if tag == "equal":
            t.append(esc(ts))
            r.append(esc(rs))
            continue
        cls = {"replace": "d-sub", "delete": "d-miss", "insert": "d-extra"}[tag]
        t.append(f'<mark class="{cls}" data-k="{k}">{esc(ts)}</mark>' if ts else f'<span class="gap" data-k="{k}"></span>')
        r.append(f'<mark class="{cls}" data-k="{k}">{esc(rs)}</mark>' if rs else f'<span class="gap" data-k="{k}"></span>')
    return "".join(t), "".join(r)


def label(ln, of="row"):
    n = ln[of]
    return str(n) if ln["box"] == "main" else f"{ln['box'][0].upper()}{n}"


def pct(eb):
    return f"<b>{100 * eb[0] / max(1, eb[1]):.1f} %</b>"


def page_html(results, meta):
    name, oname = meta["model"], meta.get("other")
    cards, sections, hint_items = [], [], []
    top = collections.Counter()
    for p in results:
        pid = p["stem"]
        folio, side = folio_side(p["stem"])
        errs = sum(p["kinds"].values())
        bar = "".join(f'<span class="seg k-{k}" style="flex:{p["kinds"][k]}" title="{KINDS[k]}: {p["kinds"][k]}"></span>'
                      for k in KINDS if p["kinds"].get(k))
        legend = "".join(f'<li><i class="sw k-{k}"></i>{KINDS[k]} <b>{p["kinds"].get(k, 0)}</b></li>' for k in KINDS)
        ink, red_share = p["ink"], p["ink"]["red"][1] / max(1, p["ink"]["red"][1] + p["ink"]["black"][1])
        ink_line = ""
        if red_share >= 0.03:
            ink_line = (f'<div class="sub">Errors per character in <b class="redink">red ink</b> {pct(ink["red"])}, in black '
                        f'{pct(ink["black"])}' + (f' ({esc(oname)}: {pct(p["ink_o"]["red"])} and {pct(p["ink_o"]["black"])})' if oname else "")
                        + f'; {100 * red_share:.0f} % of the characters are red.</div>')
        before = (f'<span class="was" title="{esc(oname)}">{p["cer_o"]:.2f}</span><span class="arrow">→</span>' if oname else "")
        warn = "".join(f'<div class="sub warnline">{esc(x)}</div>' for x in p["problems"])
        if p["unpaired_lines"]:
            n = sum(len(ln["truth"]) for ln in p["unpaired_lines"])
            warn += f'<div class="sub warnline">{len(p["unpaired_lines"])} lines of the truth ({n} characters) have no row: not scored.</div>'
        if p["not_text"]:
            warn += '<div class="sub warnline">This page is listed as not a text page (config.not_text).</div>'
        cards.append(f"""
<a class="card" href="#{pid}">
  <div class="card-head"><span class="ms">{esc(p['ms'])} · PDF page {p['pdf_page']}</span><span class="pg">folio {folio}, side {esc(side)}</span></div>
  <div class="big">{before}<span class="now">{p['cer']:.2f}<small>% CER{' · ' + esc(name) if oname else ''}</small></span></div>
  <div class="ger">{(f'<span class="was">{p["ger_o"]:.2f}</span><span class="arrow">→</span>' if oname else '')}<b>{p['ger']:.2f}</b><small>% GER, in akṣaras</small></div>
  <div class="sub">{p['n_scored']} lines scored · {p['chars']} characters, {p['aksaras']} akṣaras · {errs} akṣaras read wrong, missed or added</div>
  <div class="bar">{bar}</div>
  <ul class="legend">{legend}</ul>
  {ink_line}{warn}
</a>""")
        shapes, items = [], []
        w, h = p["size"]
        for box, rect in p["rects"].items():
            if rect and len(rect) == 4:
                x0, y0, x1, y1 = rect
                shapes.append(f'<g class="block"><rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}"/>'
                              f'<text x="{x0 + 6}" y="{y0 + 24}">{esc(box)}</text></g>')
        for n, ln in enumerate(p["lines"]):
            lid = f"{pid}-{n}"
            scored = "dist" in ln
            t_ = tone(ln["cer"]) if scored else "none"
            if "poly" in ln:
                pts = " ".join(f"{x:.0f},{y:.0f}" for x, y in ln["poly"])
                bl = " ".join(f"{x:.0f},{y:.0f}" for x, y in ln["base"])
                x0, y0 = ln["base"][0]
                shapes.append(f'<g class="row {t_}" data-n="{n}"><polygon points="{pts}"/><polyline points="{bl}"/>'
                              f'<text x="{x0 - 10:.0f}" y="{y0:.0f}">{label(ln)}</text></g>')
            for c in ln.get("changes", []):
                top[(c["t"] or "∅", c["r"] or "∅")] += 1
            hints = "".join(f'<div class="odd">Check the truth: {esc(x)}</div>' for x in ln["hints"])
            for x in ln["hints"]:
                where = f"row {label(ln)}" if ln.get("row") else f"line {label(ln, 'line')}"
                hint_items.append(f'<li><a href="#{lid}">{esc(p["stem"])} {where}</a>: {esc(x)}</li>')
            if scored:
                t, r = diff_rows(ln)
                ch = " ".join(f'<span class="chg k-{c["kind"]}" data-k="{c["k"]}" title="{KINDS[c["kind"]]}">'
                              f'{esc(c["t"]) or "∅"}<i>→</i>{esc(c["r"]) or "∅"}</span>' for c in ln["changes"])
                other = (f'<div class="txt oth"><span class="lab">{esc(oname)}</span><span class="deva">{esc(ln["other"])}</span></div>'
                         if oname else "")
                head = (f'<span class="pill {t_}">{ln["cer"]:.1f} %</span><span class="dim">{ln["dist"]} of {ln["len"]} characters, '
                        f'{ln["adist"]} of {ln["alen"]} akṣaras · confidence {ln["conf"]:.0f}' + (f' · {esc(oname)} {100 * ln["dist_o"] / ln["len"]:.1f} %' if oname else "") + "</span>")
                body = (f'<div class="txt"><span class="lab">truth</span><span class="deva">{t}</span></div>'
                        f'<div class="txt"><span class="lab">{esc(name)}</span><span class="deva">{r}</span></div>{other}'
                        + (f'<div class="chgs">{ch}</div>' if ch else ""))
            elif "read" in ln:
                head = ('<span class="pill none">not scored</span><span class="dim">'
                        + ("the truth leaves this line empty" if ln.get("line") else "no line of the truth was paired with this row") + "</span>")
                body = f'<div class="txt"><span class="lab">{esc(name)}</span><span class="deva">{esc(ln["read"])}</span></div>'
            else:
                head = '<span class="pill poor">no row</span><span class="dim">no row of the page was paired with this line of the truth: not scored</span>'
                body = f'<div class="txt"><span class="lab">truth</span><span class="deva">{esc(ln["truth"])}</span></div>'
            crop = f'<img class="crop" src="{data_uri(ln["crop"])}" alt="row {label(ln)}">' if "crop" in ln else ""
            no = label(ln) if ln.get("row") else "–"
            items.append(f"""
<li class="line {t_}{'' if ln.get('changes') or not scored else ' clean'}" id="{lid}" data-n="{n}">
  <div class="meta"><span class="no">{no}</span>{head}</div>
  {crop}{body}{hints}
</li>""")
        rowwarn = f'<p class="warnbox">Row finder: {esc(p["warn"])}</p>' if p["warn"] else ""
        sections.append(f"""
<section class="pg-sec" id="{pid}">
  <h2>{esc(p['ms'])} · PDF page {p['pdf_page']} <span class="dim">· {esc(p['stem'])} · {p['cer']:.2f} % CER, {p['ger']:.2f} % GER{f" ({esc(oname)} {p['cer_o']:.2f} %, {p['ger_o']:.2f} %)" if oname else ""}</span></h2>
  <p class="dim small">Rows from {esc(meta['seg'])}: {esc(p['note'] or '')}. Ground truth: {esc(p['gt'])}. Each outline is the band
  cut out and read; click one to go to its line.</p>{rowwarn}
  <figure class="page"><div class="pagewrap" style="aspect-ratio:{w}/{h}">
    <img src="{data_uri(Image.open(MANUSCRIPTS[p['ms']]['images'] / p['image']), 'JPEG', 2000)}" alt="{esc(p['stem'])}">
    <svg viewBox="0 0 {w} {h}" preserveAspectRatio="none">{''.join(shapes)}</svg></div></figure>
  <ol class="lines">{''.join(items)}</ol>
</section>""")
    chips = "".join(f'<span class="chg"><span class="deva sm">{esc(t)}<i>→</i>{esc(r)}</span> <b>×{n}</b></span>'
                    for (t, r), n in top.most_common() if n >= 2)
    hints = (f'<h3>Worth checking in the ground truth</h3><p class="dim small">Found by rule, not judged: a sequence of '
             f'letters that is rarely right, or a numeral that breaks the page\'s numbering where {esc(name)} read the '
             f'number that continues it. Look at the line image before changing anything.</p>'
             f'<ul class="checks">{"".join(hint_items)}</ul>') if hint_items else ""
    info = (f'<dl class="info"><dt>manuscript</dt><dd>{esc(meta["ms"])}{" · PDF of " + str(meta["pdf_pages"]) + " pages" if meta["pdf_pages"] else ""}</dd>'
            f'<dt>segmentation</dt><dd>{esc(meta["seg"])}</dd><dt>model</dt><dd>{esc(meta["model_label"])}</dd>'
            + (f'<dt>compared with</dt><dd>{esc(meta["other_label"])}</dd>' if oname else "")
            + f'<dt>made</dt><dd>{esc(meta["when"])} by msocr.pagediff</dd></dl>')
    pages = (f"PDF page {results[0]['pdf_page']}" if len(results) == 1 else f"{len(results)} pages")
    title = f"{meta['ms']} {pages} · {name}"
    toggles = ('<label><input type="checkbox" id="oe"> only lines with mistakes</label>'
               + (f'<label><input type="checkbox" id="oth"> also show {esc(oname)}\'s reading</label>' if oname else "")
               + ('<span class="dim">Go to:</span> ' + " · ".join(f'<a href="#{p["stem"]}">PDF page {p["pdf_page"]}</a>' for p in results)
                  if len(results) > 1 else ""))
    return (TEMPLATE.replace("{{TITLE}}", esc(title)).replace("{{H1}}", esc(f"{meta['ms']} {pages}, read by {name}"))
            .replace("{{INFO}}", info).replace("{{CARDS}}", "".join(cards)).replace("{{HINTS}}", hints)
            .replace("{{CHIPS}}", f'<h3>Mistakes made more than once</h3><div class="chgs">{chips}</div>' if chips else "")
            .replace("{{TOGGLES}}", toggles).replace("{{SECTIONS}}", "".join(sections)).replace("{{MODEL}}", esc(name)))


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{TITLE}}</title>
<style>
:root{--bg:#f6f3ec;--panel:#fffdf8;--ink:#23211c;--dim:#6d675b;--line:#ded7c7;--accent:#8a4b16;
 --good:#2f7d4f;--fair:#b7791f;--poor:#b33a2b;--none:#8a8478;--sub:#f6c96b55;--subb:#c98a12;--miss:#e5675655;--missb:#b33a2b;--extra:#5b8def44;--extrab:#3466c2;
 --k-letters:#b33a2b;--k-marks:#c98a12;--k-edge:#7a5cc4;--k-punct:#7d8a96}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#16150f;--panel:#201e18;--ink:#ece6d8;--dim:#a39b8b;--line:#3a362c;--accent:#e0a060;
 --sub:#f6c96b40;--miss:#e5675650;--extra:#5b8def50}}
:root[data-theme="dark"]{--bg:#16150f;--panel:#201e18;--ink:#ece6d8;--dim:#a39b8b;--line:#3a362c;--accent:#e0a060;--sub:#f6c96b40;--miss:#e5675650;--extra:#5b8def50}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:20px;margin:40px 0 6px} h3{font-size:16px;margin:28px 0 8px}
.dim{color:var(--dim);font-weight:400} .small{font-size:13px}
.info{display:grid;grid-template-columns:max-content 1fr;gap:2px 14px;font-size:13.5px;margin:8px 0 18px}.info dt{color:var(--dim)}.info dd{margin:0}
.deva{font-family:"Tiro Devanagari Sanskrit","Kohinoor Devanagari","Devanagari Sangam MN","Noto Sans Devanagari",serif;font-size:21px;line-height:1.75}
.deva.sm{font-size:17px;line-height:1.4}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.card{display:block;text-decoration:none;color:inherit;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card:hover{border-color:var(--accent)}
.card-head{display:flex;justify-content:space-between;gap:8px;font-weight:600}.pg{color:var(--dim);font-weight:400}
.big{font-size:30px;font-weight:650;margin:6px 0 2px;font-variant-numeric:tabular-nums}.was{color:var(--dim);font-size:20px;text-decoration:line-through}
.arrow{color:var(--dim);font-size:18px;margin:0 6px}.big small{font-size:13px;color:var(--dim);font-weight:400;margin-left:4px}
.ger{font-size:18px;font-weight:600;font-variant-numeric:tabular-nums;margin:0 0 2px}.ger .was{font-size:14px}.ger small{font-size:12.5px;color:var(--dim);font-weight:400;margin-left:4px}
.sub{font-size:13px;color:var(--dim);margin:4px 0}.sub b{color:var(--ink)}.warnline{color:var(--poor)}.redink{color:#c0392b !important}
.bar{display:flex;height:10px;border-radius:5px;overflow:hidden;margin:10px 0 6px;gap:2px}.seg{display:block}
.k-letters{background:var(--k-letters)}.k-marks{background:var(--k-marks)}.k-edge{background:var(--k-edge)}.k-punct{background:var(--k-punct)}
.legend{list-style:none;padding:0;margin:0;display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12.5px;color:var(--dim)}
.legend b{color:var(--ink);margin-left:3px}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.howto{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:16px 0;font-size:14px}.howto p{margin:4px 0}
mark{color:inherit;border-radius:3px;padding:0 1px}
mark.d-sub{background:var(--sub);box-shadow:inset 0 -2px var(--subb)}mark.d-miss{background:var(--miss);box-shadow:inset 0 -2px var(--missb)}
mark.d-extra{background:var(--extra);box-shadow:inset 0 -2px var(--extrab)}
.gap{display:inline-block;width:3px;height:1.1em;vertical-align:-3px;margin:0 2px;background:var(--dim);border-radius:1px}
mark.hot,.gap.hot,.chg.hot{outline:2px solid var(--accent);outline-offset:1px}
.checks{margin:6px 0 10px;padding-left:20px;max-width:62em}.checks li{margin:4px 0}.checks a{color:var(--accent)}
.page{margin:12px 0 18px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px}
.pagewrap{position:relative;width:100%}
.pagewrap img,.pagewrap svg{position:absolute;inset:0;width:100%;height:100%;display:block}
.block rect{fill:none;stroke:#3466c2;stroke-width:2;stroke-dasharray:10 6}.block text{font:600 20px system-ui;fill:#3466c2;paint-order:stroke;stroke:#fff;stroke-width:4px}
.row polygon{fill:transparent;stroke-width:2.5;cursor:pointer}.row polyline{fill:none;stroke-width:1.5;stroke-dasharray:6 5;opacity:.8}
.row text{font:700 22px system-ui;text-anchor:end;dominant-baseline:middle;paint-order:stroke;stroke:#fff;stroke-width:4px}
.row.good polygon,.row.good polyline{stroke:var(--good)}.row.good text{fill:var(--good)}
.row.fair polygon,.row.fair polyline{stroke:var(--fair)}.row.fair text{fill:var(--fair)}
.row.poor polygon,.row.poor polyline{stroke:var(--poor)}.row.poor text{fill:var(--poor)}
.row.none polygon,.row.none polyline{stroke:var(--none);stroke-dasharray:4 4}.row.none text{fill:var(--none)}
.row.hot polygon{fill:#f5a62333;stroke-width:4}
.lines{list-style:none;padding:0;margin:0;display:grid;gap:10px}
.line{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:8px;padding:8px 12px;scroll-margin-top:52px}
.line.good{border-left-color:var(--good)}.line.fair{border-left-color:var(--fair)}.line.poor{border-left-color:var(--poor)}.line.none{border-left-color:var(--none)}
.line.hot{border-color:var(--accent)}
.meta{display:flex;gap:10px;align-items:center;font-size:13px;margin-bottom:4px;flex-wrap:wrap}.no{font-weight:700;font-size:15px;min-width:1.6em}
.pill{border-radius:999px;padding:0 8px;font-weight:600;color:#fff;font-variant-numeric:tabular-nums}
.pill.good{background:var(--good)}.pill.fair{background:var(--fair)}.pill.poor{background:var(--poor)}.pill.none{background:var(--none)}
.crop{display:block;width:100%;height:auto;border-radius:4px;background:#fff;margin:2px 0 4px}
.txt{display:flex;gap:10px;align-items:baseline}.lab{flex:none;width:4.2em;font-size:12px;color:var(--dim);text-transform:uppercase;letter-spacing:.04em;overflow-wrap:anywhere}
.txt .deva{overflow-wrap:anywhere}
.oth{display:none}body.show-oth .oth{display:flex}.oth .deva{color:var(--dim)}
body.only-errors .line.clean{display:none}
.chgs{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 2px}
.chg{font-family:"Tiro Devanagari Sanskrit","Kohinoor Devanagari","Devanagari Sangam MN",serif;font-size:17px;border-radius:6px;padding:0 8px;
 background:var(--bg);border:1px solid var(--line);border-left-width:4px;cursor:default}
.chg i{font-style:normal;color:var(--dim);margin:0 4px;font-family:system-ui}.chg b{font-family:system-ui;font-size:13px;color:var(--dim);margin-left:2px}
.chg.k-letters{border-left-color:var(--k-letters);background:var(--bg)}.chg.k-marks{border-left-color:var(--k-marks);background:var(--bg)}
.chg.k-edge{border-left-color:var(--k-edge);background:var(--bg)}.chg.k-punct{border-left-color:var(--k-punct);background:var(--bg)}
.odd{font-size:13px;color:var(--accent);margin-top:4px}
.warnbox{background:var(--miss);padding:6px 10px;border-radius:6px}
.toolbar{position:sticky;top:0;z-index:5;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);display:flex;gap:18px;flex-wrap:wrap;font-size:14px;margin-top:18px}
.toolbar a{color:var(--accent)}
@media (max-width:600px){.deva{font-size:18px}.big{font-size:24px}.lab{width:3.2em}}
</style></head>
<body>
<main>
<h1>{{H1}}</h1>
{{INFO}}
<div class="cards">{{CARDS}}</div>
<div class="howto">
<p><mark class="d-sub">orange</mark> read as something else · <mark class="d-miss">red</mark> in the truth but not read ·
<mark class="d-extra">blue</mark> read but not in the truth · <span class="gap"></span> where the other line has the missing or
extra akṣara. Hover over a marked akṣara or a change to see its partner. Spaces are taken out of both before comparing,
as in training; CER is edits per 100 characters of the truth, GER per 100 akṣaras (an akṣara read wrong in any part
is one error; msocr/aksara.py).</p>
<p>Kinds: <b>letters</b>, a different consonant or conjunct; <b>vowel sign / anusvāra</b>, the same consonants with other
marks; <b>first or last akṣara</b>, an error at either end of a line, where a crop may cut into the letters;
<b>daṇḍa or numeral</b>.</p>
</div>
{{HINTS}}
{{CHIPS}}
<div class="toolbar">{{TOGGLES}}</div>
{{SECTIONS}}
</main>
<script>
const $ = (s, r = document) => [...r.querySelectorAll(s)];
const box = (id, cls) => { const e = document.getElementById(id); if (e) e.onchange = () => document.body.classList.toggle(cls, e.checked); };
box('oe', 'only-errors'); box('oth', 'show-oth');
for (const sec of $('.pg-sec')) {
  const rowOf = (n) => sec.querySelector(`.row[data-n="${n}"]`), lineOf = (n) => sec.querySelector(`.line[data-n="${n}"]`);
  for (const g of $('.row', sec)) {
    g.addEventListener('click', () => { const l = lineOf(g.dataset.n); l.scrollIntoView({behavior: 'smooth', block: 'center'}); l.classList.add('hot'); setTimeout(() => l.classList.remove('hot'), 1400); });
    g.addEventListener('mouseenter', () => g.classList.add('hot')); g.addEventListener('mouseleave', () => g.classList.remove('hot'));
  }
  for (const l of $('.line', sec)) {
    l.addEventListener('mouseenter', () => rowOf(l.dataset.n)?.classList.add('hot'));
    l.addEventListener('mouseleave', () => rowOf(l.dataset.n)?.classList.remove('hot'));
    for (const m of $('[data-k]', l)) {
      const mates = () => $(`[data-k="${m.dataset.k}"]`, l);
      m.addEventListener('mouseenter', () => mates().forEach((x) => x.classList.add('hot')));
      m.addEventListener('mouseleave', () => mates().forEach((x) => x.classList.remove('hot')));
    }
  }
}
</script>
</body></html>"""


# ------------------------------------------------------------------ command line


def run(pdf, pages, model, gt=None, seg=None, compare=None, out=None, show=False, lm=None):
    import datetime
    from .predict import model_label
    from .recognizer import load_model
    warnings.filterwarnings("ignore")
    ms = manuscript_of(pdf)
    images = page_images(ms)
    count = pdf_page_count(ms)
    if count is not None and count != len(images):
        print(f"  ! the PDF has {count} pages but {MANUSCRIPTS[ms]['images']} has {len(images)} page images: page numbers "
              f"count the images")
    chosen = [resolve_page(ms, p, images) for p in pages]
    if gt and len(gt) != len(chosen):
        raise SystemExit(f"--gt given {len(gt)} times for {len(chosen)} pages: give it once per page, or not at all")
    gts = [Path(g) for g in gt] if gt else [ms_dir(ms) / "corrected" / f"{img[:-4]}.txt" for img, _ in chosen]
    for g in gts:
        if not g.is_file():
            raise SystemExit(f"no ground truth: {g} does not exist (--gt names the file)")
    seg = seg or DEFAULT_SEG
    mp = model_path(model)
    net, chars, ck = load_model(mp)
    name = short_name(mp)
    other, oname, olabel = None, None, None
    if compare:
        op = model_path(compare)
        onet, ochars, ock = load_model(op)
        oname = short_name(op)
        if oname == name:
            name, oname = f"{mp.parent.parent.parent.name}/{name}", f"{op.parent.parent.parent.name}/{oname}"
        other, olabel = (onet, ochars), model_label(op, ock)
    results = []
    for (image, n), g in zip(chosen, gts):
        r = page_result(ms, image, n, g, seg, (net, chars, name), other, lm)
        results.append(r)
        extra = f" ({oname} {r['cer_o']:.2f} %)" if other else ""
        unp = f"; {r['unpaired_rows']} rows and {len(r['unpaired_lines'])} lines unpaired" if r["problems"] else ""
        extra += f", GER {r['ger']:.2f} %" + (f" ({oname} {r['ger_o']:.2f} %)" if other else "")
        print(f"[pagediff] {ms} PDF page {n} = {image[:-4]}: {r['cer']:.2f} % CER{extra} over {r['n_scored']} lines{unp}; "
              f"{sum(len(ln['hints']) for ln in r['lines'])} hints for the truth")
    meta = {"ms": ms, "seg": seg, "model": name, "model_label": model_label(mp, ck) + (" + language model " + ", ".join(f"{l.name} {w:.1f}" for l, w in lm.parts) if lm else " (greedy, no language model)"), "other": oname,
            "other_label": olabel, "pdf_pages": count, "when": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
    if out is None:
        first = chosen[0][0][:-4]
        tag = first if len(chosen) == 1 else f"{first}+{len(chosen) - 1}"
        out = ms_dir(ms) / "pagediff" / f"{tag}_{name.replace('/', '-')}.html"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page_html(results, meta), encoding="utf-8")
    print(f"[pagediff] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if show:
        webbrowser.open(out.resolve().as_uri())
    return out


def main():
    ap = argparse.ArgumentParser(description="A page read by a model, set against its ground truth, as an HTML page.")
    ap.add_argument("pdf", help="the manuscript: LDII_9851, LDII_9851.pdf or the PDF's path")
    ap.add_argument("pages", nargs="+", help="PDF page number (1 = first), folio_side as 66_1, or image name 009851_00066_1")
    ap.add_argument("--model", required=True, help="recogniser: a best.pt, its folder, a run name (pool1) or owner/run (LDII_19634/ft6)")
    ap.add_argument("--gt", action="append", help="the page's ground truth (corrected/ format); once per page, in order. "
                    "Default data/<ms>/corrected/<page>.txt")
    ap.add_argument("--seg", help=f"segmentation to cut rows from: seg_torch (U-Net), seg (Kraken) ... (default {DEFAULT_SEG})")
    ap.add_argument("--compare", help="a second recogniser, shown greyed and scored alongside")
    ap.add_argument("--out", help="the HTML file (default data/<ms>/pagediff/<page>_<model>.html)")
    ap.add_argument("--open", action="store_true", help="open the page in the browser when written")
    ap.add_argument("--lm", help="a language model build (default the newest under data/lm), or 'none' to read greedily")
    a = ap.parse_args()
    from .lm import load_default
    run(a.pdf, a.pages, a.model, a.gt, a.seg, a.compare, a.out, a.open, lm=load_default(a.lm))


if __name__ == "__main__":
    main()
