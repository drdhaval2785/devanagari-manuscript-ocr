"""Muktabodha's e-text of a manuscript and the scan it was typed from, made into a manuscript the pipeline can learn from.

    .venv/bin/python -m msocr.muktabodha RTJ_4387            # from its config, data/muktabodha/RTJ_4387.json
    .venv/bin/python -c "from msocr import gt; gt.extract('RTJ_4387')"
    .venv/bin/python -m msocr.segment RTJ_4387
    .venv/bin/python -m msocr.align RTJ_4387 --seg=seg_torch
    .venv/bin/python -m msocr.muktabodha RTJ_4387 --check    # every paired page read by the recogniser: is the
                                                             # transcription really the text of that page?

What comes in
- Muktabodha's diplomatic transcription (muktabodha.org, CC BY-NC 4.0): a header between lines of #, a catalogue
  note, then the manuscript side by side. A side opens with "पृ० १अ)" (folio 1 recto; ब the verso; a bare number, a
  side the scribe numbered so; the marker may carry a prefix, "उ०म्ना० पृ० १अ)", a PDF page, "पृ० ५९अ, PDF ५६)", or
  two numbers, "पृ० ९१/९२अ)"), the text of the side follows one line per manuscript line, the first on the line of
  the marker; some e-texts give the first side with no marker at all, and label the sides by the photo they are on.
  The scribe's daṇḍas are typed | and ||; the glosses in the margins stand in (round brackets) at the place they
  belong to; * is an akṣara the transcriber could not read, ..... a stretch of them, and ? follows a doubtful
  akṣara; - at the end of a line is the transcriber's, not the scribe's; a verse number stands as it is on the page.
  A side the transcriber had no image of is one line saying so ("FOLIO NO. 42a IS NOT AVAILABLE", typed through the
  Devanagari keyboard, or "यह पेज उपलब्ध नही है"); a line that starts like a marker but is a remark ("पृ० ५६ से ५८
  तक नहीं है") is left out and reported.
- the scan (eGangotri, archive.org): a PDF with one JPEG a page, the cloth cover, the wrapper and blank sides first;
  a page may carry /Rotate; a photo may hold two leaves one above the other, or a small leaf on a dark table.

What is made (nothing the pipeline has is touched)
- <into>/manuscripts/<ms>/<number>_<folio>_<side>.jpg: the page images, named as the corpus names its pages. With
  neither crop, split nor downscaling, the PDF's own JPEG bytes (turned when the page says so). The pages before the
  first side are folio 0, sides 1, 2, ...; a page that is no side (blank, a second photo of a leaf, the back cover)
  is named after the side before it with an x. data/<ms>/not_text.txt lists them.
- <into>/manuscripts/<ms>.pdf: a link to the PDF, where pagediff looks for it (when the PDF's pages are the images).
- <into>/transcriptions/<ms>.xml: the transcription in Manuscribe's TEI, which gt.extract reads: <pb n="0001_1"/>
  a side, <lb n="1" break="no"/> a line, a gloss as <note place="margin">, * and ..... as <gap reason="illegible"/>,
  a doubtful akṣara in <unclear>, daṇḍas as । and ॥, a side not available as a comment. A gloss typed over a line
  break stays with the line it opened on; a line with nothing but a gloss is left out and reported.
- data/<ms>/muktabodha.txt: the page table (PDF page, image, side, lines, and how the side was found) and every
  irregularity of the e-text; the sides to look at before training.

Which page is which side
- In order (match null): the transcription's available sides take the PDF pages from --first-text-page one each
  (the sides the transcriber had not are not in the PDF either: the count must come out exact); or the pages
  listed in "pages" ("4-22,24,23"), in that order. Pages after the last side are the last folio's further sides.
- By reading (match "monotone" or "free"): every page is cut into rows by the U-Net and read by the recogniser, and
  the reading is set against every side of the transcription (normalised edit distance); the right side is the one
  far closer than any other, even on a hand the recogniser reads with half its letters wrong. "monotone" keeps the
  PDF's order and lets pages (blank, duplicates) and sides (not photographed) be skipped; "free" takes each page's
  closest side wherever it is (a PDF with its pages out of order). The distances of every pair are kept in
  data/<ms>/match.json; the readings in data/<ms>/match_readings.json, so a second run matches at once.

The config, data/muktabodha/<ms>.json (paths relative to training_data/; the command line overrides)
  etext, pdf, into            the e-text, the PDF, the <text>/<commentary> folder under training_data
  first_text_page, pages      the plain plan (match null): the first text page, or the text pages in order
  skip                        PDF pages that are no side
  match                       null, "monotone" or "free"
  sides_per_page              2: two leaves one above the other on a page, cut apart at the gap between their lines
  crop                        "paper": the page cut down to the leaf (a small leaf on a dark table)
  max_width                   pages wider than this are scaled down (a 600 dpi scan)
  pad                         2: each leaf set in a canvas twice its height, in the paper's colour (a leaf of 4 large
                              lines, whose lines the segmenter would otherwise see three times too tall)
  sequential_ids              the sides numbered 0001_1, 0001_2, ... in the e-text's order, whatever its labels say
                              (taken anyway when the labels cannot be trusted: a side before the first marker, a
                              label without a folio number, a folio transcribed twice)
  model                       the recogniser for the matching (default predict's)

--check, once gt.extract, segment and align have run: each paired page (data/<ms>/pagexml_<seg>/) is cut into its
rows and read with the recogniser (greedy, no language model), and its reading set against the transcription line
by line: a page is paired right when each row reads closer to its own line than to any other line of the page.
Written to data/<ms>/muktabodha_check.txt.
"""
import argparse
import collections
import io
import json
import os
import re
import statistics
import time
import unicodedata
from pathlib import Path
from xml.sax.saxutils import escape

from . import NAME, __version__
from .config import DATA, TRAINING, ms_dir

CONFIGS = DATA / "muktabodha"
DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
MARKER = re.compile(r"^(?:\S{1,14}\s+)?पृ०\s*([^)]{0,40})\)\s?")
MARKER_LIKE = re.compile(r"^(?:\S{1,14}\s+)?पृ०")
# "FOLIO NO. 42a IS NOT AVAILABLE" typed through the Devanagari keyboard (ण्Oट् आVआईḻआBḻE), or in Hindi
NOT_AVAILABLE = re.compile(r"उपलब्ध\s*नही|आVआईḻआBḻE|ण्Oट्|NOT\s+AVAILABLE", re.I)
SIDE = {"अ": 1, "ब": 2, "": 1}
ALLOWED = re.compile(r"[ऀ-ॿ ।॥*?]")
# the akṣara at the end of a string: a consonant with the consonants joined to it, its vowel sign and marks, or a vowel
AKSARA_END = re.compile(r"(?:[क-हक़-य़](?:्[क-हक़-य़])*्?[ा-ौॢॣ]?"
                        r"|[ऄ-औ])[ँ-ः]?$")
LIBRARIES = {"RTJ": ("Shri Raghunath Temple MSS Library", "Jammu"), "UPSS": ("Uttar Pradesh Sanskrit Sansthan", "Lucknow")}
READER_VERSION = 4      # bumped when Reader.read changes, so cached readings of an earlier version are not reused
MATCH_LIMIT = 0.75      # a page and a side further apart than this (normalised edit distance) are no pair
SKIP_COST = 0.6         # monotone matching: the cost of passing over a page or a side
CLOSE_CALL = 0.08       # a runner-up this close to the best side makes the pair doubtful: left unpaired
LOOSE_LIMIT = 0.88      # monotone matching: a page between two matched neighbours may read this badly and still pair


# ---------------------------------------------------------------- the e-text

def parse_header(lines):
    """The 'key: value' lines of the e-text's header, lower-cased keys; the URL and the licence picked out."""
    meta, pending = {}, None
    for l in lines:
        m = re.match(r"\s*([A-Za-z][A-Za-z0-9 ]+?)\s*:\s*(.*)$", l)
        if m and not l.startswith("http"):
            meta[m.group(1).strip().lower()] = m.group(2).strip()
            pending = m.group(1).strip().lower() if not m.group(2).strip() else None   # "Notes:" with the text on the next line
        elif l.strip().startswith("http"):
            meta["url"] = l.strip()
        elif pending and l.strip() and not l.startswith("#"):
            meta[pending], pending = l.strip(), None
        m = re.search(r"Creative Commons licen[cs]e\s+(.+)", l)
        if m:
            meta["licence"] = m.group(1).strip()
    return meta


def read_etext(path):
    """-> (header meta, blocks, notes): a block is {label, folio, side ('अ', 'ब' or ''), at (its line in the file),
    raw [(line no, text)], unmarked}; notes are remarks found where a marker would stand."""
    raw = Path(path).read_text(encoding="utf-8").split("\n")
    first = next((i for i, l in enumerate(raw) if MARKER.match(l)), None)
    if first is None:
        raise SystemExit(f"{path}: no 'पृ० १अ)' side marker found; is this a Muktabodha e-text?")
    meta = parse_header(raw[:first])
    hdr_end = max((i for i, l in enumerate(raw[:first]) if l.startswith("####")), default=-1)
    pre = [(no, l) for no, l in enumerate(raw[hdr_end + 1:first], hdr_end + 2) if l.strip() and not MARKER_LIKE.match(l)]
    k = len(pre)            # the catalogue note is short lines; a side before the first marker is a run of long ones
    while k > 0 and len(pre[k - 1][1].strip()) >= 30:
        k -= 1
    blocks, notes = [], []
    if k < len(pre):
        blocks.append({"label": "", "folio": None, "side": "", "at": pre[k][0], "raw": pre[k:], "unmarked": True})
        notes.append(f"line {pre[k][0]}: {len(pre) - k} lines of text before the first side marker: taken as the first side")
    cur = None
    for no, line in enumerate(raw[first:], first + 1):
        if line.startswith("####") and "END OF FILE" in line:
            break
        m = MARKER.match(line)
        if m:
            label = m.group(1).strip()
            num = re.search(r"[०-९0-9]+", label)
            folio = int(num.group(0).translate(DIGITS)) if num else None
            s = re.search(r"[अब]", label[num.end():] if num else label)
            cur = {"label": label, "folio": folio, "side": s.group(0) if s else "", "at": no, "raw": [], "unmarked": False}
            blocks.append(cur)
            rest = line[m.end():]
            if rest.strip():
                cur["raw"].append((no, rest))
        elif MARKER_LIKE.match(line):
            notes.append(f"line {no}: a remark where a side marker would stand, left out: {line.strip()[:70]}")
        elif line.strip() and cur is not None:
            cur["raw"].append((no, line))
    return meta, blocks, notes


def split_notes(raw):
    """The lines of a side with the glosses in brackets taken apart: -> ([(line no, [('t', text) | ('n', gloss)])], anomalies).
    A gloss typed over a line break is joined (with a space) and stays with the line it opened on; a '(' inside a gloss
    (the transcriber re-opening it on the next line) is ignored."""
    lines, anomalies, note = [], [], None
    for no, s in raw:
        tokens = []
        lines.append((no, tokens))
        if note is not None:
            anomalies.append(f"line {no}: the gloss opened on line {note[0]} runs on over the line break")
            note[1] += " "
        for ch in s:
            if note is not None:
                if ch == ")":
                    note[2].append(("n", note[1]))
                    note = None
                elif ch != "(":
                    note[1] += ch
            elif ch == "(":
                note = [no, "", tokens]
            elif ch == ")":
                anomalies.append(f"line {no}: a ')' without a '(' before it")
            elif tokens and tokens[-1][0] == "t":
                tokens[-1] = ("t", tokens[-1][1] + ch)
            else:
                tokens.append(("t", ch))
    if note is not None:
        anomalies.append(f"line {note[0]}: the gloss is never closed; closed at the end of the side")
        note[2].append(("n", note[1]))
    return lines, anomalies


def norm(s):
    s = unicodedata.normalize("NFC", s).replace("\t", " ")
    s = s.replace("||", "॥").replace("|", "।")
    return re.sub(r"\s+", " ", s).strip()


def clean(s, stats):
    """The transcriber's marks that are no ink: - (a word broken at the line end), a lone . ; ..... becomes *."""
    n = s.count("-")
    if n:
        stats["hyphen"] += n
        s = s.replace("-", "")
    s = re.sub(r"\.{2,}", lambda m: "*" * max(1, len(m.group(0)) // 3), s)
    n = s.count(".")
    if n:
        stats["dot"] += n
        s = s.replace(".", "")
    return s


def xml_text(s, stats):
    """A run of main text as TEI: * an illegible akṣara (<gap/>), ? a doubtful one (<unclear> round the akṣara before it)."""
    s = escape(norm(clean(s, stats)))

    def gap(m):
        stats["gap"] += 1
        return f'<gap reason="illegible" unit="akṣara" quantity="{len(m.group(0))}"/>'
    s = re.sub(r"\*+", gap, s)
    out, pos = [], 0
    for m in re.finditer(r"\?", s):
        before = s[pos:m.start()]
        a = AKSARA_END.search(before)
        out.append(before[:a.start()] + "<unclear>" + a.group(0) + "</unclear>" if a else before)
        stats["unclear"] += 1
        pos = m.end()
    out.append(s[pos:])
    return "".join(out)


def build_sides(blocks, sequential=False):
    """-> ([side], sequential): {pid '0001_1', label, at, available, lines [(line no, tei, plain)], anomalies, stats}.
    The ids come from the labels (folio and side), or run 0001_1, 0001_2, ... in the e-text's order when asked or
    when the labels cannot be trusted."""
    sequential = sequential or any(b["unmarked"] or b["folio"] is None for b in blocks)
    sides, ids = [], collections.Counter()
    for k, b in enumerate(blocks):
        label = "(before the first marker)" if b["unmarked"] else b["label"]
        pid = f"{k // 2 + 1:04d}_{k % 2 + 1}" if sequential else f"{b['folio']:04d}_{SIDE[b['side']]}"
        side = {"pid": pid, "label": label, "at": b["at"], "available": True, "lines": [], "anomalies": [],
                "stats": collections.Counter()}
        sides.append(side)
        if len(b["raw"]) <= 1 and (not b["raw"] or NOT_AVAILABLE.search(b["raw"][0][1])):
            side["available"] = False
            side["why"] = b["raw"][0][1].strip() if b["raw"] else "no text"
            continue
        ids[pid] += 1
        lines, anomalies = split_notes(b["raw"])
        side["anomalies"] += anomalies
        for no, tokens in lines:
            plain = norm(clean("".join(t for k2, t in tokens if k2 == "t"), collections.Counter()))
            if not plain:
                side["anomalies"].append(f"line {no}: nothing but a gloss; left out")
                continue
            odd = sorted({c for c in plain if not ALLOWED.match(c)})
            if odd:
                side["anomalies"].append(f"line {no}: characters that are not Devanagari: " + " ".join(f"{c} (U+{ord(c):04X})" for c in odd))
            tei = "".join(xml_text(t, side["stats"]) if k2 == "t" else f'<note place="margin">{escape(norm(t))}</note>'
                          for k2, t in tokens)
            side["lines"].append((no, tei, plain.replace("*", "").replace("?", "")))
    def after(pid):
        folio, side = pid.split("_")
        return f"{folio}_2" if side == "1" else f"{int(folio) + 1:04d}_1"
    for pid in [q for q, n in ids.items() if n > 1]:
        # the same label twice (a slip of the transcriber): the occurrence that does not follow its neighbour is the slip,
        # and takes the id that would follow it, if that id is free
        for k in [k for k, sd in enumerate(sides) if sd["pid"] == pid and sd["available"]]:
            prev = next((sides[j]["pid"] for j in range(k - 1, -1, -1) if sides[j]["available"]), None)
            if prev is None or after(prev) == pid or ids[pid] == 1:
                continue
            want = after(prev)
            if ids[want]:
                raise SystemExit(f"the side {pid} is transcribed {ids[pid]} times (the same marker twice) and {want}, which would fit "
                                 f"the one at line {sides[k]['at']}, is taken too: correct the e-text, or set \"sequential_ids\": true in the config")
            sides[k]["anomalies"].insert(0, f"the label {sides[k]['label']} repeats {pid}: taken as {want}, which follows the side before it")
            sides[k]["pid"] = want
            ids[want] += 1
            ids[pid] -= 1
        if ids[pid] > 1:
            raise SystemExit(f"the side {pid} is transcribed {ids[pid]} times (the same marker twice): correct the e-text, or set "
                             f"\"sequential_ids\": true in the config")
    return sides, sequential


# ---------------------------------------------------------------- the PDF's pages

def page_scan(page, no):
    """The scan on a PDF page: (raw JPEG bytes or None, the image object, the page's rotation)."""
    xo = page.get("/Resources", {}).get("/XObject", {})
    best, area = None, -1
    for k in xo:
        o = xo[k].get_object()
        if o.get("/Subtype") != "/Image":
            continue
        a = (o.get("/Width") or 0) * (o.get("/Height") or 0)
        if a > area:
            best, area = o, a
    if best is None:
        raise SystemExit(f"PDF page {no}: no image on it")
    rot = int(page.get("/Rotate") or 0) % 360
    filt = best.get("/Filter")
    filt = str(filt[-1] if isinstance(filt, list) else filt)
    data = best.get_data() if filt == "/DCTDecode" else None
    if data is not None and data[:3] != b"\xff\xd8\xff":
        data = None
    return data, best, rot


def page_image(page, no, max_width=None):
    """-> the page as shown (turned by /Rotate), a PIL image in RGB, no wider than max_width."""
    from PIL import Image
    data, obj, rot = page_scan(page, no)
    im = Image.open(io.BytesIO(data)).convert("RGB") if data is not None else page.images[0].image.convert("RGB")
    turn = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}.get(rot)
    if turn is not None:
        im = im.transpose(turn)
    if max_width and im.size[0] > max_width:
        im = im.resize((max_width, round(im.size[1] * max_width / im.size[0])), Image.LANCZOS)
    return im


def _longest_run(mask):
    best, start = (0, 0), None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def crop_to_paper(im, margin=0.04):
    """The photo cut down to the leaf: the largest band of rows, then of columns, that are mostly light (paper on a
    dark table), cut just inside the leaf's edge and padded with the paper's own colour, so that the segmenter
    sees no edge against the table or the cloth (it reads such an edge as a line). -> (image, cropped?)"""
    import numpy as np
    from PIL import Image
    f = 8
    g = np.asarray(im.convert("L").resize((max(1, im.width // f), max(1, im.height // f))))
    # paper against the table: the threshold halfway between the dark and the light parts of this photo (aged paper
    # can be as dark as 100 of 255; a black table is below 50)
    flat = np.sort(g.ravel())
    thr = min(160, max(60, (int(flat[len(flat) // 5]) + int(flat[-len(flat) // 5])) / 2))
    bright = g > thr
    r0, r1 = _longest_run(bright.mean(axis=1) > 0.25)
    if r1 - r0 < 0.12 * bright.shape[0]:
        return im, False
    c0, c1 = _longest_run(bright[r0:r1].mean(axis=0) > 0.25)
    if c1 - c0 < 0.12 * bright.shape[1]:
        return im, False
    inset = 0.006
    box = (int((c0 + inset * bright.shape[1]) * f), int((r0 + inset * bright.shape[0]) * f),
           min(im.width, int((c1 - inset * bright.shape[1]) * f)), min(im.height, int((r1 - inset * bright.shape[0]) * f)))
    if box[2] - box[0] < 0.3 * im.width or box[3] - box[1] < 0.12 * im.height:
        return im, False
    leaf = im.crop(box)
    sample = np.asarray(leaf.convert("RGB").resize((max(1, leaf.width // f), max(1, leaf.height // f)))).reshape(-1, 3)
    paper = tuple(int(v) for v in np.median(sample[sample.mean(axis=1) > thr], axis=0)) if (sample.mean(axis=1) > thr).any() else (235, 225, 200)
    mw, mh = int(margin * leaf.width), int(margin * leaf.height)
    out = Image.new("RGB", (leaf.width + 2 * mw, leaf.height + 2 * mh), paper)
    out.paste(leaf.convert("RGB"), (mw, mh))
    return out, True


def pad_height(im, factor):
    """The leaf set in the middle of a canvas `factor` times its height, in the paper's own colour: a leaf of a few
    large lines fills its photo, and the segmenter, which sees every page 640 px high, then meets lines three times
    taller than any it learnt; padded, they take the share of the height a page of 10-15 lines gives them."""
    import numpy as np
    from PIL import Image
    if not factor or factor <= 1:
        return im
    g = np.asarray(im.convert("RGB").resize((max(1, im.width // 8), max(1, im.height // 8)))).reshape(-1, 3)
    light = g[g.mean(axis=1) > np.percentile(g.mean(axis=1), 40)]
    paper = tuple(int(v) for v in np.median(light if len(light) else g, axis=0))
    h = int(im.height * factor)
    out = Image.new("RGB", (im.width, h), paper)
    out.paste(im.convert("RGB"), (0, (h - im.height) // 2))
    return out


def split_leaves(im, lines, usual=None):
    """Two leaves one above the other in one photo: cut at the dark band between them (the table showing through,
    in the middle half of the photo), else at the widest gap between the lines' baselines in the middle third of
    the lines when it is clearly wider than the line pitch; only when each piece then holds three lines or more,
    and neither fewer than a sixth of all. -> [image] or [top, bottom]"""
    import numpy as np
    ys = sorted(sum(p[1] for p in l["baseline"]) / len(l["baseline"]) for l in lines if l["baseline"])
    if len(ys) < 6:
        return [im]
    if usual:          # one leaf of the usual number of lines (and a few pieces or notes) is not cut
        groups, last = 1, ys[0]
        hs = sorted(max(p[1] for p in l["boundary"]) - min(p[1] for p in l["boundary"]) for l in lines if l["boundary"])
        half = hs[len(hs) // 2] / 2
        for y in ys[1:]:
            if y - last >= half:
                groups += 1
            last = y
        if groups < 1.5 * usual:
            return [im]

    def pieces(cut):
        top = [y for y in ys if y < cut]
        bottom = [y for y in ys if y >= cut]
        ok = min(len(top), len(bottom)) >= 3 and min(len(top), len(bottom)) >= len(ys) / 6
        return [im.crop((0, 0, im.width, cut)), im.crop((0, cut, im.width, im.height))] if ok else None

    def where(k, pitch):
        """The cut between the k-th and the next baseline: a pitch and a half above the lower leaf's first line
        when the gap allows it, so that its headline marks and a margin stay with it; else halfway."""
        return int(max(ys[k] + 0.4 * pitch, min(ys[k + 1] - 1.5 * pitch, (ys[k] + ys[k + 1]) / 2)))
    f = 4
    g = np.asarray(im.convert("L").resize((max(1, im.width // f), max(1, im.height // f))))
    dark = (g < 80).mean(axis=1) > 0.6
    lo, hi = int(0.2 * len(dark)), int(0.8 * len(dark))
    runs, start = [], None
    for i in range(lo, hi + 1):
        v = dark[i] if i < hi else False
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if runs:
        a, b = max(runs, key=lambda r: r[1] - r[0])
        if b - a >= max(2, 0.004 * len(dark)):
            cut = int((a + b) / 2 * f)
            diffs = [b2 - a2 for a2, b2 in zip(ys, ys[1:])]
            k = next((i for i in range(len(ys) - 1) if ys[i] < cut <= ys[i + 1]), None)
            out = pieces(where(k, statistics.median(diffs)) if k is not None else cut)
            if out:
                return out
    # no dark band: the leaf boundary is the widest gap between lines in the middle third of the photo (two leaves of
    # about the same number of lines), when it is clearly more than the line pitch
    diffs = [b - a for a, b in zip(ys, ys[1:])]
    pitch = statistics.median(diffs)
    middle = range(len(diffs) // 3, len(diffs) - len(diffs) // 3)
    k = max(middle, key=lambda i: diffs[i])
    if diffs[k] >= 1.6 * pitch:
        out = pieces(where(k, pitch))
        if out:
            return out
    return [im]


class Reader:
    """The U-Net's lines and the recogniser's reading of a page image, for cutting and matching pages."""

    def __init__(self, model=None):
        from . import tseg
        from .predict import DEFAULT_MODEL, model_label
        from .recognizer import load_model
        self.seg_net, _ = tseg.load_net()
        self.dev = tseg.device()
        self.model = Path(model) if model else DEFAULT_MODEL
        self.net, self.chars, self.ck = load_model(self.model)
        self.label = model_label(self.model, self.ck)

    def lines(self, path):
        from . import tseg
        pb, pu, pd, s = tseg.predict_maps(self.seg_net, path, self.dev)
        return tseg.vectorize(pb, pu, pd, s)

    def read(self, path, lines=None, usual=None):
        """-> the page's rows, each cut and read (greedy): the row finder's rows (pieces joined, as predict reads a
        page), which read far closer to the transcription than the U-Net's raw pieces; where the row finder makes
        fewer than half as many rows as the pieces form (a damaged leaf, one line, no block), the pieces themselves,
        in reading order: by row (baselines within half a line height), left to right within a row."""
        import numpy as np
        from PIL import Image
        from . import tseg
        from .align import _info, _merge, find_rows
        from .linecut import band_of, cut_line
        from .predict import read_line
        lines = self.lines(path) if lines is None else lines
        lines = [l for l in lines if len(l["baseline"]) >= 2 and len(l["boundary"]) >= 3]
        if not lines:
            return []
        with Image.open(path) as im:
            size = im.size
            gray = np.asarray(im.convert("L"))
            page = np.asarray(im.convert("RGB"))
        info = []
        for l in lines:
            y = sum(p[1] for p in l["baseline"]) / len(l["baseline"])
            x = min(p[0] for p in l["baseline"])
            h = max(p[1] for p in l["boundary"]) - min(p[1] for p in l["boundary"])
            info.append((y, x, h, l))
        info.sort(key=lambda t: t[0])
        half = statistics.median(h for _, _, h, _ in info) / 2
        groups, cur = [], [info[0]]
        for t in info[1:]:
            if t[0] - cur[-1][0] < half:
                cur.append(t)
            else:
                groups.append(cur)
                cur = [t]
        groups.append(cur)
        gy = [statistics.mean(t[0] for t in g) for g in groups]
        pitch_hint = statistics.median(b - a for a, b in zip(gy, gy[1:])) if len(gy) > 1 else None
        # the row finder, with no pitch hint, with the pieces' spacing, and with the pitch the usual number of lines
        # implies; the rows kept are those nearest in number to the usual (the e-text's prior on this page)
        seg = tseg.kraken_json(path, lines)
        hints = [None] + ([pitch_hint] if pitch_hint else []) + ([0.85 * size[1] / usual] if usual else [])
        best = None
        for h in hints:
            try:
                rows, block, pitch, note = find_rows(seg, size[0], gray, pitch_hint=h)
            except Exception:
                continue
            if len(rows) < 2:
                continue
            score = abs(len(rows) - usual) if usual else -len(rows)
            if best is None or score < best[0]:
                best = (score, rows, block, pitch)
        rows, block, pitch = best[1:] if best else ([], None, None)
        if not rows or (usual and best[0] > usual / 2) or len(rows) < len(groups) / 2:
            # the row finder made nothing of the page: the pieces grouped by their baselines are the rows, each
            # merged into one baseline and band over the block's whole width, as the row finder would merge them
            x0 = min(t[1] for g in groups for t in g)
            x1 = max(max(p[0] for p in t[3]["baseline"]) for g in groups for t in g)
            pitch = pitch_hint or 2 * half / 0.8
            block = (x0, x1)
            rows = [[_info(t[3]) for t in sorted(g, key=lambda t: t[1])] for g in groups]
        out = []
        for r in rows:
            base, poly = _merge(r, block, pitch)
            above, below = band_of(base, poly)
            text, _ = read_line(self.net, self.chars, Image.fromarray(cut_line(page, base, above, below)))
            out.append(text)
        return out


# ---------------------------------------------------------------- matching pages with sides

def match_sides(texts, sides, mode, names=None):
    """texts: the pages' readings in PDF order; sides: the available sides; names: what to call the pages in the notes.
    -> (pairs {page index: side index}, D distances pages × sides, notes)."""
    names = names or [str(i) for i in range(len(texts))]
    import numpy as np
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist
    side_texts = ["".join(p for _, _, p in s["lines"]).replace(" ", "") for s in sides]
    D = cdist(texts, side_texts, scorer=Levenshtein.normalized_distance, workers=-1)
    P, S = D.shape
    pairs, notes = {}, []
    if mode == "monotone":
        INF = 1e9
        M = np.full((P + 1, S + 1), INF)
        back = np.zeros((P + 1, S + 1), dtype=np.int8)
        M[0, 0] = 0
        for i in range(P + 1):
            for j in range(S + 1):
                if i == 0 and j == 0:
                    continue
                best, b = INF, 0
                if i and j and M[i - 1, j - 1] + D[i - 1, j - 1] < best:
                    best, b = M[i - 1, j - 1] + D[i - 1, j - 1], 1
                if i and M[i - 1, j] + SKIP_COST < best:
                    best, b = M[i - 1, j] + SKIP_COST, 2
                if j and M[i, j - 1] + SKIP_COST < best:
                    best, b = M[i, j - 1] + SKIP_COST, 3
                M[i, j], back[i, j] = best, b
        i, j = P, S
        while i or j:
            b = back[i, j]
            if b == 1:
                pairs[i - 1] = j - 1
                i, j = i - 1, j - 1
            elif b == 2:
                i -= 1
            else:
                j -= 1
        for i in list(pairs):
            if D[i, pairs[i]] > MATCH_LIMIT:
                del pairs[i]
        # a page between two paired pages whose sides are as many apart takes the side between them, even when its
        # reading is poor (a damaged leaf): the order vouches for it
        done = sorted(pairs.items())
        for (i0, j0), (i1, j1) in zip(done, done[1:]):
            if i1 - i0 == j1 - j0 > 1 and all(D[i0 + k, j0 + k] <= LOOSE_LIMIT for k in range(1, i1 - i0)):
                for k in range(1, i1 - i0):
                    pairs[i0 + k] = j0 + k
                    notes.append(f"page {names[i0 + k]}: side {sides[j0 + k]['label']} by its place between pages {names[i0]} and {names[i1]}, "
                                 f"its reading {D[i0 + k, j0 + k]:.2f} away")
        for i in range(P):
            if i not in pairs and texts[i]:
                j = int(D[i].argmin())
                notes.append(f"page {names[i]}: no side; the closest reads {D[i, j]:.2f} away ({sides[j]['label']})")
    else:
        want = {}
        for i in range(P):
            if S == 0 or not texts[i]:
                continue
            j = int(D[i].argmin())
            if D[i, j] <= MATCH_LIMIT:
                want.setdefault(j, []).append(i)
        for j, pages in want.items():
            keep = min(pages, key=lambda i: D[i, j])
            pairs[keep] = j
            for i in pages:
                if i != keep:
                    notes.append(f"page {names[i]}: reads as side {sides[j]['label']} too ({D[i, j]:.2f}), which page {names[keep]} reads better ({D[keep, j]:.2f}): a second photo? left unpaired")
        # an unpaired page whose nearest side is unpaired too, and clearly nearer than any other side, takes it even
        # when the reading is poor (a dark or damaged leaf): the two are each other's only candidates
        taken = set(pairs.values())
        for i in range(P):
            if i in pairs or not texts[i] or not S:
                continue
            order = np.argsort(D[i])
            j = int(order[0])
            if j not in taken and D[i, j] <= LOOSE_LIMIT and (S == 1 or D[i, order[1]] - D[i, j] >= CLOSE_CALL):
                pairs[i] = j
                taken.add(j)
                notes.append(f"page {names[i]}: side {sides[j]['label']}, the only side left that it reads as ({D[i, j]:.2f} away, the next {D[i, order[1]]:.2f})")
        for i in range(P):
            if i not in pairs and texts[i] and S:
                j = int(D[i].argmin())
                if i not in want.get(j, []):
                    notes.append(f"page {names[i]}: no side; the closest reads {D[i, j]:.2f} away ({sides[j]['label']}), {len(texts[i])} letters read")
    for i, j in sorted(pairs.items()):
        if S > 1 and mode != "monotone":
            others = np.delete(D[i], j)
            if others.min() - D[i, j] < CLOSE_CALL:
                notes.append(f"page {names[i]}: side {sides[j]['label']} at {D[i, j]:.2f}, but another side is almost as close "
                             f"({others.min():.2f}): left unpaired, to be looked at")
                del pairs[i]
    return pairs, D, notes


# ---------------------------------------------------------------- the TEI

def tei(ms_id, meta, sides, etext, pdf, usual):
    number = ms_id.split("_")[-1]
    library, place = LIBRARIES.get(ms_id.split("_")[0], ("", ""))
    title = meta.get("uniform title") or ms_id
    author = meta.get("author", "")
    catalog = meta.get("catalog number", "")
    today = time.strftime("%Y-%m-%d")
    who = f"{NAME} {__version__}, msocr/muktabodha.py"
    rev = next((k for k in meta if k.startswith("revision")), None)
    head = ["<teiHeader>", "\t<fileDesc>", "\t\t<titleStmt>", f"\t\t\t<title>{escape(title)}</title>"]
    if author:
        head.append(f"\t\t\t<author>{escape(author)}</author>")
    head += [
        "\t\t\t<respStmt>", "\t\t\t\t<resp>diplomatic transcription</resp>",
        f"\t\t\t\t<name>Muktabodha Indological Research Institute, e-text {escape(catalog)}"
        + (f" ({escape(meta['notes'])})" if meta.get("notes") else "")
        + (f"; {escape(rev)}: {escape(meta[rev])}" if rev and meta[rev] else "") + "</name>", "\t\t\t</respStmt>",
        "\t\t\t<respStmt>", "\t\t\t\t<resp>TEI from the e-text, page images from the scan</resp>",
        f"\t\t\t\t<name>{escape(who)}</name>", "\t\t\t</respStmt>",
        "\t\t</titleStmt>", "\t\t<publicationStmt>", "\t\t\t<p>Not for distribution.</p>"]
    if meta.get("licence"):
        head += ["\t\t\t<availability>",
                 f"\t\t\t\t<licence>The transcription is Muktabodha's e-text {escape(catalog)}, Creative Commons {escape(meta['licence'])}.</licence>",
                 "\t\t\t</availability>"]
    head += [
        "\t\t</publicationStmt>", "\t\t<sourceDesc>",
        f"\t\t\t<p>Transcribed from the manuscript number {escape(number)}"
        + (f" from {escape(library)}, {escape(place)}" if library else "") + ".</p>",
        "\t\t\t<msDesc>", "\t\t\t\t<msIdentifier>"]
    if place:
        head.append(f"\t\t\t\t\t<settlement>{escape(place)}</settlement>")
    if library:
        head.append(f"\t\t\t\t\t<repository>{escape(library)}</repository>")
    head += [f"\t\t\t\t\t<idno>{escape(number)}</idno>", "\t\t\t\t</msIdentifier>",
             "\t\t\t\t<physDesc>", '\t\t\t\t\t<objectDesc form="pothi">', "\t\t\t\t\t\t<layoutDesc>",
             f'\t\t\t\t\t\t\t<layout writtenLines="{usual}"><p>Linear layout. Lines per folio side: main {usual}. Glosses in the margins.</p></layout>',
             "\t\t\t\t\t\t</layoutDesc>", "\t\t\t\t\t</objectDesc>", "\t\t\t\t</physDesc>", "\t\t\t</msDesc>",
             f"\t\t\t<p>E-text: {escape(Path(etext).name)}" + (f", {escape(meta['url'])}" if meta.get("url") else "")
             + f". Scan: {escape(Path(pdf).name)}" + (f" ({escape(meta['description'])})" if meta.get("description") else "") + ".</p>",
             "\t\t</sourceDesc>", "\t</fileDesc>", "\t<revisionDesc>",
             f'\t\t<change when="{today}">Made from the e-text by {escape(who)}: glosses in brackets as notes, * as gaps, '
             f'? as unclear, | and || as daṇḍas; nothing corrected.</change>',
             "\t</revisionDesc>", "</teiHeader>"]
    body = ['<TEI xmlns="http://www.tei-c.org/ns/1.0">'] + head + ["<text>", "<body>", "<div>"]
    for s in sides:
        if not s["available"]:
            body.append(f"<!-- folio side {s['label']} ({s['pid']}): not available to the transcriber: {escape(s['why']).replace('--', '-')} -->")
            continue
        body.append(f'<pb n="{s["pid"]}"/>')
        for i, (_, t, _) in enumerate(s["lines"], 1):
            body.append(f'<lb n="{i}" break="no"/>{t}')
    body += ["</div>", "</body>", "</text>", "</TEI>"]
    return "\n".join(body) + "\n"


# ---------------------------------------------------------------- the build

def parse_pages(spec):
    """'4-22,24,23' -> [4, 5, ..., 22, 24, 23]"""
    if isinstance(spec, list):
        return [int(p) for p in spec]
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def page_plan(sides, n_pages, first=None, pages=None, skip=()):
    """The plain plan: which PDF page becomes which side, in order. -> [{page, part, pid, side, what}]"""
    available = [s for s in sides if s["available"]]
    if pages is None:
        if not first or first < 1 or first > n_pages:
            raise SystemExit(f"--first-text-page {first}: the PDF has {n_pages} pages")
        pages = list(range(first, n_pages + 1))
    plan = [{"page": p, "part": "", "pid": f"0000_{p}", "side": None, "what": "before the text (cover, wrapper, blank side)"}
            for p in range(1, pages[0])]
    todo, last = list(available), None
    for p in pages:
        if p in skip:
            pid = (last["pid"] if last else "0000_0") + "x"
            plan.append({"page": p, "part": "", "pid": pid, "side": None, "what": "skipped (skip): no side of the transcription"})
            continue
        if todo:
            s = todo.pop(0)
            plan.append({"page": p, "part": "", "pid": s["pid"], "side": s, "what": f"side {s['label']}, {len(s['lines'])} lines"})
            last = plan[-1]
        else:
            folio, k = last["pid"].split("_") if last else ("0000", "0")
            pid = f"{folio}_{int(k.rstrip('x')) + 1}"
            plan.append({"page": p, "part": "", "pid": pid, "side": None, "what": "after the last side (blank side or back cover)"})
            last = plan[-1]
    if todo:
        raise SystemExit(f"the PDF ends {len(todo)} sides before the transcription does (its first is {todo[0]['label']}): "
                         f"{n_pages} pages, {len(pages)} of them text pages, {len(skip)} skipped, {len(available)} sides transcribed")
    return plan


def plan_from_match(entries, pairs, D, sides):
    """The entries (pages, or their parts, in PDF order) named by the sides they matched."""
    plan, used, last, k = [], set(), None, 0
    for i, e in enumerate(entries):
        if i in pairs:
            s = sides[pairs[i]]
            e = {**e, "pid": s["pid"], "side": s, "what": f"side {s['label']}, {len(s['lines'])} lines, read {D[i, pairs[i]]:.2f} away"}
            last = e
        elif last is None:
            k += 1
            e = {**e, "pid": f"0000_{k}", "side": None, "what": "before the first side matched (cover, wrapper, blank side)"}
        else:
            n, pid = 1, last["pid"] + "x"
            while pid in used:
                n += 1
                pid = f"{last['pid']}x{n}"
            e = {**e, "pid": pid, "side": None, "what": "no side matched (a blank side, a second photo of a leaf, the back cover)"}
        used.add(e["pid"])
        plan.append(e)
    return plan


def build(ms_id, cfg, redo=False):
    from PIL import Image
    etext, pdf, into = Path(cfg["etext"]), Path(cfg["pdf"]), Path(cfg["into"])
    number = f"{int(ms_id.split('_')[1]):06d}"
    meta, blocks, notes = read_etext(etext)
    sides, sequential = build_sides(blocks, cfg.get("sequential_ids", False))
    available = [s for s in sides if s["available"]]
    counts = collections.Counter(len(s["lines"]) for s in available)
    usual = counts.most_common(1)[0][0]
    import pypdf
    reader = pypdf.PdfReader(str(pdf))
    n_pages = len(reader.pages)
    skip = set(cfg.get("skip") or [])
    match = cfg.get("match")
    split, crop, max_width, pad = cfg.get("sides_per_page", 1) == 2, cfg.get("crop"), cfg.get("max_width"), cfg.get("pad")
    processed = bool(split or crop or max_width)
    images_dir = into / "manuscripts" / ms_id
    images_dir.mkdir(parents=True, exist_ok=True)
    d = ms_dir(ms_id)
    t0 = time.time()
    rep_match, reader_ = [], None

    if match:
        text_pages = parse_pages(cfg["pages"]) if cfg.get("pages") else list(range(1, n_pages + 1))
        text_pages = [p for p in text_pages if p not in skip]
        # 1. every page, cut and cut apart as configured, to tmp_pNNNN[ab].jpg
        reader_ = Reader(cfg.get("model"))
        if redo:
            for old in images_dir.glob("tmp_p*.jpg"):
                old.unlink()
        entries = []
        for p in text_pages:
            parts = [images_dir / f"tmp_p{p:04d}{suffix}.jpg" for suffix in ("", "a", "b")]
            if not redo and (parts[0].exists() or (parts[1].exists() and parts[2].exists())):
                found = [parts[0]] if parts[0].exists() else parts[1:]
            else:
                im = page_image(reader.pages[p - 1], p, max_width)
                if crop == "paper":
                    im, _ = crop_to_paper(im)
                if not split:
                    im = pad_height(im, pad)
                pieces = [im]
                if split:
                    im.save(parts[0], quality=95)
                    pieces = split_leaves(im, reader_.lines(parts[0]), usual)
                    if len(pieces) == 2:
                        parts[0].unlink()
                        pieces = [pad_height(crop_to_paper(pc)[0], pad) for pc in pieces]   # each leaf without the band between them
                found = []
                for piece, path in zip(pieces, parts[0:1] if len(pieces) == 1 else parts[1:]):
                    piece.save(path, quality=95)
                    found.append(path)
            for path in found:
                entries.append({"page": p, "part": path.stem[-1] if path.stem[-1] in "ab" else "", "tmp": path})
        # 2. read them (cached)
        cache_path = d / "match_readings.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() and not redo else {}
        for k, e in enumerate(entries, 1):
            key = f"{e['tmp'].name}:{e['tmp'].stat().st_size}:{READER_VERSION}"
            if key not in cache:
                t1 = time.time()
                cache[key] = reader_.read(e["tmp"], usual=usual)
                if time.time() - t1 > 5:
                    print(f"  ! {e['tmp'].name} took {time.time() - t1:.0f} s to read ({len(cache[key])} lines)", flush=True)
                if k % 25 == 0 or k == len(entries):
                    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                    print(f"  read {k}/{len(entries)} pages, {(time.time() - t0) / k:.1f} s a page", flush=True)
            e["lines"] = cache[key]
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        # 3. match
        t_read = time.time()
        texts = ["".join(e["lines"]).replace(" ", "") for e in entries]
        pairs, D, match_notes = match_sides(texts, available, match, [f"{e['page']}{e['part']}" for e in entries])
        plan = plan_from_match(entries, pairs, D, available)
        print(f"  pages cut and read in {t_read - t0:.0f} s, matched in {time.time() - t_read:.0f} s", flush=True)
        (d / "match.json").write_text(json.dumps({"pages": [f"{e['page']}{e['part']}" for e in entries], "sides": [s["pid"] for s in available],
                                                  "pairs": {str(i): int(j) for i, j in pairs.items()}, "distance": [[round(float(x), 3) for x in row] for row in D]}),
                                      encoding="utf-8")
        for e in plan:                 # 4. name the images
            final = images_dir / f"{number}_0{e['pid']}.jpg"
            e["tmp"].replace(final)
            e["image"] = final.name
        stale = [f for f in images_dir.glob(f"{number}_*.jpg") if f.name not in {e["image"] for e in plan}]
        for f in stale:
            f.unlink()
        if stale:
            print(f"  {len(stale)} images of an earlier run removed", flush=True)
        unmatched_sides = [s for j, s in enumerate(available) if j not in set(pairs.values())]
        rep_match = [f"  matching ({match}) : {len(pairs)} of {len(entries)} page images paired with {len(available)} sides; "
                     f"{len(entries) - len(pairs)} images without a side, {len(unmatched_sides)} sides without an image; "
                     f"read with {reader_.label}",
                     "  distance of the pairs (0 = read letter for letter): median "
                     f"{statistics.median(D[i, j] for i, j in pairs.items()):.2f}, worst {max(D[i, j] for i, j in pairs.items()):.2f}"
                     if pairs else "  no pair found"]
        rep_match += [f"    sides without an image: " + ", ".join(s["label"] for s in unmatched_sides[:30]) + (" ..." if len(unmatched_sides) > 30 else "")] if unmatched_sides else []
        rep_match += [f"    {n}" for n in match_notes]
        copied = reencoded = 0
    else:
        plan = page_plan(sides, n_pages, cfg.get("first_text_page"), parse_pages(cfg["pages"]) if cfg.get("pages") else None, skip)
        copied = reencoded = kept = 0
        for e in plan:
            out = images_dir / f"{number}_0{e['pid']}.jpg"
            e["image"] = out.name
            if out.exists() and not redo:
                kept += 1
                continue
            page = reader.pages[e["page"] - 1]
            data, obj, rot = page_scan(page, e["page"])
            if data is not None and rot == 0 and not processed and not pad:
                out.write_bytes(data)
                copied += 1
            else:
                im = page_image(page, e["page"], max_width)
                if crop == "paper":
                    im, _ = crop_to_paper(im)
                im = pad_height(im, pad)
                im.save(out, quality=95)
                reencoded += 1
        link = into / "manuscripts" / f"{ms_id}.pdf"
        if not link.exists():
            link.symlink_to(os.path.relpath(pdf.resolve(), link.parent))

    tei_path = into / "transcriptions" / f"{ms_id}.xml"
    tei_path.parent.mkdir(parents=True, exist_ok=True)
    tei_path.write_text(tei(ms_id, meta, sides, etext, pdf, usual), encoding="utf-8")
    not_text = [e for e in plan if e["side"] is None]
    (d / "not_text.txt").write_text("".join(f"{e['pid']}\t# PDF page {e['page']}{e['part']}: {e['what']}\n" for e in not_text), encoding="utf-8")

    stats = collections.Counter()
    for s in available:
        stats.update(s["stats"])
    rep = [f"[muktabodha] {ms_id}: {etext.name} + {pdf.name}", f"  made {time.strftime('%Y-%m-%d %H:%M')} by {NAME} {__version__}",
           "  e-text header      : " + "; ".join(f"{k}: {v}" for k, v in meta.items() if k in ("catalog number", "uniform title", "author", "manuscript", "licence")),
           f"  sides in the e-text: {len(sides)}, {len(available)} transcribed, {len(sides) - len(available)} not available "
           f"({', '.join(s['label'] for s in sides if not s['available']) or '-'}); ids "
           + ("0001_1, 0001_2, ... in the e-text's order" if sequential else "from the e-text's folio numbers"),
           f"  lines              : {sum(len(s['lines']) for s in available)}; lines a side "
           + ", ".join(f"{n}: {c} sides" for n, c in sorted(counts.items(), key=lambda x: -x[1])) + f" (usual {usual})",
           f"  PDF                : {n_pages} pages; {len([e for e in plan if e['side']])} page images paired with sides, "
           f"{len(not_text)} not text" + (f"; two leaves a page cut apart" if split else "") + (f"; cut down to the leaf" if crop else "")
           + (f"; scaled to {max_width} px wide" if max_width else "") + (f"; padded to {pad}x the leaf's height" if pad else ""),
           f"  page images        : {images_dir} ({len(plan)} images; {time.time() - t0:.0f} s)",
           f"  TEI                : {tei_path}", f"  not text           : {d / 'not_text.txt'} ({len(not_text)} pages)",
           f"  glosses as notes   : {sum(t.count('<note') for s in available for _, t, _ in s['lines'])}; gaps (* and ...) {stats['gap']}; "
           f"unclear (?) {stats['unclear']}; hyphens taken out {stats['hyphen']}; lone dots taken out {stats['dot']}"]
    rep += rep_match
    rep += ["", "Remarks in the e-text:"] + [f"  {n}" for n in notes] if notes else []
    rep += ["", "Sides to look at (an unusual number of lines, a gloss over a line break, odd characters):"]
    flagged = 0
    for s in available:
        ns = list(s["anomalies"])
        if len(s["lines"]) != usual:
            ns.insert(0, f"{len(s['lines'])} lines, not {usual}")
        if ns:
            flagged += 1
            rep.append(f"  {s['label'][:18]:>18} ({s['pid']}, e-text line {s['at']}): " + "; ".join(ns))
    if not flagged:
        rep.append("  none")
    rep += ["", "Page table: PDF page, image, what", ""]
    rep += [f"  {e['page']:4d}{e['part']:1s}  {e['image']}  {e['what']}" for e in plan]
    rep_path = d / "muktabodha.txt"
    rep_path.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep[:rep.index("Page table: PDF page, image, what") - 1]))
    print(f"  report             : {rep_path}")
    return plan


# ---------------------------------------------------------------- the check

def check(ms_id, model=None, seg="seg_torch", worst=12):
    """Every paired page read by the recogniser and set against the transcription. A page is paired right when each
    of its rows reads closer to its own line of the transcription than to any other line of the page: a page whose
    rows read as other lines (or as the lines before or after: a shift) is paired with the wrong side, or has a
    line missed or doubled. The CER itself says how far the hand is from what the recogniser knows."""
    import numpy as np
    from PIL import Image
    from rapidfuzz.distance import Levenshtein
    from .config import MANUSCRIPTS
    from .linecut import band_of, cut_line
    from .predict import DEFAULT_MODEL, model_label, read_line
    from .recognizer import load_model
    if seg == "seg":
        raise SystemExit("--check reads the rows align keeps beside its PAGE XML (pagexml_<seg>/*.json): --seg seg_torch or another of ours")
    xml_dir = ms_dir(ms_id) / f"pagexml_{seg}"
    pages = sorted(xml_dir.glob("*.json"))
    if not pages:
        raise SystemExit(f"{xml_dir}: no paired pages; run gt.extract, segment and align --seg={seg} first")
    path = Path(model) if model else DEFAULT_MODEL
    net, chars, ck = load_model(path)
    print(f"[check] {ms_id}: {len(pages)} paired pages ({seg}) read with {model_label(path, ck)}")
    results, t = [], time.time()
    for k, js in enumerate(pages, 1):
        rows = json.loads(js.read_text(encoding="utf-8"))
        page = np.asarray(Image.open(MANUSCRIPTS[ms_id]["images"] / f"{js.stem}.jpg").convert("RGB"))
        truths = [r["text"].replace(" ", "") for r in rows]
        lines = []
        for li, r in enumerate(rows, 1):
            above, below = band_of(r["baseline"], r["outline"])
            text, _ = read_line(net, chars, Image.fromarray(cut_line(page, r["baseline"], above, below)))
            lines.append((li, Levenshtein.distance(text, truths[li - 1]), len(truths[li - 1]), text, truths[li - 1]))
        cer = 100.0 * sum(e for _, e, _, _, _ in lines) / max(1, sum(n for _, _, n, _, _ in lines))
        fit = []                          # for each row, which line of the page its reading is closest to
        for li, _, _, text, _ in lines:
            dd = [Levenshtein.normalized_distance(text, tr) for tr in truths]
            fit.append(min(range(len(truths)), key=lambda j: dd[j]) + 1)
        own = sum(1 for li, f in zip(range(1, len(lines) + 1), fit) if f == li)
        shifted = sum(1 for li, f in zip(range(1, len(lines) + 1), fit) if f in (li - 1, li + 1))
        verdict = ("wrong side or lines out of place" if own < len(lines) / 2 and shifted < len(lines) / 2 else
                   "shifted by a line" if shifted >= len(lines) / 2 else "")
        results.append((js.stem, cer, lines, fit, verdict))
        if k % 25 == 0 or k == len(pages):
            print(f"  {k}/{len(pages)} pages, {(time.time() - t) / k:.1f} s a page", flush=True)
    cers = sorted(c for _, c, _, _, _ in results)
    median = statistics.median(cers)
    bad = [r for r in results if r[4]]
    rep = [f"[check] {ms_id}: {len(results)} paired pages read with {model_label(path, ck)} ({seg}); greedy, no language model",
           f"  CER a page: median {median:.1f} %, quartiles {cers[len(cers) // 4]:.1f} / {cers[3 * len(cers) // 4]:.1f} %, "
           f"best {cers[0]:.1f} %, worst {cers[-1]:.1f} %  (how far this hand is from what the recogniser knows)",
           f"  rows reading as their own line: {sum(sum(1 for li, f in enumerate(r[3], 1) if f == li) for r in results)} of "
           f"{sum(len(r[2]) for r in results)}",
           f"  pages whose rows do not read as their own lines: {len(bad)}"
           + (" - paired with the wrong side, or a line missed or doubled; look before training" if bad else "")]
    for stem, cer, lines, fit, verdict in sorted(bad, key=lambda r: -r[1]):
        rep.append(f"    {stem}  {cer:5.1f} %  {verdict}; row -> line: " + " ".join(f"{li}->{f}" for li, f in enumerate(fit, 1)))
    worst_lines = sorted(((100 * e / max(1, n), stem, li, text, truth) for stem, _, lines, _, _ in results for li, e, n, text, truth in lines),
                         reverse=True)[:worst]
    rep.append(f"  the {worst} lines read worst (a reading much longer than its line: a gloss cut with the row):")
    for lc, stem, li, text, truth in worst_lines:
        rep.append(f"    {stem} line {li}  {lc:5.1f} %\n      text : {truth}\n      read : {text}")
    rep += ["", "Every page: image, CER %, CER of each line, and the line each row reads as where it is not its own"]
    for stem, cer, lines, fit, verdict in results:
        odd = " ".join(f"{li}->{f}" for li, f in enumerate(fit, 1) if f != li)
        rep.append(f"  {stem}  {cer:5.1f}  " + " ".join(f"{100 * e / max(1, n):.0f}" for _, e, n, _, _ in lines) + (f"   {odd}" if odd else ""))
    out = ms_dir(ms_id) / "muktabodha_check.txt"
    out.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep[:rep.index("")]))
    print(f"  written to {out}")
    return results


# ---------------------------------------------------------------- the command line

def load_config(ms_id, a):
    cfg = {}
    f = CONFIGS / f"{ms_id}.json"
    if f.exists():
        cfg = json.loads(f.read_text(encoding="utf-8"))
    for k in ("etext", "pdf", "into", "first_text_page", "pages", "skip", "match", "sides_per_page", "crop", "max_width", "pad", "model"):
        v = getattr(a, k, None)
        if v not in (None, [], ""):
            cfg[k] = v
    if a.sequential_ids:
        cfg["sequential_ids"] = True
    for k in ("etext", "pdf", "into"):
        if k in cfg:
            p = Path(cfg[k])
            if not p.is_absolute():
                cfg[k] = str(p if p.exists() else TRAINING / p)
    for k in ("etext", "pdf", "into"):
        if k not in cfg:
            raise SystemExit(f"{ms_id}: no {k} given, on the command line or in {f}")
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ms_id", help="the manuscript's id, <library>_<number>: RTJ_4387 (Shri Raghunath Temple MSS Library, Jammu, no. 4387)")
    ap.add_argument("etext", nargs="?", help="Muktabodha's e-text (…-DEV.txt)")
    ap.add_argument("pdf", nargs="?", help="the scan, one page image a PDF page")
    ap.add_argument("--into", help="the manuscript's <text>/<commentary> folder under training_data")
    ap.add_argument("--first-text-page", type=int, help="the PDF page of the first transcribed side (1 = the PDF's first page)")
    ap.add_argument("--pages", help="the text pages in reading order, as 4-22,24,23")
    ap.add_argument("--skip", type=int, nargs="*", help="PDF pages that are no side of the transcription")
    ap.add_argument("--match", choices=["monotone", "free"], help="pair pages with sides by reading them")
    ap.add_argument("--sides-per-page", type=int, help="2: two leaves one above the other on a page")
    ap.add_argument("--crop", choices=["paper"], help="cut each page down to the leaf")
    ap.add_argument("--max-width", type=int, help="scale pages down to this width")
    ap.add_argument("--pad", type=float, help="set each leaf in a canvas this many times its height (a leaf of a few large lines)")
    ap.add_argument("--sequential-ids", action="store_true", help="number the sides in the e-text's order, whatever its labels")
    ap.add_argument("--model", help="the recogniser (matching and --check; default predict's)")
    ap.add_argument("--redo", action="store_true", help="extract the page images and read them again")
    ap.add_argument("--check", action="store_true", help="read the paired pages with the recogniser (after gt.extract, segment, align)")
    ap.add_argument("--seg", default="seg_torch", help="--check: the segmentation align paired (default seg_torch)")
    a = ap.parse_args()
    if a.check:
        check(a.ms_id, a.model, a.seg)
    else:
        build(a.ms_id, load_config(a.ms_id, a), a.redo)
