"""A manuscript of the MIDF/eGangotri dataset (huggingface.co/datasets/tadad/midf-egangotri-sanskrit, "segmentation
v1.1": page images whose lines, with their text, a reviewer accepted) made into a manuscript the pipeline learns from,
with its rows found by our own segmenter and its lines cut by our own cutter, as for every other manuscript.

    .venv/bin/python -m msocr.midf RTJ_5753 --midf M00639 --into tripurasundarisahasranama/mula
    .venv/bin/python -m msocr.midf RTJ_5753          # again, from data/midf/RTJ_5753.json

  1. The pages are fetched into training_data/midf/ (pinned to one revision, checked against the dataset's checksums;
     tools/midf_download.py fetches many manuscripts ahead, with its progress).
  2. Each photo is cut down to the leaf (muktabodha.crop_to_paper) when every line of the dataset lies inside the cut,
     else kept whole (a close-up, which the crop would cut in half): training_data/<into>/manuscripts/<id>/, named as
     the corpus names pages, the folio number being the PDF's page number. The text goes into Manuscribe's TEI,
     training_data/<into>/transcriptions/<id>.xml. A page marker of the e-text at the head of a line ('पृ० २५)'),
     which is on no leaf, is taken out.
  3. gt.extract, the U-Net (data/<ms>/seg_torch/) and the row finder (align.find_rows).
  4. Pairing, line by line, in place of align's (do not run align on such a manuscript: it would pair by row count
     alone): a line of the dataset's is paired when exactly one of our rows lies where it does, below the row of the
     line paired before it; rows that lie on no line of theirs (a library stamp, an empty strip) are dropped. Where
     their line lies is measured from our baseline down to their line along the letters' tops, in their line pitch;
     the usual distance is taken from the pages whose row count agrees. A line is left out when its row takes in
     writing standing apart from it (a margin label, a table cut with it), or stops short of writing that belongs to
     it (a verse number in the margin, a last letter: the text would say more than the picture shows).
  5. Every paired row is read by the recogniser (greedy); a line whose row reads closer to another line of the page
     than to its own, or whose text holds an illegible stretch (<gap/>), is left out. A page with fewer than half its
     lines paired is left out whole.
  6. The pages are split as align splits them (under 60 pages: a tenth to test, a tenth to validation), the record kept
     in split_pages.tsv, and the lines cut into data/<ms>/lines_seg_torch/ (--no-cut: not yet). A manuscript with cut
     training lines joins the next `recognizer pool` run.

Written to data/<ms>/midf.txt: what was left out and why, and the lines read worst; pictures of the pages with a
line left out in data/<ms>/preview_seg_torch/. The transcriptions are
Muktabodha's e-texts (CC BY-NC 4.0); the dataset's lines were reviewed, their text not character by character.
"""
import argparse
import collections
import csv
import hashlib
import json
import random
import re
import shutil
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.sax.saxutils import escape

from . import NAME, __version__, config
from .config import DATA, LIBRARIES, MANUSCRIPTS, TRAINING, ms_dir
from .muktabodha import ALLOWED, MARKER, clean, crop_to_paper, norm, pad_height, split_notes, xml_text
from .muktabodha import LIBRARIES as PLACES            # the libraries' names and places

REPO = "https://huggingface.co/datasets/tadad/midf-egangotri-sanskrit/resolve/"
REVISION = "6db1c00265dc00eb6510ac11499e67f4debb1369"       # 2026-10-11; the dataset has no tag for segmentation v1.1
SEGMENTATION = "data/segmentation/v1.1/"
CACHE = TRAINING / "midf"
CONFIGS = DATA / "midf"
SEG = "seg_torch"
TOLERANCE = 0.35        # how far, in line pitches, a row may lie from the usual distance and still be a line's row
COVER = 0.9             # the part of their line's width a row must span
PITCH_RANGE = (0.75, 1.33)      # our line pitch against theirs: outside it the row's band is cut too tall or too low
SMALL = 60              # fewer paired pages than this: a tenth each to test and validation
CROP_SLACK = 0.005      # a line may reach this far (of the width or height) outside the cut and still count as inside
GAP = 0.6               # writing past an end of their line is apart from it behind a blank of this many pitches
INKED = 0.1             # a column of the letters' bodies holds ink when this part of it is dark
SHORT = 0.15            # their line running further than this, in pitches, past an end of our row is looked at for ink
SHORT_INK = 0.3         # ... and that stretch holds writing if it is this dark against the line itself
MIN_PART = 0.5          # a page with less than this part of its lines paired is left out whole


# ---------------------------------------------------------------- the dataset

def _download(url):
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as h:
                return h.read()
        except urllib.error.HTTPError as e:
            if attempt == 5 or e.code in (401, 403, 404):
                raise
            time.sleep(int(e.headers.get("Retry-After") or 60) if e.code == 429 else 5 * (attempt + 1))
        except OSError:
            if attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))


def _fetch(path, digest=None):
    """The dataset's file `path`, from the cache or the Hub, checked against its SHA-256 when one is known."""
    local = CACHE / path
    if local.exists() and (digest is None or hashlib.sha256(local.read_bytes()).hexdigest() == digest):
        return local
    data = _download(REPO + REVISION + "/" + path)
    if digest is not None and hashlib.sha256(data).hexdigest() != digest:
        raise SystemExit(f"{path}: the file fetched does not match the dataset's checksum")
    local.parent.mkdir(parents=True, exist_ok=True)
    part = local.with_name(local.name + ".part")
    part.write_bytes(data)
    part.replace(local)
    return local


def fetch(catalog):
    """-> (the catalogue row of the manuscript, [{page, number, image, xml, annotation}] in page order)"""
    rows = {r["muktabodha_catalog_id"]: r for r in
            csv.DictReader(_fetch("metadata/manuscripts.csv").read_text(encoding="utf-8").splitlines())}
    sums = {}
    for line in _fetch(SEGMENTATION + "checksums.sha256").read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.removeprefix("./")] = digest
    manifest = json.loads(_fetch(SEGMENTATION + "manifest.json", sums.get("manifest.json")).read_text(encoding="utf-8"))
    ids = sorted(p["page_id"] for p in manifest["pages"] if p["manuscript_id"] == catalog)
    if not ids or catalog not in rows:
        raise SystemExit(f"{catalog}: no pages in segmentation v1.1")
    if not all(re.fullmatch(re.escape(catalog) + r"-p\d{4}", i) for i in ids):
        raise SystemExit(f"{catalog}: unexpected page ids in the manifest")

    def one(pid):
        files = {"image": "image.jpg", "xml": f"{pid}.xml", "annotation": "annotation.json"}
        got = {}
        for key, name in files.items():
            rel = f"pages/{pid}/{name}"
            if rel not in sums:
                raise SystemExit(f"{rel}: no checksum in the dataset's list")
            got[key] = _fetch(SEGMENTATION + rel, sums[rel])
        return {"page": pid, "number": int(pid.rsplit("p", 1)[1]), **got}

    with ThreadPoolExecutor(6) as pool:
        return rows[catalog], list(pool.map(one, ids))


def their_lines(page):
    """-> [(points of the line along the letters' tops, text)] from the page's PAGE XML."""
    from lxml import etree
    root = etree.parse(str(page["xml"]), etree.XMLParser(resolve_entities=False, no_network=True)).getroot()
    ns = root.tag.split("}")[0] + "}"
    out = []
    for line in root.iter(ns + "TextLine"):
        pts = [tuple(map(float, p.split(","))) for p in line.find(ns + "Baseline").get("points").split()]
        out.append((pts, line.findtext(f"{ns}TextEquiv/{ns}Unicode") or ""))
    return out


# ---------------------------------------------------------------- the pages and the TEI

def prepare_image(im, lines, pad=None):
    """-> (image, dx, dy, cropped): cut to the leaf when all their lines lie inside the cut; (x, y) of the photo lies
    at (x + dx, y + dy) of the image returned."""
    dx = dy = 0
    boxes, real = [], im.crop
    im.crop = lambda box: (boxes.append(box), real(box))[1]
    new, done = crop_to_paper(im)
    del im.crop
    cropped = False
    if done:
        x0, y0, x1, y1 = boxes[-1]
        sx, sy = CROP_SLACK * im.width, CROP_SLACK * im.height
        if all(x0 - sx <= x <= x1 + sx and y0 - sy <= y <= y1 + sy for pts, _ in lines for x, y in pts):
            # crop_to_paper sets the leaf in a margin of 4 % of its width and height
            dx, dy = int(0.04 * (x1 - x0)) - x0, int(0.04 * (y1 - y0)) - y0
            im, cropped = new, True
    if pad:
        new = pad_height(im, pad)
        dy += (new.height - im.height) // 2
        im = new
    return im, dx, dy, cropped


def page_text(lines, stats):
    """Their lines as TEI and as plain text, glosses in brackets taken out as notes: -> ([(curve, tei, plain)], remarks).
    The e-text's page marker at the head of a line is dropped; a line that is nothing but a gloss goes, its curve
    with it."""
    texts, marked = [], []
    for i, (_, t) in enumerate(lines, 1):
        m = MARKER.match(t)
        if m:
            marked.append(f"line {i}: the e-text's page marker taken out: {m.group(0).strip()}")
            t = t[m.end():]
        texts.append((i, t))
    split, remarks = split_notes(texts)
    remarks = marked + remarks
    out = []
    for (curve, _), (no, tokens) in zip(lines, split):
        plain = norm(clean("".join(t for k, t in tokens if k == "t"), collections.Counter()))
        if not plain:
            remarks.append(f"line {no}: nothing but a gloss; left out")
            continue
        odd = sorted({c for c in plain if not ALLOWED.match(c)})
        if "पृ०" in plain:
            remarks.append(f"line {no}: 'पृ०' inside the line: is it on the leaf?")
        if odd:
            remarks.append(f"line {no}: characters that are not Devanagari: " + " ".join(f"{c} (U+{ord(c):04X})" for c in odd))
        tei = "".join(xml_text(t, stats) if k == "t" else f'<note place="margin">{escape(norm(t))}</note>' for k, t in tokens)
        out.append((curve, tei, plain.replace("*", "").replace("?", "")))
    return out, remarks


def tei(ms_id, row, pages, usual):
    number, catalog = ms_id.split("_")[-1], row["muktabodha_catalog_id"]
    library, place = PLACES.get(ms_id.split("_")[0], ("", ""))
    title = row["uniform_title"]
    who = f"{NAME} {__version__}, msocr/midf.py"
    out = ['<TEI xmlns="http://www.tei-c.org/ns/1.0">', "<teiHeader>", "\t<fileDesc>", "\t\t<titleStmt>",
           f"\t\t\t<title>{escape(title)}</title>",
           "\t\t\t<respStmt>", "\t\t\t\t<resp>diplomatic transcription</resp>",
           f"\t\t\t\t<name>Muktabodha Indological Research Institute, e-text {escape(catalog)}</name>", "\t\t\t</respStmt>",
           "\t\t\t<respStmt>", "\t\t\t\t<resp>matching of the text to pages and lines</resp>",
           f"\t\t\t\t<name>MIDF/eGangotri Sanskrit Manuscripts, segmentation v1.1, revision {REVISION[:10]}</name>", "\t\t\t</respStmt>",
           "\t\t\t<respStmt>", "\t\t\t\t<resp>TEI and page images from the dataset</resp>",
           f"\t\t\t\t<name>{escape(who)}</name>", "\t\t\t</respStmt>",
           "\t\t</titleStmt>", "\t\t<publicationStmt>", "\t\t\t<p>Not for distribution.</p>", "\t\t\t<availability>",
           f"\t\t\t\t<licence>The transcription is Muktabodha's e-text {escape(catalog)}, Creative Commons BY-NC 4.0.</licence>",
           "\t\t\t</availability>", "\t\t</publicationStmt>", "\t\t<sourceDesc>",
           f"\t\t\t<p>Transcribed from the manuscript number {escape(number)}"
           + (f" from {escape(library)}, {escape(place)}" if library else "") + ".</p>",
           "\t\t\t<msDesc>", "\t\t\t\t<msIdentifier>"]
    if place:
        out.append(f"\t\t\t\t\t<settlement>{escape(place)}</settlement>")
    if library:
        out.append(f"\t\t\t\t\t<repository>{escape(library)}</repository>")
    out += [f"\t\t\t\t\t<idno>{escape(number)}</idno>", "\t\t\t\t</msIdentifier>",
            "\t\t\t\t<physDesc>", '\t\t\t\t\t<objectDesc form="pothi">', "\t\t\t\t\t\t<layoutDesc>",
            f'\t\t\t\t\t\t\t<layout writtenLines="{usual}"><p>Linear layout. Lines per folio side: main {usual}.</p></layout>',
            "\t\t\t\t\t\t</layoutDesc>", "\t\t\t\t\t</objectDesc>", "\t\t\t\t</physDesc>", "\t\t\t</msDesc>",
            f"\t\t\t<p>Scan: {escape(row['archive_pdf_name'])}, {escape(row['archive_item_url'])}. The folio numbers below are "
            f"the PDF's page numbers.</p>",
            "\t\t</sourceDesc>", "\t</fileDesc>", "\t<revisionDesc>",
            f'\t\t<change when="{time.strftime("%Y-%m-%d")}">Made from the dataset by {escape(who)}: glosses in brackets as notes, '
            f'* as gaps, ? as unclear, | and || as daṇḍas; nothing corrected.</change>',
            "\t</revisionDesc>", "</teiHeader>", "<text>", "<body>", "<div>"]
    for p in pages:
        out.append(f'<pb n="{p["pid"]}"/>')
        out += [f'<lb n="{i}" break="no"/>{t}' for i, (_, t, _) in enumerate(p["lines"], 1)]
    out += ["</div>", "</body>", "</text>", "</TEI>"]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- pairing our rows with their lines

def _offset(base, curve, pitch):
    """-> (how far our baseline lies below their line, in pitches; the part of their line's width the row spans)"""
    import numpy as np
    cx, cy = np.array([p[0] for p in curve]), np.array([p[1] for p in curve])
    bx, by = np.array([p[0] for p in base], float), np.array([p[1] for p in base], float)
    lo, hi = max(cx.min(), bx.min()), min(cx.max(), bx.max())
    cover = max(0.0, hi - lo) / max(1.0, cx.max() - cx.min())
    xs = np.linspace(lo, max(lo, hi), 20)
    return float(np.median(np.interp(xs, bx, by) - np.interp(xs, cx, cy)) / pitch), float(cover)


def line_pitch(curves):
    mids = [statistics.median(y for _, y in c) for c in curves]
    return statistics.median(b - a for a, b in zip(mids, mids[1:])) if len(mids) >= 2 else None


def cut_short(gray, base, curve, pitch):
    """How far, in pitches, their line runs on into writing past an end of our row: 0 when the row spans the line, or
    what it leaves of their line is blank (their line drawn longer than the writing)."""
    import numpy as np
    cx, cy = np.array([q[0] for q in curve]), np.array([q[1] for q in curve])
    x0, x1 = max(0, int(cx.min()) + 2), min(gray.shape[1] - 1, int(cx.max()) - 2)
    left, right = min(q[0] for q in base), max(q[0] for q in base)
    xs = np.arange(x0, x1 + 1)
    rows = np.clip((np.interp(xs, cx, cy)[None, :] + np.arange(int(0.5 * pitch))[:, None]).astype(int), 0, gray.shape[0] - 1)
    band = gray[rows, xs[None, :]]
    own = band[:, (xs >= left) & (xs <= right)]
    if not own.size:
        return 0.0
    dark = (float(np.percentile(own, 5)) + float(np.percentile(own, 75))) / 2
    ink = float((own < dark).mean())
    worst = 0.0
    for outside in (xs < left, xs > right):
        n = int(outside.sum())
        if n > SHORT * pitch and float((band[:, outside] < dark).mean()) > SHORT_INK * ink:
            worst = max(worst, n / pitch)
    return float(worst)


def pair_lines(gray, merged, curves, pitch, usual_offset):
    """Which of our rows (baseline, outline) is each of their lines, one by one: -> [(row index, None) or (None, why)]."""
    out, last = [], -1
    for curve in curves:
        near = []
        for k, (base, _) in enumerate(merged):
            off, cover = _offset(base, curve, pitch)
            if abs(off - usual_offset) <= TOLERANCE and cover >= COVER:
                near.append(k)
        if len(near) != 1:
            out.append((None, "no row of ours lies on it" if not near else f"{len(near)} rows lie on it"))
            continue
        k = near[0]
        if k <= last:
            out.append((None, "its row is not below the row of the line before"))
            continue
        apart = writing_apart(gray, merged[k][0], curve, pitch, usual_offset)
        short = cut_short(gray, merged[k][0], curve, pitch)
        if apart:
            out.append((None, f"its row takes in writing apart from the line, {apart:.1f} pitches past its end"))
        elif short:
            out.append((None, f"its row stops short of writing that belongs to it, by {short:.1f} pitches"))
        else:
            out.append((k, None))
            last = k
    return out


def writing_apart(gray, base, curve, pitch, offset):
    """How far past an end of their line, in pitches, the row runs into writing that stands apart from it, behind a
    blank of GAP pitches: a margin label cut with the row. 0 when there is none: blank paper after a short line, or
    writing that runs straight on (their line stops short of the line's end, which the row takes in rightly)."""
    import numpy as np
    bx, by = np.array([q[0] for q in base], float), np.array([q[1] for q in base], float)
    x0, x1 = max(0, int(bx.min())), min(gray.shape[1] - 1, int(bx.max()))
    xs = np.arange(x0, x1 + 1)
    top = np.interp(xs, bx, by) - offset * pitch          # along the letters' tops, as their line runs
    rows = np.clip((top[None, :] + np.arange(int(0.5 * pitch))[:, None]).astype(int), 0, gray.shape[0] - 1)
    band = gray[rows, xs[None, :]]                         # the letters' bodies, straightened
    left, right = min(x for x, _ in curve), max(x for x, _ in curve)
    own = band[:, (xs >= left) & (xs <= right)]
    if not own.size:
        return 0.0
    dark = (float(np.percentile(own, 5)) + float(np.percentile(own, 75))) / 2        # between ink and paper
    inked = (band < dark).mean(axis=0) > INKED
    worst = 0.0
    for end, step in ((left, -1), (right, 1)):
        k = int(np.clip(end - x0, 0, len(xs) - 1))
        blank = ink = 0
        apart = False
        while 0 <= k < len(xs):
            if inked[k]:
                ink += apart
                if ink >= 0.15 * pitch:                    # more than a speck
                    worst = max(worst, abs(xs[k] - end) / pitch)
                blank = 0
            else:
                blank += 1
                apart = apart or blank >= GAP * pitch
            k += step
    return float(worst)


# ---------------------------------------------------------------- the build

def _check_names(ms_id, catalog, into):
    if not re.fullmatch(r"(%s)_\d+" % "|".join(LIBRARIES), ms_id):
        raise SystemExit(f"{ms_id}: the id is <library>_<number>, the library one of {', '.join(LIBRARIES)}")
    if not re.fullmatch(r"[a-z0-9_-]+/[a-z0-9_-]+", into):
        raise SystemExit(f"--into {into}: give <text>/<commentary>, small letters")
    for f in (DATA / "muktabodha").glob("*.json"):
        if f"-{catalog}-" in Path(json.loads(f.read_text(encoding="utf-8"))["etext"]).name:
            raise SystemExit(f"{catalog}: this e-text is in the pipeline already, as {f.stem}; the same page must not be "
                             f"in one manuscript's training set and another's test set")
    for f in CONFIGS.glob("*.json"):
        if f.stem != ms_id and json.loads(f.read_text(encoding="utf-8"))["midf"] == catalog:
            raise SystemExit(f"{catalog}: imported already, as {f.stem}")
    if not (CONFIGS / f"{ms_id}.json").exists() and (ms_id in MANUSCRIPTS or (DATA / ms_id).exists()):
        raise SystemExit(f"{ms_id}: a manuscript of this id exists already and was not made by this tool")


def build(ms_id, cfg, model=None, cut=True):
    import numpy as np
    from PIL import Image, ImageDraw
    from rapidfuzz.distance import Levenshtein
    from . import gt, tseg
    from .align import SEED, TEST_PAGES, VAL_FRACTION, _merge, _page_xml, find_rows
    from .linecut import band_of, cut_line
    from .predict import DEFAULT_MODEL, model_label, read_line
    from .recognizer import build_cache, lines_dir, load_model

    catalog, into = cfg["midf"], cfg["into"]
    _check_names(ms_id, catalog, into)
    row, source = fetch(catalog)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    (CONFIGS / f"{ms_id}.json").write_text(json.dumps({**cfg, "title": row["uniform_title"], "revision": REVISION},
                                                      ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    d = ms_dir(ms_id)
    number = f"{int(ms_id.split('_')[1]):06d}"
    folder = TRAINING / into
    images = folder / "manuscripts" / ms_id
    images.mkdir(parents=True, exist_ok=True)

    # 2. the page images and the TEI
    pages, remarks, stats, whole = [], [], collections.Counter(), []
    for src in source:
        lines = their_lines(src)
        if json.loads(src["annotation"].read_text(encoding="utf-8"))["transcription_lines"] != [t for _, t in lines]:
            raise SystemExit(f"{src['page']}: the text of the PAGE XML and of annotation.json differ")
        kept, notes = page_text(lines, stats)
        remarks += [f"{src['page']} {n}" for n in notes]
        if not kept:
            remarks.append(f"{src['page']}: no text line; left out")
            continue
        pid = f"{src['number']:04d}_1"
        name = f"{number}_0{pid}.jpg"
        with Image.open(src["image"]) as photo:
            im, dx, dy, cropped = prepare_image(photo.convert("RGB"), lines, cfg.get("pad"))
        im.save(images / name, quality=95)
        if not cropped:
            whole.append(src["page"])
        pages.append({"page": src["page"], "pid": pid, "image": name, "lines": kept,
                      "curves": [[(x + dx, y + dy) for x, y in c] for c, _, _ in kept]})
    for stale in set(images.glob("*.jpg")) - {images / p["image"] for p in pages}:
        stale.unlink()
    usual = collections.Counter(len(p["lines"]) for p in pages).most_common(1)[0][0]
    tei_path = folder / "transcriptions" / f"{ms_id}.xml"
    tei_path.parent.mkdir(parents=True, exist_ok=True)
    tei_path.write_text(tei(ms_id, row, pages, usual), encoding="utf-8")

    # 3. register, extract the lines' text, find our rows
    MANUSCRIPTS[ms_id] = config._discover()[ms_id]
    gt.extract(ms_id, verbose=False)
    truth = collections.defaultdict(list)
    for r in gt.load(ms_id):
        truth[r["image"]].append(r["text"])
    tseg.segment_pages(ms_id, [p["image"] for p in pages], redo=True, quiet=True)
    for p in pages:
        seg = json.loads((d / SEG / p["image"].replace(".jpg", ".json")).read_text(encoding="utf-8"))
        with Image.open(images / p["image"]) as im:
            p["size"], gray = im.size, np.asarray(im.convert("L"))
        rows, block, pitch, _ = find_rows(seg, p["size"][0], gray)
        p["pitch"], p["merged"], p["gray"] = pitch, [_merge(r, block, pitch) for r in rows], gray

    # 4. pair
    pitches = [x for x in (line_pitch(p["curves"]) for p in pages) if x]
    if not pitches:
        raise SystemExit(f"{ms_id}: no page of two lines or more to measure the line pitch on")
    ms_pitch = statistics.median(pitches)
    agree = [_offset(b, c, line_pitch(p["curves"]) or ms_pitch)[0]
             for p in pages if len(p["merged"]) == len(p["curves"]) for (b, _), c in zip(p["merged"], p["curves"])]
    if not agree:
        raise SystemExit(f"{ms_id}: the row count agrees on no page; the pages need other preparation (try --pad)")
    usual_offset = statistics.median(agree)
    rejected, dropped = [], []          # pages left out whole; rows on no line
    for p in pages:
        theirs = line_pitch(p["curves"]) if len(p["curves"]) >= 3 else ms_pitch
        p["why"] = [None] * len(p["curves"])
        if not p["merged"]:
            rejected.append((p, "no row found"))
            p["pairs"] = None
        elif not PITCH_RANGE[0] <= p["pitch"] / theirs <= PITCH_RANGE[1]:
            rejected.append((p, f"line pitch {p['pitch']:.0f} px against their {theirs:.0f} px"))
            p["pairs"] = None
        else:
            p["pairs"] = pair_lines(p["gray"], p["merged"], p["curves"], theirs, usual_offset)

    # 5. read every paired row against its transcription
    path = Path(model) if model else DEFAULT_MODEL
    net, chars, ck = load_model(path)
    good, readings, left_lines = [], [], []
    for p in pages:
        if p["pairs"] is None:
            continue
        text = truth[p["image"]]
        assert len(text) == len(p["lines"]) == len(p["pairs"]), p["page"]
        page = np.asarray(Image.open(images / p["image"]).convert("RGB"))
        kept, faults = [], []
        for i, (k, why) in enumerate(p["pairs"]):
            if k is not None:
                base, outline = p["merged"][k]
                above, below = band_of(base, outline)
                read = read_line(net, chars, Image.fromarray(cut_line(page, base, above, below)))[0].replace(" ", "")
                nearest = min(range(len(text)), key=lambda j: Levenshtein.normalized_distance(read, text[j]))
                if text[nearest] != text[i]:
                    why = f"its row reads closer to line {nearest + 1} than to its own"
                elif "<gap" in p["lines"][i][1]:
                    why = "an illegible stretch (<gap/>) in the transcription"
                else:
                    kept.append((i, k, text[i], read, Levenshtein.distance(read, text[i])))
            if why:
                faults.append((i, why))
        p["why"] = dict(faults)
        if len(kept) < MIN_PART * len(text):
            i, why = faults[0]
            rejected.append((p, f"{len(kept)} of {len(text)} lines pair (line {i + 1}: {why})"))
            continue
        p["rows"] = [p["merged"][k] for _, k, _, _, _ in kept]
        p["text"] = [t for _, _, t, _, _ in kept]
        p["cer"] = 100 * sum(e for *_, e in kept) / max(1, sum(map(len, p["text"])))
        good.append(p)
        left_lines += [(p["page"], i + 1, why) for i, why in faults]
        if not faults and len(p["merged"]) > len(kept):
            dropped.append(f"{p['page']}: {len(p['merged']) - len(kept)} row(s) on no line of theirs")
        readings += [(100 * e / max(1, len(t)), p["page"], i + 1, t, r) for i, _, t, r, e in kept]

    # pictures of the pages with something left out: our rows, and their lines in green
    previews = d / f"preview_{SEG}"
    if previews.exists():
        shutil.rmtree(previews)
    previews.mkdir()
    short_pages = {page for page, _, _ in left_lines}
    for p in [p for p, _ in rejected] + [p for p in good if p["page"] in short_pages]:
        with Image.open(images / p["image"]) as src:
            im = src.convert("RGB")
        dr = ImageDraw.Draw(im)
        for i, (base, poly) in enumerate(p["merged"], 1):
            dr.polygon([tuple(q) for q in poly], outline=(0, 90, 255) if i % 2 else (255, 60, 0), width=3)
        for c in p["curves"]:
            dr.line(c, fill=(0, 200, 0), width=3)
        im.save(previews / p["image"])

    # 6. write the pairs as align writes them, split, cut
    xml_dir = d / f"pagexml_{SEG}"
    if xml_dir.exists():
        shutil.rmtree(xml_dir)
    xml_dir.mkdir()
    for p in good:
        stem = p["image"][:-4]
        triples = [(b, o, t) for (b, o), t in zip(p["rows"], p["text"])]
        (xml_dir / f"{stem}.xml").write_text(_page_xml(images / p["image"], p["size"], triples), encoding="utf-8")
        (xml_dir / f"{stem}.json").write_text(json.dumps(
            [{"baseline": [list(map(float, q)) for q in b], "outline": [list(map(float, q)) for q in o], "text": t}
             for b, o, t in triples], ensure_ascii=False), encoding="utf-8")
    stems = sorted(p["image"][:-4] for p in good)
    record_path = d / "split_pages.tsv"
    record = dict(l.split("\t") for l in record_path.read_text().splitlines() if "\t" in l) if record_path.exists() else {}
    if not record:
        shuffled = stems[:]
        random.Random(SEED).shuffle(shuffled)
        if len(shuffled) < SMALL:
            n_test = n_val = max(1, round(len(shuffled) / 10))
        else:
            n_test = TEST_PAGES
            n_val = max(1, round(VAL_FRACTION * (len(shuffled) - n_test)))
        record = {s: "test" if i < n_test else "val" if i < n_test + n_val else "train" for i, s in enumerate(shuffled)}
    for s in stems:
        record.setdefault(s, "train")
    record_path.write_text("".join(f"{s}\t{n}\n" for s, n in sorted(record.items())))
    sets = {n: [s for s in stems if record[s] == n] for n in ("train", "val", "test")}
    for n, v in sets.items():
        (d / f"{n}_{SEG}.txt").write_text("\n".join(str(xml_dir / f"{s}.xml") for s in v) + "\n")
    if lines_dir(ms_id, SEG).exists():
        shutil.rmtree(lines_dir(ms_id, SEG))
    if cut:
        build_cache(ms_id, verbose=False, seg=SEG)

    n_chars = sum(len(t) for p in good for t in p["text"])
    cer = sum(c * len(t) for c, _, _, t, _ in readings) / max(1, n_chars)
    cers = sorted(p["cer"] for p in good)
    count = {n: sum(len(p["text"]) for p in good if p["image"][:-4] in set(v)) for n, v in sets.items()}
    rep = [f"[midf] {ms_id}: {catalog}, {row['uniform_title']}",
           f"  made {time.strftime('%Y-%m-%d %H:%M')} by {NAME} {__version__} from dataset revision {REVISION[:10]}",
           f"  page images  : {images} ({len(pages)}; {len(pages) - len(whole)} cut to the leaf, {len(whole)} kept whole)",
           f"  TEI          : {tei_path} ({sum(len(p['lines']) for p in pages)} lines, usual {usual} a page)",
           f"  usual offset : our baseline {usual_offset:.2f} line pitches below their line",
           f"  paired       : {len(good)} of {len(pages)} pages, {sum(len(p['text']) for p in good)} of "
           f"{sum(len(p['lines']) for p in pages)} lines",
           "  split        : " + ", ".join(f"{n} {len(v)} pages / {count[n]} lines" for n, v in sets.items()),
           f"  lines cut    : {lines_dir(ms_id, SEG) if cut else 'not yet (--no-cut)'}",
           f"  read with {model_label(path, ck)}, greedy: CER {cer:.1f} %"
           + (f"; a page: median {statistics.median(cers):.1f} %, worst {cers[-1]:.1f} %" if cers else ""),
           f"  left out     : {len(rejected)} pages whole, and {len(left_lines)} single lines of other pages "
           f"(pictures in {previews})"]
    rep += [f"    {p['page']} ({p['image']}): {why}" for p, why in sorted(rejected, key=lambda r: r[0]["page"])]
    kinds = collections.Counter(re.sub(r",? (by )?[\d.]+ pitches.*$", "", re.sub(r"line \d+ than", "another line than", w)) for _, _, w in left_lines)
    rep += [f"  single lines left out, by reason : {len(left_lines)}"] + [f"    {n:5d}  {k}" for k, n in kinds.most_common()]
    rep += [f"      {page} line {i}: {why}" for page, i, why in left_lines]
    rep += [f"  rows dropped on paired pages : {len(dropped)} pages"] + [f"    {x}" for x in dropped]
    rep += [f"  kept whole   : {', '.join(whole) or '-'}"]
    if remarks:
        rep += ["  remarks on the text:"] + [f"    {x}" for x in remarks]
    rep += ["  the 12 lines read worst (look at these: a slip in the transcription, or damage):"]
    for c, page, i, t, r in sorted(readings, reverse=True)[:12]:
        rep.append(f"    {page} line {i}  {c:5.1f} %\n      text : {t}\n      read : {r}")
    (d / "midf.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))
    print(f"  report       : {d / 'midf.txt'}")
    return good, rejected


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ms_id", help="the id here, <library>_<manuscript number>, e.g. RTJ_5753")
    ap.add_argument("--midf", help="the dataset's manuscript id, e.g. M00639 (first run)")
    ap.add_argument("--into", help="<text>/<commentary> under training_data (first run)")
    ap.add_argument("--pad", type=float, help="set each leaf in a canvas this many times its height (a leaf of a few large lines)")
    ap.add_argument("--model", help="the recogniser for the reading check (default predict's)")
    ap.add_argument("--no-cut", action="store_true", help="pair and report, but cut no line images (stays out of the pool)")
    a = ap.parse_args()
    f = CONFIGS / f"{a.ms_id}.json"
    cfg = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    for k in ("midf", "into", "pad"):
        if getattr(a, k) is not None:
            if k != "pad" and cfg.get(k) not in (None, getattr(a, k)):
                raise SystemExit(f"{a.ms_id} was made with --{k} {cfg[k]}; to change it, remove the manuscript first")
            cfg[k] = getattr(a, k)
    if not cfg.get("midf") or not cfg.get("into"):
        ap.error("the first run needs --midf and --into")
    build(a.ms_id, cfg, a.model, not a.no_cut)
