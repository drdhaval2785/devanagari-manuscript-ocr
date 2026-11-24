"""Pictures of the segmentation, to check it by eye.

    python -m msocr.show LDII_24140                 # every segmented page -> data/LDII_24140/show/<page>.jpg,
                                                    # all of them bound into data/LDII_24140/show/LDII_24140.pdf
    python -m msocr.show LDII_24140 2_1 7_2         # only these pages (page ids as in the TEI, or image names)
    python -m msocr.show LDII_24140 --warned        # only the pages the row finder has doubts about
    python -m msocr.show LDII_24140 2_1 --lines     # the page cut into its rows, one strip under another,
                                                    # with the text read or corrected for each, if there is any
    python -m msocr.show LDII_24140 --no-pdf        # pictures only

What a page picture shows (a tripāṭha or pañcapāṭha page shows its blocks instead, each framed and
labelled in Manuscribe's colour for that box, with its rows numbered inside):
  thin blue lines    what Kraken found: its baselines. These pieces are all the row finder has to work from.
  coloured bands     the rows the recogniser will read, numbered in the left margin. Neighbours alternate
                     red and green so the overlap of the bands can be seen. A band drawn with a dashed
                     outline is a row the row finder filled in between its neighbours because Kraken
                     missed it; it is read all the same. The word "gap" in the margin marks a blank gap
                     under a row that was left unfilled.
  caption on top     the row count, the row finder's note and, in red, its warning if it has one.
The warned pages are listed at the end so they can be looked at first. Nothing here changes the
segmentation: the pictures only show what predict would do with the page.
"""
import argparse
import json
import re
import sys
import textwrap
import warnings
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import numpy as np

from .align import _merge
from .segment import page_seg, reading_from
from .blocks import COLOR, EXPECTED, blocks_for_page, layout_of, page_blocks
from .config import MANUSCRIPTS, SEG, ms_dir
from .predict import page_rows

BLUE, RED, GREEN, BLACK = (0, 90, 255), (220, 40, 0), (0, 150, 60), (0, 0, 0)
FONTS = ["/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc", "/System/Library/Fonts/Supplemental/DevanagariMT.ttc",
         "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"]


def font(size):
    for f in FONTS:
        if Path(f).exists():
            try:
                return ImageFont.truetype(f, size)
            except OSError:
                pass
    return ImageFont.load_default(size=size)


def image_of(ms_id, page):
    """'2_1', '0002_1', '024140_00002_1' or '024140_00002_1.jpg' -> the image file name."""
    m = re.fullmatch(r"(?:\d{6}_)?0?(\d{1,4})_(\d)(?:\.jpg)?", page)
    if not m:
        raise SystemExit(f"{page!r} is not a page: expected something like 2_1, 0002_1 or 024140_00002_1.jpg")
    image = MANUSCRIPTS[ms_id]["image_name"](f"{int(m.group(1)):04d}_{m.group(2)}")
    if not (MANUSCRIPTS[ms_id]["images"] / image).exists():
        raise SystemExit(f"{ms_id} has no page image {image}")
    stem = image.rsplit(".", 1)[0]
    if not (ms_dir(ms_id) / page_seg(ms_id, stem) / (stem + ".json")).exists():
        raise SystemExit(f"{image} is not segmented yet: run  python -m msocr.segment {ms_id}  first")
    return image


def all_pages(ms_id):
    """the page images segmented as each is to be read (segment.page_seg)"""
    ms = MANUSCRIPTS[ms_id]
    images = sorted(p.name for p in ms["images"].glob("*.jpg"))
    return [i for i in images if (ms_dir(ms_id) / page_seg(ms_id, i.rsplit(".", 1)[0]) / (i.rsplit(".", 1)[0] + ".json")).exists()]


def _dashed(dr, pts, fill, width, dash=9):
    """A polyline drawn in dashes."""
    on, run = True, 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        t = 0.0
        while t < length:
            step = min(dash - run, length - t)
            if on:
                dr.line([(x0 + (x1 - x0) * t / length, y0 + (y1 - y0) * t / length),
                         (x0 + (x1 - x0) * (t + step) / length, y0 + (y1 - y0) * (t + step) / length)], fill=fill, width=width)
            t += step
            run += step
            if run >= dash:
                on, run = not on, 0.0


def _caption(im, lines, fnt, pad=8):
    """The image with the given (text, colour) lines in a strip above it."""
    lh = fnt.size + 6
    out = Image.new("RGB", (im.width, im.height + pad * 2 + lh * len(lines)), (245, 242, 235))
    dr = ImageDraw.Draw(out)
    for i, (text, colour) in enumerate(lines):
        dr.text((pad, pad + i * lh), text, fill=colour, font=fnt)
    out.paste(im, (0, pad * 2 + lh * len(lines)))
    return out


def draw_blocks(ms_id, image):
    """A tripāṭha or pañcapāṭha page: its blocks in Manuscribe's box colours, each with its rows."""
    d = ms_dir(ms_id)
    stem = image.rsplit(".", 1)[0]
    seg = json.load(open(d / SEG / f"{stem}.json"))
    im = Image.open(MANUSCRIPTS[ms_id]["images"] / image).convert("RGBA")
    gray = np.asarray(im.convert("L"))
    blocks, note = page_blocks(seg, im.size, gray, EXPECTED.get(layout_of(ms_id)))
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    for ln in seg["lines"]:
        dr.line([tuple(p) for p in ln["baseline"]], fill=BLUE + (255,), width=2)
    lab = font(18)
    warns = []
    for box, b in blocks.items():
        col = COLOR[box]
        x0, y0, x1, y1 = b["rect"]
        dr.rectangle([x0 - 3, y0 - 3, x1 + 3, y1 + 3], outline=col + (255,), width=3)
        fnt = font(max(12, int(0.55 * b["pitch"]) if b["pitch"] else 14))
        for i, (base, poly) in enumerate(b["rows"], 1):
            dr.polygon(poly, fill=col + (26 if i % 2 else 50,))
            dr.line(base, fill=col + (255,), width=2)
            dr.text((x0 + 3, base[0][1] - fnt.size - 1), str(i), fill=col + (255,), font=fnt)
        title = f"{box} · {len(b['rows'])} rows"
        tw = dr.textlength(title, font=lab)
        dr.rectangle([x0 - 3, y0 - 3 - lab.size - 6, x0 - 3 + tw + 8, y0 - 3], fill=col + (255,))
        dr.text((x0 + 1, y0 - 3 - lab.size - 4), title, fill=(255, 255, 255, 255), font=lab)
        if b["warn"]:
            warns.append(f"{box}: {b['warn']}")
    im = Image.alpha_composite(im, layer).convert("RGB")
    cap = font(16)
    head = f"{ms_id}  {stem}   {layout_of(ms_id)}: " + ", ".join(f"{box} {len(b['rows'])}" for box, b in blocks.items()) + f"   ({len(seg['lines'])} Kraken pieces)"
    width = max(40, int((im.width - 16) / (cap.size * 0.48)))
    warn = "; ".join(warns) or None
    lines = [(head, BLACK)] + [(t, (90, 90, 90)) for t in textwrap.wrap(note, width)] + [(t, RED) for t in textwrap.wrap(warn or "", width)]
    return _caption(im, lines, cap), warn


def draw_page(ms_id, image):
    """-> (picture of the page with Kraken's pieces and the rows drawn on it, warning or None)."""
    if layout_of(ms_id) in ("tripatha", "panchapatha"):
        return draw_blocks(ms_id, image)
    d = ms_dir(ms_id)
    stem = image.rsplit(".", 1)[0]
    rows, block, pitch, size, note, warn = page_rows(ms_id, image)
    seg = json.load(open(d / SEG / f"{stem}.json"))
    im = Image.open(MANUSCRIPTS[ms_id]["images"] / image).convert("RGBA")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    for ln in seg["lines"]:
        dr.line([tuple(p) for p in ln["baseline"]], fill=BLUE + (255,), width=2)
    fnt, small = font(max(14, int(0.55 * pitch) if pitch else 18)), font(12)
    for i, r in enumerate(rows, 1):
        base, poly = _merge(r, block, pitch)
        col = RED if i % 2 else GREEN
        dr.polygon(poly, fill=col + (38,))
        if r[0].get("filled"):
            _dashed(dr, poly, col + (255,), 2)
        else:
            dr.line(poly + poly[:1], fill=col + (255,), width=2)
        dr.line(base, fill=col + (255,), width=2)
        label = str(i)
        tw = dr.textlength(label, font=fnt)
        dr.text((max(2, poly[0][0] - tw - 8), base[0][1] - fnt.size - 2), label, fill=col + (255,), font=fnt)
        if r[0].get("blank_after"):
            dr.text((max(2, poly[0][0] - tw - 8), base[0][1] + 4), "gap", fill=BLACK + (255,), font=small)
    im = Image.alpha_composite(im, layer).convert("RGB")
    cap = font(16)
    head = f"{ms_id}  {stem}   {len(rows)} rows, {len(seg['lines'])} Kraken pieces" + (f", pitch {pitch:.0f}px" if pitch else "")
    width = max(40, int((im.width - 16) / (cap.size * 0.48)))
    lines = [(head, BLACK)] + [(t, (90, 90, 90)) for t in textwrap.wrap(note, width)] + [(t, RED) for t in textwrap.wrap(warn or "", width)]
    return _caption(im, lines, cap), warn


def _strip(im, poly):
    """The part of the page inside the outline, everything outside it white."""
    xs, ys = [x for x, _ in poly], [y for _, y in poly]
    box = (max(0, min(xs)), max(0, min(ys)), min(im.width, max(xs) + 1), min(im.height, max(ys) + 1))
    crop = im.crop(box)
    mask = Image.new("L", crop.size, 0)
    ImageDraw.Draw(mask).polygon([(x - box[0], y - box[1]) for x, y in poly], fill=255)
    return Image.composite(crop, Image.new("RGB", crop.size, (255, 255, 255)), mask)


def draw_lines(ms_id, image):
    """-> (the page's rows as strips one under another, block by block, each with its text if the page has been read, warning)."""
    from . import pagetext
    d = ms_dir(ms_id)
    stem = image.rsplit(".", 1)[0]
    if layout_of(ms_id) in ("tripatha", "panchapatha"):
        blocks, size, note = blocks_for_page(ms_id, image)
        parts = [(box, b["rows"], [r[0].get("filled") for r in b["raw"]], [r[0].get("blank_after") for r in b["raw"]]) for box, b in blocks.items()]
        warn = "; ".join(f"{box}: {b['warn']}" for box, b in blocks.items() if b["warn"]) or None
    else:
        rows, block, pitch, size, note, warn = page_rows(ms_id, image)
        parts = [("main", [_merge(r, block, pitch) for r in rows], [r[0].get("filled") for r in rows], [r[0].get("blank_after") for r in rows])]
    im = Image.open(MANUSCRIPTS[ms_id]["images"] / image).convert("RGB")
    text, source = None, None
    for sub in ("corrected", "drafts"):
        f = d / sub / f"{stem}.txt"
        if f.exists():
            text, source = pagetext.read(f), sub
            break
    fnt, small, cap, lab = font(24), font(14), font(16), font(18)
    gutter, gap = 56, 10
    strips = [(box, [_strip(im, poly) for _, poly in rows], filled, blank) for box, rows, filled, blank in parts]
    w = gutter + max([s.width for _, ss, _, _ in strips for s in ss] or [im.width]) + 2 * gap
    th = fnt.size + 10
    h = gap
    for box, ss, filled, blank in strips:
        shown = text is not None and len(text.get(box, [])) == len(ss)
        h += (lab.size + 10 if len(strips) > 1 else 0) + sum(s.height + (th if shown else 0) + gap + (12 if b else 0) for s, b in zip(ss, blank))
    out = Image.new("RGB", (w, max(h, 40)), (245, 242, 235))
    dr = ImageDraw.Draw(out)
    y = gap
    for box, ss, filled, blank in strips:
        col = COLOR.get(box, RED)
        shown = text is not None and len(text.get(box, [])) == len(ss)
        if len(strips) > 1:
            dr.text((gutter, y), f"{box} · {len(ss)} rows" + ("" if shown or text is None else f"  ({len(text.get(box, []))} lines of text: not shown)"), fill=col, font=lab)
            y += lab.size + 10
        for i, (s, f, b) in enumerate(zip(ss, filled, blank), 1):
            out.paste(s, (gutter, y))
            dr.rectangle([gutter - 1, y - 1, gutter + s.width, y + s.height], outline=col)
            dr.text((6, y + s.height // 2 - fnt.size // 2), str(i), fill=col, font=fnt)
            if f:
                dr.text((6, y + s.height // 2 + fnt.size // 2), "filled", fill=col, font=small)
            y += s.height
            if shown:
                dr.text((gutter, y + 4), text[box][i - 1], fill=BLACK, font=fnt)
                y += th
            if b:
                dr.text((gutter, y + 2), "blank gap: nothing filled in", fill=(90, 90, 90), font=small)
                y += 12
            y += gap
    n = sum(len(ss) for _, ss, _, _ in strips)
    head = f"{ms_id}  {stem}   {n} rows" + (f", text from {source}/" if text is not None else ", not read yet")
    width = max(40, int((out.width - 16) / (cap.size * 0.48)))
    lines = [(head, BLACK)] + [(t, (90, 90, 90)) for t in textwrap.wrap(note, width)] + [(t, RED) for t in textwrap.wrap(warn or "", width)]
    return _caption(out, lines, cap), warn


def write_pdf(jpegs, out, dpi=150):
    """Bind JPEG files into one PDF, one per page, as they are (no re-encoding), at the given print size."""
    n = len(jpegs)
    offsets = {}
    with open(out, "wb") as f:
        f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

        def obj(num, body):
            offsets[num] = f.tell()
            f.write(f"{num} 0 obj\n".encode() + body + b"\nendobj\n")

        kids = " ".join(f"{5 + 3 * i} 0 R" for i in range(n))
        obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        obj(2, f"<< /Type /Pages /Count {n} /Kids [{kids}] >>".encode())
        for i, path in enumerate(jpegs):
            with Image.open(path) as im:
                w, h = im.size
                assert im.mode == "RGB" and im.format == "JPEG", path
            data = Path(path).read_bytes()
            pw, ph = w * 72 / dpi, h * 72 / dpi
            obj(3 + 3 * i, f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB "
                           f"/BitsPerComponent 8 /Filter /DCTDecode /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream")
            content = f"q {pw:.2f} 0 0 {ph:.2f} 0 0 cm /Im0 Do Q".encode()
            obj(4 + 3 * i, f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream")
            obj(5 + 3 * i, f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw:.2f} {ph:.2f}] "
                           f"/Resources << /XObject << /Im0 {3 + 3 * i} 0 R >> >> /Contents {4 + 3 * i} 0 R >>".encode())
        total = 3 + 3 * n
        xref = f.tell()
        f.write(f"xref\n0 {total}\n0000000000 65535 f \n".encode())
        for num in range(1, total):
            f.write(f"{offsets[num]:010d} 00000 n \n".encode())
        f.write(f"trailer\n<< /Size {total} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())


def show(ms_id, pages=None, lines=False, pdf=True, warned=False, verbose=True):
    d = ms_dir(ms_id)
    out = d / ("show" if SEG == "seg" else f"show_{SEG}")      # pictures per segmentation
    out.mkdir(exist_ok=True)
    images = [image_of(ms_id, p) for p in pages] if pages else all_pages(ms_id)
    if not images:
        raise SystemExit(f"{ms_id} has no segmented pages: run  python -m msocr.segment {ms_id}  first")
    made, flagged = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, image in enumerate(images, 1):
            stem = image.rsplit(".", 1)[0]
            with reading_from(page_seg(ms_id, stem)):           # the lines the page is read from
                pic, warn = (draw_lines if lines else draw_page)(ms_id, image)
            if warned and not warn:
                continue
            path = out / (f"{stem}_lines.jpg" if lines else f"{stem}.jpg")
            pic.save(path, quality=85)
            made.append(path)
            if warn:
                flagged.append((stem, warn))
            if verbose and (k % 25 == 0 or k == len(images)):
                print(f"  {k}/{len(images)} pages drawn", file=sys.stderr)
    if not made:
        print(f"[show] {ms_id}: none of the {len(images)} pages has a warning; nothing drawn")
        return []
    print(f"[show] {ms_id}: {len(made)} picture{'s' if len(made) > 1 else ''} in {out}" + ("" if len(made) > 1 else f": {made[0].name}"))
    if pdf and len(made) > 1:
        name = f"{ms_id}{'_lines' if lines else ''}{'_warned' if warned else ''}.pdf"
        write_pdf(made, out / name)
        print(f"  as one PDF   : {out / name}  ({(out / name).stat().st_size / 2**20:.0f} MB, one page per sheet, in page order)")
    if flagged:
        print(f"  {len(flagged)} page{'s' if len(flagged) > 1 else ''} with a warning, to look at first:")
        for stem, warn in flagged[:40]:
            print(f"    {stem}: {warn}")
        if len(flagged) > 40:
            print(f"    ... and {len(flagged) - 40} more; every caption carries its own warning")
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ms_id")
    ap.add_argument("pages", nargs="*", help="pages to draw, like 2_1 or 024140_00002_1.jpg; all segmented pages if none")
    ap.add_argument("--lines", action="store_true", help="cut each page into its rows, one strip under another")
    ap.add_argument("--warned", action="store_true", help="only pages the row finder has a warning for")
    ap.add_argument("--no-pdf", action="store_true", help="do not bind the pictures into a PDF")
    a = ap.parse_args()
    if a.ms_id not in MANUSCRIPTS:
        raise SystemExit(f"unknown manuscript {a.ms_id}; known: {', '.join(sorted(MANUSCRIPTS))}")
    show(a.ms_id, a.pages or None, lines=a.lines, pdf=not a.no_pdf, warned=a.warned)
