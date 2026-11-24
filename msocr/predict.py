"""Read pages that have no transcription, producing a draft to correct.

For each page:
  1. the text lines are found (skipped if already done): by our U-Net, the default, or by Kraken (msocr.segment; a
     page cut before keeps the segmentation it was cut from, see segment.page_seg)
  2. the main text rows are picked out, exactly as in align.py
  3. each row is cut out and read by the recogniser
  4. the result is written under data/<ms>/:
       to_correct/<page>.txt  one line of text per manuscript line, top to bottom (on a tripāṭha or
                              pañcapāṭha page block by block after ;box{...} markers, see pagetext.py):
                              edit this file in place, then move it to corrected/.  Never written for
                              a page that already has a corrected/ file.
       drafts/<page>.txt      the same text, kept as the model's untouched reading
       review/<page>.html     each line's image above an editable box holding the draft;
                              open in a browser, correct, press "Save" to download <page>.txt
       lines/pred/            the line images

Corrected pages go in data/<ms>/corrected/<page>.txt (same name, same number of lines).
`finetune` trains on whatever is in that folder. A page in corrected/ is never read again: its text is paired with its
line images, which must stay as they were cut.

Confidence = how sure the model was, averaged over the characters of a line (0-100).
Low-confidence lines are tinted in the review page; check those first.
"""
import argparse
import html
import json
import os
import statistics
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import pagetext
from .align import _merge, _page_xml, fill_edge_rows, find_rows, uneven
from .blocks import blocks_for_page, layout_of
from .config import DATA, MANUSCRIPTS, SEG, ms_dir, not_text
from .recognizer import load_model, to_tensor
from .segment import cut_from, ensure_segmented, page_seg, pin, reading_from

DEFAULT_MODEL = DATA / "LDII_85885" / "models" / "run3" / "best.pt"


def read_line(net, chars, crop):
    x = to_tensor(crop)[None, None]
    with torch.no_grad():
        probs = net(x)[0].softmax(-1)
    best, idx = probs.max(-1)
    out, conf, prev = [], [], 0
    for p, v in zip(best.tolist(), idx.tolist()):
        if v != prev and v != 0:
            out.append(chars[v - 1])
            conf.append(p)
        prev = v
    return "".join(out), 100 * (sum(conf) / len(conf) if conf else 0.0)


def page_rows(ms_id, image):
    """The row finder's full result for a page: (rows, block, pitch, size, note, warning or None).
    Rows are lists of detection dicts as find_rows builds them (a row may carry "filled" or
    "blank_after"); predict and show build on this."""
    d = ms_dir(ms_id)
    seg = json.load(open(d / SEG / (image.rsplit(".", 1)[0] + ".json")))
    with Image.open(MANUSCRIPTS[ms_id]["images"] / image) as im:
        size = im.size
        gray = np.asarray(im.convert("L"))
    rows, block, pitch, note = find_rows(seg, size[0], gray)
    if not rows:
        return [], block, pitch, size, note, "no text rows found"
    if len(rows) >= 3:
        rows, edge_note = fill_edge_rows(rows, block, pitch, gray)
        if edge_note:
            note += "; " + edge_note
    warn = uneven(rows, pitch)
    if warn:
        warn += ": a line may be missing or a note included"
    missed = [b for b in note.split("; ") if b.startswith("Kraken missed") or "short line may be missing" in b
              or "may have been missed" in b]
    if missed:
        warn = (warn + "; " if warn else "") + "; ".join(missed)
    return rows, block, pitch, size, note, warn


def rows_for_page(ms_id, image):
    """Find main text rows on a page. Returns (merged rows as (baseline, outline), size, note, warning or None)."""
    rows, block, pitch, size, note, warn = page_rows(ms_id, image)
    return [_merge(r, block, pitch) for r in rows], size, note, warn


def model_label(path, ck):
    """'ft3 of LDII_19634 (pass 13, validation CER 0.83%)' for data/LDII_19634/models/ft3/best.pt."""
    path = Path(path)
    run, owner = path.parent.name, path.parent.parent.parent.name
    who = f"pooled model {run}" if owner == "pool" else f"{run} of {owner}"
    return f"{who} (pass {ck['pass']}, validation CER {ck['val_cer']:.2f}%)"


CUTTER = os.environ.get("MSOCR_CUTTER", "own")   # "own": msocr/linecut.py (default since 2026-11-24); "kraken": extract_polygons


def _read_rows(ms_id, image, size, merged, net, chars, lines_dir, stem, box):
    """Cut the rows (baseline, outline) out of the page and read them. -> [{file, text, conf}]"""
    if CUTTER == "own":
        from .linecut import band_of, cut_line
        page = np.asarray(Image.open(MANUSCRIPTS[ms_id]["images"] / image).convert("RGB"))
        lines = []
        for li, (base, poly) in enumerate(merged, 1):
            above, below = band_of(base, poly)
            crop = Image.fromarray(cut_line(page, base, above, below))
            name = pagetext.image_name(stem, box, li)
            if lines_dir:
                crop.save(lines_dir / name)
            text, conf = read_line(net, chars, crop)
            lines.append({"file": name, "text": text, "conf": conf})
        return lines
    from kraken.lib.segmentation import extract_polygons
    from kraken.lib.xml import XMLPage
    with tempfile.NamedTemporaryFile("w", suffix=".xml", encoding="utf-8", delete=False) as f:
        f.write(_page_xml(MANUSCRIPTS[ms_id]["images"] / image, size, [(b, p, "?") for b, p in merged]))
    try:
        seg = XMLPage(f.name, filetype="page").to_container()
    finally:
        os.unlink(f.name)
    im = Image.open(seg.imagename)
    lines = []
    for li, (crop, _) in enumerate(extract_polygons(im, seg), 1):
        name = pagetext.image_name(stem, box, li)
        if lines_dir:
            crop.save(lines_dir / name)
        text, conf = read_line(net, chars, crop)
        lines.append({"file": name, "text": text, "conf": conf})
    return lines


def read_page(ms_id, image, net, chars, lines_dir=None):
    """Cut a page into its rows, block by block on a tripāṭha or pañcapāṭha page, and read them.
    -> ({box: [{file, text, conf}]}, the row finder's note, its warning); {} when no rows were found.
    With lines_dir the crops are saved there as <page>_NN.png (main text) or <page>_<box>_NN.png,
    replacing those of an earlier read of the page."""
    stem = image.rsplit(".", 1)[0]
    if lines_dir:
        for old in lines_dir.glob(f"{stem}_*.png"):   # a re-read may find different rows or blocks
            old.unlink()
    if layout_of(ms_id) in ("tripatha", "panchapatha"):
        blocks, size, note = blocks_for_page(ms_id, image)
        boxes, warns = {}, []
        for box, b in blocks.items():
            if not b["rows"]:
                continue
            boxes[box] = _read_rows(ms_id, image, size, b["rows"], net, chars, lines_dir, stem, box)
            if b["warn"]:
                warns.append(f"{box}: {b['warn']}")
        return boxes, note, ("; ".join(warns) or None) if boxes else "no blocks found"
    merged, size, note, warn = rows_for_page(ms_id, image)
    if not merged:
        return {}, note, warn
    return {"main": _read_rows(ms_id, image, size, merged, net, chars, lines_dir, stem, "main")}, note, warn


def _review_html(ms_id, stem, boxes, warn):
    rows = []
    for box, lines in boxes.items():
        if len(boxes) > 1:
            rows.append(f'<h4>{box} · {len(lines)} lines</h4>')
        for i, ln in enumerate(lines, 1):
            tint = "#fff" if ln["conf"] >= 90 else "#fff6d6" if ln["conf"] >= 80 else "#ffe0d6"
            rows.append(f'<div class="ln" style="background:{tint}"><span class="n">{i} · {ln["conf"]:.0f}</span>'
                        f'<img src="../lines/pred/{html.escape(ln["file"])}">'
                        f'<input value="{html.escape(ln["text"], quote=True)}"></div>')
    return f"""<!doctype html><html lang="sa"><head><meta charset="utf-8"><title>{ms_id} {stem}</title>
<style>
body{{font-family:system-ui;margin:16px;background:#f4f1ea;color:#222}}
.ln{{margin:0 0 10px;padding:6px 8px;border:1px solid #d8d2c4;border-radius:6px}}
.ln img{{display:block;width:100%;image-rendering:auto}}
.ln input{{width:100%;box-sizing:border-box;font-size:19px;padding:5px;margin-top:4px;border:1px solid #bbb;border-radius:4px;
  font-family:"Kohinoor Devanagari","Devanagari Sangam MN","Noto Sans Devanagari",sans-serif}}
.n{{font-size:12px;color:#666}} .warn{{background:#ffe0d6;padding:8px;border-radius:6px}}
button{{font-size:16px;padding:8px 18px;margin:8px 0}} p{{max-width:70em}}
</style></head><body>
<h3>{ms_id} · page {stem}</h3>
<p>Each box holds the model's reading of the line shown above it. Correct it to what the scribe wrote
(no added spaces, scribe's own spellings and daṇḍas). The number after the line number is the model's
confidence; tinted lines are the doubtful ones. Then press Save and move the downloaded file to
<code>data/{ms_id}/corrected/{stem}.txt</code>.</p>
{f'<p class="warn">Check the lines first: {html.escape(warn)}</p>' if warn else ''}
{''.join(rows)}
<button onclick="save()">Save {stem}.txt</button>
<script>
function save(){{
  const t=[...document.querySelectorAll('.ln input')].map(i=>i.value.trim()).join('\\n')+'\\n';
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([t],{{type:'text/plain;charset=utf-8'}}));
  a.download='{stem}.txt';a.click();
}}
</script></body></html>"""


def predict(ms_id, images, model=DEFAULT_MODEL, verbose=True, write_to_correct=True):
    warnings.filterwarnings("ignore")
    d = ms_dir(ms_id)
    skip = not_text(ms_id)
    if any(i.rsplit(".", 1)[0] in skip for i in images):
        print(f"[predict] {ms_id}: not text pages, not read: " + ", ".join(i for i in images if i.rsplit(".", 1)[0] in skip))
        images = [i for i in images if i.rsplit(".", 1)[0] not in skip]
    for sub in ("drafts", "to_correct", "review", "corrected", "lines/pred"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    done = [i for i in images if (d / "corrected" / f"{i.rsplit('.', 1)[0]}.txt").exists()]
    if done:
        print(f"[predict] {ms_id}: {len(done)} corrected page(s) not read again (their text is paired with their line images)")
        images = [i for i in images if i not in done]
    segs = {i: page_seg(ms_id, i.rsplit(".", 1)[0]) for i in images}
    for seg in sorted(set(segs.values())):
        ensure_segmented(ms_id, [i for i in images if segs[i] == seg], seg)
    net, chars, ck = load_model(model)
    print(f"[predict] {ms_id}: {len(images)} pages with model {model_label(model, ck)} on its own manuscript; lines from "
          + ", ".join(f"{s} on {sum(1 for v in segs.values() if v == s)}" for s in sorted(set(segs.values()))))
    summary = []
    for image in images:
        stem = image.rsplit(".", 1)[0]
        seg = segs[image]
        if cut_from(ms_id, stem) not in (None, seg):            # cut again from another segmentation: no old image may stay
            for f in (d / "lines" / "pred").glob(f"{stem}_*.png"):
                f.unlink()
        with reading_from(seg):
            boxes, note, warn = read_page(ms_id, image, net, chars, d / "lines" / "pred")
        if boxes:
            pin(ms_id, stem, seg)
        if not boxes:
            print(f"  {stem}: {warn}")
            continue
        text = pagetext.format({b: [l["text"] for l in ls] for b, ls in boxes.items()})
        (d / "drafts" / f"{stem}.txt").write_text(text, encoding="utf-8")
        if write_to_correct and not (d / "corrected" / f"{stem}.txt").exists():
            (d / "to_correct" / f"{stem}.txt").write_text(text, encoding="utf-8")
        (d / "review" / f"{stem}.html").write_text(_review_html(ms_id, stem, boxes, warn), encoding="utf-8")
        confs = [l["conf"] for ls in boxes.values() for l in ls]
        summary.append((stem, len(confs), statistics.mean(confs)))
        if verbose:
            where = ", ".join(f"{b} {len(ls)}" for b, ls in boxes.items()) if len(boxes) > 1 else f"{len(confs):2d} lines"
            print(f"  {stem}: {where}, confidence mean {statistics.mean(confs):4.1f} lowest {min(confs):4.1f}"
                  + (f"   ! {warn}" if warn else ""))
    if summary:
        print(f"[predict] {sum(n for _, n, _ in summary)} lines on {len(summary)} pages; "
              f"mean confidence {statistics.mean(c for _, _, c in summary):.1f}")
        if write_to_correct:
            print(f"  to correct : {d / 'to_correct'}   (edit in place, then move to {d / 'corrected'})")
            print(f"  drafts     : {d / 'drafts'}   (untouched copy)")
            print(f"  review     : {d / 'review'}   (same thing as .html pages with the line images)")
    return summary


def page_list(ms_id, start=1, count=10):
    files = sorted(p.name for p in MANUSCRIPTS[ms_id]["images"].glob("*.jpg"))
    return files[start:start + count]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ms_id")
    ap.add_argument("--start", type=int, default=1, help="index of the first page image (0 is usually the cover)")
    ap.add_argument("--pages", type=int, default=10)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    a = ap.parse_args()
    predict(a.ms_id, page_list(a.ms_id, a.start, a.pages), Path(a.model))
