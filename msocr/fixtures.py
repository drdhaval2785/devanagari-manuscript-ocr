"""Stage fixtures: what the reference pipeline makes of a page at every stage, with a model pack's own ONNX networks, so
that the JavaScript port can be checked stage by stage against it (Manuscribe's conformance runner).

    .venv-onnx/bin/python -m msocr.fixtures --pack data/model_pack/<pack> [--pages ms:stem ...] [--synthetic]

-> data/fixtures/<pack>/<page id>/
     fixture.json        the page, its layout, the pack, every file below with its shape, and the tolerances
     00_page.png         the input: the page already decoded (RGB, 8 bits), so a JPEG decoder's differences stay out
     01_grey.png         grey(page)                                             exact
     02_seg_input.png    the page resized for the segmenter, before inversion   within 1 grey level
     03_maps.f32         pbase, up, down (3 x h x w float32, little-endian)     within 1e-3
     04_lines.json       the lines (baselines, outlines), page px                same number, within 1 px
     05_rows.json        {box: [[baseline, outline], ...]}                       same number, within 1 px
     06_crop_<box>_<k>.png       the line image (RGB)                           within 1 grey level
     07_rec_input_<box>_<k>.png  resized for the recogniser, before inversion   within 1 grey level
     08_logits_<box>_<k>.f32     t x classes float32                            within 1e-3
     09_text.json        {box: [{"text", "conf"}]}                              identical text
     10_spaced.json      {box: the block's lines joined with no separator, spaced}   identical
The private pages (the developer's manuscripts) stay in data/; --synthetic adds a page of printed Devanagari drawn
here, which may go anywhere.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import DATA, MANUSCRIPTS
from .blocks import layout_of
from . import reference as R

SPACER = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "sanskrit_spacer"))
DEFAULT_PAGES = ["LDII_85885:085885_00008_1", "AKGM_0025383:025383_00034_1", "LDII_82328:082328_00063_1"]
TOL = {"01_grey": "exact", "02_seg_input": "max abs 1", "03_maps": "max abs 1e-3", "04_lines": "same count; points within 1 px",
       "05_rows": "same count per block; points within 1 px", "06_crop": "max abs 1", "07_rec_input": "max abs 1",
       "08_logits": "max abs 1e-3", "09_text": "identical", "10_spaced": "identical"}
SYNTH = ("धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय दृष्ट्वा तु पाण्डवानीकं व्यूढं "
         "दुर्योधनस्तदा आचार्यमुपसङ्गम्य राजा वचनमब्रवीत् पश्यैतां पाण्डुपुत्राणामाचार्य महतीं चमूम् व्यूढां द्रुपदपुत्रेण "
         "तव शिष्येण धीमता अत्र शूरा महेष्वासा भीमार्जुनसमा युधि युयुधानो विराटश्च द्रुपदश्च महारथः")


def synthetic_page():
    """A page of printed Devanagari, without spaces between the words as in the manuscripts, on paper-coloured ground
    with a little noise: public, made here, deterministic."""
    rendered = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "fixtures", "synthetic_page_1.png")
    if os.path.exists(rendered):            # shaped by Chrome (tools/fixtures/render_synthetic.js): headlines unbroken
        im = Image.open(rendered).convert("RGB")
        rng = np.random.default_rng(20261124)
        a = np.asarray(im).astype(np.int16) + rng.integers(-6, 7, size=(im.size[1], im.size[0], 1))
        return np.clip(a, 0, 255).astype(np.uint8)
    fonts = ["/System/Library/Fonts/Supplemental/DevanagariMT.ttc", "/System/Library/Fonts/Kohinoor.ttc",
             "/System/Library/Fonts/Supplemental/ITFDevanagari.ttc"]
    font = next((ImageFont.truetype(f, 34) for f in fonts if os.path.exists(f)), ImageFont.load_default())
    w, h = 1500, 700
    im = Image.new("RGB", (w, h), (226, 208, 170))
    dr = ImageDraw.Draw(im)
    words, y, line = SYNTH.split(), 70, ""
    for word in words * 3:                  # written as the manuscripts are, without spaces between the words
        if dr.textlength(line + word, font=font) > w - 260:
            dr.text((130, y), line, font=font, fill=(40, 28, 20))
            y += 46
            line = ""
            if y > h - 90:
                break
        line += word
    rng = np.random.default_rng(20261124)
    a = np.asarray(im).astype(np.int16) + rng.integers(-6, 7, size=(h, w, 1))
    return np.clip(a, 0, 255).astype(np.uint8)


def save_png(path, arr):
    Image.fromarray(arr).save(path, optimize=False)


def make(out, page, layout, seg_f, rec_f, chars, spacer_net, threshold, about):
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    save_png(out / "00_page.png", page)
    g = R.grey(page)
    save_png(out / "01_grey.png", g)
    x, s, w = R.segmenter_input(g)
    resized = np.round((1.0 - x[0, 0, :, :w]) * 255.0).astype(np.uint8)
    save_png(out / "02_seg_input.png", resized)
    raw = seg_f(x)[0]
    pb, pu, pd = R.maps(raw, w)
    np.stack([pb, pu, pd]).astype("<f4").tofile(out / "03_maps.f32")
    files["03_maps.f32"] = {"shape": [3, int(pb.shape[0]), int(pb.shape[1])], "dtype": "float32"}
    lines = R.lines(pb, pu, pd, s)
    json.dump(lines, open(out / "04_lines.json", "w"))
    rows = R.rows(lines, g, layout)
    json.dump({b: [[list(map(list, base)), list(map(list, outl))] for base, outl in rs] for b, rs in rows.items()}, open(out / "05_rows.json", "w"))
    text, spaced = {}, {}
    sys.path.insert(0, SPACER)
    import spacer_reference as SR
    for box, rs in rows.items():
        text[box] = []
        for k, (base, outl) in enumerate(rs, 1):
            crop = R.cut(page, base, outl)
            save_png(out / f"06_crop_{box}_{k:02d}.png", crop)
            xi = R.recogniser_input(crop)
            save_png(out / f"07_rec_input_{box}_{k:02d}.png", np.round((1.0 - xi[0, 0]) * 255.0).astype(np.uint8))
            logits = rec_f(xi)[0]
            logits.astype("<f4").tofile(out / f"08_logits_{box}_{k:02d}.f32")
            files[f"08_logits_{box}_{k:02d}.f32"] = {"shape": list(logits.shape), "dtype": "float32"}
            t, c = R.decode(logits, chars)
            text[box].append({"text": t, "conf": round(c, 3)})
        spaced[box] = SR.space_text("".join(l["text"] for l in text[box]), spacer_net, threshold)
    json.dump(text, open(out / "09_text.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    json.dump(spaced, open(out / "10_spaced.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    meta = dict(about, layout=layout, scale=s, seg_width=int(w), seg_input_shape=list(x.shape),
                rows={b: len(rs) for b, rs in rows.items()}, files=files, tolerance=TOL)
    json.dump(meta, open(out / "fixture.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--pages", nargs="*", default=DEFAULT_PAGES)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    pack = os.path.abspath(a.pack)
    manifest = json.load(open(os.path.join(pack, "manifest.json"), encoding="utf-8"))
    seg_f = R.onnx_function(os.path.join(pack, "segmenter.onnx"))
    rec_f = R.onnx_function(os.path.join(pack, "recogniser.onnx"))
    chars = json.load(open(os.path.join(pack, "recogniser_chars.json"), encoding="utf-8"))
    sys.path.insert(0, SPACER)
    import spacer_reference as SR
    spacer_net = SR.onnx_network(os.path.join(pack, "spacer.onnx"))
    threshold = manifest["models"]["spacer"]["threshold"]
    root = DATA / "fixtures" / manifest["pack"]
    index = []
    for p in a.pages:
        ms, stem = p.split(":")
        page = np.asarray(Image.open(MANUSCRIPTS[ms]["images"] / f"{stem}.jpg").convert("RGB"))
        meta = make(root / stem, page, layout_of(ms), seg_f, rec_f, chars, spacer_net, threshold,
                    {"page": stem, "manuscript": ms, "pack": manifest["pack"], "private": True})
        index.append({"page": stem, "dir": stem, "layout": meta["layout"], "rows": meta["rows"], "private": True})
        print(f"  {ms} {stem}: {meta['layout']}, rows {meta['rows']}", flush=True)
    if a.synthetic:
        meta = make(root / "synthetic_1", synthetic_page(), "linear", seg_f, rec_f, chars, spacer_net, threshold,
                    {"page": "synthetic_1", "manuscript": None, "pack": manifest["pack"], "private": False})
        index.append({"page": "synthetic_1", "dir": "synthetic_1", "layout": "linear", "rows": meta["rows"], "private": False})
        print(f"  synthetic page: rows {meta['rows']}", flush=True)
    json.dump({"pack": manifest["pack"], "pipeline_version": manifest["pipeline_version"], "spacer_version": manifest["spacer_version"],
               "tolerance": TOL, "pages": index}, open(root / "index.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[fixtures] {len(index)} pages: {root}")


if __name__ == "__main__":
    main()
