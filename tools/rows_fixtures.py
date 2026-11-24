"""Row-finder fixtures: what the reference's rows stage (msocr.reference.rows: align.find_rows, fill_edge_rows, _merge on a
linear page; blocks.page_blocks on a tripāṭha or pañcapāṭha one) makes of about forty segmented pages, so that Manuscribe's
JavaScript port (app/js/recogrows.js, tests/t-recog-conformance.js) can be held to it on more pages than the pack's.

    PYTHONPATH=. .venv/bin/python -W ignore tools/rows_fixtures.py

-> data/fixtures/rows/<ms>_<stem>/
     grey.png     reference.grey(page), 8-bit, lossless
     lines.json   Kraken's lines of the page (data/<ms>/seg/<stem>.json), baseline and boundary only
     rows.json    reference.rows(lines, grey, layout): {box: [[baseline, outline], ...]}, as the pack's 05_rows.json
     meta.json    {"layout", "width", "height"}, and the manuscript, the page and why it was chosen
   data/fixtures/rows/index.json lists them, with the version of msocr/pipeline.json they were made with. The pages are manuscript pages: they stay in data/, never distributed.

The pages are chosen with a fixed seed among the text pages of every manuscript with a segmentation and a layout: some of
the pages whose true rows per block the review gave (data/kraken_review/feedback/block_counts.tsv, the hard block pages),
pages where the row finder leaves out a commentary row (LDII_15152, LDII_22995, LDII_7105, LDII_82320), and a few
others of every manuscript, so that all three layouts and many hands are in. Then the pages in RARE, each the way into
a branch of the row or block finder that few pages take (found by running the reference and the port on every segmented
page, 2026-11-24), so that the fixtures reach every branch.
"""
import csv
import json
import random
import shutil

import numpy as np
from PIL import Image

from msocr import align, reference
from msocr.blocks import layout_of
from msocr.config import DATA, MANUSCRIPTS, not_text
from msocr.params import VERSION

SEED = 20261124
OUT = DATA / "fixtures" / "rows"
LAYOUTS = ("linear", "tripatha", "panchapatha")
COMMENTARY = ("LDII_15152", "LDII_22995", "LDII_7105", "LDII_82320")
HARD_PER_MS = {"AKGM_0025383": 2, "LDII_3746": 2, "LDII_38386": 2, "LDII_82328": 3}   # from block_counts.tsv
OTHERS = {"tripatha": 1, "panchapatha": 1, "linear": 1}                                  # further pages per manuscript
MORE = {"LDII_24140": 1, "LDII_45071": 1, "LDII_17963": 1}   # hands Kraken breaks into words: one page more
RARE = [
    ("LDII_38386", "038386_00004_1", "blocks by blank runs, spacing and hand (_bands), a row moved across (_rebalance)"),
    ("LDII_82328", "082328_00010_2", "blocks by blank runs, spacing and hand (_bands, _name_bands) on a pancapatha page"),
    ("AKGM_0025383", "025383_00001_1", "_bands; block from the pieces; a block's rows found again with its own pitch"),
    ("AKGM_0025383", "025383_00137_1", "main text in two half-columns (_centre)"),
    ("LDII_38386", "038386_00001_1", "main text found by its larger hand (_hand_bands)"),
    ("LDII_1806", "001806_00079_2", "pitch from the heights of the pieces (_pitch_from_pieces)"),
    ("LDII_24075", "024075_00001_1", "as many rows filled in as found: the pitch halved, nothing filled in"),
    ("LDII_17963", "017963_00032_1", "pitch from the ink, the gaps whole multiples of it (_fits_better)"),
    ("LDII_24083", "024083_00023_2", "block from the pieces, its edge on a tie of the smoothed coverage"),
    ("LDII_15152", "015152_00002_1", "a block of notes at an edge (the second rule of _small_hand_edges)"),
    ("LDII_82328", "082328_00027_2", "a row packed against a commentary row just dropped goes with it"),
    ("AKGM_0025383", "025383_00106_2", "commentary rows at the edges, but fewer than four rows would be left"),
    ("AKGM_0025383", "025383_00007_1", "two long lines at one height make one row"),
    ("LDII_3746", "003746_00062_2", "a single line: the pitch 1.3 times the height of its pieces"),
    ("LDII_3746", "003746_00051_1", "two rows of a block closer than 0.6 pitch joined (rows_in_block)"),
    ("LDII_38386", "038386_00009_2", "_hand_bands: a line below the main text taken into it by its hand"),
]


def text_pages(ms):
    seg = DATA / ms / "seg"
    skip = not_text(ms)
    return sorted(p.stem for p in seg.glob("*.json")
                  if p.stem not in skip and (MANUSCRIPTS[ms]["images"] / f"{p.stem}.jpg").exists())


def load(ms, stem):
    seg = json.load(open(DATA / ms / "seg" / f"{stem}.json", encoding="utf-8"))
    lines = [{"baseline": l["baseline"], "boundary": l["boundary"]} for l in seg["lines"]]
    page = np.asarray(Image.open(MANUSCRIPTS[ms]["images"] / f"{stem}.jpg").convert("RGB"))
    return lines, reference.grey(page)


def has_commentary(ms, stem):
    """Does the row finder leave out a row in a smaller hand on this page?"""
    lines, g = load(ms, stem)
    rows, block, pitch, note = align.find_rows({"lines": lines}, g.shape[1], g)
    return "in a smaller hand" in note


def choose():
    rng = random.Random(SEED)
    manuscripts = sorted(ms for ms in MANUSCRIPTS if (DATA / ms / "seg").is_dir() and layout_of(ms) in LAYOUTS)
    pages = {ms: text_pages(ms) for ms in manuscripts}
    chosen = {}                                                      # (ms, stem) -> why

    def take(ms, stem, why):
        if (ms, stem) not in chosen and stem in pages.get(ms, ()):
            chosen[(ms, stem)] = why
            return True
        return False
    hard = {}
    with open(DATA / "kraken_review" / "feedback" / "block_counts.tsv", encoding="utf-8") as f:
        for r in csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t"):
            hard.setdefault(r["manuscript"], []).append(r["page"])
    for ms, k in HARD_PER_MS.items():
        for stem in rng.sample(sorted(hard.get(ms, [])), min(k, len(hard.get(ms, [])))):
            take(ms, stem, "hard block page (block_counts.tsv)")
    for ms in COMMENTARY:                                            # two pages each with a commentary row left out
        found, cands = 0, list(pages.get(ms, []))
        rng.shuffle(cands)
        for stem in cands[:25]:
            if found == 2:
                break
            if has_commentary(ms, stem) and take(ms, stem, "a commentary row left out"):
                found += 1
        print(f"  {ms}: {found} page(s) with a commentary row", flush=True)
    for ms in manuscripts:
        want = OTHERS[layout_of(ms)] + MORE.get(ms, 0)
        if ms in COMMENTARY:
            want = 0
        cands = [s for s in pages[ms] if (ms, s) not in chosen]
        for stem in rng.sample(cands, min(want, len(cands))):
            take(ms, stem, "seeded sample")
    for ms, stem, why in RARE:
        if not take(ms, stem, why) and (ms, stem) not in chosen:
            print(f"  {ms} {stem}: not a segmented text page, left out", flush=True)
    return chosen


def main():
    chosen = choose()
    OUT.mkdir(parents=True, exist_ok=True)
    names = {f"{ms}_{stem}" for ms, stem in chosen}
    for d in OUT.iterdir():                                          # pages of an earlier choice
        if d.is_dir() and d.name not in names and (d / "meta.json").exists():
            shutil.rmtree(d)
    index = []
    for (ms, stem), why in sorted(chosen.items()):
        layout = layout_of(ms)
        lines, g = load(ms, stem)
        rows = reference.rows(lines, g, layout)
        d = OUT / f"{ms}_{stem}"
        d.mkdir(exist_ok=True)
        Image.fromarray(g).save(d / "grey.png")
        json.dump(lines, open(d / "lines.json", "w"))
        json.dump({b: [[list(map(list, base)), list(map(list, outl))] for base, outl in rs] for b, rs in rows.items()},
                  open(d / "rows.json", "w"))
        meta = {"layout": layout, "width": int(g.shape[1]), "height": int(g.shape[0]), "manuscript": ms, "page": stem, "chosen": why}
        json.dump(meta, open(d / "meta.json", "w"), indent=1)
        counts = {b: len(rs) for b, rs in rows.items()}
        index.append({"dir": d.name, "layout": layout, "rows": counts, "chosen": why})
        print(f"  {ms} {stem} ({layout}, {why}): rows {counts}", flush=True)
    json.dump({"seed": SEED, "pipeline_version": VERSION, "pages": index}, open(OUT / "index.json", "w"), indent=1)
    print(f"[rows fixtures] {len(index)} pages: {OUT}")


if __name__ == "__main__":
    main()
