"""How well the row finder copes with a segmented manuscript, before any page is read.

    python -m msocr.rowcheck LDII_24140 [LDII_x ...]

For every segmented page the main text rows are picked out exactly as predict does (find_rows,
fill_edge_rows) and the count is compared with the manuscript's usual count (the most common
one). On a tripāṭha or pañcapāṭha manuscript (data/layouts.csv) the page is first cut into its
blocks (blocks.py); "rows" is then the main block's count, the note lists every block with its
count, and a page is evenly spaced only when every block is. The result goes to data/<ms>/rowcheck.txt, one line per page, with a summary at the top;
LAYOUTS.md shows the summary. A page is counted as evenly spaced when its rows sit one pitch
apart, which is what a page with every line found looks like; a page with no rows, or with a
gap or a squeezed-in row (the "uneven row spacing" warning), is where Kraken's lines were too
broken for the row finder, and is read badly or not at all until looked at.
"""
import collections
import json
import sys
import warnings

from .blocks import blocks_for_page, layout_of
from .config import MANUSCRIPTS, SEG, ms_dir, not_text
from .predict import rows_for_page

REPORT = "rowcheck.txt" if SEG == "seg" else f"rowcheck_{SEG}.txt"   # a report per segmentation


def check(ms_id, verbose=True):
    d = ms_dir(ms_id)
    ms = MANUSCRIPTS[ms_id]
    results, skip = [], not_text(ms_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in sorted((d / SEG).glob("*.json")):
            image = s.stem + ".jpg"
            if not (ms["images"] / image).exists():
                continue
            pieces = len(json.load(open(s))["lines"])
            if s.stem in skip:                                   # a cover or a blank side: nothing to check
                results.append((image, pieces, 0, "not a text page", ""))
                continue
            if layout_of(ms_id) in ("tripatha", "panchapatha"):
                blocks, _, note = blocks_for_page(ms_id, image)
                counts = ", ".join(f"{box} {len(b['rows'])}" for box, b in blocks.items())
                note = (counts + ("; " + note if note else "")) if counts else note
                warn = "; ".join(f"{box}: {b['warn']}" for box, b in blocks.items() if b["warn"])
                if any(b["warn"] and b["warn"].startswith("uneven") for b in blocks.values()):
                    warn = "uneven row spacing in " + ", ".join(box for box, b in blocks.items() if b["warn"] and b["warn"].startswith("uneven")) + "; " + warn
                n = len(blocks["main"]["rows"]) if "main" in blocks else 0
                results.append((image, pieces, n, note, warn))
                continue
            rows, _, note, warn = rows_for_page(ms_id, image)
            results.append((image, pieces, len(rows), note, warn or ""))
    if not results:
        return None
    text_pages = [r for r in results if r[3] != "not a text page"]
    counts = collections.Counter(n for _, _, n, _, _ in text_pages if n)
    usual = counts.most_common(1)[0][0] if counts else 0
    none = sum(1 for _, _, n, _, _ in text_pages if n == 0)
    near = sum(1 for _, _, n, _, _ in text_pages if n and abs(n - usual) <= 1)
    even = sum(1 for _, _, n, _, w in text_pages if n >= 2 and not w.startswith("uneven row spacing"))
    summary = {"pages": len(results), "usual": usual, "near_usual": near, "even": even, "no_rows": none}
    head = (f"# {ms_id}: {len(text_pages)} text pages checked ({len(results) - len(text_pages)} not text pages left out); usual row count {usual}; "
            f"{near} pages within one row of it; {even} pages with evenly spaced rows; {none} pages with no rows\n"
            f"# image\tkraken pieces\trows\tnote\twarning\n")
    (d / REPORT).write_text(head + "".join(f"{im}\t{p}\t{n}\t{note}\t{w}\n" for im, p, n, note, w in results), encoding="utf-8")
    if verbose:
        print(f"[rowcheck] " + head.splitlines()[0][2:])
        bad = [(im, n) for im, _, n, _, w in text_pages if n < 2 or w.startswith("uneven row spacing")]
        if bad:
            print(f"  no rows or unevenly spaced: " + ", ".join(f"{im} ({n})" for im, n in bad[:15]) + (f" ... and {len(bad) - 15} more" if len(bad) > 15 else ""))
        print(f"  details: {d / REPORT}")
    return summary


def summary(ms_id):
    """The numbers from an earlier check, or None."""
    f = ms_dir(ms_id) / REPORT
    if not f.exists():
        return None
    first = f.read_text(encoding="utf-8").splitlines()[0]
    import re
    m = re.search(r"(\d+) pages checked; usual row count (\d+); (\d+) pages within one row of it; (\d+) pages with evenly spaced rows; (\d+) pages with no rows", first)
    return {"pages": int(m.group(1)), "usual": int(m.group(2)), "near_usual": int(m.group(3)), "even": int(m.group(4)), "no_rows": int(m.group(5))} if m else None


if __name__ == "__main__":
    for ms_id in sys.argv[1:]:
        check(ms_id)
