"""Kraken's segmentation repaired where it missed a line.

    python -m msocr.repair [LDII_xxx ...]      # every manuscript with a rowcheck report if none named

On some pages Kraken finds no line where the page has one. The row finder notices (a gap of two or more
pitches between neighbours that holds ink, or ink one pitch beyond the first or last line) and fills a row
in, and rowcheck warns "Kraken missed ...". Such a page reads correctly, but Kraken's own output is wrong,
and the segmenter (msocr/tseg.py) would learn the mistake. This writes, for every page whose ONLY warnings
are missed rows, a copy of Kraken's segmentation with each filled-in row added as a line of its own:

  data/<ms>/seg_repaired/<image stem>.json      Kraken's lines unchanged, plus one line per missed row, tagged
                                                "repaired": its baseline is the row finder's, its outline the
                                                baseline moved up and down as far as the outlines of the lines
                                                above and below it reach (so the hand size stays right)
  data/<ms>/repair.txt                          every such page: rows added, and whether the repair was kept

A repair is kept only if the row finder, run again on the repaired segmentation, finds the same rows in every
block with no warning left; a repair left from an earlier run whose page no longer qualifies is removed. Kraken's own data/<ms>/seg/ is never touched. Pages with any other warning (uneven
spacing, a short line that may be missing, no rows) are not repaired: they stay out of the segmenter's data.
The repaired pages can be looked at as any segmentation: MSOCR_SEG=seg_repaired python -m msocr.show <ms> <page>.
"""
import json
import re
import statistics
import sys
import warnings

import numpy as np
from PIL import Image

from .config import DATA, MANUSCRIPTS
from .align import find_rows, fill_edge_rows, uneven
from .blocks import EXPECTED, layout_of, page_blocks

OUT = "seg_repaired"


def missed_only(warn):
    """True when every warning on the page is a missed row (rowcheck's warning column)."""
    parts = [re.sub(r"^(main|top|bottom|left|right): ", "", p.strip()) for p in (warn or "").split("; ") if p.strip()]
    return bool(parts) and all(p.startswith("Kraken missed") for p in parts)


def _reach(row):
    """(up, down): how far the outlines of a row's Kraken pieces reach above and below their baselines (medians)."""
    ups, downs = [], []
    for d in row:
        line = d.get("line") or {}
        bnd, base = line.get("boundary"), line.get("baseline")
        if d.get("filled") or not bnd or len(bnd) < 3 or not base:
            continue
        yb = statistics.mean(p[1] for p in base)
        ys = [p[1] for p in bnd]
        ups.append(yb - min(ys))
        downs.append(max(ys) - yb)
    return (statistics.median(ups), statistics.median(downs)) if ups else None


def _new_lines(rows, pitch, dx=0, dy=0):
    """Kraken-style lines for the filled rows among `rows` (one block's or the page's), in page coordinates."""
    out = []
    for i, row in enumerate(rows):
        if not row[0].get("filled"):
            continue
        near = [r for r in rows[max(0, i - 2):i + 3] if not r[0].get("filled")]
        reach = [x for x in (_reach(r) for r in near) if x]
        up = statistics.median(x[0] for x in reach) if reach else 0.6 * pitch
        down = statistics.median(x[1] for x in reach) if reach else 0.2 * pitch
        (xa, ya), (xb, yb) = row[0]["line"]["baseline"][0], row[0]["line"]["baseline"][-1]
        xa, xb, ya, yb = xa + dx, xb + dx, ya + dy, yb + dy
        base = [[int(round(xa)), int(round(ya))], [int(round(xb)), int(round(yb))]]
        bnd = [[int(round(xa)), int(round(ya - up))], [int(round(xb)), int(round(yb - up))],
               [int(round(xb)), int(round(yb + down))], [int(round(xa)), int(round(ya + down))]]
        out.append({"baseline": base, "boundary": bnd})
    return out


def _rows(ms, seg, size, gray):
    """{box: number of rows} and every warning, as rowcheck sees the page."""
    if layout_of(ms) in ("tripatha", "panchapatha"):
        found, note = page_blocks(seg, size, gray, EXPECTED.get(layout_of(ms)))
        return ({b: len(v["rows"]) for b, v in found.items()},
                "; ".join(f"{b}: {v['warn']}" for b, v in found.items() if v["warn"]), found)
    rows, block, pitch, note = find_rows(seg, size[0], gray)
    if len(rows) >= 3:
        rows, edge = fill_edge_rows(rows, block, pitch, gray)
        note += ("; " + edge) if edge else ""
    warn = uneven(rows, pitch) or ""
    missed = [b for b in note.split("; ") if b.startswith("Kraken missed") or "short line may be missing" in b
              or "may have been missed" in b]
    warn = "; ".join([w for w in [warn] + missed if w])
    return {"main": len(rows)}, warn, {"main": {"raw": rows, "pitch": pitch, "offset": (0, 0)}}


def repair(ms):
    rep = DATA / ms / "rowcheck.txt"
    if not rep.exists():
        print(f"[repair] {ms}: no rowcheck report; run  python -m msocr.rowcheck {ms}  first")
        return
    out_dir = DATA / ms / OUT
    report, kept, kept_stems = [], 0, set()
    for line in rep.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t") + ["", ""]
        image, warn = f[0], f[4]
        if not missed_only(warn):
            continue
        stem = image.rsplit(".", 1)[0]
        seg = json.load(open(DATA / ms / "seg" / f"{stem}.json"))
        with Image.open(MANUSCRIPTS[ms]["images"] / image) as im:
            size = im.size
            gray = np.asarray(im.convert("L"))
        before, _, found = _rows(ms, seg, size, gray)
        added = []
        for box, b in found.items():
            dx, dy = b.get("offset", (0, 0))
            added += _new_lines(b["raw"], b["pitch"], dx, dy)
        if not added:
            report.append((image, 0, "not kept: no filled row found on the page now (report stale?)"))
            continue
        fixed = dict(seg)
        fixed["lines"] = list(seg["lines"]) + [
            {"id": f"repaired_{k}", "text": None, "base_dir": None, "imagename": None,
             "tags": {"type": [{"type": "repaired"}]}, "split": None, "regions": [], "language": None,
             "type": "baselines", "baseline": a["baseline"], "boundary": a["boundary"]} for k, a in enumerate(added)]
        fixed["repaired"] = f"{len(added)} line{'s' if len(added) > 1 else ''} added by msocr.repair where Kraken missed them"
        after, warn_after, _ = _rows(ms, fixed, size, gray)
        path = out_dir / f"{stem}.json"
        if after == before and not warn_after:
            out_dir.mkdir(exist_ok=True)
            path.write_text(json.dumps(fixed), encoding="utf-8")
            report.append((image, len(added), "kept"))
            kept += 1
            kept_stems.add(stem)
        else:
            if path.exists():
                path.unlink()
            why = []
            if after != before:
                why.append(f"rows {before} became {after}")
            if warn_after:
                why.append(f"still warns: {warn_after}")
            report.append((image, len(added), "not kept: " + "; ".join(why)))
    # a repair from an earlier run whose page is no longer one with only missed rows (the code or the layout has changed
    # since) would be used by the segmenter as if it were checked: gone
    stale = [f for f in out_dir.glob("*.json") if f.stem not in kept_stems] if out_dir.exists() else []
    for f in stale:
        f.unlink()
    if stale:
        print(f"[repair] {ms}: {len(stale)} repairs from an earlier run removed (their pages no longer qualify)", flush=True)
    (DATA / ms / "repair.txt").write_text(
        f"# {ms}: {len(report)} pages whose only warnings are missed rows; {kept} repaired and kept in {OUT}/\n"
        "# image\tlines added\tresult\n" + "".join(f"{a}\t{b}\t{c}\n" for a, b, c in report), encoding="utf-8")
    print(f"[repair] {ms}: {len(report)} pages with only missed rows; {kept} repaired and kept", flush=True)
    return len(report), kept


if __name__ == "__main__":
    ids = sys.argv[1:] or sorted(m for m in MANUSCRIPTS if (DATA / m / "rowcheck.txt").exists())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tot = [0, 0]
        for ms in ids:
            r = repair(ms)
            if r:
                tot[0] += r[0]
                tot[1] += r[1]
    print(f"[repair] in all: {tot[0]} pages with only missed rows; {tot[1]} repaired and kept")
