"""Every transcribed side of the Muktabodha manuscripts that align did not pair, with the reason, its folio label in the
e-text, its image and its PDF page. Writes data/muktabodha/unaligned.md.

    PYTHONPATH=. .venv/bin/python tools/muktabodha_unaligned.py [ms ...]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from msocr.align import find_rows, uneven
from msocr.config import DATA, MANUSCRIPTS
from msocr.muktabodha import CONFIGS, build_sides, load_config, read_etext

ORDER = ["RTJ_4387", "RTJ_4914", "RTJ_6025", "RTJ_1127", "UPSS_1322", "RTJ_4894", "RTJ_1307", "UPSS_2042"]
TITLES = {"RTJ_4387": "Mṛgendravṛtti", "RTJ_4914": "Āgamakalpalatā", "RTJ_6025": "Bagalāmukhīpañcāṅga",
          "RTJ_1127": "Svātantryadīpikā", "UPSS_1322": "Parāmbārcanacandrikā", "RTJ_4894": "Ūrdhvāmnāya",
          "RTJ_1307": "Śrīgurustutiratnāvalī", "UPSS_2042": "Tattvaśodhana"}


class A:          # load_config reads its overrides from an argparse namespace
    def __getattr__(self, k):
        return None
    sequential_ids = False


def unaligned(ms):
    d = DATA / ms
    cfg = load_config(ms, A())
    meta, blocks, _ = read_etext(cfg["etext"])
    sides, _ = build_sides(blocks, cfg.get("sequential_ids", False))
    label = {s["pid"]: (s["label"], s["at"]) for s in sides}
    pdf_page = {}
    rep = (d / "muktabodha.txt").read_text(encoding="utf-8")
    for m in re.finditer(r"^\s+(\d+)([ab]?)\s+(\S+\.jpg)\s", rep.split("Page table")[1], re.M):
        pdf_page[m.group(3)] = m.group(1) + m.group(2)
    n = {}
    for r in map(json.loads, (d / "gt_lines.jsonl").read_text(encoding="utf-8").splitlines()):
        n.setdefault(r["image"], [r["page"], 0])[1] += 1
    out = []
    for image, (pid, count) in n.items():
        if (d / "pagexml_seg_torch" / (image[:-4] + ".json")).exists():
            continue
        path = MANUSCRIPTS[ms]["images"] / image
        if not path.exists():
            why = "no page image: the matching found no photo of this side"
        else:
            seg = json.loads((d / "seg_torch" / (image[:-4] + ".json")).read_text())
            with Image.open(path) as im:
                gray = np.asarray(im.convert("L"))
                w = im.size[0]
            rows, block, pitch, note = find_rows(seg, w, gray)
            if len(rows) != count:
                why = f"the row finder found {len(rows)} rows, the transcription has {count} lines"
            else:
                why = f"{uneven(rows, pitch)}: a line probably missed and a note counted instead"
        lab, at = label.get(pid, ("?", "?"))
        out.append({"pid": pid, "label": lab, "at": at, "image": image if path.exists() else "-",
                    "pdf": pdf_page.get(image, "-"), "why": why})
    return out, len(n)


def main(mss):
    lines = ["# Muktabodha manuscripts: transcribed sides that did not align", "",
             "Made by `tools/muktabodha_unaligned.py`. Side = the side's id in the TEI (folio_side; with sequential ids "
             "the e-text's order, not the scribe's numbers); label = the e-text's own marker; e-text line = where that "
             "side starts in the e-text; PDF page = the scan's page the image came from (a/b: the upper/lower leaf of a "
             "photo cut in two).", ""]
    total = 0
    for ms in mss:
        rows, n = unaligned(ms)
        total += len(rows)
        lines += [f"## {ms} {TITLES.get(ms, '')}: {len(rows)} of {n} sides", ""]
        if not rows:
            lines += ["All aligned.", ""]
            continue
        lines += ["| side | label | e-text line | image | PDF page | reason |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['pid']} | {r['label']} | {r['at']} | {r['image']} | {r['pdf']} | {r['why']} |" for r in sorted(rows, key=lambda r: r["pid"])]
        lines.append("")
        print(f"{ms}: {len(rows)} of {n} sides unaligned", flush=True)
    out = CONFIGS / "unaligned.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{total} sides in all; written to {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or ORDER)
