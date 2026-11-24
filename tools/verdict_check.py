"""Row counts with the current code against the rowcheck reports the review pictures were drawn from, on the pages
just reviewed: an ok page whose count changes is a regression; a wrong page shows what the change did to it.

    PYTHONPATH=. .venv/bin/python tools/verdict_check.py ~/Downloads/LDII_*_feedback*.tsv
"""
import csv, glob, json, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
from msocr.config import MANUSCRIPTS, DATA
from msocr.repair import _rows
files = sys.argv[1:]
changed_ok, wrong_rows = [], []
n_ok = 0
for f in files:
    for r in csv.DictReader(open(f, encoding="utf-8", newline=""), delimiter="\t"):
        if r["verdict"] not in ("ok", "wrong"):
            continue
        ms, stem = r["manuscript"], r["page"]
        rep = {l.split("\t")[0]: l.split("\t") for l in open(DATA / ms / "rowcheck.txt", encoding="utf-8") if not l.startswith("#")}
        line = rep.get(stem + ".jpg")
        if not line:
            continue
        before = int(line[2] or 0)
        segf = DATA / ms / ("seg_repaired" if r["group"] == "REPAIRED" and (DATA / ms / "seg_repaired" / f"{stem}.json").exists() else "seg") / f"{stem}.json"
        seg = json.load(open(segf))
        with Image.open(MANUSCRIPTS[ms]["images"] / f"{stem}.jpg") as im:
            size, gray = im.size, np.asarray(im.convert("L"))
        counts, warn, found = _rows(ms, seg, size, gray)
        after = counts.get("main", 0)
        notes = "; ".join(re.findall(r"\d+ rows? in a smaller hand \w+", " ".join(v.get("note", "") for v in found.values()) if isinstance(found, dict) else ""))
        if r["verdict"] == "ok":
            n_ok += 1
            if after != before:
                changed_ok.append(f"  OK PAGE CHANGED {ms} {stem[-7:]} [{r['group']}] {before} -> {after}")
        else:
            wrong_rows.append(f"  {ms} {stem[-7:]} [{r['group']}] {before} -> {after}  | {r['note'][:90]}")
print(f"ok pages checked: {n_ok}; with a changed row count: {len(changed_ok)}")
print("\n".join(changed_ok))
print("wrong pages:")
print("\n".join(wrong_rows))
