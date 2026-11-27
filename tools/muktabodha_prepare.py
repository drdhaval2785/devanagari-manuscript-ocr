"""Ready the Muktabodha manuscripts for the recogniser's pool: their aligned pages, less the doubtful ones, cut into
line images (data/<ms>/lines_seg_torch/{train,val,test}/), the cut recognizer.pool trains on.

    PYTHONPATH=. .venv/bin/python tools/muktabodha_prepare.py [ms ...]          # default: every manuscript with a config

Run it after msocr.muktabodha's build, gt.extract, segment, align --seg=seg_torch and --check. Left out of every set:
- pages align did not pair (the row finder's count differs from the transcription's): align already left them out;
- pages with a row that reads closer to another line of the page than to its own (muktabodha_check.txt: a shift, a
  line missed or doubled, a gloss cut as a row);
- pages whose transcription has an illegible stretch (<gap/>): its letters are on the line image and not in the text.
A manuscript too small for align's 20 test pages (under 60 paired pages) is split again, a tenth to test and a tenth to
validation (at least one page each), so that it is trained on at all; the record (split_pages.tsv) follows.
The line cache is cut afresh. Writes data/<ms>/muktabodha_prepare.txt.
"""
import json
import random
import re
import shutil
import sys
from pathlib import Path

from msocr.config import DATA, MANUSCRIPTS
from msocr.recognizer import build_cache, lines_dir

SEG = "seg_torch"
SMALL = 60


def doubtful(ms):
    """image stems of the pages whose rows do not all read as their own lines (muktabodha_check.txt)"""
    f = DATA / ms / "muktabodha_check.txt"
    if not f.exists():
        raise SystemExit(f"{ms}: no muktabodha_check.txt; run python -m msocr.muktabodha {ms} --check first")
    every = f.read_text(encoding="utf-8").split("Every page", 1)[1]
    return {l.split()[0] for l in every.splitlines()[1:] if "->" in l}


def with_gaps(ms):
    """image stems of the pages whose TEI holds a <gap/>"""
    tei = MANUSCRIPTS[ms]["tei"].read_text(encoding="utf-8")
    out = set()
    for m in re.finditer(r'<pb n="([^"]+)"/>(.*?)(?=<pb |</div>)', tei, re.S):
        if "<gap" in m.group(2):
            out.add(MANUSCRIPTS[ms]["image_name"](m.group(1))[:-4])
    return out


def prepare(ms):
    d = DATA / ms
    sets = {n: [l for l in (d / f"{n}_{SEG}.txt").read_text().split() if l] for n in ("train", "val", "test")}
    paired = sum(len(v) for v in sets.values())
    bad, gaps = doubtful(ms), with_gaps(ms)
    drop = bad | gaps
    kept = {n: [p for p in v if Path(p).stem not in drop] for n, v in sets.items()}
    resplit = paired < SMALL
    if resplit:
        pages = sorted(p for v in kept.values() for p in v)
        random.Random(0).shuffle(pages)
        k = max(1, round(len(pages) / 10))
        kept = {"test": sorted(pages[:k]), "val": sorted(pages[k:2 * k]), "train": sorted(pages[2 * k:])}
    for n, v in kept.items():
        (d / f"{n}_{SEG}.txt").write_text("\n".join(v) + "\n")
    rec = d / "split_pages.tsv"
    record = dict(l.split("\t") for l in rec.read_text().splitlines() if "\t" in l) if rec.exists() else {}
    for n, v in kept.items():
        for p in v:
            record[Path(p).stem] = n
    for s in drop:
        record.pop(s, None)
    rec.write_text("".join(f"{s}\t{n}\n" for s, n in sorted(record.items())))
    if lines_dir(ms, SEG).exists():
        shutil.rmtree(lines_dir(ms, SEG))
    build_cache(ms, verbose=False, seg=SEG)
    n_lines = {n: len(list((lines_dir(ms, SEG) / n).glob("*.png"))) for n in kept}
    report = [f"[prepare] {ms}: {paired} paired pages; left out {len(bad & {Path(p).stem for v in sets.values() for p in v})} "
              f"whose rows do not all read as their own lines, {len(gaps & {Path(p).stem for v in sets.values() for p in v})} with an illegible stretch"
              + ("; split again (too small for 20 test pages)" if resplit else ""),
              "  pages : " + ", ".join(f"{n} {len(v)}" for n, v in kept.items()),
              "  lines : " + ", ".join(f"{n} {c}" for n, c in n_lines.items()),
              "  left out: " + (" ".join(sorted(drop & {Path(p).stem for v in sets.values() for p in v})) or "-")]
    (d / "muktabodha_prepare.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)
    return n_lines


if __name__ == "__main__":
    mss = sys.argv[1:] or sorted(p.stem for p in (DATA / "muktabodha").glob("*.json"))
    total = {"train": 0, "val": 0, "test": 0}
    for ms in mss:
        for n, c in prepare(ms).items():
            total[n] += c
    print(f"[prepare] {len(mss)} manuscripts: lines " + ", ".join(f"{n} {c}" for n, c in total.items()))
