"""Ready the Muktabodha manuscripts for the recogniser's pool: their rows paired with the e-text's lines one by one
(msocr/linepair.py: by what the recogniser reads in them) and cut into line images
(data/<ms>/lines_seg_torch/{train,val,test}/), the cut recognizer.pool trains on.

    PYTHONPATH=. .venv/bin/python tools/muktabodha_prepare.py [ms ...]      # default: every manuscript with a config
    PYTHONPATH=. .venv/bin/python tools/muktabodha_prepare.py --model data/pool/models/pool3b/best.pt RTJ_4387

Run it after msocr.muktabodha's build, gt.extract and segment (align is not needed). A line is left out when no row
reads as it, when an end of it and of its row do not agree, or when it has an illegible stretch (<gap/>); a page with
fewer than half its lines paired is left out whole. Every page stays in the set it is in already (else the set
split_pages.tsv remembers; else train), so that no page a model has learnt from comes to score it; a manuscript never
split is split as align splits it (under 60 pages: a tenth to test, a tenth to validation). What was there before the
first run is kept in backups/data/muktabodha_before_linepair/<ms>/. The line cache is cut afresh. Writes
data/<ms>/muktabodha_prepare.txt. Read with the best recogniser there is (--model): a weak one pairs fewer lines.
"""
import argparse
import collections
import json
import random
import shutil
from pathlib import Path

from msocr import linepair
from msocr.align import SEED, TEST_PAGES, VAL_FRACTION, _page_xml
from msocr.config import DATA, MANUSCRIPTS, ROOT
from msocr.predict import DEFAULT_MODEL, model_label
from msocr.recognizer import build_cache, lines_dir, load_model

SEG = "seg_torch"
SMALL = 60
SETS = ("train", "val", "test")
BACKUP = ROOT / "backups" / "data" / "muktabodha_before_linepair"


def backup(ms):
    """the manuscript's pairs, split and line images as they stand, copied once, before anything is changed"""
    d, out = DATA / ms, BACKUP / ms
    if out.exists() or not any((d / f"{n}_{SEG}.txt").exists() for n in SETS):
        return
    part = out.with_name(ms + ".part")
    if part.exists():
        shutil.rmtree(part)
    part.mkdir(parents=True)
    for name in [f"{n}_{SEG}.txt" for n in SETS] + ["split_pages.tsv", "muktabodha_prepare.txt", f"pagexml_{SEG}", f"lines_{SEG}"]:
        src = d / name
        if src.is_dir():
            shutil.copytree(src, part / name)
        elif src.exists():
            shutil.copy2(src, part / name)
    part.rename(out)


def prepare(ms, model):
    d = DATA / ms
    res = linepair.pair(ms, linepair.read_pages(ms, model, cache=d / "linepair_readings.json"))
    good = [r for r in res if not r["whole"]]
    before = {n: len(list((lines_dir(ms, SEG) / n).glob("*.png"))) for n in SETS}
    now = {Path(p).stem: n for n in SETS if (d / f"{n}_{SEG}.txt").exists() for p in (d / f"{n}_{SEG}.txt").read_text().split()}
    record_path = d / "split_pages.tsv"
    record = dict(l.split("\t") for l in record_path.read_text().splitlines() if "\t" in l) if record_path.exists() else {}
    backup(ms)
    stems = sorted(r["image"][:-4] for r in good)
    if not now and not record:
        shuffled = stems[:]
        random.Random(SEED).shuffle(shuffled)
        if len(shuffled) < SMALL:
            n_test = n_val = max(1, round(len(shuffled) / 10))
        else:
            n_test, n_val = TEST_PAGES, max(1, round(VAL_FRACTION * (len(shuffled) - TEST_PAGES)))
        record = {s: "test" if i < n_test else "val" if i < n_test + n_val else "train" for i, s in enumerate(shuffled)}
    record.update(now)                                  # a page left out now keeps its set for later
    for s in stems:
        record.setdefault(s, "train")
    xml_dir = d / f"pagexml_{SEG}"
    if xml_dir.exists():
        shutil.rmtree(xml_dir)
    xml_dir.mkdir()
    for r in good:
        stem = r["image"][:-4]
        triples = [(b, o, t) for _, b, o, t, _ in r["lines"]]
        (xml_dir / f"{stem}.xml").write_text(_page_xml(MANUSCRIPTS[ms]["images"] / r["image"], r["size"], triples), encoding="utf-8")
        (xml_dir / f"{stem}.json").write_text(json.dumps([{"baseline": b, "outline": o, "text": t} for b, o, t in triples],
                                                         ensure_ascii=False), encoding="utf-8")
    sets = {n: [s for s in stems if record[s] == n] for n in SETS}
    for n, v in sets.items():
        (d / f"{n}_{SEG}.txt").write_text("\n".join(str(xml_dir / f"{s}.xml") for s in v) + "\n")
    record_path.write_text("".join(f"{s}\t{n}\n" for s, n in sorted(record.items())))
    if lines_dir(ms, SEG).exists():
        shutil.rmtree(lines_dir(ms, SEG))
    build_cache(ms, verbose=False, seg=SEG)
    after = {n: len(list((lines_dir(ms, SEG) / n).glob("*.png"))) for n in SETS}

    left = [(r["image"][:-4], j + 1, why) for r in good for j, why in r["faults"]]
    whole = [(r["image"][:-4], r["whole"]) for r in res if r["whole"]]
    new_pages = [s for s in stems if s not in now]
    _, _, ck = load_model(model)
    report = [f"[prepare] {ms}: read with {model_label(model, ck)}",
              f"  paired : {len(good)} of {len(res)} pages, {sum(len(r['lines']) for r in good)} of {sum(r['total'] for r in res)} lines",
              "  pages  : " + ", ".join(f"{n} {len(v)}" for n, v in sets.items()) + f" ({len(new_pages)} not in use before)",
              "  lines  : " + ", ".join(f"{n} {before[n]} -> {after[n]}" for n in SETS),
              f"  left out : {len(whole)} pages whole, and {len(left)} single lines of other pages"]
    summary = "\n".join(report)
    report += [f"    {s}: {why}" for s, why in whole]
    report += [f"    {n:5d}  {k}" for k, n in collections.Counter(w for _, _, w in left).most_common()]
    report += [f"      {s} line {j}: {why}" for s, j, why in left]
    (d / "muktabodha_prepare.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary, flush=True)
    return before, after


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ms", nargs="*")
    ap.add_argument("--model", help="the recogniser the rows are read with (default predict's)")
    a = ap.parse_args()
    model = Path(a.model).resolve() if a.model else DEFAULT_MODEL
    mss = a.ms or sorted(p.stem for p in (DATA / "muktabodha").glob("*.json"))
    total = {"before": collections.Counter(), "after": collections.Counter()}
    for ms in mss:
        before, after = prepare(ms, model)
        total["before"].update(before)
        total["after"].update(after)
    print(f"[prepare] {len(mss)} manuscripts: lines "
          + ", ".join(f"{n} {total['before'][n]} -> {total['after'][n]}" for n in SETS))
