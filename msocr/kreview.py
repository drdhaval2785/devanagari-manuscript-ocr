"""Is Kraken's segmentation right? A review of every segmented page, before anything learns from it.

    python -m msocr.kreview [LDII_xxx ...]          # every segmented manuscript if none named
    python -m msocr.kreview --no-pictures           # the tables only

Every page falls in one of three groups, from its rowcheck report (made afresh first where it is missing or stale):
  flagged      rowcheck warns: Kraken missed a row (the row finder filled it in), the rows are unevenly spaced (a
               line missed, or a note squeezed in), a gap may hide a short line, or no text rows were found
  odd count    rowcheck passes it, but its main text has two or more rows more or fewer than the manuscript's
               usual count: a chapter's end or a colophon, or a line Kraken lost that rowcheck did not notice
  repaired     flagged only for lines Kraken missed, and repaired by msocr/repair.py (the missing lines added,
               the page checked again): drawn with the repaired segmentation, to check each repair by eye
  clean        everything else; SAMPLE of them per manuscript are drawn too, to see how often "clean" is wrong
Written to data/kraken_review/:
  summary.tsv            per manuscript: pages, clean, odd count, flagged, and the flagged by kind
  <ms>.tsv               every page: group, rows, usual, Kraken's pieces, the reason
  <ms>_review.pdf        the flagged, repaired and odd-count pages, then the sample of clean pages, each drawn as `show` draws it
                         (Kraken's pieces in blue, the rows read as numbered bands, the warning in the caption)
  <ms>_review.html       the same pages, to check in a browser (opened from this folder, nothing leaves the computer):
                         for each page, correct or wrong, the kind of error, the rows, the block and a note; "Save
                         feedback" downloads <ms>_feedback.tsv, to hand back so the code can be mended where it errs.
                         What has been marked is kept in the browser until saved.
  index.html             every manuscript's review page, with its counts
With --all-pages every segmented page is drawn, not a sample of the clean ones.
The segmenter (msocr/tseg.py) trains only on pages rowcheck passes clean, and only with --reviewed.
"""
import argparse
import collections
import random
import re
import sys
import warnings

from .config import DATA, MANUSCRIPTS
from .blocks import layout_of

SAMPLE = 3
SEED = 20261123
OUT = DATA / "kraken_review"
KINDS = [("missed rows", r"Kraken missed"), ("uneven spacing", r"uneven row spacing"),
         ("short line may be missing", r"short line may be missing"), ("no text rows", r"^no (text )?rows|no columns|no blocks")]


def kind_of(warn, note):
    for name, pat in KINDS:
        if re.search(pat, warn or "", re.I):
            return name
    return "no text rows" if not warn else "other"


def read_report(ms):
    """-> (usual row count, [(image, pieces, rows, note, warning)])"""
    lines = (DATA / ms / "rowcheck.txt").read_text(encoding="utf-8").splitlines()
    m = re.search(r"usual row count (\d+)", lines[0])
    rows = []
    for ln in lines:
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.split("\t") + ["", ""]
        rows.append((f[0], int(f[1] or 0), int(f[2] or 0), f[3], f[4]))
    return int(m.group(1)) if m else 0, rows


ERRORS = [("missed", "a text line has no row"), ("extra", "a row where there is no text line of this block (gloss, note, stain, ruling)"),
          ("merged", "one row covers two lines"), ("split", "one line spread over two rows"),
          ("short", "a row does not cover the whole line (cut at the puṣpikā, a margin or an end)"),
          ("band", "the band sits too high or low (cuts off top marks or u/ū signs)"),
          ("block", "a row in the wrong block, or a block not found"), ("order", "rows in the wrong order"),
          ("repair", "an added (repaired) line is wrong"), ("other", "anything else (say what in the note)")]


def review(ms_ids, pictures=True, refresh=True, all_pages=False):
    from .tseg import ensure_rowchecks, flagged
    if refresh:
        ensure_rowchecks()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    summary = []
    for ms in ms_ids:
        if not (DATA / ms / "rowcheck.txt").exists():
            continue
        usual, rows = read_report(ms)
        bad = flagged(ms) or {}
        blocky = layout_of(ms) in ("tripatha", "panchapatha")
        groups, table = collections.defaultdict(list), []
        for image, pieces, n, note, warn in rows:
            stem = image.rsplit(".", 1)[0]
            if note == "not a text page":
                g, why = "not text", "a cover or a blank side"
            elif stem in bad and (DATA / ms / "seg_repaired" / f"{stem}.json").exists():
                g, why = "repaired", warn
            elif stem in bad:
                g, why = "flagged", f"{kind_of(warn, note)}: {warn or note}"
            elif usual and abs(n - usual) >= 2 and not (blocky and n == 0):
                g, why = "odd count", f"{n} rows, usually {usual}"
            else:
                g, why = "clean", ""
            groups[g].append(image)
            table.append((image, g, n, usual, pieces, why))
        kinds = collections.Counter(kind_of(w, nt) for im, p, n, nt, w in rows if im.rsplit(".", 1)[0] in bad)
        kinds["repaired"] = len(groups["repaired"])
        import json
        (OUT / f"{ms}.summary.json").write_text(json.dumps(
            {"ms": ms, "layout": layout_of(ms), "pages": len(rows), "clean": len(groups["clean"]),
             "odd": len(groups["odd count"]), "flagged": len(groups["flagged"]), "kinds": dict(kinds)}), encoding="utf-8")
        with open(OUT / f"{ms}.tsv", "w", encoding="utf-8") as f:
            f.write("image\tgroup\trows\tusual\tkraken pieces\twhy\n")
            for r in table:
                f.write("\t".join(map(str, r)) + "\n")
        if pictures:
            sample = groups["clean"] if all_pages else rng.sample(groups["clean"], min(SAMPLE, len(groups["clean"])))
            items = ([("FLAGGED", i) for i in groups["flagged"]] + [("REPAIRED", i) for i in groups["repaired"]]
                     + [("ODD COUNT", i) for i in groups["odd count"]]
                     + [("CLEAN" if all_pages else "SAMPLE OF CLEAN", i) for i in sample])
            _draw(ms, items)
            _html(ms, items, {r[0]: r[5] for r in table}, layout_of(ms))
        print(f"[kreview] {ms}: {len(rows)} pages: {len(groups['clean'])} clean, {len(groups['odd count'])} odd count, "
              f"{len(groups['repaired'])} repaired, {len(groups['flagged'])} flagged {dict(kinds)}", flush=True)
    # the table covers every manuscript reviewed so far, not only those of this run
    import json
    summary = []
    for sj in sorted(OUT.glob("*.summary.json")):
        x = json.loads(sj.read_text(encoding="utf-8"))
        summary.append((x["ms"], x["layout"], x["pages"], x["clean"], x["odd"], x["flagged"], collections.Counter(x["kinds"])))
    with open(OUT / "summary.tsv", "w", encoding="utf-8") as f:
        names = [k for k, _ in KINDS] + ["other"]
        names = names + ["repaired"]
        f.write("manuscript\tlayout\tpages\tclean\todd count\tflagged (repaired ones counted in their kind too)\t" + "\t".join(names) + "\n")
        for ms, lay, n, c, o, fl, kinds in summary:
            f.write(f"{ms}\t{lay}\t{n}\t{c}\t{o}\t{fl}\t" + "\t".join(str(kinds.get(k, 0)) for k in names) + "\n")
        tot = [sum(x[i] for x in summary) for i in (2, 3, 4, 5)]
        allk = sum((x[6] for x in summary), collections.Counter())
        f.write(f"ALL\t\t{tot[0]}\t{tot[1]}\t{tot[2]}\t{tot[3]}\t" + "\t".join(str(allk.get(k, 0)) for k in names) + "\n")
    print(f"[kreview] {sum(x[2] for x in summary)} pages in {len(summary)} manuscripts; table {OUT / 'summary.tsv'}")
    _index()


def _draw(ms, items):
    """items: [(group label, image)], drawn in that order into <ms>_review.pdf; repaired pages with their repaired lines."""
    from . import blocks, predict, show
    from .show import draw_page, write_pdf
    pics = OUT / "pictures" / ms
    pics.mkdir(parents=True, exist_ok=True)
    made = []
    kraken = (show.SEG, blocks.SEG, predict.SEG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for group, image in items:
            seg = "seg_repaired" if group == "REPAIRED" else kraken[0]
            show.SEG = blocks.SEG = predict.SEG = seg              # which segmentation draw_page reads
            try:
                pic, warn = draw_page(ms, image)
            finally:
                show.SEG, blocks.SEG, predict.SEG = kraken
            from PIL import ImageDraw
            from .show import font
            dr = ImageDraw.Draw(pic)
            x0 = pic.width - 280                        # top right, clear of show's caption on the left
            dr.rectangle([x0, 0, pic.width, 34], fill=(255, 255, 255))
            dr.text((x0 + 10, 6), group, fill=(0, 120, 0) if "CLEAN" in group else (200, 0, 0), font=font(22))
            path = pics / (image.rsplit(".", 1)[0] + ".jpg")
            pic.convert("RGB").save(path, quality=80)
            made.append(path)
    if made:
        write_pdf(made, OUT / f"{ms}_review.pdf")


def _html(ms, items, why, layout):
    """<ms>_review.html: every drawn page with a form; Save downloads <ms>_feedback.tsv."""
    import html as H
    import json
    boxes = ["main", "top", "bottom", "left", "right"] if layout in ("tripatha", "panchapatha") else []
    errs = "".join(f'<label title="{H.escape(t)}"><input type="checkbox" value="{c}"> {c}</label>' for c, t in ERRORS)
    legend = "".join(f"<li><b>{c}</b>: {H.escape(t)}</li>" for c, t in ERRORS)
    pages = []
    for group, image in items:
        stem = image.rsplit(".", 1)[0]
        page = stem.split("_", 1)[1]
        blk = ("<select class=blk><option value=''>block…</option>" + "".join(f"<option>{b}</option>" for b in boxes) + "</select>") if boxes else ""
        pages.append(f'''<section class="pg" data-stem="{stem}" data-group="{group}">
<h3><span class="g {'ok' if 'CLEAN' in group else 'bad'}">{group}</span> {page} <small>{H.escape(why.get(image, ""))}</small></h3>
<a href="pictures/{ms}/{stem}.jpg" target="_blank"><img loading="lazy" src="pictures/{ms}/{stem}.jpg"></a>
<div class="f"><label><input type="radio" name="v{stem}" value="ok"> correct</label>
<label><input type="radio" name="v{stem}" value="not_text"> not a text page</label>
<label><input type="radio" name="v{stem}" value="wrong"> wrong:</label> <span class="e">{errs}</span>
<input class="rows" placeholder="rows, e.g. 3, 12-13">{blk}<input class="note" placeholder="what is wrong"></div></section>''')
    doc = rf'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{ms}: is Kraken's segmentation right?</title>
<style>
body{{font:15px system-ui,sans-serif;margin:16px auto;max-width:1400px;padding:0 16px;background:#f4f1ea;color:#222}}
.pg{{background:#fff;border:1px solid #d8d2c4;border-radius:8px;padding:10px 12px;margin:0 0 18px}}
.pg.done{{border-color:#7a9}} .pg.wrong{{border:2px solid #c33}}
h3{{margin:0 0 6px;font-size:15px}} h3 small{{color:#666;font-weight:400}}
.g{{font-size:12px;padding:2px 7px;border-radius:9px;margin-right:6px}} .g.ok{{background:#dfeedd}} .g.bad{{background:#f6dcd6}}
img{{width:100%;display:block;border-radius:4px}}
.f{{display:flex;flex-wrap:wrap;gap:6px 12px;align-items:center;margin-top:8px}}
.e label{{margin-right:8px;white-space:nowrap}} input.rows{{width:150px}} input.note{{flex:1;min-width:240px}}
input,select{{font:inherit;padding:3px 6px}}
#bar{{position:sticky;top:0;background:#f4f1ea;padding:8px 0;z-index:2;display:flex;gap:12px;align-items:center;border-bottom:1px solid #d8d2c4}}
button{{font:inherit;padding:6px 14px}} ul{{columns:2}}
</style></head><body>
<div id="bar"><b>{ms}</b> <span id="count"></span> <button onclick="save()">Save feedback ({ms}_feedback.tsv)</button></div>
<p>Each picture is Kraken's segmentation as the row finder reads it: Kraken's pieces in blue, the rows read as numbered
bands, the warning in the caption; on a repaired page the added lines are part of the bands. Mark each page correct,
not a text page (a cover, a blank side: it will no longer be read or learnt from), or wrong; for a wrong one tick what is wrong, give the row numbers (and the block), and say what in a note. Unmarked pages
are left out of the file. What you mark is kept in this browser until you save. Then hand the saved file back.</p>
<ul>{legend}</ul>
{"".join(pages)}
<script>
const MS = {json.dumps(ms)}, KEY = "kreview." + MS;
const secs = [...document.querySelectorAll(".pg")];
function state() {{
  return secs.map(s => {{
    const v = s.querySelector("input[type=radio]:checked");
    return {{stem: s.dataset.stem, group: s.dataset.group, verdict: v ? v.value : "",
      errors: [...s.querySelectorAll(".e input:checked")].map(i => i.value),
      rows: s.querySelector(".rows").value.trim(), block: (s.querySelector(".blk") || {{value: ""}}).value,
      note: s.querySelector(".note").value.trim()}};
  }});
}}
function paint() {{
  let n = 0;
  for (const s of secs) {{
    const v = s.querySelector("input[type=radio]:checked");
    s.classList.toggle("done", !!v); s.classList.toggle("wrong", !!v && v.value === "wrong"); if (v) n++;
  }}
  document.getElementById("count").textContent = n + " of " + secs.length + " pages marked";
}}
function keep() {{ try {{ localStorage.setItem(KEY, JSON.stringify(state())); }} catch (e) {{}} paint(); }}
function restore() {{
  let saved = [];
  try {{ saved = JSON.parse(localStorage.getItem(KEY) || "[]"); }} catch (e) {{}}
  const by = Object.fromEntries(saved.map(x => [x.stem, x]));
  for (const s of secs) {{
    const x = by[s.dataset.stem]; if (!x) continue;
    if (x.verdict) {{ const r = s.querySelector(`input[type=radio][value=${{x.verdict}}]`); if (r) r.checked = true; }}
    for (const i of s.querySelectorAll(".e input")) i.checked = x.errors.includes(i.value);
    s.querySelector(".rows").value = x.rows; s.querySelector(".note").value = x.note;
    if (s.querySelector(".blk")) s.querySelector(".blk").value = x.block;
  }}
}}
function tsv() {{
  const clean = v => String(v).replace(/[\t\n]/g, " ");
  const lines = ["manuscript\tpage\tgroup\tverdict\terrors\trows\tblock\tnote"];
  for (const x of state()) if (x.verdict) lines.push([MS, x.stem, x.group, x.verdict, x.errors.join(","), x.rows, x.block, x.note].map(clean).join("\t"));
  return lines.join("\n") + "\n";
}}
function save() {{
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([tsv()], {{type: "text/tab-separated-values;charset=utf-8"}}));
  a.download = MS + "_feedback.tsv"; a.click();
}}
document.addEventListener("input", keep); document.addEventListener("change", keep);
restore(); paint();
</script></body></html>'''
    (OUT / f"{ms}_review.html").write_text(doc, encoding="utf-8")


def _index():
    """index.html: a link to every manuscript's review page, with its counts from summary.tsv."""
    import html as H
    rows = [l.split("\t") for l in (OUT / "summary.tsv").read_text(encoding="utf-8").splitlines()]
    head, body = rows[0], rows[1:]
    def cell(r, i):
        v = H.escape(r[i]) if i < len(r) else ""
        if i == 0 and (OUT / f"{r[0]}_review.html").exists():
            return f'<a href="{r[0]}_review.html">{v}</a>'
        return v
    trs = "".join("<tr>" + "".join(f"<td>{cell(r, i)}</td>" for i in range(len(head))) + "</tr>" for r in body)
    (OUT / "index.html").write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Kraken review</title>
<style>body{{font:15px system-ui,sans-serif;margin:24px;background:#f4f1ea}} table{{border-collapse:collapse;background:#fff}}
td,th{{border:1px solid #d8d2c4;padding:4px 8px;text-align:right}} td:first-child,td:nth-child(2){{text-align:left}}</style></head><body>
<h1>Is Kraken's segmentation right?</h1><p>Open a manuscript, mark its pages, save the feedback file and hand it back.</p>
<table><tr>{"".join(f"<th>{H.escape(h)}</th>" for h in head)}</tr>{trs}</table></body></html>''', encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ms_ids", nargs="*")
    ap.add_argument("--no-pictures", action="store_true")
    ap.add_argument("--no-refresh", action="store_true", help="use the rowcheck reports as they are")
    ap.add_argument("--all-pages", action="store_true", help="draw every page, not a sample of the clean ones")
    a = ap.parse_args()
    ids = a.ms_ids or sorted(m for m in MANUSCRIPTS if any((DATA / m / "seg").glob("*.json")))
    review(ids, pictures=not a.no_pictures, refresh=not a.no_refresh, all_pages=a.all_pages)
