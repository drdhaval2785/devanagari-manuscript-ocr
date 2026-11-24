"""Side-by-side review of the pages where our U-Net's lines and Kraken's lead the row finder to different rows or blocks.

    PYTHONPATH=. .venv/bin/python tools/segcompare_review.py [ms ...]      # default: the block-layout manuscripts

Takes the pages marked DIFF in data/tseg/compare_<ms>.txt (python -m msocr.tseg compare), makes the U-Net's segmentation
of them where missing (data/<ms>/seg_torch/), draws each page twice with the current row finder (from Kraken's lines,
data/<ms>/seg/, and from the U-Net's), and writes data/kraken_review/seg_compare.html: the two pictures side by side
with their rows per block, and a form (Kraken right, U-Net right, both right, both wrong, a note) kept in the browser
until "Save" downloads seg_compare_feedback.tsv. Local files only: the pictures are of manuscript pages.
"""
import html as H
import json
import sys
import warnings

warnings.filterwarnings("ignore")
from msocr import blocks, predict, show
from msocr.blocks import layout_of
from msocr.config import DATA, MANUSCRIPTS

OUT = DATA / "kraken_review"
PICS = OUT / "seg_compare"
BLOCKY = ("tripatha", "panchapatha")


def pages_differing(ms):
    f = DATA / "tseg" / f"compare_{ms}.txt"
    if not f.exists():
        return []
    return [l.split("\t")[0] for l in f.read_text(encoding="utf-8").splitlines() if l.rstrip().split("\t")[-1] in ("DIFF", "DIFFER")]


def rows_and_picture(ms, image, seg):
    kraken = (show.SEG, blocks.SEG, predict.SEG)
    show.SEG = blocks.SEG = predict.SEG = seg
    try:
        if layout_of(ms) in BLOCKY:
            found, size, note = blocks.blocks_for_page(ms, image)
            counts = {b: len(v["rows"]) for b, v in found.items()}
        else:
            merged, size, note, warn = predict.rows_for_page(ms, image)
            counts = {"main": len(merged)}
        pic, warn = show.draw_page(ms, image)
    finally:
        show.SEG, blocks.SEG, predict.SEG = kraken
    return counts, pic


def main():
    from msocr.tseg import segment_pages
    ids = sys.argv[1:] or sorted(m for m in MANUSCRIPTS if layout_of(m) in BLOCKY and pages_differing(m))
    items = []
    for ms in ids:
        stems = pages_differing(ms)
        if not stems:
            continue
        imgs = [s + ".jpg" for s in stems]
        segment_pages(ms, [i for i in imgs if not (DATA / ms / "seg_torch" / i.replace(".jpg", ".json")).exists()] or [], None, False)
        (PICS / ms).mkdir(parents=True, exist_ok=True)
        for img in imgs:
            stem = img[:-4]
            res = {}
            for seg, tag in (("seg", "kraken"), ("seg_torch", "unet")):
                counts, pic = rows_and_picture(ms, img, seg)
                pic = pic.convert("RGB")
                pic.thumbnail((1400, 1400))
                pic.save(PICS / ms / f"{stem}_{tag}.jpg", quality=80)
                res[tag] = counts
            items.append((ms, stem, res["kraken"], res["unet"]))
        print(f"[segcompare] {ms}: {len(imgs)} pages drawn", flush=True)
    write_html(items)


def fmt(c):
    order = ["top", "main", "bottom", "left", "right"]
    return ", ".join(f"{b} {c[b]}" for b in order if b in c) or "no rows"


def write_html(items):
    still = sum(1 for _, _, k, u in items if k != u)
    rows = "".join(f'<tr><td><a href="#{ms}-{stem}">{ms}</a></td><td>{stem}</td><td>{fmt(k)}</td><td>{fmt(u)}</td>'
                   f'<td>{"differ" if k != u else "same now"}</td></tr>' for ms, stem, k, u in items)
    secs = []
    for ms, stem, k, u in items:
        secs.append(f'''<section class="pg" id="{ms}-{stem}" data-ms="{ms}" data-stem="{stem}" data-k="{H.escape(fmt(k))}" data-u="{H.escape(fmt(u))}">
<h2>{ms} · {stem} <small>{"differ" if k != u else "same rows now"}</small></h2>
<div class="pair"><figure><figcaption>From Kraken's lines: {fmt(k)}</figcaption><img loading="lazy" src="seg_compare/{ms}/{stem}_kraken.jpg"></figure>
<figure><figcaption>From the U-Net's lines: {fmt(u)}</figcaption><img loading="lazy" src="seg_compare/{ms}/{stem}_unet.jpg"></figure></div>
<div class="form">{"".join(f'<label><input type="radio" name="v-{ms}-{stem}" value="{v}"> {t}</label>' for v, t in (("kraken", "Kraken's is better"), ("unet", "U-Net's is better"), ("both_ok", "both equally right"), ("both_wrong", "both wrong")))}
<input class="note" placeholder="why: which block or rows are right or wrong, and on which side"></div></section>''')
    doc = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>U-Net and Kraken compared</title>
<style>
:root {{ --bg:#faf8f3; --fg:#222; --muted:#666; --line:#ddd; --accent:#8a3b12; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1c1b19; --fg:#e8e4da; --muted:#a8a294; --line:#3a3833; --accent:#e0a070; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; margin:0 auto; max-width:1500px; padding:16px; }}
h1 {{ font-size:22px; }} h2 {{ font-size:17px; margin:28px 0 8px; }} small {{ color:var(--muted); font-weight:normal; }}
table {{ border-collapse:collapse; font-size:13px; }} td, th {{ border-bottom:1px solid var(--line); padding:3px 10px; text-align:left; }}
.pair {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }} figure {{ margin:0; }}
figcaption {{ font-size:13px; color:var(--muted); margin-bottom:4px; }} img {{ width:100%; height:auto; border:1px solid var(--line); }}
.form {{ margin:8px 0 0; display:flex; flex-wrap:wrap; gap:14px; align-items:center; }}
.note {{ flex:1; min-width:240px; padding:4px 6px; background:transparent; color:var(--fg); border:1px solid var(--line); }}
.bar {{ position:sticky; top:0; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--line); z-index:1; }}
button {{ font:inherit; padding:4px 12px; }}
@media (max-width:800px) {{ .pair {{ grid-template-columns:1fr; }} }}
</style></head><body>
<h1>U-Net and Kraken compared: {len(items)} test pages</h1>
<p>Each page drawn twice with the current row finder: the rows and blocks built from Kraken's lines (left) and from the
U-Net's (right). {still} of them still differ; the rest came out the same once the U-Net's baselines were simplified
and the commentary rule changed. For each page, say which side is better and why (which block or rows are right or
wrong). Your answers stay in this browser until "Save feedback" downloads <code>seg_compare_feedback.tsv</code>.</p>
<div class="bar"><button onclick="save()">Save feedback</button> <span id="count"></span></div>
<table><tr><th>manuscript</th><th>page</th><th>from Kraken's lines</th><th>from the U-Net's lines</th><th></th></tr>{rows}</table>
{"".join(secs)}
<script>
const KEY = 'seg_compare_v1';
let st = {{}};
try {{ st = JSON.parse(localStorage.getItem(KEY) || '{{}}'); }} catch (e) {{}}
const keep = () => {{ try {{ localStorage.setItem(KEY, JSON.stringify(st)); }} catch (e) {{}} count(); }};
function count() {{ document.getElementById('count').textContent = Object.values(st).filter(v => v.verdict).length + ' of {len(items)} marked'; }}
document.querySelectorAll('.pg').forEach(sec => {{
  const id = sec.dataset.ms + '/' + sec.dataset.stem, s = st[id] || {{}};
  sec.querySelectorAll('input[type=radio]').forEach(r => {{ if (r.value === s.verdict) r.checked = true;
    r.addEventListener('change', () => {{ st[id] = Object.assign(st[id] || {{}}, {{verdict: r.value}}); keep(); }}); }});
  const n = sec.querySelector('.note'); n.value = s.note || '';
  n.addEventListener('input', () => {{ st[id] = Object.assign(st[id] || {{}}, {{note: n.value}}); keep(); }});
}});
count();
function save() {{
  const clean = (t) => (t || '').replace(/[\\t\\n]+/g, ' ');
  let tsv = 'manuscript\\tpage\\tfrom_kraken\\tfrom_unet\\tverdict\\tnote\\n';
  document.querySelectorAll('.pg').forEach(sec => {{
    const s = st[sec.dataset.ms + '/' + sec.dataset.stem] || {{}};
    if (s.verdict || s.note) tsv += [sec.dataset.ms, sec.dataset.stem, sec.dataset.k, sec.dataset.u, s.verdict || '', clean(s.note)].join('\\t') + '\\n';
  }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([tsv], {{type: 'text/tab-separated-values'}}));
  a.download = 'seg_compare_feedback.tsv'; a.click();
}}
</script></body></html>'''
    (OUT / "seg_compare.html").write_text(doc, encoding="utf-8")
    print(f"[segcompare] {len(items)} pages, {still} still differing: {OUT / 'seg_compare.html'}")


if __name__ == "__main__":
    main()
