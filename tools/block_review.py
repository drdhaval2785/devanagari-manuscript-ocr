"""Review page for the block finder on the pages whose true rows per block are known
(data/kraken_review/feedback/block_counts.tsv): what it found before (seg_compare_*.tsv) and what it finds now,
from Kraken's lines and from the U-Net's, against the true counts, each page drawn with the current code.

    PYTHONPATH=. .venv/bin/python tools/block_review.py
-> data/kraken_review/block_review.html (+ block_review/ pictures). Local files only: the pictures are manuscript pages.
"""
import csv, html as H, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, __import__("os").path.dirname(__file__))
from segcompare_review import rows_and_picture
from msocr.config import DATA

OUT = DATA / "kraken_review"
PICS = OUT / "block_review"
BOXES = ("top", "main", "bottom", "left", "right")


def parse(s):
    import re
    out = {b: 0 for b in BOXES}
    for name, n in re.findall(r"(top|main|bottom|left|right)\s+(\d+)", s or ""):
        out[name] = int(n)
    return out


def fmt(c):
    return ", ".join(f"{b} {c[b]}" for b in BOXES if c.get(b)) or "no rows"


def main():
    lines = [l for l in open(OUT / "feedback" / "block_counts.tsv", encoding="utf-8") if not l.startswith("#")]
    truth = [(r["manuscript"], r["page"], {b: int(r[b]) for b in BOXES}) for r in csv.DictReader(lines, delimiter="\t")]
    before = {}
    for f in sorted((OUT / "feedback").glob("seg_compare_*.tsv")):
        for r in csv.DictReader(open(f, encoding="utf-8"), delimiter="\t"):
            before[(r["manuscript"], r["page"])] = (parse(r["from_kraken"]), parse(r["from_unet"]))
    items = []
    for ms, stem, t in truth:
        (PICS / ms).mkdir(parents=True, exist_ok=True)
        now = {}
        for seg, tag in (("seg", "kraken"), ("seg_torch", "unet")):
            counts, pic = rows_and_picture(ms, stem + ".jpg", seg)
            counts = {b: counts.get(b, 0) for b in BOXES}
            pic = pic.convert("RGB")
            pic.thumbnail((1400, 1400))
            pic.save(PICS / ms / f"{stem}_{tag}.jpg", quality=80)
            now[tag] = counts
        bk, bu = before.get((ms, stem), ({}, {}))
        items.append((ms, stem, t, {b: bk.get(b, 0) for b in BOXES}, {b: bu.get(b, 0) for b in BOXES}, now["kraken"], now["unet"]))
        print(f"  {ms} {stem}", flush=True)
    write(items)


def write(items):
    ok = lambda c, t: all(c[b] == t[b] for b in BOXES)
    n = len(items)
    tot = {k: sum(ok(it[i], it[2]) for it in items) for k, i in (("bk", 3), ("bu", 4), ("nk", 5), ("nu", 6))}
    mark = lambda c, t: ('<span class="ok">✓</span>' if ok(c, t) else '<span class="no">✗</span>')
    trs = "".join(
        f'<tr class="{"fixed" if ok(nk, t) and not ok(bk, t) else ("still" if not ok(nk, t) else "same")}"><td><a href="#{ms}-{stem}">{ms}</a></td><td>{stem}</td><td>{fmt(t)}</td>'
        f'<td>{mark(bk, t)} {fmt(bk)}</td><td>{mark(nk, t)} {fmt(nk)}</td><td>{mark(bu, t)} {fmt(bu)}</td><td>{mark(nu, t)} {fmt(nu)}</td></tr>'
        for ms, stem, t, bk, bu, nk, nu in items)
    secs = []
    for ms, stem, t, bk, bu, nk, nu in items:
        status = "fixed" if ok(nk, t) and not ok(bk, t) else ("still" if not ok(nk, t) else "same")
        label = {"fixed": "corrected now (from Kraken's lines)", "still": "still wrong (from Kraken's lines)", "same": "right before and now"}[status]
        secs.append(f'''<section class="pg {status}" id="{ms}-{stem}" data-ms="{ms}" data-stem="{stem}" data-t="{H.escape(fmt(t))}" data-k="{H.escape(fmt(nk))}" data-u="{H.escape(fmt(nu))}">
<h2>{ms} · {stem} <small>{label} · your count: {fmt(t)}</small></h2>
<div class="pair"><figure><figcaption>From Kraken's lines, now: {mark(nk, t)} {fmt(nk)} <span class="was">(before: {fmt(bk)})</span></figcaption><img loading="lazy" src="block_review/{ms}/{stem}_kraken.jpg"></figure>
<figure><figcaption>From the U-Net's lines, now: {mark(nu, t)} {fmt(nu)} <span class="was">(before: {fmt(bu)})</span></figcaption><img loading="lazy" src="block_review/{ms}/{stem}_unet.jpg"></figure></div>
<div class="form">{"".join(f'<label><input type="radio" name="v-{ms}-{stem}" value="{v}"> {lbl}</label>' for v, lbl in (("both_right", "both right now"), ("kraken_right", "only Kraken's right"), ("unet_right", "only U-Net's right"), ("both_wrong", "both still wrong")))}
<input class="note" placeholder="what is still wrong: which block, which rows, and why"></div></section>''')
    doc = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Block finder review</title>
<style>
:root {{ --bg:#faf8f3; --fg:#222; --muted:#666; --line:#ddd; --ok:#1a7f37; --no:#b42318; --hl:#fff4d6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1c1b19; --fg:#e8e4da; --muted:#a8a294; --line:#3a3833; --ok:#5fd07a; --no:#ff8a7a; --hl:#3a3320; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; margin:0 auto; max-width:1500px; padding:16px; }}
h1 {{ font-size:22px; }} h2 {{ font-size:17px; margin:28px 0 8px; }} small, .was {{ color:var(--muted); font-weight:normal; }}
table {{ border-collapse:collapse; font-size:13px; width:100%; }} td, th {{ border-bottom:1px solid var(--line); padding:3px 8px; text-align:left; }}
tr.fixed td {{ background:var(--hl); }} .ok {{ color:var(--ok); font-weight:bold; }} .no {{ color:var(--no); font-weight:bold; }}
.pair {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }} figure {{ margin:0; }}
figcaption {{ font-size:13px; margin-bottom:4px; }} img {{ width:100%; height:auto; border:1px solid var(--line); }}
.form {{ margin:8px 0 0; display:flex; flex-wrap:wrap; gap:14px; align-items:center; }}
.note {{ flex:1; min-width:240px; padding:4px 6px; background:transparent; color:var(--fg); border:1px solid var(--line); }}
.bar {{ position:sticky; top:0; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--line); z-index:1; display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
button {{ font:inherit; padding:4px 12px; }} .hide {{ display:none; }}
@media (max-width:800px) {{ .pair {{ grid-template-columns:1fr; }} }}
</style></head><body>
<h1>Block finder review: {n} pages with your counts</h1>
<p>The blocks are now cut along the white space between them, stepping round the end of a half line. Pages fully right
in every block, from Kraken's lines: <b>{tot["bk"]} before, {tot["nk"]} now</b>; from the U-Net's lines:
<b>{tot["bu"]} before, {tot["nu"]} now</b> (of {n}). Rows highlighted in the table are the ones corrected now. For each
page, say what is right now and what is still wrong; "Save feedback" downloads <code>block_review_feedback.tsv</code>.</p>
<div class="bar"><button onclick="show('all')">all pages</button><button onclick="show('still')">still wrong</button><button onclick="show('fixed')">corrected now</button>
<button onclick="save()">Save feedback</button> <span id="count"></span></div>
<table><tr><th>manuscript</th><th>page</th><th>your count</th><th>Kraken's lines, before</th><th>Kraken's lines, now</th><th>U-Net's lines, before</th><th>U-Net's lines, now</th></tr>{trs}</table>
{"".join(secs)}
<script>
const KEY = 'block_review_v1';
let st = {{}};
try {{ st = JSON.parse(localStorage.getItem(KEY) || '{{}}'); }} catch (e) {{}}
const keep = () => {{ try {{ localStorage.setItem(KEY, JSON.stringify(st)); }} catch (e) {{}} count(); }};
function count() {{ document.getElementById('count').textContent = Object.values(st).filter(v => v.verdict).length + ' of {n} marked'; }}
function show(which) {{ document.querySelectorAll('.pg').forEach(s => s.classList.toggle('hide', which !== 'all' && !s.classList.contains(which))); }}
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
  let tsv = 'manuscript\\tpage\\tyour_count\\tfrom_kraken_now\\tfrom_unet_now\\tverdict\\tnote\\n';
  document.querySelectorAll('.pg').forEach(sec => {{
    const s = st[sec.dataset.ms + '/' + sec.dataset.stem] || {{}};
    if (s.verdict || s.note) tsv += [sec.dataset.ms, sec.dataset.stem, sec.dataset.t, sec.dataset.k, sec.dataset.u, s.verdict || '', clean(s.note)].join('\\t') + '\\n';
  }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([tsv], {{type: 'text/tab-separated-values'}}));
  a.download = 'block_review_feedback.tsv'; a.click();
}}
</script></body></html>'''
    (OUT / "block_review.html").write_text(doc, encoding="utf-8")
    print(f"[block_review] {n} pages; fully right from Kraken's lines {tot['bk']} -> {tot['nk']}, from the U-Net's {tot['bu']} -> {tot['nu']}: {OUT / 'block_review.html'}")


if __name__ == "__main__":
    main()
