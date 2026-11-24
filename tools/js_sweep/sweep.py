"""Every segmented text page: reference.rows against the JS port (server.js, which loads Manuscribe's app/js/recog*.js),
notes of the reference logged. The full-corpus check of the row and block finders' port (2026-11-24: 3,497 pages, all
exact after one fix); worth running again before every release and after any change to align.py, blocks.py or
recogrows.js.

    PYTHONPATH=. .venv/bin/python tools/js_sweep/sweep.py [results folder] [only ms]   # about 25 min
-> <results folder>/sweep.jsonl (one line per page: ok, largest difference, why) and sweep.out (progress);
   default results folder data/benchmarks/js_sweep/
"""
import sys, json, os, subprocess, time
sys.path.insert(0, '.')
import numpy as np
from PIL import Image
from msocr import reference, align, blocks
from msocr.config import DATA, MANUSCRIPTS, not_text
from msocr.blocks import layout_of
HERE = os.path.dirname(os.path.abspath(__file__))                          # server.js lives here
S = sys.argv[1] if len(sys.argv) > 1 else str(DATA / 'benchmarks' / 'js_sweep')  # where the results go
os.makedirs(S, exist_ok=True)
TMP = DATA / 'fixtures' / 'rows_sweep'
TMP.mkdir(parents=True, exist_ok=True)
notes = []
_fr, _fe, _rib, _fb, _reb, _bands = align.find_rows, align.fill_edge_rows, blocks.rows_in_block, blocks.find_blocks, blocks._rebalance, blocks._bands
def fr(*a, **k):
    r = _fr(*a, **k); notes.append(('find_rows', ('hint ' if k.get('pitch_hint') else '') + r[3])); return r
def fe(*a, **k):
    r = _fe(*a, **k); notes.append(('fill_edge', r[1])); return r
def fb(*a, **k):
    r = _fb(*a, **k); notes.append(('find_blocks', r[1])); return r
def reb(bands, *a):
    before = [len(b['rows']) for b in bands]; r = _reb(bands, *a); after = [len(b['rows']) for b in r]
    notes.append(('rebalance', 'moved' if before != after else 'same')); return r
def bd(*a):
    r = _bands(*a); notes.append(('bands', f'{len(r)} bands')); return r
align.find_rows = blocks.find_rows = fr
align.fill_edge_rows = blocks.fill_edge_rows = fe
blocks.find_blocks = fb
blocks._rebalance = reb
blocks._bands = bd
pages = []
for ms in sorted(MANUSCRIPTS):
    if not (DATA / ms / 'seg').is_dir() or layout_of(ms) not in ('linear', 'tripatha', 'panchapatha'):
        continue
    skip = not_text(ms)
    for p in sorted((DATA / ms / 'seg').glob('*.json')):
        if p.stem not in skip and (MANUSCRIPTS[ms]['images'] / f'{p.stem}.jpg').exists():
            pages.append((ms, p.stem))
only = sys.argv[2] if len(sys.argv) > 2 else None
if only: pages = [p for p in pages if p[0] in only.split(',')]
print(len(pages), 'pages', flush=True)
node = subprocess.Popen(['node', os.path.join(HERE, 'server.js')], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
log = open(os.path.join(S, 'sweep.jsonl'), 'w')
t0 = time.time(); bad = 0
for n, (ms, stem) in enumerate(pages):
    seg = json.load(open(DATA / ms / 'seg' / f'{stem}.json'))
    lines = [{'baseline': l['baseline'], 'boundary': l['boundary']} for l in seg['lines']]
    g = reference.grey(np.asarray(Image.open(MANUSCRIPTS[ms]['images'] / f'{stem}.jpg').convert('RGB')))
    layout = layout_of(ms)
    notes.clear()
    try:
        rows = reference.rows(lines, g, layout)
    except Exception as e:
        log.write(json.dumps({'ms': ms, 'stem': stem, 'pyerror': repr(e)}) + '\n'); continue
    ref = {b: [[list(map(list, base)), list(map(list, outl))] for base, outl in rs] for b, rs in rows.items()}
    d = TMP / 'page'; d.mkdir(exist_ok=True)
    g.tofile(d / 'grey.raw')
    json.dump({'lines': lines, 'ref': ref, 'layout': layout, 'width': int(g.shape[1]), 'height': int(g.shape[0])}, open(d / 'page.json', 'w'))
    node.stdin.write(str(d) + '\n'); node.stdin.flush()
    res = json.loads(node.stdout.readline())
    res.update(ms=ms, stem=stem, layout=layout, notes=list(notes))
    if not res['ok']: bad += 1
    log.write(json.dumps(res) + '\n'); log.flush()
    if n % 100 == 0: print(n, ms, stem, f'{time.time() - t0:.0f}s', 'bad', bad, flush=True)
node.stdin.close(); node.wait()
for f in (TMP / 'page').iterdir(): f.unlink()
(TMP / 'page').rmdir(); TMP.rmdir()
print('done', len(pages), 'bad', bad, f'{time.time() - t0:.0f}s')
