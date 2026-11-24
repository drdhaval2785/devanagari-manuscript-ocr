// reads "<dir>" lines on stdin; each dir has grey.raw, page.json {lines, ref, layout, width, height}; prints a JSON result
const fs = require('fs'), path = require('path'), vm = require('vm'), readline = require('readline');
// Manuscribe's folder: MANUSCRIBE in the environment, else where it sits on the machine this was written on
const T = process.env.MANUSCRIBE || '/Users/Shared/drdhaval2785/edit_manuscripts/manuscript_editor';
const ctx = { window: { ME: {} } };
vm.createContext(ctx);
for (const f of ['recogimage.js', 'recogvector.js', 'recogrows.js']) vm.runInContext(fs.readFileSync(path.join(T, 'app/js', f), 'utf8'), ctx, { filename: f });
const RR = ctx.window.ME.RecogRows;
const P = JSON.parse(fs.readFileSync(path.join(__dirname, '..', '..', 'msocr', 'pipeline.json'), 'utf8'));
function cmp(a, b) {
  const ka = Object.keys(a).sort(), kb = Object.keys(b).sort();
  if (ka.join() !== kb.join()) return { ok: false, why: 'boxes ' + ka + ' vs ' + kb };
  let worst = 0; const msgs = [];
  for (const k of ka) {
    if (a[k].length !== b[k].length) { msgs.push(k + ' rows ' + a[k].length + ' vs ' + b[k].length); continue; }
    a[k].forEach((r, i) => [0, 1].forEach((j) => {
      if (r[j].length !== b[k][i][j].length) { msgs.push(k + ' row ' + i + ' part ' + j + ' points ' + r[j].length + ' vs ' + b[k][i][j].length); return; }
      r[j].forEach((p, m) => { worst = Math.max(worst, Math.abs(p[0] - b[k][i][j][m][0]), Math.abs(p[1] - b[k][i][j][m][1])); });
    }));
  }
  return msgs.length ? { ok: false, why: msgs.join('; '), worst } : { ok: worst <= 1, worst };
}
const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (d) => {
  const pg = JSON.parse(fs.readFileSync(path.join(d, 'page.json'), 'utf8'));
  const grey = new Uint8Array(fs.readFileSync(path.join(d, 'grey.raw')));
  const t = Date.now();
  let res;
  try { const out = RR.rows(pg.lines, grey, pg.width, pg.height, pg.layout, P); res = cmp(out, pg.ref); if (!res.ok) res.mine = out; }
  catch (e) { res = { ok: false, why: 'ERROR ' + e.stack.split('\n').slice(0, 3).join(' | ') }; }
  res.ms = Date.now() - t;
  process.stdout.write(JSON.stringify(res) + '\n');
});
