// Times the three models in Chrome with onnxruntime-web: WebAssembly on one thread (a page opened from a file, or a host
// without isolation headers), WebAssembly on 8 threads (a host sending the isolation headers, as this server does),
// and WebGPU with WebAssembly as the fallback. node run.js
const puppeteer = require('/Users/Shared/drdhaval2785/edit_manuscripts/manuscript_editor/tests/node_modules/puppeteer-core');
const http = require('http');
const fs = require('fs');
const path = require('path');

const DIR = __dirname;
const PORT = 8765;
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const TYPES = { '.html': 'text/html; charset=utf-8', '.onnx': 'application/octet-stream', '.js': 'text/javascript' };

const server = http.createServer((req, res) => {
  const p = path.join(DIR, decodeURIComponent(req.url.split('?')[0]));
  if (!p.startsWith(DIR) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, {
    'Content-Type': TYPES[path.extname(p)] || 'application/octet-stream',
    'Cross-Origin-Opener-Policy': 'same-origin',         // these two make the page cross-origin isolated:
    'Cross-Origin-Embedder-Policy': 'credentialless',    // WebAssembly threads are allowed only then
  });
  fs.createReadStream(p).pipe(res);
});

(async () => {
  await new Promise((r) => server.listen(PORT, r));
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new',
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-features=WebGPU'] });
  const runs = [{ ep: 'wasm', threads: 1 }, { ep: 'wasm', threads: 8 }, { ep: 'webgpu', threads: 1 }];
  for (const cfg of runs) {
    const page = await browser.newPage();
    page.on('pageerror', (e) => console.log('  page error:', e.message));
    await page.goto(`http://localhost:${PORT}/index.html?ep=${cfg.ep}&threads=${cfg.threads}`, { waitUntil: 'load' });
    const { env, out } = await page.evaluate(() => window.runBench());
    console.log(`\n== ${cfg.ep}, ${cfg.threads} thread(s): isolated ${env.isolated}, cores ${env.cores}, WebGPU in browser ${env.webgpu}` +
                (env.adapter !== undefined ? `, adapter ${env.adapter}` : '') + `, ${env.ua}`);
    for (const r of out) {
      console.log(r.error ? `   ${r.model.padEnd(36)} ERROR ${r.error}`
        : `   ${r.model.padEnd(36)} load ${String(r.load_ms).padStart(5)} ms   first run ${String(r.first_ms).padStart(6)} ms   then ${String(r.run_ms).padStart(6)} ms a run`);
    }
    await page.close();
  }
  await browser.close();
  server.close();
})().catch((e) => { console.error(e); server.close(); process.exit(1); });
