// The public synthetic page for the stage fixtures: printed Devanagari, properly shaped (Chrome's text layout), written
// without spaces between the words as the manuscripts are, on paper-coloured ground. node tools/fixtures/render_synthetic.js
const puppeteer = require('/Users/Shared/drdhaval2785/edit_manuscripts/manuscript_editor/tests/node_modules/puppeteer-core');
const path = require('path');
const TEXT = ('धर्मक्षेत्रेकुरुक्षेत्रेसमवेतायुयुत्सवःमामकाःपाण्डवाश्चैवकिमकुर्वतसञ्जय' +
  'दृष्ट्वातुपाण्डवानीकंव्यूढंदुर्योधनस्तदाआचार्यमुपसङ्गम्यराजावचनमब्रवीत्' +
  'पश्यैतांपाण्डुपुत्राणामाचार्यमहतींचमूम्व्यूढांद्रुपदपुत्रेणतवशिष्येणधीमता' +
  'अत्रशूरामहेष्वासाभीमार्जुनसमायुधियुयुधानोविराटश्चद्रुपदश्चमहारथः').repeat(3);
const html = `<!doctype html><meta charset="utf-8"><style>
  html,body{margin:0;background:#e2d0aa}
  p{position:absolute;left:130px;top:40px;width:1240px;margin:0;font:34px "Kohinoor Devanagari","Devanagari Sangam MN","ITF Devanagari",serif;
    line-height:46px;color:#281c14;word-break:break-all;height:620px;overflow:hidden}
</style><p>${TEXT}</p>`;
(async () => {
  const b = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: 'new' });
  const p = await b.newPage();
  await p.setViewport({ width: 1500, height: 700, deviceScaleFactor: 1 });
  await p.setContent(html);
  await p.evaluate(() => document.fonts.ready);
  await p.screenshot({ path: path.join(__dirname, 'synthetic_page_1.png') });
  await b.close();
  console.log('synthetic_page_1.png written');
})();
