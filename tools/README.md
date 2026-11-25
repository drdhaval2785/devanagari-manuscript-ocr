# tools/

Checks, benchmarks and experiments that sit beside the pipeline (`msocr/`). Run them from the project folder; the
Python ones need `PYTHONPATH=.`. None of them writes to `data/<ms>/seg/` or to anything a manuscript's export reads.

| Script | What it does |
|---|---|
| `segocr_test.py` | Which segmentation reads better: LDII_85885's 19 held-out test pages read by its recogniser (run3) on the rows built from Kraken's lines and on those built from our U-Net's (`data/LDII_85885/seg_torch/`, made by `python -m msocr.tseg segment LDII_85885 --pages ...`), each scored against the corrected text (CER per page, lines joined, spaces ignored) |
| `segcompare_review.py [ms ...]` | Side-by-side review of the pages where the U-Net's lines and Kraken's give different rows or blocks (from `data/tseg/compare_<ms>.txt`): both drawn with the current row finder, with a form (which is better, and why) that downloads `seg_compare_feedback.tsv`. Writes `data/kraken_review/seg_compare.html` and its pictures |
| `block_eval.py [--seg=seg,seg_torch] [-v]` | Scores the block finder on the pages whose true rows per block the review gave (`data/kraken_review/feedback/block_counts.tsv`). 2026-11-24, after the blocks were cut along the white space: 30 of 50 pages right in every block from Kraken's lines (5 before), 27 from the U-Net's (4 before); the rest judged exceptional and left |
| `block_review.py` | Review page for those pages: your counts, before and now, from both segmentations, drawn with the current code (`data/kraken_review/block_review.html`) |
| `verdict_check.py <feedback.tsv> ...` | Before a change to the row finder is kept: the row count of every page in a review batch, now against the `rowcheck` report its picture was drawn from. An ok page whose count changes is a regression |
| `commentary_fit.py <feedback.tsv> ...` | Fits the edge-row commentary classifier to the review verdicts (leave one manuscript out); rows and labels kept in `data/commentary/clf_data.npz` |
| `kraken_jobs_bench.py` | Kraken throughput with 1, 2 and 4 parallel jobs on 16 random pages (`data/benchmarks/kraken_jobs/`). 2026-11-23, kraken 7.2, M2 Pro 16 GB: 14.2, 9.3 and 10.0 s a page, so `segment` runs 2 jobs |
| `kraken_mps_test.py` | Kraken on the Mac GPU (MPS): time, peak memory, swap, and whether the lines equal the CPU segmentation (`data/benchmarks/kraken_mps/`). 2026-11-24, kraken 7.2: 4.4 s a page against 14.2 s on one CPU process, 1.9 GB peak, no swap; lines within 1.5 px of the CPU's and the same rows on all 4 pages (not byte-identical) |
| `browser_bench/export.py` | Exports segmenter (U-Net), recogniser (CRNN) and spacer to ONNX in `browser_bench/models/`, checked against PyTorch and timed natively. Needs `.venv-onnx` (see its docstring) |
| `browser_bench/run.js` | Times the three ONNX models in Chrome with onnxruntime-web 1.31.1: WebAssembly on 1 thread, on 8 threads (the local server sends the isolation headers), and WebGPU. Uses Manuscribe's `tests/node_modules/puppeteer-core`. 2026-11-24, Chrome 157 on the M2 Pro, a run after the first: segmenter (a 640×1456 page) 3.12 s / 0.56 s / 0.13 s; recogniser (a 48×1800 line) 91 / 42 / 47 ms; spacer (20 sentences of 120 letters) 437 / 120 / 21 ms (WebAssembly 1 thread / 8 threads / WebGPU). Model files 7.5, 8.0 and 13.1 MB |
| `rows_fixtures.py` | Row-finder fixtures for Manuscribe's JavaScript port (`app/js/recogrows.js`, checked by `tests/t-recog-conformance.js`): about forty pages chosen with a fixed seed (every layout, the hard block pages of `block_counts.tsv`, pages where a commentary row is left out, pages that reach the rarer branches), each with its grey page, Kraken's lines and `reference.rows`' output, in `data/fixtures/rows/` |
| `fixtures/render_synthetic.js`, `fixtures/synthetic_page_1.png` | The public synthetic page for the stage fixtures (`python -m msocr.fixtures --synthetic`): printed Devanagari without word spaces, shaped by Chrome |
| `js_sweep/sweep.py`, `js_sweep/server.js` | The JavaScript row and block finders (Manuscribe's `app/js/recogrows.js`) against the reference on every segmented text page: 2026-11-24, 3,497 pages (Kraken's lines) and 71 (the U-Net's), all exact. Run before every release and after any change to `align.py`, `blocks.py` or `recogrows.js` (about 25 min; results in `data/benchmarks/js_sweep/`) |
| `experiments/align_commentary_v7c.py` | A version of `msocr/align.py` with the stem/pen/density commentary rule (2026-11-24); set aside because it changed 7 of 155 pages the review had passed. Kept for its `_hand` (row-own ink threshold that counts red ink, smooth stem length) |

`../backups/code/` keeps earlier versions of the pipeline's code (the project is not under version control).

## experiments/lm — reading with a language model (2026-11-25)

Whether the letter patterns of proof-read Sanskrit text (the spacer's Pramāṇa sentences, the lexica texts, the
manuscripts' own transcriptions) mend the recogniser's recurring mistakes: the mistakes mined by character and akṣara,
character n-gram and LSTM language models, CTC prefix beam search with the LM over the saved probabilities, measured on
every held-out line of ft6 and pool1. Findings, numbers and what to do next in `experiments/lm/README.md`; result
files in `experiments/lm/results/`.
