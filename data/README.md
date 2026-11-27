# data/

Everything the pipeline reads and writes about manuscripts lives here. None of it is part of the tool's repository
(see `../.gitignore`): the manuscript data is kept and versioned in a repository of its own, and released Kraken
segmentations and trained weights in another. The manuscript images themselves stay outside this folder altogether
(each manuscript's image folder is registered in `msocr/config.py`) and are never copied here.

| Path | What it holds | Made by |
|---|---|---|
| `layouts.csv` | each manuscript's layout: linear, tripatha, panchapatha (and a note) | by hand |
| `kraken_models/blla_kraken7.1.1.mlmodel` | the Kraken baseline model segmentation always uses (`KRAKEN_SEG_MODEL`); see the README's Setup for how to fetch it | by hand |
| `<ms>/seg/` | Kraken's segmentation, one JSON per page | `msocr.segment` |
| `<ms>/seg_repaired/`, `<ms>/repair.txt` | Kraken's lines with the missed ones added, where checked | `msocr.repair` |
| `<ms>/seg_torch/` | our U-Net's segmentation | `msocr.tseg segment` |
| `<ms>/rowcheck.txt` | the row finder's report on every page: rows, blocks, warnings | `msocr.rowcheck` |
| `<ms>/not_text.txt` | pages that are not text pages (covers, blank sides) | `msocr.kfeedback`, or by hand |
| `<ms>/lines/`, `<ms>/drafts/`, `<ms>/to_correct/`, `<ms>/review/` | line images, machine readings, pages waiting for correction | `msocr.predict` |
| `<ms>/corrected/`, `<ms>/corrected_upto.txt` | corrected page transcriptions (never re-read) | by hand |
| `<ms>/<ms>.xml` | the manuscript in Manuscribe TEI | `msocr.export` (hand-edited ones are managed with `msocr.update`) |
| `<ms>/muktabodha.txt`, `<ms>/muktabodha_check.txt` | for a manuscript made from a Muktabodha e-text and its scan: the page table and the e-text's irregularities; every paired page read by the recogniser against the transcription | `msocr.muktabodha` |
| `<ms>/models/`, `pool/models/`, `tseg/models/` | trained recognisers and segmenters | `msocr.recognizer`, `msocr.tseg` |
| `<ms>/show/` | pictures of the rows found | `msocr.show` |
| `kraken_review/` | review pages and their pictures; `feedback/` the verdicts taken in | `msocr.kreview`, `msocr.kfeedback` |
| `benchmarks/`, `commentary/`, `rowcheck_history/` | outputs of the checks in `../tools/` | `tools/` |
