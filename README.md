# devanagari-manuscript-ocr

OCR for handwritten Sanskrit manuscripts in Devanagari: https://github.com/drdhaval2785/devanagari-manuscript-ocr
(the folder is called `manuscript_ocr` on the developer's computer, and the paths in these notes say so). Plan and tool survey: [ROADMAP.md](ROADMAP.md).

Version 0.12.1 (24 November 2026; `msocr.__version__`), released with Manuscribe 0.12.1, whose recognition in the
browser it is the reference of. Manuscribe and its companions (this, sanskrit_spacer, manuscribe-models,
manuscribe-contributions) always carry the same version number: use the same version of each.

The code is under the GNU General Public License v3 ([LICENSE](LICENSE)), as Manuscribe's is; the trained weights it
makes are published separately under CC BY-SA 4.0.

## Current state

- A line recogniser trained on one manuscript (`LDII_85885`) from its TEI transcription: run3,
  1.70% character error rate on test pages it never saw. Kraken finds the lines on a page; our own
  PyTorch model reads them.
- `LDII_19634` read in full with a model fine-tuned on corrected pages of its hand (ft3, about 1%);
  its TEI file is being corrected page by page with `update`.
- Every page of the 19 linear manuscripts in the corpus is segmented and row-checked, ready to be
  read and fine-tuned on; the tripatha, panchapatha and special ones are not yet ([LAYOUTS.md](LAYOUTS.md)).
- Five clean pages each of eight more hands (LDII_9851, 45364, 82668, 1806, 4278, 82320, 6487, 80136)
  are read and waiting in their `to_correct/` folders; once corrected, `pool` trains on all hands at once.

## Setup

```
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install "kraken==7.2" lxml rapidfuzz
```

Segmentation always uses `data/kraken_models/blla_kraken7.1.1.mlmodel` (`KRAKEN_SEG_MODEL` in `msocr/config.py`): the
baseline model bundled with kraken 7.1.1, which segmented every manuscript so far. It is not part of this repository;
fetch it from kraken 7.1.1's package (Apache-2.0) and check it:

```
.venv/bin/pip download --no-deps "kraken==7.1.1" -d /tmp/k711
mkdir -p data/kraken_models && unzip -p /tmp/k711/kraken-7.1.1-py3-none-any.whl kraken/blla.mlmodel > data/kraken_models/blla_kraken7.1.1.mlmodel
shasum -a 256 data/kraken_models/blla_kraken7.1.1.mlmodel   # 77a638a83c9e535620827a09e410ed36391e9e8e8126d5796a0f15b978186056
```
 kraken 7.2 bundles a different
default model (trained on Latin, Arabic and Greek only), so the model is passed explicitly. With it, 7.2 gives the same
lines as 7.1.1 and cuts byte-identical line images (checked on 7 random pages and 156 lines, 2026-11-23), and runs
about 1.3× faster (15.7 s a page against 20.2 s on one CPU process) with a peak of about 7.7 GB per process. On the
16 GB M2 Pro, 16 random pages took 14.2 s a page with one job, 9.3 s with two and 10.0 s with four (swap peaking at
2.3, 4.8 and 10 GB; identical lines every time), so on a CPU `segment` runs two jobs. Since 2026-11-24 it runs on the
Mac GPU (MPS) where there is one, one job at a time (`KRAKEN_DEVICE` in `msocr/config.py`; `MSOCR_KRAKEN_DEVICE=cpu`
overrides): 4.4 s a page, no swap. The GPU's lines are within a pixel or so of the CPU's and give the same rows
(`tools/kraken_mps_test.py`), but are not byte-identical, so segmentations made before that date came from the CPU.

## Steps (run from this folder)

| Step | Command | What it does | Time |
|---|---|---|---|
| 1 | `.venv/bin/python -c "from msocr import gt; gt.extract('LDII_85885')"` | TEI → one text string per manuscript line | seconds |
| 2 | `.venv/bin/python -m msocr.segment LDII_85885 --kraken` | Kraken finds text lines on each page (align pairs Kraken's lines with the transcription) | ~40 min |
| 3 | `.venv/bin/python -m msocr.align LDII_85885` | Pairs found lines with transcribed lines; splits pages into train / validation / test | seconds |
| 4 | `.venv/bin/python -m msocr.recognizer train LDII_85885 --run run1` | Trains the recogniser, reporting after every pass | hours |
| 5 | `.venv/bin/python -m msocr.recognizer test LDII_85885 --run run1` | Final score on 20 pages never used in training | seconds |

## Reading a new manuscript and adapting the model to its hand

| Step | Command | What it does |
|---|---|---|
| A | `.venv/bin/python -m msocr.predict LDII_19634 --start 1 --pages 12` | Finds lines and reads them with the current model. Writes `to_correct/<page>.txt` (one text line per manuscript line), plus `review/<page>.html` and an untouched copy in `drafts/` |
| B | Edit `data/LDII_19634/to_correct/<page>.txt` against the page image, then move it to `data/LDII_19634/corrected/` | Gives the model correct text for this hand. 3–5 pages to start |
| C | `.venv/bin/python -m msocr.recognizer finetune LDII_19634 --run ft1` | Continues training on the corrected pages; the last corrected page is held back to score before and after |
| D | `.venv/bin/python -m msocr.predict LDII_19634 --start 13 --pages 20 --model data/LDII_19634/models/ft1/best.pt` | Reads further pages with the adapted model |

| E | `.venv/bin/python -m msocr.recognizer pool --run pool1 --base data/LDII_19634/models/ft3/best.pt` | Continues a model on every manuscript that has training text at once, each hand given the same share of every pass; the road to a model for hands it has never seen (below) |

Rules for correcting: type what the scribe wrote (scribe's spellings, anusvāra and daṇḍas as
written), keep one text line per manuscript line, and do not add or remove lines — a page whose
line count differs from the number of line images is skipped, and `finetune` says so. Spaces are
ignored. `--start` is the index of the first image in the folder (0 is the cover; folio N side S is
index 2N−2+S for this manuscript).
Every `LDII_*` (Lālabhāī Dalapatabhāī Institute) and `AKGM_*` (Kailāsasāgarasūri Jñānamandira, Kobā) folder
in the corpus is registered automatically (`msocr/config.py`); a manuscript with a transcription is listed there
by hand. A manuscript that came as one PDF has its page images extracted beside it, named like the others
(`<6-digit number>_<5-digit folio>_<side>.jpg`, the cover as folio 0): see `AKGM_0025383` in
`abhidhanachintamani_commentaries/charitrasimha/manuscripts/`.

## Training on all hands at once

A model fine-tuned on one hand after another leans towards the last one. `pool` instead gathers
every manuscript with training text: the transcribed one (its `lines/train` split) and every one
with pages in `corrected/`. Each pass draws the same number of lines from each manuscript
(`--per-pass` lines in all, 3000 by default, shared equally): a hand with 60 corrected lines has
them repeated a few times under different augmentation, one with 4,000 lines contributes a
different draw each pass, so no hand drowns the others. Whole pages are held out per manuscript
(the last `--holdout` corrected pages, the `val` split of a transcribed one) and scored after
every pass; the mean over the manuscripts picks `best.pt`. Everything goes to
`data/pool/models/<run>/` (`train.log`, `history.csv` with one CER column per manuscript, `best.pt`).

```
.venv/bin/python -m msocr.recognizer pool --run pool1 --base data/LDII_19634/models/ft3/best.pt
.venv/bin/python -m msocr.recognizer pool --run pool2 --base data/pool/models/pool1/best.pt --exclude LDII_4278
.venv/bin/python -m msocr.recognizer test LDII_85885 --model data/pool/models/pool1/best.pt
.venv/bin/python -m msocr.predict LDII_45364 --start 1 --pages 20 --model data/pool/models/pool1/best.pt
```

`--exclude` keeps a manuscript out of training and scores it as an unseen hand on all its
corrected pages: the number that says how generic the model has become. Name manuscripts after
`pool` to use only those. A pass of 3000 lines takes about 3 minutes; the run stops after 8 passes
without gain or `--passes` (40). A corrected page whose line count does not match its line images
is skipped and reported, as in `finetune`.

## Tripāṭha and pañcapāṭha manuscripts: blocks

A page of these layouts holds the main text in the middle and a commentary in blocks around it.
For a manuscript marked tripatha or panchapatha in `data/layouts.csv`, every step above works
block by block, and nothing changes for a linear manuscript. Plan, traps and status:
[ROADMAP_layouts.md](ROADMAP_layouts.md).

- `msocr/blocks.py` finds the blocks from Kraken's pieces and the page's ink: columns from blank
  vertical bands (side columns, the red rules), bands within the centre column from blank
  horizontal runs of its ink profile and from jumps of hand size, names from hand size and pitch
  (main text, top, bottom) and position (left, right). Each block is then handed to the row
  finder as a small page of its own. Blocks are named as Manuscribe names its boxes.
- A page's text file (`drafts/`, `to_correct/`, `corrected/`, `machine/`) holds its blocks after
  `;box{main}`, `;box{top}` ... markers (`msocr/pagetext.py`); a page with main text only has no
  marker, so linear manuscripts' files are as before. Line images are `<page>_NN.png` for the
  main text and `<page>_top_NN.png` and so on for the blocks, and `finetune` and `pool` train on
  all of them.
- The TEI takes the form Manuscribe writes and reads for these layouts: a
  `<div type="base-text">` with every page's main text, then a `<div type="commentary">` with
  every page's commentary blocks after `<cb n="top"/>`, `<cb n="right"/>`, `<cb n="left"/>`,
  `<cb n="bottom"/>`, each page with its `<pb/>` in both streams, lines numbered from 1 in each
  block, the usual line counts in the header, and `level="#ocr"` on machine-read pages.
  Manuscribe's import puts the blocks into its boxes and sets the folio's layout from them;
  `update` reads such a file back and rewrites a machine page in both streams.
- `show` frames each block in Manuscribe's colour for that box with its rows numbered inside;
  `show --lines` stacks the strips block by block. `rowcheck` lists every block's count per page
  and counts a page as evenly spaced only when every block is.

## Preparing a manuscript: segmentation and its check

Step A segments the pages it reads on demand, so none of this is required; it is how a whole
manuscript is prepared in advance and how the segmentation is checked before any page is read.

| Command | What it does |
|---|---|
| `.venv/bin/python -m msocr.segment LDII_45364 LDII_9851 ...` | Finds the text lines on every page with our U-Net (`data/<ms>/seg_torch/`, the default since 2026-11-24; see below), 0.13 s a page, under a minute for a 300-page manuscript; `--kraken` runs Kraken instead (`data/<ms>/seg/`, four pages per process on the Mac GPU, about 4 s a page, some 20 min for 300 pages), which `align` and the U-Net's own training need. Pages already done are skipped |
| `.venv/bin/python -m msocr.layouts` | Prints the table of manuscripts and rewrites [LAYOUTS.md](LAYOUTS.md) from `data/layouts.csv` (the layout of each manuscript, judged by eye, and a note) with live counts: pages, segmented pages, rows found |
| `.venv/bin/python -m msocr.rowcheck LDII_45364 ...` | Runs the row finder over every segmented page, before any page is read, and writes `data/<ms>/rowcheck.txt`: rows found per page, the manuscript's usual count, and the pages with no rows or with unevenly spaced ones (a gap where a line was missed, or a note squeezed in). Its summary is the "rows found" column of LAYOUTS.md |
| `.venv/bin/python -m msocr.show LDII_45364` | Draws the segmentation on every page image (`data/<ms>/show/<page>.jpg`) and binds the pictures into `data/<ms>/show/LDII_45364.pdf`, one page per sheet in page order. About 0.3 s and 0.4 MB per page. `--no-pdf` for the pictures only |
| `.venv/bin/python -m msocr.show LDII_45364 2_1 7_2` | The same for the named pages only (page ids as in the TEI, or image file names) |
| `.venv/bin/python -m msocr.show LDII_45364 --warned` | Only the pages the row finder has a warning for, as `LDII_45364_warned.pdf`: the ones to look at first |
| `.venv/bin/python -m msocr.show LDII_45364 2_1 --lines` | The page cut into its row strips, one under another (`show/<page>_lines.jpg`), with the page's draft or corrected text under each strip once it has been read: the quickest way to see whether a band cuts off the marks above the headline |

What a page picture shows: Kraken's pieces as thin blue lines; the rows the recogniser will read
as numbered bands, red and green in turn so their overlap can be seen; a dashed band is a row the
row finder filled in because Kraken missed it, and "gap" in the margin marks a blank gap under a
row that was left unfilled. The caption carries the row count, the row finder's note and, in red,
its warning. The pictures only show what `predict` would do with the page; nothing is changed.

The row finder (`find_rows` in `msocr/align.py`) takes the lines Kraken found that span more than
half the page as the text rows, glues fragments at the same height into them, and discards marginal
and interlinear notes. Kraken breaks some hands into word-sized pieces; the row finder then takes
the line pitch from the page's own ink, the text block from where the pieces cluster, builds rows
from pieces that together cover half the block, and fills a missed row in between neighbours only
where the page has ink. Lines cut in two by the puṣpikā (the blank hole left in the middle of a leaf for
the binding string: a diamond, a circle or a band, with no letter lost) are read as one: when the text
block found holds only one side of the hole, the wide pieces on the other side that are level with a
wide piece inside are taken as the other halves of those lines, and the block is widened to them (the
page note says "lines cut by the puṣpikā"). Letters written inside the hole itself belong to the line
and are read in its middle, as the line is meant to be read. The block found from Kraken's pieces often stops short of
the writing (their ends are taken as a median, and a hand broken into word-sized pieces leaves the first and last
letters out); the block is then pushed out, across the band of the text lines, as far as the page's own ink goes:
it stops at a rule drawn down the page (dark through the gaps between the lines too, where writing never is), at a
blank gap wider than half a line pitch (which keeps marginal notes out), or a sixth of the width beyond Kraken's
lines, and a row never reaches across a rule (the page note says "block widened to the ink"). The widening is the
last step: which pieces make a row, and where a missed row is filled in, are decided on the block Kraken's lines give.

A commentary written above or below the text in a smaller hand (common in the margins of a leaf) is not a line of
the text. Each of the outer four rows at the top and the bottom is measured against the page's own rows (`_hand`:
ink density over the baseline, length of the letters' vertical stems, pen width, all at the row's own darkness so red
ink counts) and scored by a small classifier (a logistic regression, `COMMENTARY_W` in `msocr/align.py`), fitted to the
review verdicts with `tools/commentary_fit.py` (2026-11-24: 1,447 edge rows of 13 manuscripts; held out by
manuscript, it caught 56 of 62 commentary rows and took 4 of 1,385 rows of text for commentary). From each edge, the
rows up to the innermost clear commentary row (score 0.9 or more) among the outer three go, provided every row from
the edge to it scores 0.8 or more; two or more rows of thin ink packed closer than three quarters of a pitch go too (a
block of notes), and a row scoring 0.8 or more packed that close against a commentary row just dropped goes with it.
At most four rows go at an edge ("rows in a smaller hand above/below the text left out"), and no row is added at an
edge where a commentary was left out. The thresholds lean towards keeping a doubtful row: a commentary line read as
text costs a deletion, a line of text left out costs the text. Before a change here is kept, `tools/verdict_check.py`
must show no page the review passed changing its rows. When Kraken found only a few long lines, the median of
their gaps can mix one and two pitches; if the page's ink gives a pitch that the gaps are whole multiples of, that
pitch is used. A block made from Kraken's pieces that is under a quarter of the page wide (a stamp, a number: Kraken
found no line of the text) is warned about: "the text may have been missed".
A page that still comes out with no rows, or with far too few, needs looking
at before it is read; `rowcheck` lists them and `show --warned` draws them. Short lines under half
the block's width (the last line of a verse, a list) are the ones it still misses.

`.venv/bin/python -m msocr.compare LDII_19634` reports how much each corrected page differed from
the model's draft (CER per page, commonest mistakes): the honest score of the model that read
those pages, on pages it never trained on.

## Is Kraken's segmentation right? Check, repair, review

Kraken is wrong on some pages: a line missed, two lines merged, a note taken for a line. The OCR mostly survives this
(the row finder fills in a missed row from the ink), but anything that learns from Kraken's output must not see it.

| Command | What it does |
|---|---|
| `.venv/bin/python -m msocr.rowcheck <ms> ...` | The row finder over every segmented page: `data/<ms>/rowcheck.txt`, one line per page with its warning (Kraken missed a row, uneven spacing, a gap that may hide a short line, no rows) |
| `.venv/bin/python -m msocr.repair [<ms> ...]` | For every page whose only warnings are missed rows: a copy of Kraken's segmentation with each missed line added (its baseline from the row finder, its outline as tall as its neighbours'), in `data/<ms>/seg_repaired/`; kept only if the row finder, run again on it, finds the same rows with no warning. Kraken's `seg/` is never touched; `data/<ms>/repair.txt` lists every page and the outcome |
| `.venv/bin/python -m msocr.kreview [<ms> ...]` | Every page put in a group: flagged (a warning, not repaired), repaired, odd count (passes, but two or more rows off the usual: a chapter end, or a miss nobody noticed), clean. `data/kraken_review/summary.tsv`, `<ms>.tsv` page by page, and `<ms>_review.pdf`: the flagged, repaired and odd-count pages and a sample of clean ones, each labelled, the repaired ones drawn with their added lines |
| `MSOCR_SEG=seg_repaired .venv/bin/python -m msocr.show <ms> <page>` | One repaired page drawn |
| open `data/kraken_review/index.html`, then a manuscript's `<ms>_review.html` | Each drawn page with a form: correct, not a text page, or wrong (missed, extra, merged, split, short, band, block, order, repair, other; the rows, the block, a note). Kept in the browser until "Save feedback" downloads `<ms>_feedback.tsv` |
| `.venv/bin/python -m msocr.kfeedback <ms>_feedback.tsv ...` | Takes the saved feedback in: kept in `data/kraken_review/feedback/<ms>.tsv` (a later verdict on a page replaces an earlier one), "not a text page" added to `data/<ms>/not_text.txt`, and the wrong pages listed by kind of error, to mend the code from |

**Pages that are not text pages** (the cover, a blank side) are neither checked, read, exported nor learnt from: the cover
(folio 0) always, and every page listed in `data/<ms>/not_text.txt`. `rowcheck` reports them as "not a text page" and
leaves them out of its counts; `predict` skips them; `export` writes no text for them.

**The reference's numerics are those of Python 3.11 and NumPy 2.4** (the `.venv` of 2026-11-24). The JavaScript port in Manuscribe reproduces them exactly, down to NumPy's pairwise float32 sums, its float32 percentile with a single q, its type promotion of Python numbers (NEP 50), `statistics.mean`/`pstdev` and Python's round-half-to-even. Python 3.12 makes the built-in `sum()` of floats compensated, and a newer NumPy may change its sums: before upgrading either, regenerate the fixtures and run `tools/js_sweep/sweep.py`.

**The export contract and the stage fixtures** (2026-11-24). `.venv-onnx/bin/python -m msocr.export_onnx` builds a model pack, `data/model_pack/pack-<date>-p<pipeline version>-s<spacer version>/`: the three networks in ONNX (segmenter, recogniser, spacer; inputs "x", outputs "y"), `pipeline.json` and `spacer.json`, the recogniser's characters, and `manifest.json` (each model's source, sha256, inputs and outputs and what they mean, its largest difference from PyTorch, the versions, the licence, CC BY-SA 4.0). The spacer is exported with its length free (1 to 256 letters), by the newer exporter (`dynamo=True`, which needs `onnxscript` in `.venv-onnx`; the older one fixes the traced length in the attention): a window then costs what its letters cost, where padded to 256 a page's spacing took 5 s on one browser thread, most of it padding (now 0.3–0.5 s); unpadded and padded outputs differ by 3·10⁻⁶ and gave the same spaced text on all 459 fixture texts. `spacer_reference.onnx_network` pads only for a spacer exported at a fixed length (the first export). The pack is the unit Manuscribe fetches. `.venv-onnx/bin/python -m msocr.fixtures --pack <pack> [--pages ms:stem ...] --synthetic` runs the reference pipeline with the pack's own ONNX networks and writes what every stage makes of a page, to `data/fixtures/<pack>/<page>/`: the decoded page, grey, the segmenter's resized input, the maps, the lines, the rows, every line image, its resized input, its logits, the text, the spaced text, and `fixture.json` with the tolerances (grey exact; resized images within one grey level; maps and logits within 1e-3; lines and rows the same number within 1 px; text identical). The JavaScript port is held to these stage by stage. The private pages (a linear, a tripāṭha and a pañcapāṭha page of the developer's manuscripts) stay in `data/`; the synthetic page, printed Devanagari without word spaces rendered by Chrome (`tools/fixtures/render_synthetic.js`, `tools/fixtures/synthetic_page_1.png`), is the tool's own and may go anywhere.

**The reference pipeline** (`msocr/reference.py`, 2026-11-24) is one folio side from image to text as pure functions, one per stage (grey → segmenter input → network → maps → lines → rows and blocks → line images → recogniser input → network → text), reading and writing no files, the networks passed in as functions (PyTorch or ONNX Runtime). It is what the JavaScript port in Manuscribe copies stage by stage and is checked against; on LDII_85885's 19 held-out pages it gives the same rows and the same text as `predict`. The parameters both must share are in `msocr/pipeline.json` (read through `msocr/params.py`: input height, padding, distance scale, threshold, smoothing, simplification, the band, the commentary classifier, the cutter, the recogniser's height and stride, and the resampling both must use); a value is changed there, never in either codebase, and the version bumped. The vectoriser labels connected components itself (`tseg.components`, runs joined row by row; the same regions as scikit-image's on 23 pages, and faster).

**Line images are cut by our own cutter** (`msocr/linecut.py`, default since 2026-11-24; `MSOCR_CUTTER=kraken` brings back Kraken's `extract_polygons`): the band of a row (0.9 pitch above its baseline, 0.3 below) straightened along the baseline, bilinear, white outside the page; simple enough to be the same in JavaScript. On LDII_85885's 19 held-out pages it reads better than Kraken's cutter with the same recogniser: 2.96 % of characters wrong on Kraken's lines (3.38 % before), 2.60 % on the U-Net's (3.28 %). So the reading path needs no Kraken: U-Net lines, our cutter, our recogniser. Recognisers are still trained on Kraken's crops; rebuild the line cache with this cutter at the next training.

**Blocks are cut along the white space between them** (tripāṭha and pañcapāṭha, `blocks._seam_bands`, 2026-11-24). Where a block ends in a half line, the next block begins beside it, a little lower, so no straight blank band runs across; the white space between the blocks still does, stepping round the end of the half line like a sigmoid. From every gap between two lines a path is found across the centre column through the clearest white space within a line and a fifth of that gap (each pixel's vertical distance from the nearest ink; stepping up or down a column costs extra), and the paths whose white space is clearly wider than that of most paths (the gaps between lines) are the cuts; each piece goes to the block on its side. When no cut stands out, the larger hand of the main text decides, as before. On the 50 pages counted in the review it got 30 right in every block (5 before); the pages still wrong were judged exceptional and left as they are.

**A page with very few lines** (a last page of two lines, a colophon) shows no line pitch in its ink; the row finder
then takes the pitch from the heights of Kraken's pieces instead (the note says so), so such lines are still read.

**The recognition site** (2026-11-24). `.venv/bin/python -m msocr.site [--pack <pack>] [--site ../manuscribe-models]` writes the folder Manuscribe fetches its recognition pack from, to be pushed by hand as its own repository on GitHub Pages (GitHub's release assets cannot be fetched by a browser; Pages, Zenodo and Hugging Face can): `index.json` (the packs and runtimes), `packs/<pack>/` (the pack's files, each checked against its manifest first, and the manifest completed with every file's size, the runtime it needs, the median height of the segmenter's training images, at which Manuscribe renders a side, and a card per model: the manuscripts the segmenter learned from, the recogniser's runs back to the first, the spacer's texts), `MODEL_CARD.md`, `runtime/onnxruntime-web-<version>/` (ONNX Runtime Web's two bundles and their WebAssembly, unchanged from npm, MIT, from Manuscribe's `tests/node_modules`), `contribute.json` (where contributed lines go: the inbox repository, the maintainer's address, the consent wordings taken, and the public key that seals them; written only where the key exists), `README.md`, `LICENSE.md`, `.nojekyll`. The index and each manifest name the version of manuscript_ocr that wrote them (`software`). About 73 MB; a browser fetches 45 MB of it (the WebGPU runtime) or 30 MB (WebAssembly only). Packs already on the site are kept and listed.

**Contributed lines** (2026-11-24; by GitHub since 0.11.0). Manuscribe's "Contribute corrected lines" makes a package of a manuscript's corrected lines (their images, recut from the PDF; their corrected and proposed text; the editor's consent record), only once its editor has said yes for that manuscript, and **seals** the lines to the maintainer's public key (AES-256-GCM, the key wrapped by RSA-OAEP with SHA-256; `msocr/seal.py`); the editor attaches the file to an issue in the inbox repository, `manuscribe-contributions` (beside this folder, pushed by hand), whose workflow checks the package, keeps it in a draft release, registers it and publishes a data release; a withdrawal comes as an issue too and deletes it. The key pair: `.venv/bin/python -m msocr.seal keygen`, once (the private key at `~/.config/manuscribe/contribution-key.pem`, outside every repository: keep a copy somewhere safe; made 24 November 2026, id fed7ed31e9afc8c5); `msocr.site` publishes the public key in `contribute.json`. `.venv/bin/python -m msocr.contributions sync` brings `data/contributions/` (`register.tsv`, each package's lines, opened) up to the inbox's register through `gh`: new packages fetched, opened and checked whole, withdrawals made here too; `add <zip>` registers a package by hand (sealed or not) only if its consent says yes, confirms the right to share the images, is not withdrawn, and was given on an approved wording (`APPROVED` in `msocr/contributions.py`: `1`, the wording of 24 November 2026; `--accept-wording` takes another for a trial, and the row says so); `withdraw <id>` deletes its lines and keeps its row; `list`. **Before training, every line is checked** (`msocr/linequality.py`; `contributions quality`, run by `sync` and `add` on their own; results in `data/contributions/<id>/quality.json`): the image (a PNG, 16-160 px high, not blank or solid), the text (NFC, spaces taken out, Devanagari only, at least 2 letters), the two against each other (8-24 px of line a character at the recogniser's height, room for CTC), copies of a line already taken, and a judge's reading: the recogniser of the pack that read the line (else the newest pack's) reads the image, and its reading may differ from the text by at most 60 % (measured with ft4 on four manuscripts' corrected pages: the true text 0-23 %, another line's 76 % or more), and must be what Manuscribe proposed then, within 10 % (the image is the one Manuscribe cut; a package from the browser reads exactly so). Lines that fail are left out; a package where more than 25 % fail, whose median reading difference exceeds 35 %, where more than 20 % do not read as proposed, or where more than 20 % share one text, is quarantined whole until `contributions accept <id>` (then lines that failed only on the reading are taken; the others stay out). Tested on four packages (good, texts shifted a line, garbage, images swapped): only the good one passed. `contributions.allowed()` gives the lines training may use, and **`recognizer pool` trains on them** (unless `--no-contributions`): each contributed manuscript a source of its own, drawn at most 3 times a line per pass, its last folio sides (10 %) held out for scoring once it has 30 lines on two sides; a pass that reads the developer's own manuscripts more than 0.5 CER points worse than the model began cannot become `best.pt` (OWN_GUARD), so that contributed lines cannot spoil what the model reads; `best.pt` lists the packages it learned from, and `msocr.site` puts them on the recogniser's card with their data release. **Folio sides for the segmenter, and languages kept apart** (since consent wording 2, 2026-11-24). Manuscribe also contributes the image of each folio side whose blocks its editor drew on that side itself (Calibrate; a calibration drawn on one side and applied to the whole manuscript only roughly fits the others, so those sides are not sent, and `sidequality` and the inbox refuse a side not marked `"drawn": "here"`) and whose every block holds as many lines of text as it was given, with the blocks and their counts (`sides/<folio>.jpg` and `.json`, at the segmenter's page height). `msocr/sidequality.py` checks the image and the blocks, has the U-Net find the lines and the block finder's row counter count them block by block, and only where every block's count is borne out writes the side's target (`sides/<folio>.seg.json`, Kraken's format); `tseg train` then learns from those (`contributions.allowed_sides()`, one in 25 to validation; `--no-contributions` leaves them out), and its `best.pt` and card name them. Tested: the block finder's own rows, counted in their rectangles, agree with the reviewed counts on 60 of 60 linear pages and 76 of 81 tripāṭha and pañcapāṭha blocks; a miss only keeps a side out. What the counts confirm is that the U-Net got a side right, so a side teaches it a new hand or scan as it already reads it where right, not its mistakes (that would need lines corrected by hand). Every package states the manuscript's language and script, which its editor gives with the consent (a tag as `sa-Deva`); the inbox registers it, and here `MODELS_SCRIPT` (`sa-Deva`) decides: a package of another language or script is kept, not checked and never used by these models, whose `allowed()` and `allowed_sides()` take their own language and script only. The pack declares its language and script (`script` in the manifest), and Manuscribe asks before reading a manuscript marked otherwise. The spacer learns from no contributions (the consent does not name it). Needs the `cryptography` package (in `.venv` since 24 November 2026).

## Training on the U-Net's lines (2026-11-24)

The recogniser so far learned from lines found by Kraken and cut by Kraken (`lines/`), but reads lines found by the
U-Net and cut by our own cutter (`predict`, and Manuscribe pixel for pixel). `align <ms> --seg=seg_torch` pairs a
transcription with the U-Net's lines into a set of its own (`pagexml_seg_torch/`, each page's exact rows beside its PAGE
XML; `train_seg_torch.txt` ..., the pages kept in the sets Kraken's set gave them), and `recognizer train|test <ms>
--seg seg_torch` cut them into `lines_seg_torch/` as `predict` cuts them (checked identical to the pixel). On
LDII_85885 282 of 286 pages pair (Kraken's lines: 266), 4,509 training lines against 4,218, the same 25 validation and
19 test pages; run4 trains from scratch on them. `recognizer recut <ms> --seg seg_torch` cuts a manuscript's corrected
pages again from the U-Net's lines into `lines_seg_torch/corrected/`, taking a page only where every block has as many
rows as its corrected text has lines and the judge (ft4) reads each line as its text (a page with its text shifted a
line is refused); `corrected/` and `lines/pred/` are not touched. All 71 corrected pages of LDII_19634, 9851, 45364
and 82668 qualify. `finetune` and `pool` take `--seg seg_torch` to learn from these; a manuscript without lines of that
cut is left out rather than mixed in.

## Our own segmenter (PyTorch), the default since 2026-11-24

**The default segmentation** (`config.SEG`, `MSOCR_SEG` to choose: `seg` for Kraken's). `segment`, `predict`, `export`
and `show` make and read the U-Net's lines (`data/<ms>/seg_torch/`), some 30-70 times faster than Kraken's. A page whose
line images were cut already keeps the segmentation they were cut from (`segment.page_seg`; recorded in
`lines/pred/segmentation.tsv`, and every page cut before the switch was cut from Kraken's), so that its `to_correct`
text and review page stay paired with its line images; `MSOCR_SEG` set by hand overrules that for pages not yet
corrected. **A corrected page is never read again** by `predict` (since 2026-11-24): its text is paired with its line
images for training. Kraken stays for `align` (training lines of a transcribed manuscript) and as the U-Net's teacher
(`tseg train` learns from Kraken's reviewed segmentation). On the 50 tripāṭha and pañcapāṭha pages whose blocks were
reviewed (`tools/block_eval.py`), the U-Net's lines give 27 pages split right in every block, Kraken's 30 (main block
right on 35 and 38); on linear pages the rows are the same.


**tseg1 (trained 2026-11-23/24, 24 passes, 2,929 pages; best validation F1 0.974 against Kraken's lines).** On the 210
held-out test pages: baseline agreement with Kraken 0.975 (linear 0.970, tripāṭha 0.979, pañcapāṭha 0.990), 0.13 s a
page; the row finder builds the same rows from its lines as from Kraken's on all 124 linear pages, on 20 of 65
tripāṭha and 13 of 21 pañcapāṭha pages (where the blocks are split differently; which split is right is for the
review to say). Read by LDII_85885's recogniser on its 19 held-out pages (`tools/segocr_test.py`), its lines give
3.28 % of characters wrong, Kraken's 3.38 %. That needs the vectoriser to simplify each baseline to the few points it
needs (`tseg.simplify`, 1.5 px): with a point every 8 px, each half a pixel off, the straightened line images had a
kink at every point and 4.90 % of characters came out wrong.

Kraken's segmenter runs on the CPU here (its Mac GPU path exhausts memory): about 45 s a page per job, and each
job can take some 10 GB, too much to run beside a training job. `msocr/tseg.py` is a small U-Net (1.9 M parameters)
that runs on the Mac's GPU and writes the same JSON as Kraken, to `data/<ms>/seg_torch/`, so the row finder, the
block finder, `predict`, `show` and `rowcheck` work on it unchanged. It predicts each line's baseline and, along it,
how far that line's outline reaches up and down (the hand size the block finder needs). It learns from Kraken's own
segmentation of the pages segmented so far. Kept out for the final test, so every hand and layout is tested: one
page in 20 of every manuscript, every fifth page of AKGM_0025383, and for LDII_85885 exactly the recogniser's
19 test pages; `compare` scores only those pages (`compare all`: every manuscript, with a summary in
`data/tseg/compare_summary.tsv`).
Kraken is wrong on some pages, and the network should neither learn nor be scored on those: only pages that `rowcheck`
passes clean (text rows found, no warning: no row filled in where Kraken missed one, no uneven spacing, no gap that may
hide a short line) are used, in training, validation and test alike, and a page whose only fault was a missed line in
its repaired form (`msocr.repair`, above). A page marked wrong in the review (`msocr.kfeedback`) is left out too,
repaired or not, until a later review marks it otherwise. Stale or missing reports are made afresh first; the pages
left out are listed with their reasons in `data/tseg/flagged_pages.tsv`.

| Command | What it does |
|---|---|
| `.venv/bin/python -m msocr.tseg train --run tseg1 --reviewed` | Trains on the GPU, and only with `--reviewed`: Kraken's segmentation must be checked first, since the network learns whatever Kraken did; `data/tseg/models/tseg1/` gets `train.log`, `history.csv` (agreement with Kraken on the validation pages after every pass), `best.pt` |
| `.venv/bin/python -m msocr.tseg segment AKGM_0025383 [--pages 0001_2 ...]` | Segments every page image (or those named) into `data/<ms>/seg_torch/`; pages done are skipped (`--redo` to redo) |
| `.venv/bin/python -m msocr.tseg compare AKGM_0025383 LDII_85885` | On the test pages (never trained on; `--all` for every page Kraken has segmented): how far the baselines agree, whether the row finder makes the same rows of both, time a page; `data/tseg/compare_<ms>.txt` |
| `MSOCR_SEG=seg_torch .venv/bin/python -m msocr.predict ...` (or `show`, `rowcheck`, `export --model`) | Any tool on our segmentation instead of Kraken's: reads `seg_torch/`, segments missing pages with `tseg`, and writes `show_seg_torch/` and `rowcheck_seg_torch.txt` beside Kraken's. Without the variable nothing changes |

`predict` with `MSOCR_SEG=seg_torch` replaces a page's line images, drafts and `to_correct` text like any re-read
(never a `corrected/` page): try it first on a manuscript or pages without corrections.

## Gathering a whole manuscript

| Command | What it does |
|---|---|
| `.venv/bin/python -m msocr.predict LDII_19634 --start 1 --pages 400 --model data/LDII_19634/models/ft3/best.pt` | Reads every page (pages already segmented are not segmented again) |
| `.venv/bin/python -m msocr.export LDII_19634` | Writes `data/LDII_19634/LDII_19634.txt` (page markers `;p{0001_2}`, one line per manuscript line; corrected pages where they exist, otherwise the model's reading) and `data/LDII_19634/LDII_19634.xml` (the same in the TEI layout of `LDII_85885.xml`) |
| `.venv/bin/python -m msocr.export LDII_19634 --from data/LDII_19634/LDII_19634.txt` | Rebuilds the TEI file from a composite text file corrected by hand |
| `.venv/bin/python -m msocr.export LDII_19634 --model data/LDII_19634/models/ft3/best.pt --out <file>` | Reads every page afresh with that model and writes the reading alone, nothing corrected, as a TEI file (default `data/LDII_19634/LDII_19634.machine.xml`): same `<pb>` and `<lb>` layout, the model named in the header. About 3 minutes for 360 pages; touches nothing else |

To follow how training changes the reading, make that file for each model in turn and keep it
under version control in a repository of its own (the manuscripts are not for distribution, so a
private one), one commit per model: the difference between two commits is exactly what the
training changed, and the difference from the corrected TEI is what remains to fix. The CER it
prints against the corrected pages is only a check that the right model was used, since the model
may have trained on those pages; the honest scores are the held-out pages of `pool` and `test`.

## Correcting in the TEI file itself

Once `export` has produced `data/<ms>/<ms>.xml`, that file can be the only thing you edit, in
any editor or transcription service; nothing in the XML has to record what has been checked.
One round, after correcting up to page 0020_2:

```
.venv/bin/python -m msocr.update LDII_19634 --run ft4 --done 0020_2
```

1. Reads the XML and decides which pages are corrected: every page up to `--done` (also
   written `20_2`, or `20` for both sides of folio 20), every page whose `<pb>` carries the
   editor's `level="..."` with a value other than `#ocr` (proofread), every page already in
   `corrected/`, and any page whose text differs from what the model wrote for it. `--done` is
   remembered in `data/<ms>/corrected_upto.txt`, so give it only when it moves on.
2. Copies them to `corrected/` and reports how much each newly corrected page had to be changed.
3. Fine-tunes a new model on them (two pages held out) — skip with `--no-train --model <path>`.
4. Re-reads every other page with that model and rewrites the XML: corrected pages stay byte for
   byte as you left them, machine pages are replaced, nothing else is touched. The previous file
   is kept in `backups/`.

`--dry-run` does steps 1–2 only; run it first after a new export. If many pages beyond `--done`
differ from the model's text, the round stops before training and shows the first difference:
either the editor changed the text itself, or `--done` should be further on. Keep one `<lb/>` per
manuscript line on corrected pages; a page whose line count no longer matches its line images is
reported and not trained on. `--marks` additionally writes `resp="#ocr"` on the `<pb>` of every
machine-read page, for an editor that keeps unknown attributes; deleting the mark then also
counts a page as checked.

## Where things are

- `msocr/` — code; each module's docstring explains its rules and assumptions
- `LAYOUTS.md`, `data/layouts.csv` — which manuscripts are linear, tripatha, panchapatha or special, and how far each is prepared
- `data/<manuscript>/` — everything generated for one manuscript:
  - `seg/` Kraken's lines per page, `rowcheck.txt` the row finder's report (per block on a tripāṭha or pañcapāṭha), `show/` the segmentation drawn on the pages and its PDF
  - `gt_lines.jsonl`, `pagexml/`, `lines/`, `preview/`, `train.txt` `val.txt` `test.txt` — training data of a transcribed manuscript
  - `drafts/`, `to_correct/`, `review/`, `corrected/`, `machine/`, `lines/pred/` — the model's readings and their corrections
  - `<manuscript>.txt`, `<manuscript>.xml`, `backups/`, `corrected_upto.txt` — the gathered text and the TEI round; `<manuscript>.machine.xml` the whole manuscript as one model read it
  - `models/<run>/` (`train.log`, `history.csv`, `best.pt`)
- `data/pool/` — `models/<run>/` of the pooled training, `pages.txt` the pages chosen for correction, `review.html` links to their review pages
- `logs/` — console output of long runs

## Following a training run

- Live: `tail -f logs/train_run1.out`
- Table of passes: `data/LDII_85885/models/run1/history.csv`

CER (character error rate) is the number to watch: errors per 100 characters on validation
pages the model never trains on. The verdict column says whether it is still improving.
