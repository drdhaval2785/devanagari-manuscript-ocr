# Errors at the margins (2026-11-26)

The question: pagediff shows many errors at the left and right ends of lines. Does the U-Net segmenter (seg_torch, the
default since 2026-11-24) cut lines short at the margins where Kraken's did not, or is it something else?

Material: the 40 corrected pages of the pool's eight manuscripts (LDII_1806, 4278, 45364, 6487, 80136, 82320, 82668,
9851; 650 lines). Every line was read from the rows found on the U-Net's lines and from those found on Kraken's (same
row finder, same cutter), by pool1 and by ft6, greedily. pool1 never trained on LDII_4278, 6487, 80136 or 82320, nor on
the last page of the other four ("unseen" below); ft6 never trained on any of them.

## How a row's ends are set

The segmenter does not decide where a row starts and stops. The row finder (`msocr/align.py`, `find_rows` and
`_merge`) gives every row of a page the full width of the text block: the block is taken from the median ends of the
segmenter's long lines, then pushed out column by column to where the page's ink ends (`_to_the_ink`, since 0.11.0 of
2026-11-24), stopping 2 px short of a rule drawn down the page ("a rule in a line image could be read as a daṇḍa").
All 40 pages are ruled at both margins.

## Findings

1. **The ends are where the errors are.** On unseen pages (pool1, greedy) the first akṣara of a line is wrong in 24 %
   of lines and the last in 42 %, the akṣaras between them 12 %. The first and last akṣara are 4 % of the akṣaras and
   11 % of the edits. ft6 shows the same shape (31 % / 48 % / 18 %).
2. **It is not the segmenter.** The blocks found from the U-Net's lines and from Kraken's end at the same place (40
   pages: median difference 0, at most 0.11 of a line pitch), and the end errors are the same or a little worse from
   Kraken's rows: last akṣara wrong 41.8 % (U-Net) vs 44.5 % (Kraken), first 23.8 % vs 25.4 %. Line by line, the right
   end is wrong only from the U-Net's rows in 14 lines and only from Kraken's in 28 (ft6: 36 vs 62).
3. **Some "errors" are the truth's: it inherited an old cut.** The drafts were made on 2026-11-16, before the block was
   widened to the ink, from rows that stopped short of the margin on some pages; the corrections kept the drafts'
   line ends. Today's rows reach the ink and the model reads the akṣaras the truth lacks. Clear on two pages:
   - 004278_00017_2, left end, 10 of 11 lines: प्रम|दवनं (truth वनं), रलयोरैक्यात् (truth योरैक्यात्), सौधपंक्तिः,
     क्रीडम्, पूर्ववत् ...
   - 082320_00067_1, right end, 11 of 15 lines: अज्ञाननाम अ|विद्या (truth stops at नाम), परगुणोत्कर्षनाम, पातकं पु ...
   `results/inherited.txt` lists every candidate (36 ends; the two pages are 21 of them, the rest mostly misreads).
4. **Letters at a ruled margin are clipped.** Scribes write up to the rule and squeeze the last akṣaras against it or
   across it; the cut stops 2 px before it. Cutting a quarter pitch further (across the rule) on unseen hands: first
   akṣara wrong 20.5 → 13.7 %, last 38.3 → 28.4 % (LDII_6487 left 34 → 14 %, LDII_4278 right 22 → 7 %), the middle
   unchanged. But on the hands pool1 trained on (cut short of the rule) the rule is then read as ा, ी or ।
   (पवर्ग → पवर्गी, तस्य → तस्या): LDII_82668 right 3 → 12 %, LDII_9851 6 → 11 %. Painting the rule over with paper
   colour is worse than not (it takes the crossing strokes too); a strip of blank paper added at the ends (no wider
   cut) is worse at the left and no better at the right, so the recogniser is not weak at an image's edge as such.
   0.15 pitch instead of 0.25: no better.
5. What remains is the ends being harder anyway: squeezed letters, worn leaf edges, context on one side only.

## What follows

- The two pages' truths (finding 3) need their missing akṣaras; until then pagediff counts correct readings there as
  errors, and a recogniser trained on them (recut pairs today's wide rows with the short truth) would learn to drop the
  first or last akṣara.
- A recogniser trained on rows cut across the rule (a quarter pitch past the block's end) would learn that a rule at
  the end of a line is not a letter, and keep the letters that touch it: the wide cut already helps unseen hands
  without training. It changes predict, Manuscribe's cutter and the training cache together (one pipeline.json
  value), and needs a training run to measure.

## Files

- `edges.py`: each line read from both segmentations; where its errors fall; what the cut does at each end.
- `summarize.py`: the tables of `results/edges_pool1.txt` and `results/edges_ft6.txt`.
- `inherited.py`: ends where today's reading runs further than the truth and the draft stopped where the truth stops.
- `variants.py`, `summarize_variants.py`: the cut as now, padded with paper, a quarter (or `--frac`) pitch wider, wider
  with the rule erased (`results/variants_025.txt`, `results/variants_015.txt`).
- The raw records (`results/*.jsonl`) are not kept in git: `edges.py` and `variants.py` remake them in a few minutes.
