# Tripāṭha and pañcapāṭha manuscripts: OCR by block, imported into Manuscribe

Goal: read the 7 tripāṭha and 23 pañcapāṭha manuscripts so that each page's blocks (main
text in the middle; commentary top, right, left, bottom as the page has them) come out
separately, each with its line count and its lines, in a file that Manuscribe imports into the
right boxes for proofreading. The proofread file comes back the same way, so the model can be
fine-tuned on these hands exactly as with the linear manuscripts.

## 1. What Manuscribe expects (read from `edit_manuscripts/manuscript_editor/app/js`)

Manuscribe's data model (`model.js`) has four layouts. A folio side is a set of **boxes**;
boxes belong to **streams**:

| layout | boxes | streams |
|---|---|---|
| linear | main | main |
| tripatha | top, main, bottom | main = [main]; commentary = [top, bottom] |
| pancapatha | top, left, main, right, bottom | main = [main]; commentary = [top, right, left, bottom] |
| interlinear | main, gloss | main; gloss |

Two doors bring machine text in.

**Door 1, TEI import** (`importxml.js`, "Bringing in an existing transcription"). Fully
automatic. The file is Manuscribe's own export format:

- one `<div type="base-text">` holding the main stream of every folio, then one
  `<div type="commentary">` holding the commentary stream of every folio. So **every folio
  side has a `<pb/>` in each stream**: `<pb n="0005_1" level="#ocr"/>` appears twice in the file;
- inside the commentary stream each block that has text starts with `<cb n="top"/>`,
  `<cb n="right"/>`, `<cb n="left"/>`, `<cb n="bottom"/>`, in that order (the order is asserted by
  Manuscribe's own test `tests/t-xml.js`; the older specs file says top, right, bottom, left,
  the code and its test say top, right, left, bottom). A block without text has no `<cb/>`. The
  main stream, having one box, has no `<cb/>`;
- lines are `<lb n="k" break="no"/>text`, numbered from 1 within each block;
- the header says the usual line counts: `<layout writtenLines="8"><p>… layout. Lines per
  folio side: main text 8, top 5, right 12, left 12, bottom 5.</p></layout>`, matched by the
  box labels of `Model.BOX_LABEL` in lower case;
- `level="#ocr"` on a `<pb/>` marks a folio side as machine-read, any other level as proofread
  (this is what `msocr.update` already uses on the linear manuscripts);
- the importer sets a folio's layout from the boxes it finds: left or right present →
  pañcapāṭha; otherwise a commentary stream or top/bottom → tripāṭha. A page's layout can
  differ from the project's.

**Door 2, "Import recognised text"** (`htr.js`, `app.js › importProposed`). PAGE-XML or ALTO
with regions. Lines come in as *proposed* (dotted, accepted one by one with Alt+Y, error rate
per source). But the dialog asks, per import, which folio side each page is (pre-selected in
sequence from the current folio, not from the image name) and which box each region goes to,
**by region index, the same for every page**. Usable for a few pages, not for 200 without a
small change to Manuscribe (see step 8).

## 2. What the pages look like (one page of each of four manuscripts, with Kraken's pieces)

- Kraken finds the lines of every block, including the tiny side-column hands (8–15 px), and
  its pieces stop at the red rules and the blank bands between blocks. Nothing crosses from a
  side column into the centre.
- Blocks are separated by **blank bands**: vertical ones (red rules or empty strips) between side
  columns and the centre, horizontal ones between the top commentary, the main text and the
  bottom commentary. That is the reliable signal.
- Hand size is **not** a reliable separator on its own: on LDII_82328 the main text is only a
  little larger than the commentary (median piece height 29 px against about 22 px). It serves
  to decide which band is the main text when there are only two, with position as the fallback.
- The main text often has a **string-hole gap** in the middle (LDII_17964, 10215) or stands in
  two half-width columns (LDII_3746): pieces of one line on either side of the gap. The row
  finder already joins pieces at one height into one row, so these read as one line each.
- Tripāṭha is not one shape: LDII_3746/46320/4639/5710/87222/87440 have commentary above and
  below; LDII_10215 has it in two side columns and nothing above or below. Manuscribe calls the
  latter pañcapāṭha (left and right boxes, top and bottom empty), and that is what we emit.
- Some pañcapāṭha pages lack one block (LDII_17964's sample page has no bottom commentary;
  LDII_5419 and 81899 are noted as "some below"). A missing block is simply not emitted.
- Marginal notes outside the blocks and glosses between the lines are left out: Manuscribe has
  no box for them (they are `<note>`s typed by the editor), and a line crop mixing two texts
  would only mislead the model.

## 3. Traps found while checking, and what they changed

1. **Two `<pb/>` per folio side.** `msocr/tei.py` cuts the file at every `<pb/>` into one chunk
   per page. With streams a page has a chunk in each stream, and the `</div><div
   type="commentary">` boundary sits inside a chunk, which makes that page "not plain" and
   `update` would never rewrite it. → tei.py gets a stream-aware reader: cut at `<div>` and
   `<pb>`, collect a page's lines as {box: [lines]} across both streams, rewrite each chunk in
   its stream. Linear files must read and rewrite exactly as today (regression test on
   LDII_19634.xml).
2. **`<cb/>` only where a stream has several boxes**, and only for blocks with text; line
   numbers restart per block; the header carries the usual counts. → the writer follows this
   exactly; a round trip through Manuscribe (import, export, compare bodies) is the acceptance
   test, done with Manuscribe's own test harness or by hand.
3. **Commentary order** is top, right, left, bottom in the code and its test. → emit in that
   order; nothing on our side depends on it, but the file must not be rejected or reordered.
4. **The PAGE-XML door is manual** per import. → TEI is the primary door; PAGE-XML is written
   as well (we build it per page anyway) for anyone who wants proposed lines, and step 8 offers
   the small Manuscribe change that would make it automatic.
5. **Hand size alone misclassifies** (LDII_82328). → blocks come from blank bands first, hand
   size and position decide names; per-manuscript expectations (layouts.csv) and a check
   report flag pages whose blocks differ from the manuscript's usual set.
6. **Per-box line images and corrected text.** `load_corrected` pairs `corrected/<page>.txt`
   with `lines/pred/<page>_NN.png` by index. → box-aware names: `<page>_NN.png` stays for the
   main box (nothing changes for linear manuscripts), `<page>_top_NN.png` etc. for the others;
   a page's text file carries `;box{top}` markers before each non-main block, like the
   composite's `;p{…}` page markers. `finetune` and `pool` then train on commentary lines too.
7. **The recogniser has only seen 30–45 px hands.** Commentary hands are 10–20 px and dense;
   crops are scaled to 48 px, so size itself is handled, but the first readings will be poor.
   → read the pilot pages, have a few proofread, fine-tune, measure; the pooled trainer gives
   these hands their share.
8. **`level` is per folio side**, not per box. A side proofread only in its main box cannot
   be expressed. → accepted; `update` treats a side as corrected or not, as now.
9. **Volume.** 30 manuscripts, about 3,900 pages; Kraken takes about 8 s a page, so some
   9 hours in the background. → started first, pilot manuscripts first, so nothing waits on it.
10. **Found in the first review of LDII_82328's pictures** (the user checked a dozen pages):
    the side columns were right everywhere, the cutting of the centre column was not. Three
    causes, all fixed: Kraken's line outlines hardly shrink for a small hand, so hand size
    is now measured on the page as the dense band of each line's ink profile; a change from
    wide to narrow spacing (main text to the commentary below) was not a cut, so the cut is
    now a symmetric change point in the row spacing with the boundary rows assigned by the
    spacing they fit and the hand size breaking ties; and a short last line of a commentary
    block seeded no row, so a piece of some width with ink at a height no row has is kept.
11. **Nothing must disturb the linear workflow** (LDII_19634's rounds, the 8 manuscripts being
    corrected, the pooled training). → every change is checked against the existing data before
    the pilot, and the box-aware code paths are only taken when a page has more than one box.

## 4. Steps

Each step ends with a check that can be shown. Training only when asked. Scope, as asked on
22 November 2026: the sample manuscripts only, that is the two pilots LDII_82328 (pañcapāṭha)
and LDII_3746 (tripāṭha) in full, with one page each of LDII_17964 and LDII_10215 as further
test pages; the other 26 manuscripts wait until asked for.

Status (23 November 2026): steps 1 to 5 done for the pilots; step 6 reached the point where
both pilots' TEI files are written for Manuscribe (`data/LDII_82328/LDII_82328.xml`,
`data/LDII_3746/LDII_3746.xml`) and await proofreading. Block check: LDII_82328 219 of 237
pages with evenly spaced rows in every block, all five blocks on 159; LDII_3746 221 of 231,
top/main/bottom on 190.

1. **Segment** the pilot manuscripts with Kraken (the others only when asked). Check: `seg/`
   count equals page count.
2. **Block finder** `msocr/blocks.py`: columns from the horizontal coverage of Kraken's pieces
   (blank vertical bands, red rules), bands within the centre column from the vertical coverage
   (blank horizontal bands), names from hand size and position, expectations from
   `layouts.csv`. Rows per block with the existing row finder run on the block as a small page.
   `msocr.show` draws the blocks in Manuscribe's own box colours. Check: the four sample pages
   and a sheet of pages from several manuscripts, by eye.
3. **Block check** `msocr.blockcheck` (or rowcheck extended): per manuscript, the usual set of
   blocks and lines per block, pages that deviate; the "rows found" column of LAYOUTS.md for
   these manuscripts. Check: the two pilot manuscripts' reports.
4. **Reading per block**: `predict.read_page` returns lines per box; line images and page text
   files box-aware (trap 6); review HTML per box. Check: linear pages unchanged byte for byte.
5. **TEI per stream and block**: `export.tei` writes the two streams with `<cb/>`, per-block
   numbering, header counts; `export --model` and the machine-only file work per box;
   `tei.py` reads such files back (trap 1); `update` rewrites machine pages in both streams.
   Check: round trip of our own file (write, read, write: identical); Manuscribe import of
   a pilot file, export, body compared.
6. **Pilot**: read every page of LDII_82328 (pañcapāṭha) and LDII_3746 (tripāṭha) with the
   best current model; TEI with `level="#ocr"`; you import into Manuscribe and proofread a few
   folio sides of each; `update` imports them, fine-tunes, re-reads, rewrites. Check:
   error rate before and after on held-out sides, per box.
7. **The rest, when asked**: the other manuscripts the same way, in the order of the check
   report (cleanest first), and all corrected pages into the pooled trainer.
8. **Manuscribe, optional**: in *Import recognised text*, pre-select the folio side from the
   image file name (`…_00005_1.jpg` → `0005_1`) and the box from the region's type when it is a
   box name. Two small changes in `app.js › importProposed`; then PAGE-XML import is automatic
   too. Also, if wanted, a Manuscribe export of the calibrated block rectangles, which the block
   finder could take as a prior for a manuscript whose pages it finds hard.

## 5. Not in scope now

The five "special" manuscripts (interlinear glosses, base text in a ruled box, the empty
LDII_16249). The interlinear layout exists in Manuscribe, but line crops of a base line with a
gloss line above it would mix two texts; it needs its own treatment.
