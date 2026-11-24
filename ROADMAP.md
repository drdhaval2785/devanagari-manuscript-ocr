# Sanskrit manuscript OCR — tool survey and implementation roadmap

Date: 2026-11-13

## 1. Summary

- **No free, off-the-shelf model reads these manuscripts well enough to use unchecked.** The best published zero-shot result on handwritten Sanskrit pages is Gemini Flash at 35–51% page-level character error rate (CER); every other general model tested was worse.
- **Models fine-tuned on the target hand reach roughly 3–8% CER** in comparable projects. Fine-tuning needs line-aligned transcriptions.
- **You already have that training data.** `LDII_85885.xml` holds 286 pages and 5,421 lines, transcribed line by line and keyed to the image file names. That is about five times the ~1,000 lines usually quoted as the minimum for training a recogniser from scratch.
- **Recommended design:** one command-line tool with interchangeable engines.
  - Engine A, usable on day one: a vision-language model (Gemini, free API key) transcribes whole pages.
  - Engine B, the accuracy path: Kraken (free, local) segments lines and reads them with a recogniser trained on your own transcriptions.
  - A scoring command measures both against your ground truth, so every choice below is decided by numbers from your manuscripts and not by vendor claims.
- **First step is a bake-off, not a build.** Phase 1 runs seven candidate engines on 30 pages and costs nothing. Its result decides how much of Phase 2 onwards is needed.

## 2. What is in the corpus

| Item | Finding |
|---|---|
| Volume | 102 PDFs (13,894 pages) and 7,287 JPGs. 7,280 PDF pages also exist as JPGs, so there are about 13,900 unique pages. |
| Script | Jaina Nāgarī: pṛṣṭhamātrā vowel signs, no word spacing, red-ink daṇḍas and headings, folio numbers and running titles in the margins. |
| Layout | Pothi (wide) folios, 11–17 lines. Some are tripāṭha: root text in large letters in the middle, commentary above and below (e.g. `LDII_3746`). |
| Condition | Varies from good to moth-eaten with broken edges and bleed-through. |
| Resolution | LDII JPGs are small: median 1559 × 674 px at 150 dpi, so a text line is only about 40 px tall. This is at the lower limit for handwriting recognition. The PDF-only scans (Koba, BORI, vFlat) have not been measured yet. |
| Damaged file | `RD_D_117_P_005665.pdf` reports no page count and may be truncated. |

Existing transcriptions, which serve as ground truth:

| File | Content | Use |
|---|---|---|
| `charitrasimha/transcriptions/LDII_85885.xml` | TEI, 286 `<pb>` pages, 5,421 `<lb>` lines; diplomatic (keeps `संबंध` spellings, marks red ink, marginal notes, deletions, `<orig>`/`<corr>`) | Training and test data for line recognition; test data for page transcription |
| `shrivallabhagani/transcriptions/LDII_3746.txt` | 59 page markers; spelling is normalised (`सम्बन्ध`) and lines are not manuscript lines | Page-level test of a second hand and of tripāṭha layout; not usable for line training as it stands |
| `saroddhara_AKGM13339.txt`, `saroddhaara_AKGM13354.txt` | Short samples (5–6 KB) | Small cross-hand spot checks |

## 3. Survey of tools

"Free" means usable for this corpus without paying. Accuracy figures come from the cited papers or vendors and were measured on other manuscripts; none is a measurement on yours.

### 3.1 Vision-language models through an API

| Tool | Evidence on handwritten Sanskrit | Cost and limits | Verdict |
|---|---|---|---|
| **Gemini Flash** (`gemini-3.8-flash`; Lite: `gemini-3.5-flash-lite`) | Best of four VLMs in the only direct study (arXiv 2608.18696): page CER 0.35 / 0.44 / 0.51 on moderate / dense / circular layouts. A Penn project found Gemini and ChatGPT hallucinate on Sanskrit commentary and substitute the standard printed text. | Free tier with an API key. Google does not publish free limits; third-party reports from October 2026 give 20 requests/day for Flash and 500/day for Flash-Lite. Paid: $0.75 in / $3.75 out per 1M tokens until 31 Dec 2026, then double; batch is half price. | **Include.** Best zero-shot candidate. |
| **Sarvam Vision 2.1** | Second in the same study (0.39 / 0.64 / 1.44, with runaway generation on the hardest layout). Sanskrit is a listed language; vendor's own benchmark puts it ahead of Gemini on Sanskrit (84.05 vs 81.07), mix of print and handwriting not stated. | API only. ₹100 free credit at signup = 200 pages; then ₹0.50/page. | **Include in bake-off.** |
| Claude Sonnet 5, GPT-5.6 | Page CER 0.64–0.97 and 0.74–0.79 in the same study. | Paid. | Skip. |
| Mistral OCR, Google Cloud Vision, Azure | Only vendor-benchmark numbers for Sanskrit (71, 71, 57), none on manuscripts. A 2018 comparison found Google's OCR strong on printed Sanskrit. | Cloud Vision has a monthly free quota (check current size). | Cloud Vision in bake-off as a baseline; skip the rest. |

### 3.2 Open models that run locally (no key, no cost)

| Tool | Evidence | Fits a 16 GB M2 Pro? | Verdict |
|---|---|---|---|
| **Qwen3-VL-8B** | Best open model in a 2026 Devanagari benchmark (printed Hindi), behind only Gemini and Claude. No manuscript test. | Yes, 4-bit via Ollama or MLX; about a minute per page reported on an M4. | **Include in bake-off.** |
| PaddleOCR-VL 1.6 | Tops general document benchmarks; weak on handwriting without fine-tuning (Arabic test). | Yes (0.9B parameters). | Bake-off, low expectations. |
| olmOCR, DeepSeek-OCR | olmOCR collapses on real Devanagari scans; DeepSeek-OCR collapses at low resolution. | — | Skip. |
| Chandra OCR 2, Surya OCR 2 | About 60 on Sarvam's Sanskrit benchmark. | Chandra is 9B, tight. | Skip. |
| Tesseract (`san`, `script/Devanagari`) | Built for print; 8.8% CER on printed Sanskrit, reported to fail on degraded manuscripts. | Yes. | Bake-off as the floor. |

### 3.3 Trainable handwriting-recognition (HTR) toolkits

| Tool | Evidence | Cost | Verdict |
|---|---|---|---|
| **Kraken** (+ optional **eScriptorium** web interface) | Standard open HTR stack: trainable baseline segmentation and line recognition. No working public Devanagari handwriting model exists. The Penn project trained its own and went from 78% accuracy with ~200 lines to over 97% after a further manuscript. | Free, local. Training on Apple Silicon GPU is unconfirmed; CPU works, a free Colab/Kaggle GPU is the fallback. | **Engine B.** |
| TrOCR fine-tuning | Old Nepali manuscripts (Devanagari): CER 9.6% → 4.9%; pre-training on printed Nagari was essential. | Free; needs a GPU to train. | Alternative recogniser if Kraken plateaus. |
| FLAME pipeline (CRAFT + GNN line grouping + CNN-BiLSTM-CTC) | Built for Sanskrit manuscripts with complex layouts. Fine-tuning on just 3 pages cut page CER by up to 37.5% and beat Gemini on all three test manuscripts. Code and data are public. | Free; trained on a laptop GPU. | Alternative if Kraken's segmentation fails on tripāṭha pages. |
| UniLipi (IIIT Hyderabad, ICDAR 2026) | One model for 13 Indic manuscript scripts, including Devanagari (8.4% CER) and a class called "Jaini". Reads line crops only. | Release of weights not confirmed. | **Watch.** Check the project page before Phase 2; if weights are out, test it. |
| Transkribus | Public Devanagari models are for print only. | 50 free credits/month; training and bulk use are paid; data sits on their servers. | Skip, apart from a bake-off look. |

### 3.4 Supporting resources

| Resource | Use |
|---|---|
| ByT5-Sanskrit and the IIT Bombay post-OCR dataset (218k sentences) | Optional post-correction. Reported to bring printed-Sanskrit OCR to about 2.7% CER. Risk for this project: it pulls rare lexicon words and scribal spellings towards common forms, so it must stay a separate, switchable layer. |
| AnciDev (3,000 lines, ancient Devanagari), VedOCR (57 pages, Vedic) | Extra Devanagari manuscript lines for pre-training the recogniser. Download links and licences not yet checked. |
| Gyan Bharatam "Gyan-Setu" challenge | Government programme funding OCR for manuscripts; no released model yet. |

### 3.5 What the survey means for the design

1. Zero-shot VLM output is a first draft with one character in three wrong on hard pages. It also fails in a way that matters for an edition: it writes fluent Sanskrit that is not on the page.
2. A recogniser trained on the hand makes fewer and more honest errors, and gives line positions, which your TEI needs.
3. Published learning curves show a few corrected pages per manuscript move accuracy substantially. The tool therefore needs a correct-and-retrain loop, not only a one-shot OCR command.

## 4. Tool design

A Python command-line tool, `msocr`, in this folder.

```
msocr ingest  <pdf|folder>        # extract page images, build a manifest
msocr ocr     <ms-id> --engine gemini|kraken|sarvam|qwen|tesseract
msocr eval    <ms-id> --engine X  # CER/WER against ground truth
msocr export  <ms-id> --format txt|tei|pagexml
msocr train   <ms-id>             # fine-tune the Kraken recogniser on corrected lines
```

- **Ingest** extracts the embedded scan images from PDFs without re-rendering, and uses the existing JPG folder where one exists.
- **Engines** share one interface: page image in, lines of text out, with line polygons where the engine provides them.
- **Output** is one file per page, in the same page and line numbering as your TEI (`<pb n="0002_2"/>`, `<lb n="1"/>`), so OCR output drops into your current editing workflow.
- **Runs are resumable**: a manifest records which pages each engine has done, so daily API quotas and interruptions lose nothing.
- **Prompts for VLM engines** ask for a diplomatic transcription: line for line, no sandhi or spelling correction, anusvāra as written, a fixed mark for illegible characters.
- **Environment**: the system Python is 3.14, which Kraken and PyTorch may not support. Use `uv` with Python 3.11 or 3.12.

## 5. Roadmap

### Phase 0 — Setup and ground truth (1–2 days)
- Project skeleton, `uv` environment, `msocr ingest`.
- Convert `LDII_85885.xml` to plain per-page, per-line text. Rules: take `<orig>` over `<corr>`, keep `<surplus>`, drop `<del>`, keep marginal `<note>` text separately.
- Choose the test set: 20 pages of `LDII_85885` held out from all training, plus 10 pages of `LDII_3746`.
- Write `msocr eval` (CER and WER after Unicode normalisation; a second score that ignores daṇḍas and spaces).
- **Done when:** scoring a transcription against itself gives 0%, and against a deliberately damaged copy gives the expected figure.

### Phase 1 — Bake-off (2–3 days, no cost)
- Run on the 30 test pages: Gemini Flash, Gemini Flash-Lite, Sarvam Vision, Qwen3-VL-8B (local), PaddleOCR-VL (local), Google Cloud Vision, Tesseract.
- For the two best, try three variants: whole page, page cut into horizontal strips, and 2× upscaled input.
- Read ten pages of the best output by eye for invented text and skipped lines, which CER alone hides.
- **Decision:** below about 10% CER, the VLM is the main engine and Phase 2 becomes optional. Above that (the likely outcome), go on to Phase 2 and keep the VLM as a first-draft engine.

### Phase 2 — Train a recogniser on LDII_85885 (about 1 week)
- Segment all 286 pages with Kraken's default baseline model.
- Pair segmented lines with TEI lines automatically where the line counts agree; fix or drop the rest. If fewer than ~70% of pages pair cleanly, correct segmentation in eScriptorium or switch to the FLAME line grouping.
- Train the recogniser on ~90% of pages; test on the held-out 20.
- **Done when:** CER on held-out `LDII_85885` pages is at or below 5%. If it stalls above 8%, try TrOCR and check whether resolution is the limit by testing on a higher-resolution scan of the same folios.

### Phase 3 — Carry over to other hands (1–2 weeks, then ongoing)
- Run the model on `LDII_3746` and one Koba and one BORI manuscript; measure the drop in accuracy.
- Build the loop: OCR 5 pages → correct them → `msocr train` → OCR the next pages. Record CER after each round to learn how many corrected pages a new manuscript needs.
- Train the segmenter on tripāṭha pages so root text and commentary come out as separate regions.
- **Done when:** a new manuscript reaches under 8% CER after correcting 10 pages or fewer.

### Phase 4 — Finish the tool (1 week)
- TEI and PAGE-XML export, batch runs over whole folders, per-line confidence so reviewers see doubtful lines first.
- A correction view: line image above editable text. eScriptorium already provides this; build a custom one only if it proves awkward.
- User guide.

### Phase 5 — Optional improvements
- Run both engines and flag lines where they disagree; agreement is a cheap confidence signal.
- ByT5-Sanskrit post-correction as a separate output layer.
- Use the parallel witnesses: several manuscripts carry the same text, so aligning their OCR output exposes errors in each.

## 6. Risks and open decisions

| Issue | Detail | Proposed handling |
|---|---|---|
| **Sending images to Google's free tier** | Free-tier content is used to improve Google's products. The TEI header says "Not for distribution", and library terms may restrict sharing scans. | Your decision before Phase 1. Alternatives: paid tier (content not used), or local engines only. |
| Free-tier quota | At 500 pages/day the corpus takes about 28 days on Flash-Lite; at 20/day Flash is only usable for tests. | Check your real limits in AI Studio. Rough estimate for the full corpus on paid Flash: $150–400, half that in batch mode, before the January price rise. |
| Low resolution of LDII scans | ~40 px per line may cap accuracy regardless of model. | Phase 2 tests this directly; ask LDII for higher-resolution images if it proves the limit. |
| Ground truth is one scribe | A model trained only on `LDII_85885` may read other hands poorly. | Phase 3 measures it; AnciDev lines may help as extra training data. |
| Kraken training speed on Mac | GPU support on Apple Silicon unconfirmed. | Free Colab or Kaggle GPU. |

## 7. Sources

- [Impact of Iterative Fine-Tuning on Transcription Accuracy in Complex Historical Sanskrit Manuscripts (arXiv 2608.18696)](https://arxiv.org/html/2608.18696) — code: <https://github.com/flame-cai/gnn-synthetic-layout-historical/>
- [A Case Study of Handwritten Text Recognition from Early Modern Sanskrit Manuscripts (WSC 2025)](https://aclanthology.org/2025.wsc-csdh.4/) — not read in full (PDF too large to fetch)
- [UniLipi: A Unified Multi-Script OCR for Historical Indic Manuscripts (arXiv 2608.28195)](https://arxiv.org/abs/2608.28195) — project page: <https://ihdia.iiit.ac.in/unilipi/>
- [AnciDev (BHASHA 2025)](https://aclanthology.org/2025.bhasha-1.8/) — abstract only
- [VedOCR dataset](https://www.kaggle.com/datasets/suyashkumarbhagat/vedocr-vedic-sanskrit-manuscript-ocr-dataset)
- [Digitizing Nepal's Written Heritage: HTR pipeline for Old Nepali manuscripts (arXiv 2512.17111)](https://arxiv.org/html/2512.17111)
- [Penn SIMS fellows: eScriptorium on Devanāgarī manuscripts](https://uniqueatpenn.wordpress.com/2026/09/11/2025-2026-sims-fellows-applying-escriptoriums-handwritten-text-recognition-techniques-to-premodern-italian-and-devanagari-manuscripts/)
- [Can OCR-VLMs Read Devanagari? (arXiv 2606.29213)](https://arxiv.org/pdf/2606.29213)
- [Sarvam Vision 2.1](https://www.sarvam.ai/blogs/sarvam-vision-2-1) and [pricing](https://www.sarvam.ai/api-pricing)
- [Gemini API rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) and [pricing](https://ai.google.dev/gemini-api/docs/pricing); third-party free-tier figures: [imisofts](https://imisofts.com/blog/gemini-api-free-tier-limits/)
- [Kraken](https://kraken.re/main/ketos.html), [HTR-United catalog](https://htr-united.github.io/catalog.html)
- [Transkribus Devanagari mixed M1A](https://www.transkribus.org/models/devanagari-mixed-m1a)
- [A Benchmark and Dataset for Post-OCR text correction in Sanskrit (EMNLP Findings 2022)](https://aclanthology.org/2022.findings-emnlp.466)
- [Gyan Bharatam Gyan-Setu](https://www.gbm-moc.in/gyansetu)
- [SanskritOCR vs. Google OCR (2018)](https://sanskrit-coders.github.io/posts/2018-06-05-ocr-comparison)
