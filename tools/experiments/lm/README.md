# Reading with a language model: do the corpus's letter patterns mend the recogniser's mistakes?

An experiment of 2026-11-25 on the question: the recogniser's misreadings recur (सृ for मृ, ह for द, a dropped virama);
would the co-occurrence of letters in proof-read Sanskrit text tell the right reading from the wrong one, and by how
much? Everything here is measured on lines the models never trained on, with the models as they are (ft6, pool1): the
recogniser is not retrained, only its reading is decoded differently.

## What was done

1. **The models' readings, with their probabilities kept** (`dump_logits.py`): every held-out line of ft6 and pool1
   (LDII_85885's 19 test pages and its 120 validation lines; the corrected pages of LDII_1806, 19634, 45364, 82668,
   9851 cut from the U-Net lines), the model's log-probabilities per image column saved.
2. **The recurring mistakes** (`mine_errors.py`): every edit between the greedy reading and the truth, by character,
   by akṣara (with the truth's neighbours), by kind.
3. **Three character language models** (`build_text.py`, `charlm.py`): 7-gram models with interpolated absolute
   discounting over Devanagari letters (daṇḍas, digits and marks are one boundary symbol), from
   - *Pramāṇa*: the spacer's 1.69 million proof-read sentences (61 M letters, general Sanskrit literature, no lexica),
   - *lexica*: the 115 IAST texts in `~/sanskrit-lexica-for-annotation` (15 M letters: Abhidhānacintāmaṇi with its
     commentaries, Amarakośa commentaries, Anekārthasaṃgraha...),
   - *own*: the manuscripts' own training lines (0.7 M letters: LDII_85885's training split and the corrected pages
     the model trained on; for ft6 only LDII_85885 and LDII_19634),
   and a small LSTM (`lstmlm.py`, 0.9 M parameters, trained on Pramāṇa + lexica on the Mac GPU).
4. **Decoding with the LM** (`decode.py`): CTC prefix beam search (beam 16) over the saved probabilities, each
   hypothesis scored by `log P(recogniser) + α · log P(LM) + β · length`; α and β tuned on LDII_85885's validation
   lines only, every other set untouched. `lm_prefers.py` asks the LM directly, without decoding, which of two readings
   of a line it prefers; `compare_decoded.py` lists what decoding mended and what it broke.

Rerun (paths are the session's scratch folder in these notes; any folder will do):

    .venv/bin/python tools/experiments/lm/dump_logits.py --model data/pool/models/pool1/best.pt --out /tmp/lm/pool1
    .venv/bin/python tools/experiments/lm/mine_errors.py /tmp/lm/pool1.jsonl
    .venv/bin/python tools/experiments/lm/build_text.py pramana ../sanskrit_spacer/data/interim/sentences.jsonl /tmp/lm/pramana.txt
    .venv/bin/python tools/experiments/lm/build_text.py lexica ~/sanskrit-lexica-for-annotation /tmp/lm/lexica.txt
    .venv/bin/python tools/experiments/lm/charlm.py build 7 /tmp/lm/lm_lexica7.npz /tmp/lm/lexica.txt
    .venv/bin/python tools/experiments/lm/charlm.py build 7 /tmp/lm/lm_own7.npz /tmp/lm/ms_text_pool.txt --min-count 1
    .venv/bin/python tools/experiments/lm/decode.py /tmp/lm/pool1 --lm /tmp/lm/lm_lexica7.npz:0.7,/tmp/lm/lm_own7.npz:0.3 --alpha 0.4 --beta 1.0 --out /tmp/lm/decoded.jsonl
    .venv/bin/python tools/experiments/lm/compare_decoded.py /tmp/lm/decoded.jsonl

(`ms_text_pool.txt` is the training lines' text: `lines_seg_torch/train/*.gt.txt` of LDII_85885 and every corrected
page but the last of the other five, one line each, as the session built it.) The result files are in `results/`.

## The recurring mistakes

ft6 on its 848 held-out lines (108,173 characters, CER 5.28 %, GER 8.84 %) makes 5,707 character edits:

| kind of edit | share | kind of edit | share |
|---|---:|---|---:|
| consonant for consonant | 22.0 % | vowel sign for vowel sign | 6.3 % |
| consonant missed | 12.8 % | virama missed | 6.2 % |
| other substitution (daṇḍa for daṇḍa, digit for digit, letter for sign) | 9.7 % | vowel sign vs other | 4.8 % |
| consonant invented | 8.9 % | vowel sign invented | 3.7 % |
| vowel sign missed | 7.8 % | anusvāra missed | 3.5 % |
| virama invented | 6.4 % | daṇḍa invented or missed | 3.0 % |

The commonest single edits (truth → read): ∅→् 6.4 %, ्→∅ 6.2 %, ं→∅ 3.5 %, ा→∅ 2.2 %, ∅→र 2.2 %, र→∅ 1.9 %, त→∅
1.9 %, ∅→त 1.8 %, ∅→ा 1.8 %, े→∅ 1.6 %, **द→ह 1.6 %**, ि→∅ 1.5 %, ∅→। 1.4 %, **ि→े 1.4 %**, ।→∅ 1.4 %, ः→∅ 1.3 %,
।→॥ 1.0 %, **प→य 0.9 %**, **ो→ा 0.8 %**, **ी→ा 0.7 %**, ौ→ो, व→च, त→न, े→ो, ध→व, च→व, ब→व, न→त (0.4–0.6 % each).
pool1 has the same ranking with fewer of each (2,168 edits on 564 lines, CER 2.81 %).

By akṣara the picture is a long tail: ft6's 4,640 akṣara edits fall into 2,827 distinct patterns, 2,247 of them seen
once; the 10 commonest patterns cover 9 % of the edits, the 200 commonest 34 % (pool1: 39 %). The commonest are
daṇḍa missed or invented, । for ॥, नां→ना (the Gujarati anusvāra of LDII_1806), ऽ dropped, digits dropped or read as
letters (२→र), then दी→ही, णि→ण, स्तु→सु, ते→त, ए→प, म→स, द→ह, श→त्रा, दि→हि, द्वि→द्वे. Rules for the commonest
patterns would therefore touch a tenth of the mistakes; only something that judges every reading in its context can
reach the rest.

## Does the corpus tell the right reading?

Asked directly (`lm_prefers.py`, ft6's 3,595 akṣara substitutions, the rest of the line kept as written): the Pramāṇa
7-gram prefers the truth to the misreading in 71 % of cases (67 % by more than a factor e), the lexica 7-gram in 74 %
(72 %). For letter confusions it is higher still: दी→ही, दि→हि, द्वि→द्वे, श→त्रा 100 %, द→ह 91 %, २→र 94 %, णि→ण 93 %, म→स
73 %; and 0 % for what no Sanskrit text can know: । for ॥, २ for ३, and 5 % for नां→ना, which is right Gujarati and
wrong Sanskrit. The odds are large: अमृत is rated 400 (lexica) to 1,800 (Pramāṇa) times more probable than असृत,
इत्यादीनि 57,000 to 200,000 times more than इत्याहीनि.

How well each corpus knows these texts, as perplexity per letter of the held-out truth (lower is better; each line
scored from nothing known, as a line is read):

| held-out set | Pramāṇa 7-gram | lexica 7-gram | lexica + own (0.7/0.3) | LSTM (Pramāṇa + lexica) |
|---|---:|---:|---:|---:|
| LDII_9851 | 36.9 | 14.5 | 8.7 | 15.7 |
| LDII_45364 | 26.9 | 13.5 | 8.6 | 14.7 |
| LDII_19634 | 28.0 | 13.3 | 6.7 | 14.2 |
| LDII_1806 (Old Gujarati) | 39.6 | 40.9 | 12.3 | 18.9 |
| LDII_85885 test | 20.9 | 11.6 | 6.9 | 11.2 |
| LDII_85885 val | 14.9 | 11.2 | 6.6 | 9.7 |
| LDII_82668 | 20.0 | 13.0 | 6.9 | 10.7 |

The 15 MB of lexica texts beat the 61 MB of general literature by far (these manuscripts are all commentaries on the
Abhidhānacintāmaṇi, and the Pramāṇa release holds no lexicon at all), and 0.7 MB of the manuscripts' own text, mixed
in, halves the perplexity again. LDII_1806's bālāvabodha is Old Gujarati: no Sanskrit corpus knows it, only its own
four training pages do.

## Decoding with the LM

CER / GER in % on every held-out line; α, β tuned on LDII_85885 val only.

| decoding | pool1 (564 lines) | ft6 (848 lines) |
|---|---:|---:|
| greedy (as now) | 2.81 / 4.52 | 5.28 / 8.84 |
| beam search, no LM | 2.79 / 4.46 | 5.20 / 8.66 |
| Pramāṇa 7-gram (α 0.2, β 1.5) | 2.58 / 3.97 | 4.87 / 7.68 |
| lexica 7-gram (α 0.2, β 1.0) | 2.47 / 3.74 | 4.59 / 7.24 |
| Pramāṇa + lexica (α 0.3, β 1.5) | 2.46 / 3.75 | 4.53 / 7.14 |
| **lexica + own 0.7/0.3 (α 0.4, β 1.0)** | **2.37 / 3.55** | **4.28 / 6.74** |
| the same, each line given the previous line's reading as context | 2.35 / 3.52 | 4.25 / 6.68 |
| the same with daṇḍas and digits as LM symbols (`--punct`, α 0.4, β 1.5) | 2.37 / 3.53 | 4.33 / 6.80 |

By manuscript, with the adapted mixture (lexica + own), greedy → decoded:

| set | pool1 CER | pool1 GER | ft6 CER | ft6 GER |
|---|---:|---:|---:|---:|
| LDII_9851 | 5.54 → 4.21 | 8.64 → 6.33 | 7.21 → 4.87 | 12.58 → 7.81 |
| LDII_45364 | 3.41 → 2.96 | 4.67 → 3.54 | 5.72 → 4.88 | 8.57 → 6.53 |
| LDII_19634 | 4.31 → 3.82 | 5.20 → 4.30 | 4.60 → 3.82 | 5.91 → 4.48 |
| LDII_1806 | 5.08 → 3.91 | 7.50 → 5.48 | 13.47 → 12.37 | 21.10 → 19.34 |
| LDII_82668 | 2.92 → 2.23 | 5.74 → 3.92 | 11.43 → 8.41 | 18.83 → 12.81 |
| LDII_85885 test | 2.50 → 2.13 | 4.08 → 3.29 | 2.24 → 1.94 | 3.64 → 2.95 |
| LDII_85885 val (tuned on) | 3.30 → 2.77 | 4.94 → 3.83 | 3.08 → 2.64 | 4.64 → 3.59 |

What changes: letter edits fall by 18 % (pool1) and 23 % (ft6), consonant-for-consonant edits by a third; daṇḍa and
digit edits (a fifth of ft6's edits) hardly move, by design (the LM sees them as a boundary), and giving the LM
daṇḍas and digits as symbols does not help either (last row above). The LM also breaks some readings: on ft6 it
mends 2,193 akṣara mistakes and makes 1,090 new ones, chiefly where the scribe's spelling is not the printed one:
नां→ना (the Gujarati anusvāra, 42 cases), र्ण्ण→र्ण, dropped digits and avagrahas; the own-text component in the
mixture is what keeps most such habits (LDII_1806 with pool1, whose own pages are in the mixture, gains 23 %; with
ft6, whose mixture lacks them, 8 %).

Sensitivity to the LM weight (ft6, lexica alone): see `results/alpha_ft6.txt`.

## Statistics or a small neural model?

A 2-layer LSTM of 256 units (0.9 M parameters, `lstmlm.py`) was trained on Pramāṇa + lexica (78 M letters) for
20,000 steps of 128 × 256 letters on the Mac GPU, 28 minutes; its best checkpoint (step 18,500) has a perplexity of
6.8 on held-out corpus text. On the manuscripts' truth it is about level with the lexica 7-gram (last column of the
perplexity table above: better on LDII_85885, 82668 and the Gujarati 1806, worse on 9851, 45364, 19634), and so in
decoding (same α, β; CER / GER):

| decoding | pool1 | ft6 |
|---|---:|---:|
| lexica 7-gram | 2.47 / 3.74 | 4.59 / 7.24 |
| LSTM | 2.57 / 3.91 | 4.65 / 7.34 |
| LSTM + own 7-gram 0.7/0.3 | 2.48 / 3.75 | 4.51 / 7.08 |
| lexica 7-gram + own 7-gram 0.7/0.3 | **2.37 / 3.55** | **4.28 / 6.74** |

The LSTM generalises better to text it has not seen (LDII_1806: perplexity 18 against 41) but knows the lexica less
exactly than the counts do, and the counts win where it matters. It would need more in-domain text, longer training
and its own adaptation step to catch up, and it costs 5 times the decoding time (one LSTM step per hypothesis letter
against a table lookup) and a second trained model to ship and to keep in step. With the corpus we have, the n-gram
is the right tool; a neural LM becomes worth revisiting when the in-domain corpus is several times larger (the
lexica folder plus every transcription) or when a Gujarati corpus arrives that the n-gram cannot smooth over.

## What to do with this

1. **Decode with the LM in the pipeline.** A character 7-gram built from the lexica texts and the manuscripts' own
   transcriptions, mixed 0.7/0.3, with prefix beam search (beam 16, α 0.4, β 1.0) takes 16 % off pool1's CER and
   19 % off ft6's, 21–24 % off their GER, on lines never trained on, for about 0.1 s of CPU per line in Python. It
   mends what no rule could list: the long tail of one-off confusions. The place for it is `predict` (and pagediff,
   and the test and pool scores, so that every figure is the figure of what the reader gets). Beam search alone,
   without an LM, gains almost nothing (2.81 → 2.79).
2. **The corpus that matters is the one in the manuscripts' own domain.** The 61 MB Pramāṇa release is worth less
   than the 15 MB of lexica texts, and 0.7 MB of the manuscripts' own transcriptions, mixed in, is worth as much
   again: every page corrected in Manuscribe improves the LM as well as the recogniser, at no cost (counting is a
   minute; no training). So: build the LM from the lexica folder plus every corrected page and every TEI
   transcription; add Pramāṇa only as a third, lightly weighted component (it adds nothing measurable here).
3. **One LM per language.** A Sanskrit LM damages Old Gujarati (LDII_1806's bālāvabodha: नां→ना, छइ, इं): its gain
   there comes only from its own four pages. The bālāvabodha manuscripts need an Old Gujarati text corpus of their
   own (the corrected bālāvabodha pages are the start of one), and the mixture weights should follow the manuscript's
   declared language, which Manuscribe already records.
4. **Leave daṇḍas and digits to the recogniser.** They are a fifth of the remaining edits and the LM cannot judge
   them (giving it daṇḍas and digits as symbols did not help). They will fall with more corrected pages in training,
   or with a layout rule (verse numbers between double daṇḍas), not with text statistics.
5. **Statistics, not a neural LM, for now.** The character n-gram with absolute discounting, built in a minute by counting, is level with the trained
   LSTM in perplexity and ahead of it in decoding (pool1 2.47 against 2.57 CER; with the own text 2.37 against 2.48);
   it adapts to a new manuscript by counting its corrected pages, ships as a table, and needs nothing trained. See
   the section above.
6. **For the browser.** A 7-gram of the lexica texts has 5.1 M entries (34 MB deflated, 89 MB in memory as packed keys and
   float16 probabilities); a 6-gram pruned of n-grams seen fewer than three times has 2.1 M entries (12 MB deflated,
   31 MB in memory) and gives nearly the same result (pool1 2.40 / 3.58 against 2.37 / 3.55, ft6 4.34 / 6.82 against
   4.28 / 6.74); a 5-gram (1.4 M entries, 5 MB deflated) keeps two thirds of the gain (2.45 / 3.67, 4.45 / 6.98). The
   own-text 7-gram is 9 MB deflated. The tables are sorted integer keys with a binary search, so the JavaScript port
   is a small matter, and the prefix beam search is 150 lines; the recogniser's probabilities are already there in
   `recogread.js`. A pruned 6-gram of the lexica texts plus the own-text model, about 20 MB, is the pack to ship,
   downloaded once like the models.
7. **Honest tuning.** α and β were tuned on LDII_85885's validation lines only and transferred unchanged to six
   other sets; the gain is not sensitive to them (α 0.3–0.5 all within 0.1 CER point on ft6). Re-tune once, on the
   validation lines, whenever the recogniser or the LM changes.

What the LM cannot do is also clear from the numbers: the recogniser's own errors remain the larger part (ft6's
CER falls from 5.3 to 4.3, not to 2), and the model that has seen a hand (pool1) still beats the unseen-hand model
with an LM (ft6 + LM 4.28 against pool1 alone 2.81 on the same manuscripts' pages). More corrected pages per hand
remain the main road; the LM is a free fifth on top of it.
