# Quick adaptation to a new hand from a few corrected pages: tried, and dropped

An experiment of 28 November 2026. **Outcome: dropped.** The gain was too small, and too uncertain on the first pages,
to be worth putting into Manuscribe. A new hand is learned by full training on the corrected lines users contribute
(`msocr.contributions`, then `recognizer pool`).

## The question

A Manuscribe user reads one or two pages with the shipped models and corrects them. Could the next pages come out
better at once, with an update that takes seconds in the browser (no server), instead of 1–3 hours of fine-tuning?

## What was tried

`run.py` imitates that setting on a manuscript the model had never seen. pool3b did not train on RTJ_5520; pool4 did,
for hours, on its 5,956 training lines, and serves as the yardstick. The pages are taken in order. Page k is read with
what pages 1..k-1 taught, and is then counted as corrected. Two cheap kinds of adaptation were tested, both of which the
browser could run:

- **A, page LM:** a character 6-gram built from the corrected lines so far, mixed into the language model's two parts
  (lexica, own) with weight w. The recogniser is not touched.
- **C, head:** only the recogniser's last layer (384 → 90 classes, 34,650 weights) is trained, with the CTC loss and
  Adam, on the corrected lines. Every column's 384 features are kept fixed, computed once as the page was read. The
  layer starts afresh from the shipped weights after every page, can be pulled back toward them by lam·|W − W0|², and
  is trained on all lines corrected so far.

The language model is the 20261127 build, which has no line of RTJ_5520. The 20261128 build has RTJ_5520's training
split and would flatter the baseline.

The settings were tuned on RTJ_5520's 38 validation pages (`tune`; results/tune_RTJ_5520_val*.json). The best were
lr 0.003, no pull-back, 100 steps and w 0.1. More steps, a larger rate, or any pull-back did worse. These settings were
then run once on the 20 test pages (`run`; results/run_RTJ_5520_test.json).

    .venv/bin/python tools/experiments/adapt/run.py tune --lm-weights 0.1 0.2 0.35 0.5
    .venv/bin/python tools/experiments/adapt/run.py tune --lrs 0.003 0.01 --lams 0 --at 50 100 200 300 --tag _ext
    .venv/bin/python tools/experiments/adapt/run.py run --split test --lr 0.003 --lam 0 --steps 100 --w 0.1

## What it gave

CER on test pages 2–20 (340 lines):

| reading | greedy | with the LM |
|---|---|---|
| pool3b as shipped | 6.47 | 5.08 |
| + page LM (A) | | 5.01 |
| + head adapted (C) | 5.64 | 4.27 |
| + both (A+C) | | 4.25 |
| pool4, fully trained on this manuscript | 2.41 | 2.08 |

- **The head adaptation pays off only once about 8 pages (about 150 lines) are corrected.** Over test pages 11–20 it
  read with about a quarter fewer errors.
- **With 1–4 corrected pages it read slightly worse,** which is exactly the case it was meant for: +0.57, +0.98, +0.30
  and +0.16 CER points on test pages 2–5. On the validation pages, learning from a single bad first page (26 % CER)
  made the next page much worse (7.1 → 12.0).
- **The page LM adds almost nothing** (0.07 points alone, 0.02 on top of the head).
- **Fully trained, pool4 is far better.** After 19 corrected pages, adaptation closed only about 28 % of the gap
  between pool3b and pool4.
- **It is fast enough:** the head took 1.1 s after one page and 7.6 s after 19 (Python, 2 threads), and the page LM
  0.03 s.

Untried, and the way to go if this is ever taken up again: a guard (keep the update only if it reads the
last-corrected page better), steps and pull-back that scale with the number of lines, imprinting (class means) as a
start, a second unseen manuscript, and training more of the network (which needs back-propagation in the browser).

The results hold only numbers and page names, never the manuscripts' text.
