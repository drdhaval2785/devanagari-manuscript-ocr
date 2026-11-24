"""One round of the correct-in-the-XML workflow for a manuscript.

    python -m msocr.update LDII_19634 --run ft4 --done 0020_2

  1. reads data/LDII_19634/LDII_19634.xml and decides which pages are corrected. A page is
     corrected when any of these holds:
       - it is at or before the page given with --done ("I have corrected up to here");
         the value is remembered in data/<ms>/corrected_upto.txt, so it need only be
         given again when it moves on
       - it is already in corrected/ (once corrected, always corrected)
       - its text differs from what the model wrote for it (kept in machine/<page>.txt)
       - its <pb> carries the editor's level="..." with a value other than #ocr (proofread)
       - with --marks: its <pb> has lost the resp="#ocr" mark
     So the XML itself need not record what has been checked: an editor that drops
     unknown attributes or reformats the tags does no harm.
  2. copies the corrected pages to corrected/<page>.txt and reports how much each newly
     corrected page differed from the model's reading (the model's score on it)
  3. fine-tunes a fresh model on them, two pages held out (--holdout), unless --no-train
  4. re-reads every other page with the new model (--model to use a given one instead)
  5. rewrites the XML: corrected pages byte for byte as they were, machine pages replaced by
     the new reading, everything else untouched. The previous file is kept in backups/.

A corrected page whose number of lines no longer matches the line images is reported and
left out of training. A machine page that holds any markup besides <lb/> is left as it is.
    --dry-run   do step 1 and 2 only
    --marks     also write resp="#ocr" on the <pb> of every machine-read page (and the
                respStmt it points to), so a page can be ticked off by deleting its mark.
                Off by default; without it any old marks are removed.
"""
import argparse
import collections
import re
import shutil
import time
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from . import pagetext
from .config import MANUSCRIPTS, ms_dir
from .predict import DEFAULT_MODEL, predict
from .recognizer import finetune
from .tei import by_page, norm, page_boxes, read, with_resp_stmt, without_resp_stmt

PROGRESS = "corrected_upto.txt"


def stem_of(ms_id, pid):
    return MANUSCRIPTS[ms_id]["image_name"](pid).rsplit(".", 1)[0]


def machine_lines(d, stem):
    """What the model wrote for the page, all blocks in order, or None."""
    for sub in ("machine", "drafts"):
        p = d / sub / f"{stem}.txt"
        if p.exists():
            return pagetext.flat(pagetext.read(p))
    return None


def page_key(pid):
    """'0020_2' -> (20, 2), so pages can be ordered; None for a page id of another shape."""
    m = re.fullmatch(r"(\d+)_(\d+)", pid or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_done(value, pages):
    """'0020_2', '20_2' or '20' (= 0020_2) -> that page's id as the XML writes it."""
    m = re.fullmatch(r"\s*(\d+)(?:_(\d+))?\s*", str(value))
    if not m:
        raise SystemExit(f"--done: expected a page like 0020_2 or 20_2, got {value!r}")
    key = (int(m.group(1)), int(m.group(2) or 2))
    for p in pages:
        if page_key(p.pid) == key:
            return p.pid
    raise SystemExit(f"--done: there is no page {key[0]:04d}_{key[1]} in the XML")


def cutoff(d, pages, done):
    """-> (page up to which everything is corrected or None, where that came from)."""
    f = d / PROGRESS
    old = f.read_text(encoding="utf-8").strip() if f.exists() else None
    if done is None:
        return old, f"remembered in {PROGRESS}; pass --done to move it" if old else None
    pid = parse_done(done, pages)
    if old and page_key(old) and page_key(old) > page_key(pid):
        print(f"  ! --done {pid} is before the remembered {old}: pages between them stay corrected only "
              f"because they are in corrected/ already")
    f.write_text(pid + "\n", encoding="utf-8")
    return pid, f"from --done, now remembered in {PROGRESS}"


def import_corrected(ms_id, xml, done=None, dry_run=False):
    """Decide which pages of the XML are corrected; write them to corrected/. Returns (head, chunks, corrected set).
    A page is its <pb> chunk, or its two chunks in a two-stream file (one per stream)."""
    d = ms_dir(ms_id)
    (d / "corrected").mkdir(exist_ok=True)
    head, chunks = read(xml)
    groups = by_page(chunks)
    firsts = [cs[0] for cs in groups.values()]
    upto, how = cutoff(d, firsts, done)
    upto_key = page_key(upto) if upto else None
    uses_marks = any(c.marked for c in chunks)
    print(f"[update] {ms_id}: {len(groups)} pages in {xml.name}" + (f" ({len(chunks)} chunks in {len({c.stream for c in chunks})} streams)" if len(chunks) != len(groups) else ""))
    print(f"  corrected up to : {upto}  ({how})" if upto else
          "  corrected up to : not set  (pass --done <page> once pages have been checked)")

    # 1. decide
    decided, why, beyond, after, last_diff = [], collections.Counter(), [], 0, None
    for pid, cs in groups.items():
        boxes = page_boxes(cs)
        lines = pagetext.flat(boxes)
        if not pid or not lines:
            continue
        stem = stem_of(ms_id, pid)
        machine = machine_lines(d, stem)
        differs = machine is None or norm(lines) != norm(machine)
        if differs and machine is not None:
            last_diff = pid
        key = page_key(pid)
        if upto_key and key and key > upto_key:
            after += 1
        if upto_key and key and key <= upto_key:
            reason = f"up to {upto}"
        elif any(c.proofread for c in cs):
            reason = f"proofread (level {next(c.level for c in cs if c.proofread)})"
        elif (d / "corrected" / f"{stem}.txt").exists():
            reason = "already in corrected/"
        elif differs:
            reason = "text differs from the model's"
            if upto_key:
                beyond.append(pid)
        elif uses_marks and not any(c.marked for c in cs):
            reason = "mark removed"
        else:
            continue
        why[reason] += 1
        decided.append((pid, stem, boxes, lines, machine))
    corrected = {pid for pid, _, _, _, _ in decided}
    print(f"  corrected pages : {len(corrected)}  (" + ", ".join(f"{k}: {v}" for k, v in why.items()) + ")"
          if corrected else "  corrected pages : none")
    if beyond:
        shown = ", ".join(beyond[:12]) + (f" ... and {len(beyond) - 12} more" if len(beyond) > 12 else "")
        print(f"  beyond {upto} but changed, so kept as corrected too: {shown}")
        if len(beyond) >= 10 and len(beyond) >= 0.5 * after:
            _, _, _, lines, machine = next(t for t in decided if t[0] == beyond[0])
            a, b = next(((a, b) for a, b in zip(norm(machine), norm(lines)) if a != b), ("", ""))
            print(f"  ! {len(beyond)} of the {after} pages beyond {upto} differ from the model's text. If you did not correct them,\n"
                  f"    the editor has changed the text itself (spacing and Unicode normalisation are already ignored). First\n"
                  f"    difference on {beyond[0]}:\n      model : {a}\n      XML   : {b}\n"
                  f"    Nothing was imported. Check what changed, or move --done on if those pages really are corrected.")
            if dry_run:
                return head, chunks, corrected
            raise SystemExit("  stopping before training.")
    elif not upto and last_diff:
        print(f"  hint: the last page that differs from the model's text is {last_diff}; if every page up to there has been\n"
              f"        checked, run again with --done {last_diff} so pages the model read perfectly count as corrected too")

    # 2. import and score
    new_scores, problems = [], []
    for pid, stem, boxes, lines, machine in decided:
        names = sorted(f.name for f in (d / "lines" / "pred").glob(f"{stem}_*.png"))
        for box, texts in boxes.items():
            n_img = sum(1 for n in names if pagetext.image_pattern(stem, box).match(n))
            while len(texts) > n_img and texts and not texts[-1].strip():
                texts = texts[:-1]
            if n_img and n_img != len(texts):
                problems.append(f"{pid}: {len(texts)} lines in the XML but {n_img} line images" + (f" in the {box} block" if len(boxes) > 1 else "") + " - not usable for training until the lines match")
        old = d / "corrected" / f"{stem}.txt"
        if not old.exists() or norm(pagetext.flat(pagetext.read(old))) != norm(lines):
            if machine is not None and len(machine) == len(lines):
                err = sum(Levenshtein.distance(a, b) for a, b in zip(norm(machine), norm(lines)))
                new_scores.append((pid, sum(len(t) for t in norm(lines)), err))
            else:
                new_scores.append((pid, sum(len(t) for t in norm(lines)), None))
            pagetext.write(old, boxes)
    if new_scores:
        print(f"  {len(new_scores)} new or changed since last time:")
        print(f"  {'page':<8} {'chars':>6} {'edits':>5} {'CER %':>6}   how much the model's reading had to be changed")
        for pid, chars, err in new_scores:
            print(f"  {pid:<8} {chars:>6} " + (f"{err:>5} {100 * err / max(1, chars):6.2f}" if err is not None else "    -      -   (line count differs)"))
        scored = [(c, e) for _, c, e in new_scores if e is not None]
        if len(scored) > 1:
            print(f"  {'all':<8} {sum(c for c, _ in scored):>6} {sum(e for _, e in scored):>5} {100 * sum(e for _, e in scored) / max(1, sum(c for c, _ in scored)):6.2f}")
    for pr in problems:
        print(f"  ! {pr}")
    return head, chunks, corrected


def rewrite(ms_id, xml, head, chunks, corrected, marks=False):
    d = ms_dir(ms_id)
    (d / "machine").mkdir(exist_ok=True)
    backups = d / "backups"
    backups.mkdir(exist_ok=True)
    shutil.copy2(xml, backups / f"{xml.stem}.{time.strftime('%Y%m%d-%H%M%S')}{xml.suffix}")
    out = [with_resp_stmt(head) if marks else without_resp_stmt(head)]
    kept, replaced, markup, written = set(), set(), [], set()
    for c in chunks:
        if c.pid in corrected:
            out.append(c.unmarked())
            kept.add(c.pid)
            continue
        stem = stem_of(ms_id, c.pid) if c.pid else None
        draft = d / "drafts" / f"{stem}.txt" if stem else None
        if not c.plain or not draft or not draft.exists():
            out.append(c.pb + c.body)
            if c.pid and not c.plain:
                markup.append(c.pid)
            continue
        boxes = pagetext.read(draft)
        if stem not in written:
            pagetext.write(d / "machine" / f"{stem}.txt", boxes)
            written.add(stem)
        out.append(c.rewrite(boxes, mark=marks))
        replaced.add(c.pid)
    xml.write_text("".join(out), encoding="utf-8")
    print(f"[update] wrote {xml}: {len(kept)} corrected pages kept as they were, {len(replaced)} pages re-read by the model"
          + (' (marked resp="#ocr")' if marks else ""))
    if markup:
        print(f"  left untouched because they hold markup besides <lb/>: {', '.join(dict.fromkeys(markup))}")
    print(f"  previous version: {backups}")


def update(ms_id, run, holdout=2, base=DEFAULT_MODEL, train=True, model=None, dry_run=False, done=None, marks=False):
    d = ms_dir(ms_id)
    xml = d / f"{ms_id}.xml"
    head, chunks, corrected = import_corrected(ms_id, xml, done, dry_run)
    if dry_run:
        return
    if train:
        hist = finetune(ms_id, base, run, holdout=holdout)
        best = d / "models" / run / "best.pt"
        model = best if hist and best.exists() else base
        if model == base:
            print("  fine-tuning gave no improvement; reading with the base model")
    model = Path(model or base)
    todo = [MANUSCRIPTS[ms_id]["image_name"](pid) for pid, cs in by_page(chunks).items()
            if pid and pid not in corrected and all(c.plain for c in cs)]
    predict(ms_id, todo, model, verbose=False, write_to_correct=False)
    rewrite(ms_id, xml, head, chunks, corrected, marks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ms_id")
    ap.add_argument("--done", metavar="PAGE", help="pages up to and including this one are corrected: 0020_2, 20_2 or 20 (= 0020_2). "
                                                   "Remembered in data/<ms>/corrected_upto.txt")
    ap.add_argument("--run", default="ft4", help="name for the new model (data/<ms>/models/<run>)")
    ap.add_argument("--holdout", type=int, default=2)
    ap.add_argument("--base", default=str(DEFAULT_MODEL), help="model to fine-tune from")
    ap.add_argument("--no-train", action="store_true", help="skip fine-tuning; read with --model")
    ap.add_argument("--model", help="model to read the uncorrected pages with (with --no-train)")
    ap.add_argument("--dry-run", action="store_true", help="only report which pages count as corrected")
    ap.add_argument("--marks", action="store_true", help='write resp="#ocr" on the <pb> of machine-read pages')
    a = ap.parse_args()
    update(a.ms_id, a.run, a.holdout, Path(a.base), train=not a.no_train, model=a.model, dry_run=a.dry_run,
           done=a.done, marks=a.marks)
