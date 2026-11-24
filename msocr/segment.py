"""Step 2: find the text lines on every page: with our own U-Net (msocr/tseg.py; data/<ms>/seg_torch/), the default
since 2026-11-24, or with Kraken's stock baseline segmenter (data/<ms>/seg/), both in Kraken's segmentation format.

    .venv/bin/python -m msocr.segment LDII_45364 [...]            the default segmenter (config.SEG: the U-Net)
    .venv/bin/python -m msocr.segment LDII_45364 --kraken         Kraken (needed by align, and as tseg's teacher)

Pages already segmented are skipped, so the step can be interrupted and re-run.

Which segmentation a page is read from (page_seg): a page never cut into lines, the default; a page whose line images
were cut already (lines/pred), the one they were cut from, which lines/pred/segmentation.tsv records (pages cut before
the record was kept were all cut from Kraken's), so that its to_correct text and review page stay paired with its line
images; MSOCR_SEG set by hand overrules that, but never for a corrected page, whose text is paired with its images for
training. reading_from(seg) makes predict, blocks and show read a given segmentation for a while.
"""
import argparse
import contextlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import KRAKEN_DEVICE, KRAKEN_SEG, KRAKEN_SEG_MODEL, MANUSCRIPTS, SEG, SEG_CHOSEN, VENV_BIN, ms_dir
from . import gt

PINS = "segmentation.tsv"       # in data/<ms>/lines/pred/: page stem, the segmentation its line images were cut from


def _run_batch(pairs, device=KRAKEN_DEVICE):
    cmd = [str(VENV_BIN / "kraken"), "-d", device]
    for img, out in pairs:
        cmd += ["-i", str(img), str(out)]
    cmd += ["segment", "-bl", "-i", str(KRAKEN_SEG_MODEL)]   # the same model whatever kraken's default
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    return pairs, time.time() - t, r


def segment(ms_id: str, jobs: int = None, batch: int = 4, images=None):
    """images: file names to segment; default = every page that has a transcription, or,
    for a manuscript without one, every image in its folder. jobs: Kraken processes at once; by default one on the
    GPU (they would share it) and two on the CPU (2026-11-23, 16 GB M2 Pro: 14.2, 9.3 and 10.0 s a page with 1, 2, 4)."""
    jobs = jobs or (1 if KRAKEN_DEVICE == "mps" else 2)
    ms = MANUSCRIPTS[ms_id]
    out_dir = ms_dir(ms_id) / "seg"
    out_dir.mkdir(exist_ok=True)
    if images is None:
        images = sorted({r["image"] for r in gt.load(ms_id)}) if "tei" in ms else sorted(p.name for p in ms["images"].glob("*.jpg"))
    todo = [(ms["images"] / i, out_dir / (i.rsplit(".", 1)[0] + ".json")) for i in images]
    done = [p for p in todo if p[1].exists()]
    todo = [p for p in todo if not p[1].exists()]
    print(f"[segment] {ms_id}: {len(images)} pages requested, "
          f"{len(done)} already segmented, {len(todo)} to do, {jobs} parallel job{'s' if jobs > 1 else ''} on {KRAKEN_DEVICE}")
    if not todo:
        return
    batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    start, finished = time.time(), 0
    with ThreadPoolExecutor(jobs) as pool:
        for fut in as_completed([pool.submit(_run_batch, b) for b in batches]):
            pairs, secs, r = fut.result()
            ok = [p for p in pairs if p[1].exists()]
            finished += len(pairs)
            elapsed = time.time() - start
            eta = elapsed / finished * (len(todo) - finished)
            print(f"  {finished:4d}/{len(todo)}  {pairs[0][0].name} .. {pairs[-1][0].name}  "
                  f"{secs:5.0f}s  ok {len(ok)}/{len(pairs)}  elapsed {elapsed/60:5.1f} min  "
                  f"remaining ~{eta/60:4.1f} min", flush=True)
            if len(ok) < len(pairs):
                print("    ! kraken said: " + (r.stderr.strip().splitlines() or ["(nothing)"])[-1])
    print(f"[segment] finished in {(time.time()-start)/60:.1f} min")


def ensure_segmented(ms_id: str, images, seg=None):
    """The pages segmented by the segmenter of `seg` (default config.SEG): Kraken (seg) or ours (seg_torch,
    msocr/tseg.py). Pages already done are skipped either way; a segmentation made otherwise (seg_repaired) is only read."""
    seg = seg or SEG
    if seg == KRAKEN_SEG:
        return segment(ms_id, images=images)
    from . import tseg
    if seg == tseg.OUT_DIR:
        return tseg.segment_pages(ms_id, images)
    missing = [i for i in images or [] if not (ms_dir(ms_id) / seg / (i.rsplit(".", 1)[0] + ".json")).exists()]
    if missing:
        raise SystemExit(f"{seg}: no segmenter makes it, and {len(missing)} page(s) lack it ({missing[0]} ...)")


def _pins(ms_id):
    f = ms_dir(ms_id) / "lines" / "pred" / PINS
    if not f.exists():
        return {}
    return dict(l.split("\t", 1) for l in f.read_text(encoding="utf-8").splitlines() if "\t" in l)


def pin(ms_id, stem, seg):
    """record that the page's line images were cut from `seg`"""
    pins = _pins(ms_id)
    if pins.get(stem) == seg:
        return
    pins[stem] = seg
    f = ms_dir(ms_id) / "lines" / "pred" / PINS
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("".join(f"{k}\t{v}\n" for k, v in sorted(pins.items())), encoding="utf-8")


def cut_from(ms_id, stem):
    """the segmentation the page's line images were cut from: as recorded; Kraken's for a page cut before the record was
    kept (every one was); None for a page never cut"""
    p = _pins(ms_id).get(stem)
    if p:
        return p
    if any((ms_dir(ms_id) / "lines" / "pred").glob(f"{stem}_*.png")):
        return KRAKEN_SEG
    return None


def page_seg(ms_id, stem):
    """the segmentation the page is to be read from (see the module's notes)"""
    cut = cut_from(ms_id, stem)
    if cut is None:
        return SEG
    if (ms_dir(ms_id) / "corrected" / f"{stem}.txt").exists():
        return cut
    return SEG if SEG_CHOSEN else cut


@contextlib.contextmanager
def reading_from(seg):
    """predict, blocks and show (as loaded, also as __main__) read `seg` until the block ends"""
    mods = [sys.modules.get(f"msocr.{n}") for n in ("predict", "blocks", "show")]
    main = sys.modules.get("__main__")
    if main is not None and getattr(main, "__package__", None) == "msocr":
        mods.append(main)
    mods = [m for m in mods if m is not None and hasattr(m, "SEG")]
    old = [m.SEG for m in mods]
    for m in mods:
        m.SEG = seg
    try:
        yield
    finally:
        for m, s in zip(mods, old):
            m.SEG = s


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ms_id", nargs="+")
    ap.add_argument("--kraken", action="store_true", help="Kraken's segmenter, whatever the default")
    a = ap.parse_args()
    for ms_id in a.ms_id:
        if a.kraken:
            segment(ms_id)
        else:
            ensure_segmented(ms_id, None)
