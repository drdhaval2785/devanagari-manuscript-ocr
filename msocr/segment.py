"""Step 2: find text lines on every page with Kraken's stock baseline segmenter.

Output: data/<ms>/seg/<image stem>.json (Kraken's native segmentation format).
Pages already segmented are skipped, so the step can be interrupted and re-run.
"""
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import KRAKEN_DEVICE, KRAKEN_SEG_MODEL, MANUSCRIPTS, VENV_BIN, ms_dir
from . import gt


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


def ensure_segmented(ms_id: str, images):
    """The pages segmented by whichever segmenter config.SEG names: Kraken (seg, the default) or ours
    (seg_torch, msocr/tseg.py). Pages already done are skipped either way."""
    from .config import SEG
    if SEG == "seg":
        return segment(ms_id, images=images)
    from . import tseg
    if SEG != tseg.OUT_DIR:
        raise SystemExit(f"MSOCR_SEG={SEG}: only 'seg' (Kraken) and '{tseg.OUT_DIR}' are known")
    return tseg.segment_pages(ms_id, images)


if __name__ == "__main__":
    for ms_id in sys.argv[1:]:
        segment(ms_id)
