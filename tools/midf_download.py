"""Fetch the pages of the MIDF/eGangotri dataset's segmentation v1.1 (page image, PAGE XML, annotation) into the
cache msocr.midf reads (training_data/midf/), saying every minute how far it is. Nothing is imported.

    PYTHONPATH=. caffeinate -i .venv/bin/python tools/midf_download.py                # every manuscript, about 2.3 GB
    PYTHONPATH=. .venv/bin/python tools/midf_download.py --new                        # without those muktabodha.py made
    PYTHONPATH=. .venv/bin/python tools/midf_download.py M00622 M00627 --interval 30

Every file is checked against the dataset's SHA-256. A file already there and right is not fetched again, so the run
may be stopped (Ctrl-C) and started again. When the Hub limits the requests, the run waits and goes on.
"""
import argparse
import collections
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from msocr import midf
from msocr.config import DATA

PASSES = 3              # a file that fails is tried again, in this many passes over what is missing


def wanted(ids, new):
    """-> [(manuscript, page, path in the dataset, SHA-256)]"""
    midf._fetch("metadata/manuscripts.csv")
    sums = {}
    for line in midf._fetch(midf.SEGMENTATION + "checksums.sha256").read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.removeprefix("./")] = digest
    manifest = json.loads(midf._fetch(midf.SEGMENTATION + "manifest.json", sums.get("manifest.json")).read_text(encoding="utf-8"))
    pages = sorted((p["manuscript_id"], p["page_id"]) for p in manifest["pages"])
    known = {m for m, _ in pages}
    if set(ids) - known:
        raise SystemExit(f"not in segmentation v1.1: {' '.join(sorted(set(ids) - known))} (it has {' '.join(sorted(known))})")
    skip = set()
    if new:
        for f in (DATA / "muktabodha").glob("*.json"):
            skip |= set(re.findall(r"-(M\d{5})-", Path(json.loads(f.read_text(encoding="utf-8"))["etext"]).name))
    out = []
    for ms, page in pages:
        if (ids and ms not in ids) or ms in skip:
            continue
        if not re.fullmatch(re.escape(ms) + r"-p\d{4}", page) or not re.fullmatch(r"M\d{5}", ms):
            raise SystemExit(f"unexpected page id in the manifest: {page!r}")
        for name in ("image.jpg", f"{page}.xml", "annotation.json"):
            rel = f"pages/{page}/{name}"
            if rel not in sums:
                raise SystemExit(f"{rel}: no checksum in the dataset's list")
            out.append((ms, page, midf.SEGMENTATION + rel, sums[rel]))
    return out


def one(item):
    """-> (item, bytes of the file, fetched now?, error or None)"""
    _, _, path, digest = item
    local = midf.CACHE / path
    before = local.stat().st_mtime_ns if local.exists() else None
    try:
        midf._fetch(path, digest)
    except (Exception, SystemExit) as e:
        return item, 0, False, f"{type(e).__name__}: {e}"
    st = local.stat()
    return item, st.st_size, st.st_mtime_ns != before, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ms", nargs="*", help="the dataset's manuscript ids (default: all)")
    ap.add_argument("--new", action="store_true", help="leave out the manuscripts msocr.muktabodha has made already")
    ap.add_argument("--interval", type=float, default=60, help="seconds between status lines (default 60)")
    ap.add_argument("--jobs", type=int, default=6, help="files fetched at once")
    a = ap.parse_args()

    def say(text):
        print(f"{time.strftime('%H:%M:%S')}  {text}", flush=True)

    say("reading the dataset's list of pages ...")
    files = wanted(a.ms, a.new)
    pages_of = collections.Counter(ms for ms, page in {(f[0], f[1]) for f in files})
    left = collections.Counter(f[1] for f in files)                 # files still missing, by page
    pages_left = collections.Counter(pages_of)                      # pages still incomplete, by manuscript
    say(f"{len(pages_of)} manuscripts, {len(left)} pages, {len(files)} files -> {midf.CACHE}")
    say(f"free disk space: {shutil.disk_usage(midf.CACHE).free / 1e9:.0f} GB")

    start = last = time.time()
    done = fetched = size = new_size = last_done = 0
    errors, todo = {}, files
    for attempt in range(1, PASSES + 1):
        if attempt > 1:
            say(f"pass {attempt}: trying {len(todo)} failed file(s) again")
        errors = {}
        pool = ThreadPoolExecutor(a.jobs)
        pending = {pool.submit(one, f) for f in todo}
        try:
            while pending:
                finished, pending = wait(pending, timeout=max(0.0, last + a.interval - time.time()), return_when=FIRST_COMPLETED)
                for fut in finished:
                    item, n, is_new, error = fut.result()
                    if error:
                        errors[item] = error
                        continue
                    done += 1
                    size += n
                    fetched += is_new
                    new_size += n if is_new else 0
                    left[item[1]] -= 1
                    if not left[item[1]]:
                        pages_left[item[0]] -= 1
                        if not pages_left[item[0]]:
                            say(f"  {item[0]} complete ({pages_of[item[0]]} pages)")
                if time.time() - last >= a.interval:
                    elapsed = time.time() - start
                    eta = (f", about {max(1, round((len(files) - done) * elapsed / fetched / 60))} min to go" if fetched else "")
                    quiet = "  (no file finished since the last line: a slow link, or the Hub's request limit; it retries by itself)" if done == last_done else ""
                    say(f"{done}/{len(files)} files ({100 * done / len(files):.0f} %), "
                        f"{sum(1 for v in left.values() if not v)}/{len(left)} pages, "
                        f"{sum(1 for v in pages_left.values() if not v)}/{len(pages_of)} manuscripts; "
                        f"{size / 1e6:.0f} MB here, {new_size / 1e6:.0f} MB fetched at {new_size / 1e6 / elapsed:.1f} MB/s, "
                        f"failed {len(errors)}{eta}{quiet}")
                    last, last_done = time.time(), done
        except KeyboardInterrupt:
            say(f"stopped at {done}/{len(files)} files; run the same command again to go on")
            os._exit(130)             # the threads are not waited for: a half-fetched file is only a .part, overwritten next time
        pool.shutdown()
        todo = list(errors)
        if not todo:
            break

    say(f"{'finished' if not errors else 'finished WITH FAILURES'}: {done}/{len(files)} files, {size / 1e6:.0f} MB "
        f"({fetched} fetched now, {done - fetched} there already) in {(time.time() - start) / 60:.1f} min")
    if errors:
        for item, error in sorted(errors.items())[:20]:
            say(f"  {item[2]}: {error}")
        say(f"{len(errors)} file(s) missing; run the same command again")
        sys.exit(1)


if __name__ == "__main__":
    main()
