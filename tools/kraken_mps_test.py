"""Kraken on the Mac GPU (MPS): a few pages in one process, with the same model as the pipeline, timed, with peak
memory and swap recorded, and the lines checked against Kraken's CPU output already in data/<ms>/seg/. The run is
stopped after --limit seconds (default 300) if it has not finished, as the MPS path once exhausted memory.

    PYTHONPATH=. .venv/bin/python tools/kraken_mps_test.py                   # 4 pages: LDII_3746 and LDII_9851
    PYTHONPATH=. .venv/bin/python tools/kraken_mps_test.py LDII_1806:001806_00051_1 ...
Outputs go to data/benchmarks/kraken_mps/, never to data/<ms>/seg/.
"""
import json, re, subprocess, sys, time
from msocr.config import DATA, KRAKEN_SEG_MODEL, MANUSCRIPTS, VENV_BIN

PAGES = ["LDII_3746:003746_00091_2", "LDII_3746:003746_00093_2", "LDII_9851:009851_00013_2", "LDII_9851:009851_00064_1"]


def swap():
    m = re.search(r"used = ([\d.]+)M", subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout)
    return float(m.group(1)) / 1024 if m else 0.0


def geometry(path):
    return sorted((json.dumps(l["baseline"]), json.dumps(l["boundary"])) for l in json.load(open(path))["lines"])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--limit=")), 300)
    pages = [p.split(":") for p in (args or PAGES)]
    out = DATA / "benchmarks" / "kraken_mps"
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["/usr/bin/time", "-l", str(VENV_BIN / "kraken"), "-d", "mps"]
    for ms, stem in pages:
        o = out / f"{stem}.json"
        if o.exists():
            o.unlink()
        cmd += ["-i", str(MANUSCRIPTS[ms]["images"] / f"{stem}.jpg"), str(o)]
    cmd += ["segment", "-bl", "-i", str(KRAKEN_SEG_MODEL)]
    s0, t0 = swap(), time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=limit)
        err = r.stderr
    except subprocess.TimeoutExpired as e:
        err = (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        print(f"stopped after {limit} s")
    secs = time.time() - t0
    rss = re.search(r"(\d+)\s+maximum resident set size", err)
    done = [(ms, stem) for ms, stem in pages if (out / f"{stem}.json").exists()]
    same = [s for ms, s in done if geometry(out / f"{s}.json") == geometry(DATA / ms / "seg" / f"{s}.json")]
    print(f"kraken -d mps: {len(done)} of {len(pages)} pages in {secs:.0f} s ({secs / max(1, len(done)):.1f} s a page, "
          f"model loading included); peak memory {int(rss.group(1)) / 1e9:.1f} GB" if rss else f"{len(done)} pages in {secs:.0f} s")
    print(f"swap {s0:.1f} GB before, {swap():.1f} GB after; lines identical to the CPU segmentation on {len(same)} of {len(done)} pages")
    if not done:
        print("kraken said:", "\n".join(err.strip().splitlines()[-5:]))


if __name__ == "__main__":
    main()
