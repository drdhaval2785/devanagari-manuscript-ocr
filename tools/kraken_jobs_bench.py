"""Kraken throughput with 1, 2 and 4 parallel jobs (4 pages per process, as msocr.segment runs it), on the same 16 pages.

    PYTHONPATH=. .venv/bin/python tools/kraken_jobs_bench.py      # 2026-11-23: 14.2 / 9.3 / 10.0 s a page
"""
import random, re, subprocess, sys, threading, time, json, shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from msocr.config import MANUSCRIPTS, DATA, not_text
from msocr.segment import _run_batch

S = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "benchmarks" / "kraken_jobs"   # outputs, never data/<ms>/seg
S.mkdir(parents=True, exist_ok=True)
rng = random.Random(4278)
mss = sorted(m for m in MANUSCRIPTS if (DATA / m / "seg").exists() and any((DATA / m / "seg").glob("*.json")))
pages = []
for ms in rng.sample(mss, 8):
    stems = sorted(j.stem for j in (DATA / ms / "seg").glob("*.json") if j.stem not in not_text(ms) and not j.stem.endswith("_00000_1"))
    pages += [(ms, s) for s in rng.sample(stems, 2)]
print("pages:", [f"{m}:{s[-7:]}" for m, s in pages], flush=True)

def swap_used():
    m = re.search(r"used = ([\d.]+)M", subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout)
    return float(m.group(1)) / 1024 if m else 0.0

def pageouts():
    m = re.search(r"Pageouts:\s+(\d+)", subprocess.run(["vm_stat"], capture_output=True, text=True).stdout)
    return int(m.group(1)) if m else 0

def run(jobs, cap=None):
    out = S / f"bench_j{jobs}"
    shutil.rmtree(out, ignore_errors=True); out.mkdir()
    todo = [(MANUSCRIPTS[m]["images"] / f"{s}.jpg", out / f"{s}.json") for m, s in pages]
    batches = [todo[i:i + 4] for i in range(0, len(todo), 4)]
    peak, stop = [swap_used()], threading.Event()
    def watch():
        while not stop.wait(3):
            peak.append(swap_used())
    th = threading.Thread(target=watch, daemon=True); th.start()
    po0, t0 = pageouts(), time.time()
    with ThreadPoolExecutor(jobs) as pool:
        futs = [pool.submit(_run_batch, b, "cpu") for b in batches]   # a CPU benchmark, whatever the default device
        for f in futs:
            left = None if cap is None else max(1, cap - (time.time() - t0))
            try:
                f.result(timeout=left)
            except Exception:
                print(f"  jobs={jobs}: over the {cap:.0f}s cap, stopped", flush=True)
                subprocess.run(["pkill", "-f", str(out)])
                stop.set()
                return None
    secs = time.time() - t0
    stop.set(); th.join()
    ok = sum(1 for _, o in todo if o.exists())
    print(f"jobs={jobs}: {secs:6.1f} s for {ok}/{len(todo)} pages = {secs / len(todo):5.1f} s a page; "
          f"swap used {peak[0]:.1f} -> peak {max(peak):.1f} GB; pageouts {(pageouts() - po0) * 16 / 1024:.0f} MB", flush=True)
    return secs

t1 = run(1)
t2 = run(2)
run(4, cap=2 * t2)
# the outputs are the same whatever the number of jobs
same = all(json.load(open(S / "bench_j1" / f"{s}.json"))["lines"] == json.load(open(S / "bench_j2" / f"{s}.json"))["lines"] for m, s in pages)
print("jobs=1 and jobs=2 give identical lines:", same)
print(time.strftime("%H:%M"), "all done", flush=True)
