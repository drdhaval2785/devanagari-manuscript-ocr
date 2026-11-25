"""The best alpha/beta of a tuning grid written by decode.py (lowest decoded CER on the set)."""
import re
import sys

best = None
for path in sys.argv[1:]:
    head = None
    for line in open(path, encoding="utf-8"):
        if line.startswith("["):
            head = line
        elif head and not line.startswith("  all") and not line.startswith("  set"):
            parts = line.split()
            cer = float(parts[-4])
            m = re.search(r"alpha ([\d.]+) beta ([\d.]+)", head)
            if best is None or cer < best[0]:
                best = (cer, m.group(1), m.group(2), path)
print(f"{best[1]} {best[2]}  # decoded CER {best[0]} in {best[3].rsplit('/', 1)[-1]}")
