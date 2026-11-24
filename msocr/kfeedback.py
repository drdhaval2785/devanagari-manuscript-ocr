"""Take in the feedback saved from the review pages (data/kraken_review/<ms>_review.html).

    python -m msocr.kfeedback ~/Downloads/LDII_1806_feedback.tsv [more.tsv ...]

Each file (manuscript, page, group, verdict, errors, rows, block, note; one line per page marked) is kept in
data/kraken_review/feedback/<ms>.tsv, a later verdict on a page replacing an earlier one, so the record grows as
pages are looked at again after the code is mended. Pages marked "not a text page" are added to data/<ms>/not_text.txt,
and from then on are neither checked, read nor learnt from. The rest is summed up: how many pages of each group were
found correct, and every page found wrong, by kind of error, with its rows, block and note: the list to mend the
code from.
"""
import collections
import csv
import sys
from pathlib import Path

from .config import DATA, MANUSCRIPTS

FIELDS = ["manuscript", "page", "group", "verdict", "errors", "rows", "block", "note"]
STORE = DATA / "kraken_review" / "feedback"


def _read(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    missing = [c for c in FIELDS if rows and c not in rows[0]]
    if missing:
        raise SystemExit(f"{path}: not a feedback file (no {', '.join(missing)} column)")
    return rows


def take(paths):
    STORE.mkdir(parents=True, exist_ok=True)
    by_ms = collections.defaultdict(dict)
    for p in paths:
        for r in _read(p):
            ms = r["manuscript"]
            if ms not in MANUSCRIPTS:
                raise SystemExit(f"{p}: unknown manuscript {ms}")
            by_ms[ms][r["page"]] = r
    for ms, new in by_ms.items():
        store = STORE / f"{ms}.tsv"
        old = {r["page"]: r for r in _read(store)} if store.exists() else {}
        old.update(new)
        with open(store, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, FIELDS, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            for page in sorted(old):
                w.writerow(old[page])
        # pages that are not text pages, into data/<ms>/not_text.txt
        nt = DATA / ms / "not_text.txt"
        listed = set(nt.read_text(encoding="utf-8").split()) if nt.exists() else set()
        add = [p for p, r in sorted(old.items()) if r["verdict"] == "not_text" and p not in listed]
        if add:
            with open(nt, "a", encoding="utf-8") as f:
                if not listed:
                    f.write("# pages that are not text pages (from the review); one image stem or page id per line\n")
                for p in add:
                    f.write(p + "\n")
        print(_summary(ms, old, add))


def _summary(ms, rows, added):
    lines = [f"== {ms}: {len(rows)} pages marked in all"]
    by_group = collections.defaultdict(collections.Counter)
    for r in rows.values():
        by_group[r["group"]][r["verdict"]] += 1
    for g, c in sorted(by_group.items()):
        lines.append(f"   {g:16s} " + ", ".join(f"{v or 'unmarked'} {n}" for v, n in sorted(c.items())))
    if added:
        lines.append(f"   added to data/{ms}/not_text.txt: {', '.join(added)}")
    wrong = [r for r in rows.values() if r["verdict"] == "wrong"]
    if wrong:
        lines.append(f"   wrong pages ({len(wrong)}), by kind of error:")
        kinds = collections.defaultdict(list)
        for r in wrong:
            for k in (r["errors"] or "unspecified").split(","):
                kinds[k].append(r)
        for k, rs in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"     {k} ({len(rs)})")
            for r in rs:
                where = ", ".join(x for x in (f"rows {r['rows']}" if r["rows"] else "", r["block"]) if x)
                lines.append(f"        {r['page']} [{r['group']}]" + (f" {where}" if where else "") + (f": {r['note']}" if r["note"] else ""))
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    take(sys.argv[1:])
