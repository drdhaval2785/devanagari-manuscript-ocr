"""Which manuscripts are linear, and which have been segmented.

data/layouts.csv holds one line per manuscript: its page layout (linear = one block of text,
tripatha = base text with commentary above and below or at both sides, panchapatha = commentary
on all four sides, special = anything else) and a note from looking at its pages.

    python -m msocr.layouts          prints the table and writes LAYOUTS.md with live counts
"""
import csv
from collections import Counter

from .config import DATA, MANUSCRIPTS, ROOT
from .rowcheck import summary as rowcheck_summary

ORDER = ["linear", "tripatha", "panchapatha", "special"]


def rows():
    out = []
    with open(DATA / "layouts.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ms = MANUSCRIPTS[r["ms_id"]]
            pages = len(list(ms["images"].glob("*.jpg")))
            seg = len(list((DATA / r["ms_id"] / "seg").glob("*.json")))
            out.append({**r, "text": ms["text"].replace("_commentaries", ""), "commentary": ms["commentary"],
                        "pages": pages, "segmented": seg, "rows": rowcheck_summary(r["ms_id"])})
    missing = set(MANUSCRIPTS) - {r["ms_id"] for r in out}
    if missing:
        print("not in layouts.csv:", ", ".join(sorted(missing)))
    return sorted(out, key=lambda r: (ORDER.index(r["layout"]), r["text"], r["commentary"], r["ms_id"]))


def render(rs):
    n = Counter(r["layout"] for r in rs)
    seg_ms = [r for r in rs if r["segmented"]]
    lines = ["# Manuscript layouts and segmentation", "",
             f"{len(rs)} manuscripts in the corpus: " + ", ".join(f"{n[k]} {k}" for k in ORDER) + ".",
             f"Segmented: {len(seg_ms)} manuscripts, {sum(r['segmented'] for r in seg_ms)} pages "
             f"(`data/<ms>/seg/`). Layout judged from a page near the middle of each manuscript, and a second "
             f"page for the linear ones.",
             "\"Rows found\" is from `python -m msocr.rowcheck <ms>`: of the segmented pages, how many the row finder "
             "gives evenly spaced rows (every line found, nothing squeezed in) - the pages that can be read and corrected "
             "as they are; the rest have a gap or an extra row to look at, or no rows at all. In brackets, the usual "
             "number of lines on a page.", "",
             "| manuscript | text | commentary | layout | pages | segmented | rows found | note |", "|---|---|---|---|---:|---:|---:|---|"]
    for r in rs:
        seg = "all" if r["segmented"] == r["pages"] and r["pages"] else (
            f"{r['segmented']} of {r['pages']}" if r["segmented"] else "no")
        rc = r["rows"]
        found = f"{rc['even']} of {rc['pages']} ({rc['usual']} lines)" if rc else ("-" if not r["segmented"] else "not checked")
        lines.append(f"| {r['ms_id']} | {r['text']} | {r['commentary']} | {r['layout']} | {r['pages']} | {seg} | {found} | {r['note']} |")
    lines += ["", "Linear manuscripts are segmented so the recogniser can be fine-tuned on them "
              "(`python -m msocr.predict <ms> --start 1 --pages N`, then correct, then `finetune`). "
              "Tripatha and panchapatha manuscripts are left for later: the row finder expects one block of "
              "text per page. Special ones need their own treatment, noted per manuscript.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    rs = rows()
    text = render(rs)
    (ROOT / "LAYOUTS.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"written to {ROOT / 'LAYOUTS.md'}")
