"""Line ends the truth may have dropped: every corrected page of the pool's manuscripts read as pagediff reads it,
rows cut a quarter pitch wider (across the margin rule), by two recognisers with the language model; an end is listed
where a reading has akṣaras with letters in them before the truth's first akṣara or after its last, with whether the
other recogniser read them too and whether the old draft stopped where the truth stops.

    .venv/bin/python tools/experiments/margins/scan.py [--widen 0.25] [--out results/scan.json]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from msocr import pagetext                                                    # noqa: E402
from msocr.config import ms_dir                                               # noqa: E402
from msocr.lm import load_default                                             # noqa: E402
from msocr.pagediff import (MARK, PUNCT, VIRAMA, align, model_path, page_images, page_result,   # noqa: E402
                            short_name)
from msocr.recognizer import load_model                                       # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from edges import MSS                                                         # noqa: E402


def letter(x):
    return any(c not in PUNCT and c not in MARK and c != VIRAMA for c in x)


def beyond(a, b, ops):
    """the reading's akṣaras before the truth's first matched one and after its last, beyond what the truth has there"""
    eq = [o for o in ops if o[0] == "equal"]
    if not eq:
        return [], []
    return b[:max(0, eq[0][3] - eq[0][1])], b[eq[-1][4]:][len(a) - eq[-1][2]:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widen", type=float, default=0.25)
    ap.add_argument("--models", default="pool1,LDII_19634/ft6")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "scan.json"))
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    lm = load_default(None)
    nets = [(load_model(model_path(m)), short_name(model_path(m))) for m in a.models.split(",")]
    found = {}
    for ms in MSS:
        images = page_images(ms)
        for gt in sorted((ms_dir(ms) / "corrected").glob("*.txt")):
            image = gt.stem + ".jpg"
            draft_f = ms_dir(ms) / "drafts" / gt.name
            drafts = [x.strip().replace(" ", "") for x in pagetext.read(draft_f).get("main", [])] if draft_f.exists() else []
            for (net, chars, _), name in nets:
                res = page_result(ms, image, images.index(image) + 1, gt, "seg_torch", (net, chars, name), [], lm, a.widen)
                for ln in res["lines"]:
                    if not ln.get("truth") or "read" not in ln or not ln.get("ops"):
                        continue
                    lead, trail = beyond(ln["a"], ln["b"], ln["ops"])
                    d = drafts[ln["line"] - 1] if ln["line"] and ln["line"] - 1 < len(drafts) else None
                    for side, extra in (("left", lead), ("right", trail)):
                        key = (ms, gt.stem, ln["line"], side)
                        rec = found.setdefault(key, {"ms": ms, "page": gt.stem, "line": ln["line"], "row": ln["row"],
                                                     "side": side, "truth": ln["truth"], "read": {}, "extra": {}})
                        rec["read"][name] = ln["read"]
                        rec["extra"][name] = "".join(extra) if extra and any(map(letter, extra)) else ""
                        if d is not None and "draft_stops" not in rec:
                            da, db, dops = align(ln["truth"], d)
                            dl, dr = beyond(da, db, dops)
                            rec["draft_stops"] = not (dl if side == "left" else dr)
            print(f"  {ms} {gt.stem}", flush=True)
    flagged = [r for r in found.values() if any(r["extra"].values())]
    json.dump(flagged, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(flagged)} line ends flagged of {len(found)}")


if __name__ == "__main__":
    main()
