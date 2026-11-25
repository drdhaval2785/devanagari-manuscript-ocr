"""How much did the corrected pages differ from what the model read?

Compares data/<ms>/drafts/<page>.txt (the model's reading, kept untouched) with
data/<ms>/corrected/<page>.txt (what the scribe wrote, per the corrector) for every page
that has both, and reports the character error rate (CER: edits needed per 100 characters) and the grapheme error rate (GER: per 100
akṣaras, msocr/aksara.py).
This is the honest score of the model that produced the drafts, on pages it never trained on.

    python -m msocr.compare LDII_19634                 all pages with both files
    python -m msocr.compare LDII_19634 019634_00006_1 019634_00006_2 ...
"""
import argparse
import collections

from rapidfuzz.distance import Levenshtein

from .aksara import errors as aksara_errors
from .config import ms_dir


def _lines(path):
    from . import pagetext
    return [t.strip().replace(" ", "") for t in pagetext.flat(pagetext.read(path)) if t.strip()]


def compare(ms_id, pages=None, top=20):
    d = ms_dir(ms_id)
    if not pages:
        pages = sorted(p.stem for p in (d / "corrected").glob("*.txt") if (d / "drafts" / p.name).exists())
    tot_err = tot_chars = tot_aerr = tot_aks = 0
    conf = collections.Counter()
    line_cers = []
    print(f"[compare] {ms_id}: model drafts vs corrected pages\n")
    print(f"  {'page':<16} {'lines':>5} {'chars':>6} {'edits':>5} {'CER %':>6} {'akṣaras':>7} {'GER %':>6}   worst line")
    for p in pages:
        draft, truth = _lines(d / "drafts" / f"{p}.txt"), _lines(d / "corrected" / f"{p}.txt")
        if len(draft) != len(truth):
            print(f"  {p:<16} skipped: {len(draft)} draft lines but {len(truth)} corrected lines")
            continue
        err = chars = aerr = aks = 0
        worst = (0, 0.0)
        for i, (a, t) in enumerate(zip(draft, truth), 1):
            e = Levenshtein.distance(a, t)
            err += e
            chars += len(t)
            ae, an = aksara_errors(a, t)
            aerr, aks = aerr + ae, aks + an
            line_cers.append(100 * e / max(1, len(t)))
            if e / max(1, len(t)) > worst[1]:
                worst = (i, e / max(1, len(t)))
            for op, ti, ai in Levenshtein.editops(t, a):
                conf[f"{t[ti]}→{a[ai]}" if op == "replace" else f"{t[ti]} missed" if op == "delete" else f"{a[ai]} invented"] += 1
        tot_err += err
        tot_chars += chars
        tot_aerr, tot_aks = tot_aerr + aerr, tot_aks + aks
        print(f"  {p:<16} {len(truth):>5} {chars:>6} {err:>5} {100 * err / max(1, chars):6.2f} {aks:>7} {100 * aerr / max(1, aks):6.2f}"
              f"   line {worst[0]} ({100 * worst[1]:.1f}%)")
    if not line_cers:
        print("  nothing to compare")
        return None
    cer = 100 * tot_err / tot_chars
    print(f"  {'all':<16} {len(line_cers):>5} {tot_chars:>6} {tot_err:>5} {cer:6.2f} {tot_aks:>7} {100 * tot_aerr / max(1, tot_aks):6.2f}")
    line_cers.sort()
    n = len(line_cers)
    print(f"\n  per line: median {line_cers[n // 2]:.1f}%, worst {line_cers[-1]:.1f}%; "
          f"{sum(c == 0 for c in line_cers)} of {n} lines needed no correction")
    kinds = collections.Counter()
    for k, v in conf.items():
        kinds["missed" if k.endswith("missed") else "invented" if k.endswith("invented") else "substituted"] += v
    print(f"  {sum(conf.values())} edits: {kinds['missed']} characters missed, {kinds['invented']} invented, {kinds['substituted']} substituted")
    print(f"  edits involving virama: {sum(v for k, v in conf.items() if chr(0x94D) in k)}, "
          f"anusvara: {sum(v for k, v in conf.items() if chr(0x902) in k)}")
    print("  commonest: " + ", ".join(f"{k} ×{v}" for k, v in conf.most_common(top)))
    return cer


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ms_id")
    ap.add_argument("pages", nargs="*", help="page stems; default: every page with both files")
    a = ap.parse_args()
    compare(a.ms_id, a.pages)
