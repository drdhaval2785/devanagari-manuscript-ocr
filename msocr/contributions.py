"""Contributed lines and folio sides: the packages Manuscribe makes of a manuscript's corrected lines and of its folio
sides whose drawn blocks each hold as many lines of text as their editor gave them ("Contribute corrected lines", in the
Manuscript tab, once its editor has said yes; sides only on a consent wording that gives them, SIDES), registered here
before anything may be trained on them. Every package states the manuscript's language and script (a tag as sa-Deva),
and everything is kept apart by it: the models here (MODELS_SCRIPT) learn from their own language and script only, and
another's contributions are kept for a model of their own and touch nothing here.

They come by GitHub: Manuscribe seals the lines to the maintainer's key (msocr.seal) and the contributor attaches the
file to an issue in the inbox repository (manuscribe-contributions), whose workflow checks the package, keeps it as an
asset of a draft release, registers it in its register.tsv and publishes a data release; a withdrawal comes the same
way and deletes the asset. `sync` brings this register up to the inbox's: it fetches the packages registered there and
not here, opens them with the private key, and withdraws here what was withdrawn there. Through `gh`, signed in as the
maintainer (the draft releases are seen by no one else).

    .venv/bin/python -m msocr.contributions sync [--inbox owner/repo]      (then quality, by itself)
    .venv/bin/python -m msocr.contributions quality                       check every package's lines (msocr.linequality)
    .venv/bin/python -m msocr.contributions accept <id> [--why "..."]      take a quarantined package, having looked at it
    .venv/bin/python -m msocr.contributions add <package.zip> [--accept-wording draft-1]
    .venv/bin/python -m msocr.contributions list
    .venv/bin/python -m msocr.contributions withdraw <id> [--why "..."]

-> data/contributions/register.tsv     a row per package: id, manuscript (the name the editor gave for the model cards),
                                       its language and script, lines, sides, the consent's wording and date, when it
                                       was added, its state, the issue and GitHub account it came from, what is usable
   data/contributions/<id>/            as opened: contribution.json, lines/*.png and *.txt, sides/*.jpg and *.json (the
                                       blocks drawn and their line counts), and quality.json: each line's checks and the
                                       judge's reading, each side's counts, the package's verdict; sides/*.seg.json, the
                                       segmenter's target for each side its editor's counts bore out
A package is taken only if its consent record says yes, confirms the right to share the images, is not withdrawn, and
was given on a wording approved here (APPROVED; --accept-wording takes another for a trial, and the row says so). A
sealed package (format version 2, as Manuscribe makes them since 0.11.0) is opened with the private key and must be
whole; an unsealed one (version 1) is taken as it is. Withdrawal deletes the package's lines and keeps its row, as
withdrawn, so that the next model's card can say what it no longer learns from; models trained before stay as they are.

Before training may use a package, its lines are checked (msocr.linequality: the image, the text, the two against
each other, copies, and a recogniser's reading of the image against the text): the lines that fail are left out, and a
package with too many failures, or whose texts the judge's reading does not bear out, is quarantined whole until the
maintainer accepts it (`accept`, after looking at its lines in data/contributions/<id>/). Its folio sides are checked
too (msocr.sidequality): the image and the blocks, and the U-Net's rows in every block against the count its editor
gave; a side whose counts are borne out gets its target, the others are kept and not used.

    allowed(script) -> [{manuscript, png, text, id, folio, release}]    the lines a recogniser of `script` may learn
                 from: registered, approved, not withdrawn, of that language and script, of a package checked ok or
                 accepted, each line usable; text as the recogniser learns it
    allowed_sides(script) -> [{manuscript, image, target, id, folio, release}]    the sides a segmenter may learn from

`recognizer pool` trains on allowed() alongside the developer's own manuscripts (each contributed manuscript a source of
its own, its last folio sides held out for scoring), and `tseg train` on allowed_sides() beside Kraken's reviewed pages;
the models they save list the packages they learned from, which the model cards name.
"""
import argparse
import datetime
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from . import seal
from .config import DATA

ROOT = DATA / "contributions"
REGISTER = ROOT / "register.tsv"
FIELDS = ["id", "manuscript", "script", "lines", "sides", "wording", "consent_date", "added", "state", "note", "issue", "contributor",
          "release", "quality", "usable", "usable_sides"]
APPROVED = {"3"}        # consent wordings approved by the maintainer (Manuscribe's recogview.js, CONSENT.version); 1 and 2
                        # named CC BY-SA 4.0 for the models, which since 2026-11-27 are CC BY-NC-SA 4.0: no package was
                        # given on them, and none is taken
SIDES = {"3"}           # ... of which those that give whole folio sides, not lines only
TAG = __import__("re").compile(r"^[a-z]{2,3}-[A-Z][a-z]{3}$")       # a manuscript's language and script, as sa-Deva
MODELS_SCRIPT = "sa-Deva"   # the language and script of the models trained here: only their data trains them
INBOX = "drdhaval2785/manuscribe-contributions"


def _rows():
    if not REGISTER.exists():
        return []
    lines = REGISTER.read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t") if lines else FIELDS
    return [dict(zip(head, l.split("\t"))) for l in lines[1:] if l.strip()]


def _write(rows):
    ROOT.mkdir(parents=True, exist_ok=True)
    REGISTER.write_text("\t".join(FIELDS) + "\n" + "".join("\t".join(str(r.get(k, "")).replace("\t", " ") for k in FIELDS) + "\n" for r in rows), encoding="utf-8")


def check(about, accept=()):
    """-> the reasons a package cannot be taken (none: it can)."""
    why = []
    if about.get("format") != "manuscribe-contribution" or about.get("version") not in (1, 2):
        why.append("not a Manuscribe contribution package (format/version)")
        return why
    c = about.get("consent") or {}
    if c.get("answer") != "yes":
        why.append(f"its consent says {c.get('answer')!r}, not yes")
    if not c.get("rights"):
        why.append("the right to share the images is not confirmed")
    if c.get("withdrawn"):
        why.append("its consent was withdrawn")
    if c.get("wording") not in APPROVED | set(accept):
        why.append(f"consent given on wording {c.get('wording')!r}, which is not approved here")
    if about.get("sealed"):
        if not TAG.match(str((about.get("manuscript") or {}).get("script", ""))):
            why.append("no language and script stated for the manuscript (as sa-Deva)")
        if about.get("sides") and c.get("wording") not in SIDES:
            why.append(f"folio sides given on wording {c.get('wording')!r}, which gives lines only")
        if not about.get("count") and not about.get("sides"):
            why.append("no lines and no folio sides")
    elif not about.get("lines"):
        why.append("no lines")
    return why


def open_package(data, priv=None):
    """a package's bytes -> (about, inner zip, id or None): a sealed one opened with the private key and checked whole"""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        about = json.loads(z.read("contribution.json").decode("utf-8"))
        if about.get("version") != 2:
            return about, data, None
        sealed = z.read("sealed.bin")
    sha = hashlib.sha256(sealed).hexdigest()
    if about.get("id") != seal.package_id(about, sha):
        raise SystemExit(f"[contributions] {about.get('id')}: its id is not the one its date and checksum give")
    inner = seal.open_sealed(about.get("seal") or {}, sealed, priv or seal.load_private())
    with zipfile.ZipFile(io.BytesIO(inner)) as z:
        a = json.loads(z.read("contribution.json").decode("utf-8"))
    if ((a.get("consent") or {}) != (about.get("consent") or {}) or len(a.get("lines") or []) != about.get("count")
            or len(a.get("sides") or []) != (about.get("sides") or 0) or (a.get("manuscript") or {}).get("script") != (about.get("manuscript") or {}).get("script")):
        raise SystemExit(f"[contributions] {about['id']}: what is sealed is not what the package says (consent, counts, or script)")
    return about, inner, about["id"]


def add(path, accept=(), came=None, priv=None):
    """register a package (a sealed one opened first); came: {issue, contributor, release} from the inbox"""
    data = Path(path).read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        outer = json.loads(z.read("contribution.json").decode("utf-8"))
    why = check(outer, accept)
    if why:
        raise SystemExit(f"[contributions] {path}: not taken: " + "; ".join(why))
    about, inner, cid = open_package(data, priv)
    with zipfile.ZipFile(io.BytesIO(inner)) as z:
        a = json.loads(z.read("contribution.json").decode("utf-8"))
        names = set(z.namelist())
        missing = [l["image"] for l in a["lines"] if l["image"] not in names] + [x for sd in a.get("sides") or [] for x in (sd["image"], sd["geometry"]) if x not in names]
        if missing:
            raise SystemExit(f"[contributions] {path}: not taken: missing {', '.join(missing[:5])}")
        cid = cid or f"c{datetime.date.today():%Y%m%d}-{hashlib.sha256(data).hexdigest()[:12]}"
        if any(r["id"] == cid for r in _rows()):
            raise SystemExit(f"[contributions] {path}: already registered as {cid}")
        out = ROOT / cid
        out.mkdir(parents=True, exist_ok=True)
        for n in names:
            if n == "contribution.json" or (n.startswith(("lines/", "sides/")) and not n.endswith("/") and ".." not in n):
                (out / n).parent.mkdir(parents=True, exist_ok=True)
                (out / n).write_bytes(z.read(n))
    c = a["consent"]
    rows = _rows()
    rows.append(dict({"id": cid, "manuscript": c.get("name", ""), "script": (a.get("manuscript") or {}).get("script", ""), "lines": len(a["lines"]),
                      "sides": len(a.get("sides") or []), "wording": c.get("wording", ""),
                      "consent_date": c.get("date", "")[:10], "added": datetime.datetime.now().isoformat(timespec="seconds"), "state": "registered",
                      "note": "" if c.get("wording") in APPROVED else f"wording {c.get('wording')} taken for a trial (--accept-wording)"}, **(came or {})))
    _write(rows)
    print(f"[contributions] {cid}: {len(a['lines'])} lines and {len(a.get('sides') or [])} folio sides of {c.get('name')!r} "
          f"({(a.get('manuscript') or {}).get('script') or 'no script stated'}) registered ({out})")
    return cid


def withdraw(cid, why=""):
    rows = _rows()
    row = next((r for r in rows if r["id"] == cid), None)
    if not row:
        raise SystemExit(f"[contributions] no contribution {cid}")
    shutil.rmtree(ROOT / cid / "lines", ignore_errors=True)
    shutil.rmtree(ROOT / cid / "sides", ignore_errors=True)
    row["state"] = "withdrawn"
    row["note"] = (f"withdrawn {datetime.date.today()}" + (f": {why}" if why else "")).strip()
    _write(rows)
    print(f"[contributions] {cid}: withdrawn; its lines and sides deleted, its row kept")


_JUDGES = {}


def _judge(pack):
    """-> (pack the judge is, net, chars): the recogniser of the pack that read the lines where that pack is here (then
    its reading can be held to what Manuscribe proposed), else the newest pack's"""
    from . import recognizer as R
    repo = Path(__file__).resolve().parent.parent
    packs = {mf.parent.name: mf for mf in sorted((DATA / "model_pack").glob("pack-*/manifest.json"))}
    if not packs:
        raise SystemExit("[contributions] no recognition pack in data/model_pack to judge the lines with")
    name = pack if pack in packs else sorted(packs)[-1]
    if name not in _JUDGES:
        src = json.loads(packs[name].read_text(encoding="utf-8"))["models"]["recogniser"]["source"]
        net, chars, _ = R.load_model(repo / src)
        _JUDGES[name] = (net, chars, src)
    net, chars, src = _JUDGES[name]
    return name, net, chars, src


_SEGMENTER = []


def _check_sides(r, d, about, seen):
    """each folio side checked (msocr.sidequality) and, where the segmenter's rows bear out its editor's counts in every
    block, its target written beside it (sides/<folio>.seg.json) -> [{image, folio, reasons, counts, target, usable}]"""
    import numpy as np
    from . import sidequality as SQ, tseg
    if not _SEGMENTER:
        _SEGMENTER.append((tseg.load_net()[0], tseg.device()))
    net, dev = _SEGMENTER[0]
    out = []
    for sd in about.get("sides") or []:
        rec = {"image": sd["image"], "folio": sd.get("folio", ""), "layout": sd.get("layout", ""), "reasons": []}
        jpg, geo = d / sd["image"], d / sd["geometry"]
        try:
            geom = json.loads(geo.read_text(encoding="utf-8"))
        except Exception:
            geom = {}
        g, why = SQ.check(jpg.read_bytes(), geom) if jpg.exists() and ".." not in sd["image"] else (None, ["image: missing"])
        rec["reasons"] += why
        if g is not None:
            px = hashlib.sha256(np.asarray(g).tobytes()).hexdigest()
            if px in seen:
                rec["reasons"].append(f"copy: the same side as {seen[px]}")
            else:
                seen[px] = f"{r['id']}/{sd['image']}"
            seg, counts, why = SQ.verify(jpg, geom, net, dev)
            rec["counts"] = counts
            rec["reasons"] += why
            if not rec["reasons"]:
                target = jpg.with_suffix(".seg.json")
                target.write_text(json.dumps(seg), encoding="utf-8")
                rec["target"] = str(target.relative_to(d))
        rec["usable"] = not rec["reasons"]
        out.append(rec)
    return out


def quality(verbose=True):
    """check every registered package (msocr.linequality for its lines, msocr.sidequality for its folio sides), in the
    order they came: quality.json in each package's folder, and the verdict and what is usable in the register. Only a
    package of the language and script the models here are for (MODELS_SCRIPT) is checked and can be used; another's
    is kept for a model of its own, and touches nothing here."""
    from . import linequality as LQ
    rows = _rows()
    seen = {}
    for r in rows:
        if r["state"] != "registered":
            continue
        d = ROOT / r["id"]
        about = json.loads((d / "contribution.json").read_text(encoding="utf-8"))
        if r.get("script") != MODELS_SCRIPT:
            out = {"checked": datetime.datetime.now().isoformat(timespec="seconds"), "verdict": "other-script",
                   "reasons": [f"{r.get('script') or 'no language and script stated'}: no model here is for it, so nothing here uses it"]}
            (d / "quality.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
            r["quality"], r["usable"], r["usable_sides"] = "other-script", 0, 0
            if verbose:
                print(f"[quality] {r['id']} {r['manuscript']!r}: kept, not used: {out['reasons'][0]}")
            continue
        lines, images = [], []
        for l in about["lines"]:
            rec = {"image": l["image"], "folio": l.get("folio", ""), "block": l.get("block", ""), "reasons": []}
            png = d / l["image"]
            g, st = None, {}
            if png.exists() and ".." not in l["image"]:
                g, st, why = LQ.check_image(png.read_bytes())
                rec["reasons"] += why
            else:
                rec["reasons"].append("image: missing")
            t, why = LQ.check_text(l.get("text", ""))
            rec["reasons"] += why
            if g is not None and not any(x.startswith("image") for x in rec["reasons"]):
                rec["reasons"] += LQ.check_fit(st, t)
            px = st.pop("pixels", None)
            if px:
                if px in seen:
                    rec["reasons"].append(f"copy: the same image as {seen[px]}")
                else:
                    seen[px] = f"{r['id']}/{l['image']}"
            rec.update(text=t, proposed=LQ.normalise(l.get("proposed", "")), pack=str(l.get("model", "")).split(" ")[0], stats=st)
            lines.append(rec)
            images.append(g if g is not None and not any(x.startswith("image") for x in rec["reasons"]) else None)
        judged = {}
        for pack in sorted({x["pack"] for x in lines}):
            ks = [k for k, x in enumerate(lines) if x["pack"] == pack and images[k] is not None]
            if not ks:
                continue
            name, net, chars, src = _judge(pack)
            judged[pack] = {"judge": name, "model": src}
            for k, text in zip(ks, LQ.judge_readings(net, chars, [images[k] for k in ks])):
                x = lines[k]
                x["read"] = text
                if x["text"]:
                    x["reading"] = round(LQ.cer(text, x["text"]), 3)
                    if x["reading"] > LQ.LIMITS["reading_max"]:
                        x["reasons"].append(f"reading: the judge reads {text[:40]!r}, {100 * x['reading']:.0f} % away from the text")
                if name == pack and x["proposed"]:
                    x["as_read"] = round(LQ.cer(text, x["proposed"]), 3)
                    if x["as_read"] > LQ.LIMITS["as_read_max"]:
                        x["reasons"].append(f"as read: the pack that read it reads it {100 * x['as_read']:.0f} % otherwise than Manuscribe proposed")
        accepted = r.get("quality") == "accepted"
        v, why, summary = LQ.verdict(lines, accepted=accepted)
        state = "accepted" if accepted else v
        sides = _check_sides(r, d, about, seen) if about.get("sides") and r.get("wording") in SIDES else []
        summary.update(sides=len(sides), sides_usable=sum(x["usable"] for x in sides))
        out = {"checked": datetime.datetime.now().isoformat(timespec="seconds"), "verdict": state, "reasons": why, "summary": summary,
               "judges": judged, "limits": LQ.LIMITS, "lines": lines, "sides": sides}
        (d / "quality.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        good = state in ("ok", "accepted")
        r["quality"], r["usable"], r["usable_sides"] = state, summary["usable"] if good else 0, summary["sides_usable"] if good else 0
        if verbose:
            print(f"[quality] {r['id']} {r['manuscript']!r} ({r.get('script')}): {state}; {summary['usable']} of {summary['lines']} lines usable"
                  + (f", reading {100 * summary['reading_median']:.0f} % at the median" if summary["reading_median"] is not None else "")
                  + (f"; {summary['sides_usable']} of {len(sides)} folio sides borne out by their editor's counts" if sides else "")
                  + ("".join(f"\n    ! {w}" for w in why)))
            fails = {}
            for x in lines + sides:
                for w in x["reasons"]:
                    fails[w.split(":")[0]] = fails.get(w.split(":")[0], 0) + 1
            if fails:
                print("    failing: " + ", ".join(f"{n} on {k}" for k, n in sorted(fails.items())))
    _write(rows)
    return rows


def accept(cid, why=""):
    """the maintainer takes a quarantined package, having looked at its lines: its lines that failed only on the judge's
    reading are then used too; the others stay out"""
    rows = _rows()
    row = next((r for r in rows if r["id"] == cid), None)
    if not row or row["state"] != "registered":
        raise SystemExit(f"[contributions] no registered contribution {cid}")
    row["quality"] = "accepted"
    row["note"] = (row.get("note", "") + f"; accepted by the maintainer {datetime.date.today()}" + (f": {why}" if why else "")).strip("; ")
    _write(rows)
    quality(verbose=False)
    r = next(r for r in _rows() if r["id"] == cid)
    print(f"[contributions] {cid}: {r['quality']}, {r['usable']} line(s) usable")


def _gh(*args):
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"[contributions] gh {' '.join(args[:3])}: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout


def sync(inbox=INBOX):
    """bring this register up to the inbox's: its new packages fetched and opened, its withdrawals made here"""
    remote = _gh("api", f"repos/{inbox}/contents/register.tsv", "-H", "Accept: application/vnd.github.raw")
    lines = remote.splitlines()
    head = lines[0].split("\t")
    there = [dict(zip(head, l.split("\t"))) for l in lines[1:] if l.strip()]
    try:
        tag = _gh("release", "view", "-R", inbox, "--json", "tagName", "--jq", ".tagName").strip()
    except SystemExit:
        tag = ""
    here = {r["id"]: r for r in _rows()}
    priv = None
    got = gone = 0
    for r in there:
        mine = here.get(r["id"])
        if r["state"] == "withdrawn":
            if mine and mine["state"] != "withdrawn":
                withdraw(r["id"], f"in the inbox, {r['note']}")
                gone += 1
            continue
        if r["state"] != "registered" or mine:
            continue
        priv = priv or seal.load_private()
        with tempfile.TemporaryDirectory() as d:
            _gh("release", "download", r["store"], "-R", inbox, "-p", f"{r['id']}.zip", "-D", d)
            f = Path(d) / f"{r['id']}.zip"
            if hashlib.sha256(f.read_bytes()).hexdigest() != r["sha256"]:
                raise SystemExit(f"[contributions] {r['id']}: not the file the inbox registered (checksum)")
            add(f, came={"issue": r["issue"], "contributor": r["contributor"], "release": tag}, priv=priv)
        got += 1
    if got:
        quality()
    print(f"[contributions] synced with {inbox} ({tag or 'no data release yet'}): {got} package(s) added, {gone} withdrawn; "
          f"{len(allowed())} line(s) training may use")


def allowed(script=MODELS_SCRIPT):
    """the lines a recogniser of `script` may learn from"""
    out = []
    for r in _rows():
        if (r["state"] != "registered" or r["wording"] not in APPROVED or r.get("quality") not in ("ok", "accepted")
                or r.get("script") != script):
            continue
        q = json.loads((ROOT / r["id"] / "quality.json").read_text(encoding="utf-8"))
        for l in q["lines"]:
            png = ROOT / r["id"] / l["image"]
            if l["usable"] and png.exists():
                out.append({"manuscript": r["manuscript"], "png": png, "text": l["text"], "id": r["id"], "folio": l["folio"], "release": r.get("release", "")})
    return out


def allowed_sides(script=MODELS_SCRIPT):
    """the folio sides a segmenter of `script` may learn from: registered on a wording that gives sides, approved, of a
    package checked ok or accepted, borne out by their editor's counts -> [{manuscript, image, target, id, folio, release}]"""
    out = []
    for r in _rows():
        if (r["state"] != "registered" or r["wording"] not in SIDES or r.get("quality") not in ("ok", "accepted")
                or r.get("script") != script):
            continue
        q = json.loads((ROOT / r["id"] / "quality.json").read_text(encoding="utf-8"))
        for x in q.get("sides") or []:
            if x.get("usable") and (ROOT / r["id"] / x["target"]).exists():
                out.append({"manuscript": r["manuscript"], "image": ROOT / r["id"] / x["image"], "target": ROOT / r["id"] / x["target"],
                            "id": r["id"], "folio": x["folio"], "release": r.get("release", "")})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync"); s.add_argument("--inbox", default=INBOX)
    a = sub.add_parser("add"); a.add_argument("package"); a.add_argument("--accept-wording", action="append", default=[])
    sub.add_parser("list")
    w = sub.add_parser("withdraw"); w.add_argument("id"); w.add_argument("--why", default="")
    sub.add_parser("quality")
    c = sub.add_parser("accept"); c.add_argument("id"); c.add_argument("--why", default="")
    args = ap.parse_args()
    if args.cmd == "sync":
        sync(args.inbox)
    elif args.cmd == "quality":
        quality()
    elif args.cmd == "accept":
        accept(args.id, args.why)
    elif args.cmd == "add":
        add(args.package, args.accept_wording)
        quality()
    elif args.cmd == "withdraw":
        withdraw(args.id, args.why)
    else:
        rows = _rows()
        for r in rows:
            print("\t".join(str(r.get(k, "")) for k in FIELDS))
        print(f"[contributions] {len(rows)} package(s); {len(allowed())} line(s) training may use (approved wordings: {', '.join(sorted(APPROVED)) or 'none yet'})")


if __name__ == "__main__":
    sys.exit(main())
