"""The recognition site: the folder Manuscribe fetches its recognition pack from, published as its own repository
(manuscribe-models, on GitHub Pages: GitHub's release assets cannot be fetched by a browser, Pages can).

    .venv/bin/python -m msocr.site [--pack data/model_pack/<pack>] [--site ../manuscribe-models] [--ort <dist folder>]

-> <site>/  (default: manuscribe-models beside this repository's folder)
     index.json                      what Manuscribe reads first: {schema: 1, updated, latest, packs: [...], runtimes: {...}}
     packs/<pack>/                   the pack's files (msocr.export_onnx), each checked against its manifest first, and
                                     the manifest completed: every file's checksum and size, the runtime it needs, and a
                                     card per model saying what it learned from; MODEL_CARD.md, the same for people
     runtime/onnxruntime-web-<v>/    ONNX Runtime Web as published on npm (MIT): two variants, each a bundle and its
                                     WebAssembly: webgpu (ort.webgpu.bundle.min.mjs, ort-wasm-simd-threaded.asyncify.wasm)
                                     and wasm (ort.wasm.bundle.min.mjs, ort-wasm-simd-threaded.wasm); runtime.json
     contribute.json                 where contributed lines go and the key that seals them: {schema: 1, inbox (the GitHub
                                     repository whose issues take them), email, wordings (the consent wordings taken),
                                     key: {id, alg, spki}} (msocr.seal; the private key stays on this computer)
     README.md, LICENSE.md, .nojekyll
Packs already on the site are kept, and listed in index.json with the new one; the newest is `latest`. The runtime is
taken from Manuscribe's tests (tests/node_modules/onnxruntime-web, pinned there) unless --ort names another dist folder.
Nothing is published by this: the folder is pushed by hand.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import statistics
from pathlib import Path

from . import NAME, REPOSITORY, __version__, seal
from .config import DATA, MANUSCRIPTS

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT.parent / "manuscribe-models"
MANUSCRIBE = Path(os.environ.get("MANUSCRIBE", "/Users/Shared/drdhaval2785/edit_manuscripts/manuscript_editor"))
LICENCE = "CC BY-SA 4.0"
VARIANTS = {"webgpu": ("ort.webgpu.bundle.min.mjs", "ort-wasm-simd-threaded.asyncify.wasm"),
            "wasm": ("ort.wasm.bundle.min.mjs", "ort-wasm-simd-threaded.wasm")}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _entry(path):
    return {"sha256": sha256(path), "bytes": os.path.getsize(path)}


# ---------------------------------------------------------------- what each model learned from

def _recogniser_lineage(source):
    """The recogniser's runs back to the first: [{run, manuscript, lines, contributions}] from each run's train.log, and
    the contributed packages each run's checkpoint says it learned from (recognizer.pool)."""
    out, seen = [], set()
    p = (ROOT / source) if not os.path.isabs(source) else Path(source)
    while p and p not in seen:
        seen.add(p)
        run_dir = p.parent
        log = run_dir / "train.log"
        ms = run_dir.parent.parent.name
        step = {"run": run_dir.name, "manuscript": ms}
        base = None
        if log.exists():
            text = log.read_text(encoding="utf-8", errors="replace")
            starts = re.findall(r"starting from (\S+)", text)
            base = Path(starts[-1]) if starts else None
            m = re.findall(r"training on \d+ \((\d+) lines\)", text) or re.findall(r"training lines\s*:\s*([\d,]+)", text)
            if m:
                step["lines"] = int(m[-1].replace(",", ""))
        if p.exists():
            import torch
            ck = torch.load(p, map_location="cpu")
            if ck.get("contributions"):
                step["contributions"] = ck["contributions"]
            if ck.get("lines"):
                step["lines_from"] = ck["lines"]          # seg: cut from Kraken's segmentation; seg_torch: the U-Net's
        out.append(step)
        p = base
    return out


def _segmenter_pages():
    """The segmenter's training pages as the split stands now (tseg.page_list), by manuscript, and the heights of
    the training images (the scale Manuscribe should render a side at)."""
    from . import tseg
    from PIL import Image
    pages = tseg.page_list()
    by_ms, heights = {}, []
    for ms, stem, img, js in pages["train"]:
        by_ms[ms] = by_ms.get(ms, 0) + 1
        try:
            with Image.open(img) as im:
                heights.append(im.size[1])
        except OSError:
            pass
    heights.sort()
    q = lambda f: heights[min(len(heights) - 1, int(f * len(heights)))]
    return by_ms, {"median": int(statistics.median(heights)), "p10": q(0.1), "p90": q(0.9), "pages": len(heights)}


def _cards(manifest):
    seg_src = manifest["models"]["segmenter"]["source"]
    run = Path(seg_src).parent.name
    log = (ROOT / Path(seg_src).parent / "train.log")
    head = log.read_text(encoding="utf-8").splitlines()[0] if log.exists() else ""
    by_ms, heights = _segmenter_pages()
    seg = {"run": run, "log": head.strip(), "manuscripts": by_ms,
           "note": "the training split as it stands at this pack's build (tseg.page_list); the run's own log gives its counts",
           "page_height": heights}
    seg_ck = ROOT / seg_src
    if seg_ck.exists():
        import torch
        seg["contributions"] = torch.load(seg_ck, map_location="cpu").get("contributions") or []
    rec = {"lineage": _recogniser_lineage(manifest["models"]["recogniser"]["source"])}
    got = {}
    for step in rec["lineage"]:
        for c in step.get("contributions", []):
            got.setdefault(c["id"], c)
    rec["contributions"] = list(got.values())
    sp = {"texts": "Pramāṇa's open release: Sanskrit Wikisource (CC BY-SA), SARIT (CC BY-SA), Ambuda (CC0)",
          "run": Path(manifest["models"]["spacer"]["source"]).parent.name}
    for k, c in (("segmenter", seg), ("recogniser", rec), ("spacer", sp)):
        c["licence"] = LICENCE
        c["consent"] = ("the developer's own manuscripts only; no contributed data" if k != "spacer"
                        else "openly licensed texts; no manuscript images")
    if seg.get("contributions"):
        seg["consent"] = (f"the developer's own manuscripts, and folio sides contributed from Manuscribe with their editors' consent: "
                          f"{sum(c['sides'] for c in seg['contributions'])} sides of {len({c['manuscript'] for c in seg['contributions']})} manuscript(s)")
    if rec["contributions"]:
        ms = sorted({c["manuscript"] for c in rec["contributions"]})
        rel = sorted({c.get("release") or "-" for c in rec["contributions"]})
        rec["consent"] = (f"the developer's own manuscripts, and lines contributed from Manuscribe with their editors' consent: "
                          f"{len(rec['contributions'])} package(s) of {len(ms)} manuscript(s), from data release(s) {', '.join(rel)}")
    return {"segmenter": seg, "recogniser": rec, "spacer": sp}


def _card_md(m):
    c = {k: m["models"][k]["card"] for k in ("segmenter", "recogniser", "spacer")}
    seg, rec, sp = c["segmenter"], c["recogniser"], c["spacer"]
    lines = [f"# {m['pack']}", "",
             f"A recognition pack for Manuscribe: the line segmenter, the line recogniser and the word spacer, with the "
             f"parameters of the stages between them (pipeline version {m['pipeline_version']}, spacer version "
             f"{m['spacer_version']}). Weights under {LICENCE}. Made with {NAME} "
             f"{m.get('software', {}).get(NAME, '')} ({REPOSITORY}).", "",
             "## Segmenter", "", f"Run `{seg['run']}`. {seg['log']}", "",
             f"Trained on pages of these manuscripts (pages each, {sum(seg['manuscripts'].values())} in all, as the training "
             f"split stands at this pack's build; the run's log above gives the split it was trained on):", ""]
    lines += [f"- {ms}: {n}" for ms, n in sorted(seg["manuscripts"].items())]
    h = seg["page_height"]
    lines += ["", f"Training images are {h['p10']}–{h['p90']} px high (median {h['median']}); Manuscribe renders a side "
              f"at about the median.", "", "## Recogniser", "",
              f"Characters: {m['models']['recogniser']['classes'] - 1}. Trained, run after run:", ""]
    for s in reversed(rec["lineage"]):
        lines.append(f"- `{s['run']}` on {s['manuscript']}" + (f", {s['lines']} lines" if s.get("lines") else "")
                     + (", cut from the U-Net's segmentation as Manuscribe cuts them" if s.get("lines_from") == "seg_torch" else ""))
    lines += ["", "It reads the hands it learned from well and other hands poorly: a model pooled over many hands is "
              "to replace it.", "", "## Word spacer", "", f"Run `{sp['run']}`, trained on {sp['texts']}. Threshold "
              f"{m['models']['spacer']['threshold']}.", "", "## Consent", ""]
    if seg.get("contributions"):
        lines += ["The segmenter learned also from folio sides contributed from Manuscribe, each manuscript's given by its editor "
                  "with consent, and used only where its lines as the segmenter found them bore out its editor's line counts "
                  "in every block (msocr.sidequality):", ""]
        lines += [f"- {c['manuscript']}: {c['sides']} sides (package `{c['id']}`)" for c in sorted(seg["contributions"], key=lambda c: (c["manuscript"], c["id"]))]
        lines += [""]
    if rec.get("contributions"):
        lines += [("The segmenter learned from the developer's own manuscripts only. " if not seg.get("contributions") else "")
                  + "The recogniser learned also from lines "
                  "contributed from Manuscribe, each manuscript's given by its editor with consent to train the models, and "
                  "checked before training (msocr.linequality):", ""]
        lines += [f"- {c['manuscript']}: {c.get('trained', c.get('lines'))} lines (package `{c['id']}`, data release "
                  f"`{c.get('release') or '-'}`)" for c in sorted(rec["contributions"], key=lambda c: (c["manuscript"], c["id"]))]
        lines += ["", "A contribution withdrawn later is left out of models trained after its withdrawal; this model, trained "
                  "while its consent stood, remains as it is.", ""]
    elif not seg.get("contributions"):
        lines += ["No contributed data: the segmenter and the recogniser learned from the developer's own manuscripts only.", ""]
    lines += [f"Language and script: {m.get('script', 'sa-Deva')} only. Lines and pages of another language or script are never "
              "mixed into these models' training.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- the site

def add_pack(pack_dir, site):
    pack_dir = Path(pack_dir)
    m = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    for k, mod in m["models"].items():
        if sha256(pack_dir / mod["file"]) != mod["sha256"]:
            raise SystemExit(f"{pack_dir / mod['file']}: not the file its manifest names")
    for name, h in m["files"].items():
        if sha256(pack_dir / name) != (h if isinstance(h, str) else h["sha256"]):
            raise SystemExit(f"{pack_dir / name}: not the file its manifest names")
    out = site / "packs" / m["pack"]
    out.mkdir(parents=True, exist_ok=True)
    for mod in m["models"].values():
        shutil.copy2(pack_dir / mod["file"], out / mod["file"])
    for name in m["files"]:
        shutil.copy2(pack_dir / name, out / name)
    m["files"] = {name: _entry(out / name) for name in m["files"]}
    m["needs"] = {"runtime": f"onnxruntime-web-{RUNTIME_VERSION}", "pipeline_version": m["pipeline_version"], "spacer_version": m["spacer_version"]}
    cards = _cards(m)
    for k in cards:
        m["models"][k]["card"] = cards[k]
    m["page_height"] = cards["segmenter"]["page_height"]
    m["software"] = {NAME: __version__, "repository": REPOSITORY}
    from .contributions import MODELS_SCRIPT
    m["script"] = MODELS_SCRIPT          # the language and script its models read (BCP 47): Manuscribe asks before reading another's
    (out / "MODEL_CARD.md").write_text(_card_md(m), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    return m


RUNTIME_VERSION = None


def add_runtime(dist, site):
    global RUNTIME_VERSION
    dist = Path(dist)
    pkg = json.loads((dist.parent / "package.json").read_text(encoding="utf-8"))
    RUNTIME_VERSION = pkg["version"]
    rid = f"onnxruntime-web-{RUNTIME_VERSION}"
    out = site / "runtime" / rid
    out.mkdir(parents=True, exist_ok=True)
    variants = {}
    for v, (js, wasm) in VARIANTS.items():
        for f in (js, wasm):
            shutil.copy2(dist / f, out / f)
        variants[v] = {"js": dict(file=js, **_entry(out / js)), "wasm": dict(file=wasm, **_entry(out / wasm))}
    lic = dist.parent / "LICENSE"
    if lic.exists():
        shutil.copy2(lic, out / "LICENSE")
    rt = {"id": rid, "name": "onnxruntime-web", "version": RUNTIME_VERSION, "licence": pkg.get("license", "MIT"),
          "source": f"npm onnxruntime-web@{RUNTIME_VERSION}, files of dist/ unchanged", "variants": variants}
    (out / "runtime.json").write_text(json.dumps(rt, indent=1), encoding="utf-8")
    return rt


def write_index(site):
    packs = []
    for mf in sorted((site / "packs").glob("*/manifest.json")):
        m = json.loads(mf.read_text(encoding="utf-8"))
        files = [mod["bytes"] for mod in m["models"].values()] + [h["bytes"] for h in m["files"].values()]
        packs.append({"id": m["pack"], "manifest": f"packs/{m['pack']}/manifest.json", "created": m["created"],
                      "pipeline_version": m["pipeline_version"], "spacer_version": m["spacer_version"],
                      "needs": m.get("needs", {}), "bytes": sum(files),
                      "models": {k: Path(mod["source"]).parent.name for k, mod in m["models"].items()},
                      "licence": LICENCE})
    runtimes = {}
    for rf in sorted((site / "runtime").glob("*/runtime.json")):
        r = json.loads(rf.read_text(encoding="utf-8"))
        runtimes[r["id"]] = {"path": f"runtime/{r['id']}/", "version": r["version"], "licence": r["licence"], "variants": r["variants"]}
    packs.sort(key=lambda p: p["created"])
    idx = {"schema": 1, "updated": datetime.datetime.now().isoformat(timespec="seconds"), "software": {NAME: __version__, "repository": REPOSITORY},
           "latest": packs[-1]["id"] if packs else None, "packs": packs, "runtimes": runtimes}
    (site / "index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
    return idx


def write_contribute(site, inbox, email):
    """contribute.json: where Manuscribe sends contributed lines, and the public key it seals them to"""
    from .contributions import APPROVED, MODELS_SCRIPT, SIDES
    c = {"schema": 1, "inbox": inbox, "email": email, "wordings": sorted(APPROVED), "sides": sorted(SIDES), "script": MODELS_SCRIPT,
         "key": seal.public_info(seal.load_public())}
    (site / "contribute.json").write_text(json.dumps(c, indent=1), encoding="utf-8")
    return c


README = """# Manuscribe's recognition models

The files Manuscribe fetches to read manuscript pages in the browser: a line segmenter, a line recogniser for
Devanagari manuscripts and a word spacer for Sanskrit, in ONNX, with the parameters of the steps between them, and
ONNX Runtime Web to run them. Manuscribe (Settings › Recognition) reads `index.json`, fetches the newest pack it can
read and the runtime the browser needs, checks every file's SHA-256 against the manifest, and keeps them in the
browser; recognition then runs on the device, offline, and nothing is sent anywhere.

- `index.json`: the packs and runtimes on this site.
- `packs/<pack>/`: a pack, its `manifest.json` (every file's checksum, the inputs and outputs of each network, what
  each model learned from) and `MODEL_CARD.md`.
- `runtime/onnxruntime-web-<version>/`: ONNX Runtime Web's files as published on npm, unchanged (MIT licence).
- `contribute.json`: where Manuscribe sends corrected lines that their editors agree to contribute for training (an
  issue in the inbox repository it names), and the public key it seals them to, so that only the maintainer can open
  them.

The packs are made by [devanagari-manuscript-ocr](@REPOSITORY@) (`python -m msocr.export_onnx`, then
`python -m msocr.site`); its reference pipeline in Python and Manuscribe's JavaScript are held to give the same output
by stage fixtures. Made with devanagari-manuscript-ocr @VERSION@, for Manuscribe @VERSION@.

Licences: the models' weights are under CC BY-SA 4.0 (see LICENSE.md); ONNX Runtime Web under MIT
(runtime/*/LICENSE).
"""

LICENSE_MD = """# Licences

The models' weights in `packs/` (the segmenter, the recogniser and the word spacer, and the parameter files beside
them) are licensed under the Creative Commons Attribution-ShareAlike 4.0 International licence (CC BY-SA 4.0):
https://creativecommons.org/licenses/by-sa/4.0/. Each pack's `MODEL_CARD.md` says what each model learned from; the
word spacer learned from openly licensed texts (Sanskrit Wikisource and SARIT, CC BY-SA; Ambuda, CC0) through
Pramāṇa's open release.

ONNX Runtime Web, in `runtime/`, is Microsoft's, under the MIT licence: see the LICENSE file beside it.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=None, help="a pack folder made by msocr.export_onnx (default: the newest)")
    ap.add_argument("--site", default=str(SITE))
    ap.add_argument("--ort", default=str(MANUSCRIBE / "tests" / "node_modules" / "onnxruntime-web" / "dist"))
    ap.add_argument("--inbox", default="drdhaval2785/manuscribe-contributions", help="the GitHub repository whose issues take contributed lines")
    ap.add_argument("--email", default="drdhaval2785@gmail.com", help="the maintainer's address, for those without a GitHub account")
    a = ap.parse_args()
    site = Path(a.site)
    site.mkdir(parents=True, exist_ok=True)
    pack = a.pack or str(sorted((DATA / "model_pack").glob("pack-*"))[-1])
    rt = add_runtime(a.ort, site)
    m = add_pack(pack, site)
    idx = write_index(site)
    if seal.KEY.exists() or seal.KEY.with_suffix(".pub").exists():
        c = write_contribute(site, a.inbox, a.email)
        print(f"[site] contribute.json: inbox {c['inbox']}, key {c['key']['id']}, wordings {', '.join(c['wordings'])}")
    else:
        print(f"[site] no contribute.json: no key at {seal.KEY} (python -m msocr.seal keygen)")
    (site / "README.md").write_text(README.replace("@VERSION@", __version__).replace("@REPOSITORY@", REPOSITORY), encoding="utf-8")
    (site / "LICENSE.md").write_text(LICENSE_MD, encoding="utf-8")
    (site / ".nojekyll").write_text("", encoding="utf-8")
    total = sum(f.stat().st_size for f in site.rglob("*") if f.is_file())
    print(f"[site] {site}: pack {m['pack']} added; {len(idx['packs'])} pack(s), runtime {rt['id']}; {total / 1e6:.1f} MB in all")
    print(f"       page height of the segmenter's training images: median {m['page_height']['median']} px "
          f"({m['page_height']['p10']}-{m['page_height']['p90']})")


if __name__ == "__main__":
    main()
