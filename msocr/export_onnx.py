"""The export contract: a model pack for Manuscribe (and anything else reading the reference pipeline's networks).

    .venv-onnx/bin/python -m msocr.export_onnx [--seg tseg1] [--rec data/pool/models/pool2/best.pt] [--spacer full_e10_s1]

-> data/model_pack/<pack>/ (pack = pack-<date>-p<pipeline version>-s<spacer version>; a second pack made the same day is
   pack-<date>b-..., a third pack-<date>c-..., so that no pack a browser may keep is ever replaced under its own name;
   --overwrite makes the day's first again):
     segmenter.onnx, recogniser.onnx, spacer.onnx    the networks, opset 17, inputs named "x", outputs "y"
     pipeline.json, spacer.json                      the parameters the stages between the networks use
     recogniser_chars.json                           the recogniser's characters (class k + 1 is chars[k]; 0 is blank)
     lexica.bin.gz, own.bin.gz                       the language model's two parts (msocr/lm.py; --lm names the build,
                                                     the newest under data/lm by default, 'none' leaves it out)
     manifest.json                                   what each file is: source, sha256, inputs and outputs, the largest
                                                     difference from PyTorch on a test input, the versions, the licence
The pack is the unit Manuscribe fetches; fixtures (msocr.fixtures) are made with it and name it. Needs onnx and
onnxruntime (.venv-onnx; see tools/browser_bench/export.py).
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from .config import DATA
from .params import P, VERSION as PIPELINE_VERSION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACER = os.path.abspath(os.path.join(ROOT, "..", "sanskrit_spacer"))
LICENCE = "CC BY-SA 4.0"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def export(net, x, path, dyn):
    import onnx
    import onnxruntime as ort
    torch.onnx.export(net, x, path, input_names=["x"], output_names=["y"], dynamic_axes=dyn, opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    with torch.no_grad():
        ref = net(x).numpy()
    got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"x": x.numpy()})[0]
    return float(np.abs(ref - got).max())


def export_dynamic(net, path, max_len):
    """The spacer with its batch and its length both free (1 to max_len): the newer exporter (torch.export; needs
    onnxscript), as the older one fixes the traced length in the attention's reshapes. Weights inside the file. Checked
    against PyTorch at several lengths, and unpadded against padded to max_len. -> the largest difference"""
    import onnx
    import onnxruntime as ort
    x = torch.randint(2, 54, (2, 40))
    torch.onnx.export(net, (x,), path, input_names=["x"], output_names=["y"], opset_version=18, dynamo=True, external_data=False,
                      dynamic_shapes={"ids": {0: torch.export.Dim("n", min=1, max=64), 1: torch.export.Dim("L", min=1, max=max_len)}})
    onnx.checker.check_model(onnx.load(path))
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    worst = 0.0
    g = torch.Generator().manual_seed(20261124)
    for n, L in ((1, 1), (1, 7), (1, 50), (3, 120), (1, max_len), (16, max_len)):
        ids = torch.randint(2, 54, (n, L), generator=g)
        with torch.no_grad():
            ref = net(ids).numpy()
        worst = max(worst, float(np.abs(ref - s.run(None, {"x": ids.numpy()})[0]).max()))
    ids = torch.randint(2, 54, (1, 30), generator=g)
    pad = torch.zeros(1, max_len, dtype=torch.long)
    pad[0, :30] = ids
    a, b = s.run(None, {"x": ids.numpy()})[0][0], s.run(None, {"x": pad.numpy()})[0][0, :30]
    return max(worst, float(np.abs(a - b).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="tseg1")
    ap.add_argument("--rec", default=str(DATA / "pool" / "models" / "pool2" / "best.pt"))
    ap.add_argument("--spacer", default="full_e10_s1")
    ap.add_argument("--overwrite", action="store_true", help="make the day's pack again under its own name")
    ap.add_argument("--lm", help="the language model build to put in the pack (data/lm/<build>; default the newest; 'none' for none)")
    a = ap.parse_args()
    sys.path.insert(0, SPACER)
    spacer_spec = json.load(open(os.path.join(SPACER, "spacer.json"), encoding="utf-8"))
    day, k = f"{datetime.date.today():%Y%m%d}", 0
    name = f"pack-{day}-p{PIPELINE_VERSION}-s{spacer_spec['version']}"
    while (DATA / "model_pack" / name).exists() and not a.overwrite:
        k += 1
        name = f"pack-{day}{chr(ord('a') + k)}-p{PIPELINE_VERSION}-s{spacer_spec['version']}"
    out = DATA / "model_pack" / name
    out.mkdir(parents=True, exist_ok=True)

    from .tseg import load_net
    from .recognizer import load_model as load_rec
    seg = load_net(str(DATA / "tseg" / "models" / a.seg / "best.pt"))
    seg = (seg[0] if isinstance(seg, tuple) else seg).float().eval().to("cpu")
    rec, chars, rec_ck = load_rec(a.rec)
    rec.eval()
    cwd = os.getcwd()
    os.chdir(SPACER)
    from src.model import load_model as load_spacer
    spacer = load_spacer(os.path.join("runs", a.spacer), "cpu").eval()
    os.chdir(cwd)

    models = {}
    h = P["segmenter"]["img_h"]
    d = export(seg, torch.rand(1, 1, h, 1456), str(out / "segmenter.onnx"), {"x": {0: "n", 3: "w"}, "y": {0: "n", 3: "w"}})
    models["segmenter"] = {
        "file": "segmenter.onnx", "source": f"data/tseg/models/{a.seg}/best.pt", "max_diff_from_pytorch": d,
        "input": {"name": "x", "dtype": "float32", "shape": [1, 1, h, "w"],
                  "meaning": f"the page in grey, ink high (1 - grey/255), scaled to height {h} with the resize of pipeline.json, padded with 0 to a width that is a multiple of {P['segmenter']['pad_to']}"},
        "output": {"name": "y", "dtype": "float32", "shape": [1, 3, h, "w"],
                   "meaning": f"channel 0: baseline logit; 1 and 2: distance up and down from the baseline, divided by {P['segmenter']['norm']}"}}
    d = export(rec, torch.rand(1, 1, P["recogniser"]["height"], 1800), str(out / "recogniser.onnx"), {"x": {0: "n", 3: "w"}, "y": {0: "n", 1: "t"}})
    models["recogniser"] = {
        "file": "recogniser.onnx", "source": os.path.relpath(a.rec, ROOT), "max_diff_from_pytorch": d,
        "pass": rec_ck.get("pass"), "val_cer": rec_ck.get("val_cer"), "chars": "recogniser_chars.json", "classes": len(chars) + 1,
        "input": {"name": "x", "dtype": "float32", "shape": [1, 1, P["recogniser"]["height"], "w"],
                  "meaning": f"a line image in grey, ink high, scaled to height {P['recogniser']['height']} (width at least {2 * P['recogniser']['stride']})"},
        "output": {"name": "y", "dtype": "float32", "shape": [1, "t", len(chars) + 1],
                   "meaning": f"logits per column (one for every {P['recogniser']['stride']} input columns); class 0 is the CTC blank, class k + 1 is chars[k]"}}
    # any length up to max_len: a window costs what its letters cost (a page's text took 5 s on one browser thread
    # with every window padded to max_len, most of it padding)
    d = export_dynamic(spacer, str(out / "spacer.onnx"), spacer_spec["max_len"])
    models["spacer"] = {
        "file": "spacer.onnx", "source": f"sanskrit_spacer/runs/{a.spacer}/best.pt", "max_diff_from_pytorch": d,
        "threshold": spacer_spec["threshold"],
        "max_len": spacer_spec["max_len"],
        "input": {"name": "x", "dtype": "int64", "shape": ["n", "L"], "meaning": f"letter ids: symbol i of the alphabet in spacer.json is i + 2, 1 unknown; L from 1 to {spacer_spec['max_len']}, no padding needed (a batch of windows of different lengths is padded with 0 to the longest: padding is masked)"},
        "output": {"name": "y", "dtype": "float32", "shape": ["n", "L"], "meaning": "logit of a word space after each letter; those of any padding are ignored"}}
    json.dump(chars, open(out / "recogniser_chars.json", "w", encoding="utf-8"), ensure_ascii=False)
    # the language model: its two files and what they hold, mixed as lm.json weighs them (pipeline.json's weights)
    from .lm import latest as latest_lm
    lm = None
    lm_dir = None if a.lm in ("none", "no", "off") else (Path(a.lm) if a.lm else latest_lm())
    if lm_dir:
        spec = json.load(open(lm_dir / "lm.json", encoding="utf-8"))
        comps = []
        for c in spec["components"]:
            shutil.copy(lm_dir / c["file"], out / c["file"])
            comps.append(dict(c, sha256=sha256(out / c["file"]), bytes=os.path.getsize(out / c["file"]), licence=LICENCE))
        lm = {"source": os.path.relpath(lm_dir, ROOT), "built": spec["built"], "order": spec["order"], "components": comps, "decoding": spec["decoding"],
              "meaning": "character n-grams over the letters of the Devanagari block (daṇḍas, digits and marks one boundary symbol), mixed by weight; lines are read by a prefix beam search over the recogniser's probabilities with pipeline.json's lm parameters (msocr/lm.py, Manuscribe's recoglm.js)"}
        parts = ", ".join("{} {:.1f} MB".format(c["name"], c["bytes"] / 1e6) for c in comps)
        print(f"  language model: {lm_dir.name} ({parts})")
    else:
        print("  language model: none (lines will be read greedily)")
    shutil.copy(os.path.join(ROOT, "msocr", "pipeline.json"), out / "pipeline.json")
    shutil.copy(os.path.join(SPACER, "spacer.json"), out / "spacer.json")
    for m in models.values():
        m["sha256"] = sha256(out / m["file"])
        m["bytes"] = os.path.getsize(out / m["file"])
        m["licence"] = LICENCE
    manifest = {
        "pack": name, "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "pipeline_version": PIPELINE_VERSION, "spacer_version": spacer_spec["version"],
        "runtime": "ONNX Runtime (Web 1.31.1 measured); opset 17",
        "models": models,
        "lm": lm,
        "files": {f: sha256(out / f) for f in ("pipeline.json", "spacer.json", "recogniser_chars.json")},
        "licence": f"weights {LICENCE}; training data: the developer's own manuscripts and, for the spacer, CC BY-SA/CC0 texts (Pramāṇa)",
    }
    json.dump(manifest, open(out / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k, m in models.items():
        print(f"{k:10s} {m['bytes'] / 1e6:5.1f} MB  max |torch - onnx| {m['max_diff_from_pytorch']:.1e}  sha256 {m['sha256'][:12]}")
    print(f"[export_onnx] {out}")


if __name__ == "__main__":
    main()
