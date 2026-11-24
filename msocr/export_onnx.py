"""The export contract: a model pack for Manuscribe (and anything else reading the reference pipeline's networks).

    .venv-onnx/bin/python -m msocr.export_onnx [--seg tseg1] [--rec data/LDII_19634/models/ft4/best.pt] [--spacer full_e10_s1]

-> data/model_pack/<pack>/ (pack = pack-<date>-p<pipeline version>-s<spacer version>):
     segmenter.onnx, recogniser.onnx, spacer.onnx    the networks, opset 17, inputs named "x", outputs "y"
     pipeline.json, spacer.json                      the parameters the stages between the networks use
     recogniser_chars.json                           the recogniser's characters (class k + 1 is chars[k]; 0 is blank)
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
    ap.add_argument("--rec", default=str(DATA / "LDII_19634" / "models" / "ft4" / "best.pt"))
    ap.add_argument("--spacer", default="full_e10_s1")
    a = ap.parse_args()
    sys.path.insert(0, SPACER)
    spacer_spec = json.load(open(os.path.join(SPACER, "spacer.json"), encoding="utf-8"))
    name = f"pack-{datetime.date.today():%Y%m%d}-p{PIPELINE_VERSION}-s{spacer_spec['version']}"
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
        "files": {f: sha256(out / f) for f in ("pipeline.json", "spacer.json", "recogniser_chars.json")},
        "licence": f"weights {LICENCE}; training data: the developer's own manuscripts and, for the spacer, CC BY-SA/CC0 texts (Pramāṇa)",
    }
    json.dump(manifest, open(out / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k, m in models.items():
        print(f"{k:10s} {m['bytes'] / 1e6:5.1f} MB  max |torch - onnx| {m['max_diff_from_pytorch']:.1e}  sha256 {m['sha256'][:12]}")
    print(f"[export_onnx] {out}")


if __name__ == "__main__":
    main()
