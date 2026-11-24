"""Export the three networks of the browser workflow to ONNX (opset 17) in tools/browser_bench/models/, check each
against PyTorch, and time each with native ONNX Runtime on one thread and on all threads.

  segmenter   our U-Net, data/tseg/models/<run>/best.pt (default tseg1)
  recogniser  a CRNN-CTC line recogniser (default data/LDII_19634/models/ft4/best.pt)
  spacer      the word spacer, sanskrit_spacer/runs/<run>/best.pt (default full_e10_s1)

Needs onnx and onnxruntime, which the pipeline's .venv does not have (kept apart from kraken's pinned packages):
    /opt/homebrew/bin/python3.11 -m venv .venv-onnx && .venv-onnx/bin/pip install torch onnx onnxruntime pillow numpy rapidfuzz lxml
    .venv-onnx/bin/python tools/browser_bench/export.py [--seg tseg1] [--rec path] [--spacer full_e10_s1]
Then the browser timings: node tools/browser_bench/run.js
"""
import argparse, os, statistics, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OCR = os.path.abspath(os.path.join(HERE, "..", ".."))
SPACER = os.path.abspath(os.path.join(OCR, "..", "sanskrit_spacer"))
OUT = os.path.join(HERE, "models")


def export(net, x, name, dyn):
    import onnx
    path = os.path.join(OUT, f"{name}.onnx")
    torch.onnx.export(net, x, path, input_names=["x"], output_names=["y"], dynamic_axes=dyn, opset_version=17, dynamo=False)
    m = onnx.load(path)
    onnx.checker.check_model(m)
    return path, sorted({n.op_type for n in m.graph.node})


def bench(path, x, threads, reps):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    s.run(None, {"x": x})
    ts = []
    for _ in range(reps):
        t = time.time()
        s.run(None, {"x": x})
        ts.append(time.time() - t)
    return statistics.median(ts)


def check(net, x, path):
    import onnxruntime as ort
    with torch.no_grad():
        ref = net(x).numpy()
    got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"x": x.numpy()})[0]
    return float(np.abs(ref - got).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="tseg1")
    ap.add_argument("--rec", default=os.path.join(OCR, "data", "LDII_19634", "models", "ft4", "best.pt"))
    ap.add_argument("--spacer", default="full_e10_s1")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sys.path.insert(0, OCR)
    from msocr.tseg import load_net
    from msocr.recognizer import load_model as load_rec
    seg = load_net(os.path.join(OCR, "data", "tseg", "models", a.seg, "best.pt"))
    seg = (seg[0] if isinstance(seg, tuple) else seg).float().eval().to("cpu")
    rec = load_rec(a.rec)[0].eval()
    sys.path.insert(0, SPACER)
    cwd = os.getcwd()
    os.chdir(SPACER)
    from src.model import load_model as load_spacer
    spacer = load_spacer(os.path.join("runs", a.spacer), "cpu").eval()
    os.chdir(cwd)
    jobs = [("segmenter", seg, torch.rand(1, 1, 640, 1456), {"x": {0: "n", 2: "h", 3: "w"}, "y": {0: "n", 2: "h", 3: "w"}},
             f"one page 640x1456 ({a.seg})", 3),
            ("recogniser", rec, torch.rand(1, 1, 48, 1800), {"x": {0: "n", 3: "w"}, "y": {0: "n", 1: "t"}},
             f"one line 48x1800 ({os.path.basename(os.path.dirname(a.rec))})", 5),
            ("spacer", spacer, torch.randint(2, 54, (20, 120)), {"x": {0: "n", 1: "l"}, "y": {0: "n", 1: "l"}},
             f"20 sentences x 120 letters ({a.spacer})", 5)]
    for name, net, x, dyn, what, reps in jobs:
        path, ops = export(net, x, name, dyn)
        err = check(net, x, path)
        t1, tn = bench(path, x.numpy(), 1, reps), bench(path, x.numpy(), os.cpu_count(), reps)
        print(f"{name:10s} {what:42s} {os.path.getsize(path) / 1e6:5.1f} MB  max |torch-onnx| {err:.1e}  "
              f"1 thread {t1:6.3f} s  {os.cpu_count()} threads {tn:6.3f} s")
        print(f"           ops: {', '.join(ops)}")


if __name__ == "__main__":
    main()
