"""Paths and the registry of manuscripts that have ground truth."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the repository's folder (manuscript_ocr/ here)
CORPUS = ROOT.parent                                   # lexicon_manuscripts/
DATA = ROOT / "data"                                   # everything generated lives here
VENV_BIN = ROOT / ".venv" / "bin"
# Kraken's baseline segmentation model, named explicitly: the one bundled with kraken 7.1.1, which segmented every
# manuscript so far. kraken 7.2 bundles a different default model (Latin, Arabic and Greek only); passing this one keeps
# 7.2's output identical to 7.1.1's (checked on 7 pages, 2026-11-23) while its post-processing runs ~1.3x faster.
KRAKEN_SEG_MODEL = DATA / "kraken_models" / "blla_kraken7.1.1.mlmodel"
# Kraken's device: the Mac GPU (MPS) on Apple silicon, else the CPU; MSOCR_KRAKEN_DEVICE=cpu (or mps) overrides. With
# kraken 7.2 the GPU is ~3x faster than one CPU process and light on memory (2026-11-24: 4.4 s a page against 14.2 s,
# no swap). Its lines differ from the CPU's by a pixel at most and give the same rows (tools/kraken_mps_test.py), but
# are not byte-identical: segmentations made before 2026-11-24 came from the CPU.
KRAKEN_DEVICE = os.environ.get("MSOCR_KRAKEN_DEVICE") or (
    "mps" if __import__("sys").platform == "darwin" and __import__("platform").machine() == "arm64" else "cpu")
# Which segmentation the tools read: "seg" (Kraken's, data/<ms>/seg/) unless MSOCR_SEG says otherwise,
# e.g. MSOCR_SEG=seg_torch for our own PyTorch segmenter (msocr/tseg.py, data/<ms>/seg_torch/).
SEG = os.environ.get("MSOCR_SEG", "seg")

def _image_name(num):
    # <pb n="0002_2"/>  ->  085885_00002_2.jpg
    return lambda pb: "{}_0{}_{}.jpg".format(num, *pb.split("_"))


def _discover():
    """Every manuscript folder in the corpus: <text>/<commentary>/manuscripts/LDII_<n>/ (Lālabhāī Dalapatabhāī
    Institute, Ahmedabad) or AKGM_<n>/ (Kailāsasāgarasūri Jñānamandira, Kobā). Page images are named
    <6-digit number>_<5-digit folio>_<side>.jpg."""
    found = {}
    for sub in ("manuscripts", "manuscript"):
        for d in sorted(p for pat in ("LDII_*", "AKGM_*") for p in CORPUS.glob(f"*/*/{sub}/{pat}")):
            if d.is_dir():
                found[d.name] = {"images": d, "image_name": _image_name(f"{int(d.name.split('_')[1]):06d}"),
                                 "text": d.parts[-4], "commentary": d.parts[-3]}
    return found


MANUSCRIPTS = _discover()
# manuscripts with a transcription (used for training from the TEI)
MANUSCRIPTS["LDII_85885"]["tei"] = CORPUS / "abhidhanachintamani_commentaries/charitrasimha/transcriptions/LDII_85885.xml"


def not_text(ms_id: str) -> set:
    """Image stems of the pages that are not text pages, and are neither read, checked nor learnt from: the cover
    (folio 0, by the corpus's numbering) and every page listed in data/<ms>/not_text.txt (one page id like 0001_1,
    or an image stem, per line; '#' starts a comment). The review's "not a text page" verdict adds to that file."""
    ms = MANUSCRIPTS[ms_id]
    stems = {p.stem for p in ms["images"].glob("*.jpg") if p.stem.split("_")[-2].lstrip("0") == ""}
    f = DATA / ms_id / "not_text.txt"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            page = line.split("#", 1)[0].strip()
            if page:
                stems.add(page if page.count("_") >= 2 else ms["image_name"](page).rsplit(".", 1)[0])
    return stems


def ms_dir(ms_id: str) -> Path:
    d = DATA / ms_id
    d.mkdir(parents=True, exist_ok=True)
    return d
