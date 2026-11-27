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
# Which segmentation the tools make and read: "seg_torch", our own U-Net (msocr/tseg.py, data/<ms>/seg_torch/), the
# default since 2026-11-24 (tseg1: baselines 0.974 F1 against Kraken's, the same rows on every linear test page, 3.28 %
# of characters wrong against Kraken's 3.38 %, 0.13 s a page against 4-10 s; on the 50 reviewed tripāṭha and
# pañcapāṭha pages 27 split right against 30); "seg", Kraken's (data/<ms>/seg/); or another folder of segmentations in
# Kraken's format (seg_repaired). MSOCR_SEG chooses. A page whose line images were cut already keeps the segmentation
# they were cut from (segment.page_seg), unless MSOCR_SEG is set, and a corrected page always keeps it.
SEG = os.environ.get("MSOCR_SEG", "seg_torch")
SEG_CHOSEN = "MSOCR_SEG" in os.environ      # set by hand: then it applies to pages cut before too (not corrected ones)
KRAKEN_SEG = "seg"

def _image_name(num):
    # <pb n="0002_2"/>  ->  085885_00002_2.jpg
    return lambda pb: "{}_0{}_{}.jpg".format(num, *pb.split("_"))


# Manuscripts from other libraries, which come with a transcription (Muktabodha's e-texts, made into TEI and page
# images by msocr/muktabodha.py): training_data/<text>/<commentary>/manuscripts/<library>_<n>/ beside the corpus, the
# transcription at training_data/<text>/<commentary>/transcriptions/<id>.xml, registered with it when it is there.
TRAINING = CORPUS.parent / "training_data"
LIBRARIES = ("LDII", "AKGM", "RTJ", "UPSS")     # LDII and AKGM as below; RTJ = Shri Raghunath Temple MSS Library, Jammu;
                                                # UPSS = Uttar Pradesh Sanskrit Sansthan, Lucknow


def _discover():
    """Every manuscript folder in the corpus: <text>/<commentary>/manuscripts/LDII_<n>/ (Lālabhāī Dalapatabhāī
    Institute, Ahmedabad) or AKGM_<n>/ (Kailāsasāgarasūri Jñānamandira, Kobā), and in training_data (any library in
    LIBRARIES, its transcription beside it). Page images are named <6-digit number>_<5-digit folio>_<side>.jpg."""
    found = {}
    for root in (CORPUS, TRAINING):
        for sub in ("manuscripts", "manuscript"):
            for d in sorted(p for lib in LIBRARIES for p in root.glob(f"*/*/{sub}/{lib}_*")):
                if d.is_dir():
                    found[d.name] = {"images": d, "image_name": _image_name(f"{int(d.name.split('_')[1]):06d}"),
                                     "text": d.parts[-4], "commentary": d.parts[-3]}
                    tei = d.parent.parent / "transcriptions" / f"{d.name}.xml"
                    if root == TRAINING and tei.exists():
                        found[d.name]["tei"] = tei
    return found


MANUSCRIPTS = _discover()
# manuscripts with a transcription (used for training from the TEI): the editor's own git-tracked file, in the editions
# folder beside the corpus (edit_manuscripts/<text>/<commentary>/transcriptions/), not the corpus's copy, which is an older
# snapshot (2026-10-10). MSOCR_TEI_LDII_85885 names another file. After it changes: gt.extract, align (seg and
# seg_torch) and the line caches again (README, "The transcription").
EDITIONS = CORPUS.parent / "edit_manuscripts"
MANUSCRIPTS["LDII_85885"]["tei"] = Path(os.environ.get("MSOCR_TEI_LDII_85885") or
                                        EDITIONS / "abhidhanachintamani_commentaries/charitrasimha/transcriptions/LDII_85885.xml")


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
