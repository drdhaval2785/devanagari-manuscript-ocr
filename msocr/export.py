"""Gather a manuscript's pages into one composite text file and one TEI file.

    python -m msocr.export LDII_19634
        writes data/LDII_19634/LDII_19634.txt   page marker ;p{0001_2}, then one line per
                                                 manuscript line, for every page image in order
               data/LDII_19634/LDII_19634.xml   the same text in the TEI layout of LDII_85885.xml
                                                 (<pb n="0001_2"/>, <lb n="1" break="no"/> ...)
    python -m msocr.export LDII_19634 --from data/LDII_19634/LDII_19634.txt
        rebuilds the TEI file from a composite text file that has been corrected by hand
    python -m msocr.export AKGM_0025383 --spaces
        the same, with word spaces put in by the newest model pack's word spacer, as Manuscribe puts them in
    python -m msocr.export LDII_19634 --model data/LDII_19634/models/ft3/best.pt [--out FILE]
        reads EVERY page afresh with that model and writes the reading alone, nothing corrected,
        as a TEI file (default data/LDII_19634/LDII_19634.machine.xml): the same page and line
        breaks as above, the model named in the header. Nothing else is touched, so the file can
        be made for each model in turn and kept under version control elsewhere; the difference
        between two of them is exactly what the training changed.

Word spaces (--spaces), as Manuscribe puts them in when it reads a side: each block of a page is spaced as one text,
its lines joined with nothing between them, by the word spacer of the newest pack in data/model_pack (its spacer.onnx
at its threshold; spacer_reference.py, beside this folder in sanskrit_spacer), and the spaces are given back to the
lines, a space that falls between two lines ending the first. A line that ends in a space or a daṇḍa ends its word, and
the next line opens with <lb n=".."/>; a line that does not runs on, and the next opens with <lb n=".." break="no"/>
(Manuscribe's rule, both ways). In the text file a line that ends a word keeps one space at its end, so that
--from FILE --spaces builds the same TEI again. A block with a space in it already (typed in a correction) keeps its
own spaces; nothing in corrected/, to_correct/ or drafts/ is changed. Without --spaces the text is written as read,
every line opening with break="no".

For each page the text comes from the first of: corrected/<page>.txt, to_correct/<page>.txt,
drafts/<page>.txt. Pages where no text was found (cover, blank sides) get a marker and no lines
in the text file, and are left out of the TEI body, as in LDII_85885.xml.
The TEI header is copied from LDII_85885.xml with the title and source changed; kāṇḍa divisions
are not known to the tool, so the body is one <div> to be split by hand.

A tripāṭha or pañcapāṭha manuscript has blocks: in the text file each block of a page follows a
;box{main} / ;box{top} ... marker (pagetext.py); in the TEI the file takes the form Manuscribe
writes and reads: a <div type="base-text"> holding every page's main text, then a
<div type="commentary"> holding every page's commentary with <cb n="top"/>, <cb n="right"/>,
<cb n="left"/>, <cb n="bottom"/> before each block, lines numbered from 1 in each block, every
page with its <pb/> in both streams, and the usual line count of each block in the header
("Lines per folio side: main text 8, top 5, ..."). Machine-read pages carry level="#ocr" on
their <pb/>, which Manuscribe shows as the OCR state and update.py takes as "not corrected".
"""
import argparse
import re
import sys
import time
import warnings
from pathlib import Path
from xml.sax.saxutils import escape

from . import pagetext
from .config import CORPUS, MANUSCRIPTS, ms_dir, not_text

# the header every exported TEI starts from (title and source replaced): the corpus's copy of LDII_85885.xml, whose
# header is the plain one this layout was made for. Not the transcription training reads (config), which is now a
# Manuscribe export whose <sourceDesc> describes LDII_85885 itself (siglum, layout) and would be carried into others.
REFERENCE_TEI = CORPUS / "abhidhanachintamani_commentaries/charitrasimha/transcriptions/LDII_85885.xml"
TITLES = {"LDII_19634": "हेमाभिधानचिंतामणिनाममाला - बालावबोध",
          "AKGM_0025383": "हेमाभिधानचिंतामणिनाममालादीपिका - तात्पर्यार्थप्रकाशिका (चारित्रसिंही)"}
# the library a manuscript comes from, by the prefix of its id
COLLECTIONS = {"LDII": "Lālabhāī Dalapatabhāī Institute of Indology, Ahmedabad",
               "AKGM": "Ācārya Śrī Kailāsasāgarasūri Jñānamandira, Śrī Mahāvīra Jaina Ārādhanā Kendra, Kobā",
               "RTJ": "Shri Raghunath Temple MSS Library, Jammu",
               "UPSS": "Uttar Pradesh Sanskrit Sansthan, Lucknow"}


def page_id(stem):
    """019634_00001_2 -> 0001_2 (the form used in <pb n=.../>)."""
    folio, side = stem.split("_")[-2:]
    return f"{int(folio):04d}_{side}"


def _page_text(d, stem):
    """-> (source folder or None, {box: [lines]})"""
    for sub in ("corrected", "to_correct", "drafts"):
        p = d / sub / f"{stem}.txt"
        if p.exists():
            return sub, pagetext.read(p)
    return None, {}


def gather(ms_id):
    """[(page id, source folder or None, {box: [lines]})] for every page image, in order."""
    d = ms_dir(ms_id)
    out = []
    skip = not_text(ms_id)
    for img in sorted(p.name for p in MANUSCRIPTS[ms_id]["images"].glob("*.jpg")):
        stem = img.rsplit(".", 1)[0]
        src, lines = (None, {}) if stem in skip else _page_text(d, stem)     # a cover or blank side: no text
        out.append((page_id(stem), src, lines))
    return out


def composite(pages):
    parts = []
    for pid, _, boxes in pages:
        parts.append(f";p{{{pid}}}\n")
        if boxes:
            parts.append(pagetext.format(boxes))
    return "".join(parts)


def parse_composite(text, keep_end=False):
    """keep_end: a spaced composite, whose lines that end a word end in a space"""
    pages, cur = [], None
    for raw in text.splitlines():
        m = re.fullmatch(r"\s*;p\{(\w+)\}\s*", raw)
        if m:
            cur = (m.group(1), "composite", [])
            pages.append(cur)
        elif raw.strip():
            if cur is None:
                raise ValueError("text before the first ;p{...} marker")
            cur[2].append(re.sub(r"\s+$", " ", raw) if keep_end else raw.rstrip())
    return [(pid, src, pagetext.parse("\n".join(lines), keep_end)) for pid, src, lines in pages]


# ---------------------------------------------------------------- word spaces

SPACER_DIR = Path(__file__).resolve().parent.parent.parent / "sanskrit_spacer"
DANDA = "|।॥"


def split_spaced(rows, spaced):
    """The spaced text of a block given back to its rows (Manuscribe's RecogPage.splitSpaced, step for step): a space
    falling between two rows ends the first; a row begins with no space and ends in one at most. -> [row text], or
    None if the letters do not match (the rows are then left as they were)."""
    out, pos = [], 0
    for row in rows:
        line = ""
        for c in row:
            if c.isspace():
                continue
            while pos < len(spaced) and spaced[pos].isspace():      # whitespace before a row's first letter ends the row before it
                if line or not out:
                    line += spaced[pos]
                else:
                    out[-1] += spaced[pos]
                pos += 1
            if pos >= len(spaced) or spaced[pos] != c:
                return None
            line += c
            pos += 1
        out.append(line)
    while pos < len(spaced) and spaced[pos].isspace():
        pos += 1
    if pos != len(spaced):
        return None
    return [re.sub(r"\s+$", " ", l.lstrip()) for l in out]


def word_spacer(pack=None):
    """The word spacer of a model pack (the newest in data/model_pack by default), run as Manuscribe runs it.
    -> (space(text) -> spaced text, the pack's name)"""
    import json
    from .config import DATA
    pack = Path(pack) if pack else sorted(p.parent for p in (DATA / "model_pack").glob("pack-*/manifest.json"))[-1]
    m = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(SPACER_DIR))
    import spacer_reference as SR
    if m["spacer_version"] != SR.VERSION:
        raise SystemExit(f"{pack.name}: spacer version {m['spacer_version']}, but {SPACER_DIR / 'spacer_reference.py'} "
                         f"is version {SR.VERSION}")
    sp = m["models"]["spacer"]
    net = SR.onnx_network(pack / sp["file"])
    return (lambda text: SR.space_text(text, net, sp["threshold"])), pack.name


def space_pages(pages, space):
    """pages [(pid, source, {box: [lines]})] with word spaces put in, block by block -> (pages, counts)"""
    out, n = [], {"spaced": 0, "own spaces kept": 0, "left as read": 0}
    for pid, src, boxes in pages:
        new = {}
        for b, lines in boxes.items():
            if not any(t.strip() for t in lines):
                new[b] = lines
            elif any(" " in t.strip() for t in lines):           # spaced by a person already
                new[b] = lines
                n["own spaces kept"] += 1
            else:
                back = split_spaced(lines, space("".join(t.strip() for t in lines)))
                new[b] = back if back is not None else lines
                n["spaced" if back is not None else "left as read"] += 1
        out.append((pid, src, new))
    return out, n


def ends_word(t):
    """a line ends its word: it is empty, or ends in whitespace or a daṇḍa (Manuscribe's lineEnd)"""
    return not t.strip() or t[-1].isspace() or t.rstrip()[-1] in DANDA


LAYOUT_LABEL = {"linear": "Linear", "tripatha": "Commentary above and below (tripāṭha)", "panchapatha": "Commentary on four sides (pañcapāṭha)"}
BOX_LABEL = {"main": "main text", "top": "top", "right": "right", "left": "left", "bottom": "bottom"}


def usual_lines(pages):
    """The most common line count of each block over the pages that have it. -> {box: n}"""
    import collections
    counts = collections.defaultdict(collections.Counter)
    for _, _, boxes in pages:
        for b, lines in boxes.items():
            if lines:
                counts[b][len(lines)] += 1
    with_text = sum(1 for _, _, boxes in pages if any(boxes.values()))
    # a block seen on a page or two is a stray of the block finder, not part of the layout
    return {b: counts[b].most_common(1)[0][0] for b in pagetext.BOXES if b in counts and sum(counts[b].values()) >= max(2, 0.05 * with_text)}


def _layout_element(ms_id, pages):
    from .blocks import layout_of
    lines = usual_lines(pages)
    layout = layout_of(ms_id) if any(set(b) - {"main"} for _, _, b in pages) else "linear"
    if layout == "tripatha" and ("left" in lines or "right" in lines):
        layout = "panchapatha"
    label = LAYOUT_LABEL.get(layout, "Linear")
    stated = ", ".join(f"{BOX_LABEL[b]} {n}" for b, n in lines.items())
    return f'<layout writtenLines="{lines.get("main", 0)}"><p>{label} layout. Lines per folio side: {stated}.</p></layout>'


def _with_layout(head, layout):
    """The header with this <layout> element: in place of the one it has, or added where Manuscribe
    looks for it (sourceDesc/msDesc/physDesc/objectDesc/layoutDesc)."""
    if re.search(r"<layout\b", head):
        return re.sub(r"<layout\b[^>]*>.*?</layout>", layout, head, count=1, flags=re.S)
    if "<layoutDesc>" in head:
        return head.replace("<layoutDesc>", "<layoutDesc>" + layout, 1)
    if re.search(r"<objectDesc\b", head):
        return re.sub(r"(<objectDesc\b[^>]*>)", r"\1<layoutDesc>" + layout.replace("\\", "\\\\") + "</layoutDesc>", head, count=1)
    block = f'<physDesc><objectDesc form="pothi"><layoutDesc>{layout}</layoutDesc></objectDesc></physDesc>'
    if "</msDesc>" in head:
        return head.replace("</msDesc>", block + "</msDesc>", 1)
    if "</sourceDesc>" in head:
        return head.replace("</sourceDesc>", f"<msDesc><msIdentifier/>{block}</msDesc></sourceDesc>", 1)
    return head


def _header(ms_id, pages=None):
    number = ms_id.split("_")[-1]
    title = TITLES.get(ms_id, ms_id)
    if REFERENCE_TEI.exists():
        ref = REFERENCE_TEI.read_text(encoding="utf-8")
        head = ref[ref.index("<teiHeader>"):ref.index("</teiHeader>") + len("</teiHeader>")]
        head = re.sub(r"<title>.*?</title>", f"<title>{escape(title)}</title>", head, count=1)
        library = COLLECTIONS.get(ms_id.split("_")[0], COLLECTIONS["LDII"])
        head = re.sub(r"<sourceDesc>\s*<p>.*?</p>", f"<sourceDesc>\n\t\t\t<p>Transcribed from the manuscript number {number} "
                      f"from {library}.</p>", head, count=1, flags=re.S)
        if pages:
            head = _with_layout(head, _layout_element(ms_id, pages))
        return head
    return (f"<teiHeader>\n\t<fileDesc>\n\t\t<titleStmt>\n\t\t\t<title>{escape(title)}</title>\n\t\t</titleStmt>\n"
            f"\t\t<publicationStmt>\n\t\t\t<p>Not for distribution.</p>\n\t\t</publicationStmt>\n\t\t<sourceDesc>\n"
            f"\t\t\t<p>Transcribed from the manuscript number {number}.</p>\n\t\t</sourceDesc>\n\t</fileDesc>\n</teiHeader>")


def _machine_header(head, label):
    """The header with the recogniser named as responsible and the reading recorded as a change."""
    from .tei import with_resp_stmt
    head = with_resp_stmt(head, who=f"devanagari-manuscript-ocr recogniser, {label}")
    change = f'<change when="{time.strftime("%Y-%m-%d")}">Every page read by the recogniser, {escape(label)}. Nothing corrected.</change>'
    if "<revisionDesc>" in head:
        return head.replace("<revisionDesc>", "<revisionDesc>\n      " + change, 1)
    return head.replace("</teiHeader>", f"  <revisionDesc>\n    {change}\n  </revisionDesc>\n</teiHeader>", 1)


def _pb(pid, src):
    return f'<pb n="{pid}"' + (' level="#ocr"' if src in ("drafts", "to_correct", "machine") else "") + "/>"


def tei(ms_id, pages, reading=None, spaced=False):
    """pages: [(pid, source, {box: [lines]})]. One <div> when every page has only main text; else the
    two streams Manuscribe writes, base-text and commentary, each with every page's <pb/>. spaced: the lines carry
    word spaces, and a line opens with break="no" only where the line before it in its stream runs on (ends_word);
    else every line does (text as read, no word known to end)."""
    head = _machine_header(_header(ms_id, pages), reading) if reading else _header(ms_id, pages)
    out = ['<TEI xmlns="http://www.tei-c.org/ns/1.0">', head, "<text>", "<body>"]
    multi = any(set(boxes) - {"main"} for _, _, boxes in pages)
    state = {"open": False}               # the stream's last line runs on (its word not ended)

    def lbs(lines):
        for i, t in enumerate(lines, 1):
            brk = state["open"] if spaced else True
            out.append(f'<lb n="{i}"' + (' break="no"' if brk else "") + f'/>{escape(t.rstrip() if spaced else t)}')
            state["open"] = not ends_word(t)
    if not multi:
        out.append('<div type="kāṇḍa" n="1" id="Devādhidevakāṇḍa">')
        for pid, src, boxes in pages:
            lines = boxes.get("main", [])
            if not lines:
                continue
            out.append(_pb(pid, src))
            lbs(lines)
        out.append("</div>")
    else:
        out.append('<div type="base-text">')
        for pid, src, boxes in pages:
            lines = boxes.get("main", [])
            if lines:
                out.append(_pb(pid, src))
                lbs(lines)
        out += ["</div>", '<div type="commentary">']
        state["open"] = False
        for pid, src, boxes in pages:
            comm = {b: boxes[b] for b in ("top", "right", "left", "bottom") if boxes.get(b)}
            if not comm:
                continue
            out.append(_pb(pid, src))
            for b, lines in comm.items():
                out.append(f'<cb n="{b}"/>')
                lbs(lines)
        out.append("</div>")
    out += ["</body>", "</text>", "</TEI>"]
    return "\n".join(out) + "\n"


def export(ms_id, from_composite=None, spaces=False):
    """spaces: put word spaces in (or, from a composite, take its spaces as they stand) and write the line breaks
    they give"""
    d = ms_dir(ms_id)
    if from_composite:
        pages = parse_composite(open(from_composite, encoding="utf-8").read(), keep_end=spaces)
        print(f"[export] {ms_id}: {len(pages)} pages read from {from_composite}" + (" (spaced as it stands)" if spaces else ""))
    else:
        pages = gather(ms_id)
        if spaces:
            space, pack = word_spacer()
            pages, n = space_pages(pages, space)
            print(f"[export] {ms_id}: word spaces from {pack}'s spacer: blocks " + ", ".join(f"{k} {v}" for k, v in n.items()))
        txt = d / f"{ms_id}.txt"
        txt.write_text(composite(pages), encoding="utf-8")
        by_src = {s: sum(1 for _, src, _ in pages if src == s) for s in ("corrected", "to_correct", "drafts", None)}
        if any(set(b) - {"main"} for _, _, b in pages):
            print(f"  blocks: usual lines " + ", ".join(f"{b} {n}" for b, n in usual_lines(pages).items()))
        print(f"[export] {ms_id}: {len(pages)} page images -> {txt}")
        print(f"  pages with text: {by_src['corrected']} corrected, {by_src['to_correct']} uncorrected (to_correct), "
              f"{by_src['drafts']} from drafts; {by_src[None]} without text: "
              + ", ".join(pid for pid, src, _ in pages if src is None))
    xml = d / f"{ms_id}.xml"
    xml.write_text(tei(ms_id, pages, spaced=spaces), encoding="utf-8")
    n_lines = sum(len(t) for _, _, b in pages for t in b.values())
    print(f"  TEI: {xml}  ({sum(1 for _, _, b in pages if b)} pages, {n_lines} lines)")
    return pages


def machine(ms_id, model, out=None, spaces=False):
    """Read every page of the manuscript afresh with `model`; write the reading alone as a TEI file."""
    from rapidfuzz.distance import Levenshtein
    from .predict import model_label, read_page
    from .recognizer import load_model
    from .segment import ensure_segmented
    from .tei import norm
    warnings.filterwarnings("ignore")
    d = ms_dir(ms_id)
    model = Path(model)
    net, chars, ck = load_model(model)
    label = model_label(model, ck)
    images = sorted(p.name for p in MANUSCRIPTS[ms_id]["images"].glob("*.jpg"))
    from .segment import page_seg, reading_from
    segs = {i: page_seg(ms_id, i.rsplit(".", 1)[0]) for i in images}
    for seg in sorted(set(segs.values())):
        ensure_segmented(ms_id, [i for i in images if segs[i] == seg], seg)
    print(f"[export] {ms_id}: reading {len(images)} pages with {label}")
    pages, warned, t = [], [], time.time()
    for k, image in enumerate(images, 1):
        stem = image.rsplit(".", 1)[0]
        if stem in not_text(ms_id):
            pages.append((page_id(stem), None, {}))
            continue
        with reading_from(segs[image]):
            boxes, note, warn = read_page(ms_id, image, net, chars)
        pages.append((page_id(stem), "machine" if boxes else None, {b: [l["text"] for l in ls] for b, ls in boxes.items()}))
        if warn and boxes:
            warned.append(stem)
        if k % 25 == 0 or k == len(images):
            print(f"  {k}/{len(images)} pages, {(time.time() - t) / 60:.1f} min", file=sys.stderr)
    written = pages
    if spaces:                            # the score below stays on the reading itself
        space, pack = word_spacer()
        written, n = space_pages(pages, space)
        print(f"  word spaces from {pack}'s spacer: blocks " + ", ".join(f"{k} {v}" for k, v in n.items()))
    out = Path(out) if out else d / f"{ms_id}.machine.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tei(ms_id, written, reading=label, spaced=spaces), encoding="utf-8")
    n_lines = sum(len(t) for _, _, b in pages for t in b.values())
    print(f"  TEI: {out}  ({sum(1 for _, s, _ in pages if s)} pages, {n_lines} lines; "
          f"{len(warned)} pages with a row-finder warning; without rows: {', '.join(pid for pid, s, _ in pages if not s) or 'none'})")
    err = total = n = 0
    for pid, _, boxes in pages:
        stem = MANUSCRIPTS[ms_id]["image_name"](pid).rsplit(".", 1)[0]
        f = d / "corrected" / f"{stem}.txt"
        if f.exists():
            lines = pagetext.flat(boxes)
            truth = norm([t for t in pagetext.flat(pagetext.read(f)) if t.strip()])
            if len(truth) == len(lines):
                err += sum(Levenshtein.distance(a, b) for a, b in zip(norm(lines), truth))
                total += sum(len(t) for t in truth)
                n += 1
    if n:
        print(f"  against the {n} corrected pages: CER {100 * err / max(1, total):.2f}% (not a test score: the model may have trained on them)")
    return pages


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ms_id")
    ap.add_argument("--from", dest="from_composite", help="composite .txt to build the TEI from")
    ap.add_argument("--model", help="read every page afresh with this model and write the reading alone")
    ap.add_argument("--out", help="with --model: where to write the TEI (default data/<ms>/<ms>.machine.xml)")
    ap.add_argument("--spaces", action="store_true", help="word spaces put in by the newest pack's spacer, as Manuscribe "
                    "puts them in, and the line breaks they give (with --from: the composite's spaces as they stand)")
    a = ap.parse_args()
    if a.model:
        machine(a.ms_id, a.model, a.out, spaces=a.spaces)
    else:
        export(a.ms_id, a.from_composite, spaces=a.spaces)
