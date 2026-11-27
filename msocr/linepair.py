"""Rows paired with the lines of a transcription one by one, by what a recogniser reads in them: for a manuscript with
a transcription page by page and no line positions (Muktabodha's e-texts), in place of align's pairing by row count,
which loses a whole page for one missed row or one gloss cut as a row.

The rows are the row finder's (align.find_rows on data/<ms>/seg_torch/), each read greedily. Rows and lines of a page
are matched in order (a row or a line may be passed over) by the edit distance between reading and line. A matched
pair is left out when
- the reading is further than LIMIT from the line (not that line, or a row too damaged to teach anything);
- one end of the line holds END letters more than the reading, or the reading END more than the line (a part of the
  line left outside the row, or writing cut with it that the line does not have);
- the line has an illegible stretch (<gap/>) outside its glosses.
A page with less than half its lines paired is left out whole. A recogniser reads the pages it trained on better than
new ones, and may give a learnt line in full where the row shows less of it: the end test is weaker on those.
"""
import hashlib
import json
import re

from .config import MANUSCRIPTS, ms_dir

SEG = "seg_torch"
LIMIT = 0.5             # normalised edit distance beyond which a reading and a line are no pair
END = 4                 # letters one end of the line (or of the reading) may hold over the other, less than this
MIN_PART = 0.5          # a page with less than this part of its lines paired is left out whole
ANCHOR = 2              # an end is what lies outside the first and last stretch of this many letters read alike


def gap_lines(ms_id):
    """-> {(image, index of the line on its page)} of the lines with a <gap/> outside a <note>."""
    ms = MANUSCRIPTS[ms_id]
    tei = ms["tei"].read_text(encoding="utf-8")
    out = set()
    for m in re.finditer(r'<pb n="([^"]+)"[^>]*/>(.*?)(?=<pb |</body>)', tei, re.S):
        for i, line in enumerate(re.split(r"<lb [^>]*/>", m.group(2))[1:]):
            if "<gap" in re.sub(r"<note\b[^>]*?(?<!/)>.*?</note>", "", line, flags=re.S):
                out.add((ms["image_name"](m.group(1)), i))
    return out


def end_excess(read, truth):
    """-> the most letters by which one end of the line outruns that end of the reading, or the other way round."""
    from rapidfuzz.distance import Levenshtein
    same = [b for b in Levenshtein.opcodes(read, truth) if b.tag == "equal" and b.src_end - b.src_start >= ANCHOR]
    if not same:
        return abs(len(read) - len(truth))
    head = abs(same[0].src_start - same[0].dest_start)
    tail = abs((len(read) - same[-1].src_end) - (len(truth) - same[-1].dest_end))
    return max(head, tail)


def in_order(reads, truths, limit=LIMIT):
    """Rows and lines matched in order, as many and as close as can be: -> [(row, line, distance)]."""
    from rapidfuzz.distance import Levenshtein
    d = [[Levenshtein.normalized_distance(r, t) for t in truths] for r in reads]
    n, m = len(reads), len(truths)
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best[i][j] = max(best[i - 1][j], best[i][j - 1])
            if d[i - 1][j - 1] <= limit:
                best[i][j] = max(best[i][j], best[i - 1][j - 1] + 1 - d[i - 1][j - 1])
    out, i, j = [], n, m
    while i and j:
        if d[i - 1][j - 1] <= limit and best[i][j] == best[i - 1][j - 1] + 1 - d[i - 1][j - 1]:
            out.append((i - 1, j - 1, d[i - 1][j - 1]))
            i, j = i - 1, j - 1
        elif best[i][j] == best[i - 1][j]:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def read_pages(ms_id, model, cache=None, quiet=False):
    """-> {image: {"size", "rows": [(baseline, outline)], "reads": [text]}} for every segmented page with text. The
    readings are kept in `cache` (a JSON path), keyed by the model's file and the segmentation of the page."""
    import numpy as np
    from PIL import Image
    from . import gt
    from .align import _merge, find_rows
    from .linecut import band_of, cut_line
    from .predict import read_line
    from .recognizer import load_model
    d, ms = ms_dir(ms_id), MANUSCRIPTS[ms_id]
    images = sorted({r["image"] for r in gt.load(ms_id)})
    tag = f"{model}:{model.stat().st_size}"
    have = json.loads(cache.read_text(encoding="utf-8")) if cache and cache.exists() else {}
    net = chars = None
    out = {}
    for k, image in enumerate(images, 1):
        seg_path = d / SEG / (image.rsplit(".", 1)[0] + ".json")
        if not seg_path.exists():
            continue
        raw = seg_path.read_bytes()
        key = hashlib.sha256(tag.encode() + raw).hexdigest()
        if have.get(image, {}).get("key") != key:
            if net is None:
                net, chars, _ = load_model(model)
            with Image.open(ms["images"] / image) as im:
                rgb = im.convert("RGB")
            gray = np.asarray(rgb.convert("L"))
            rows, block, pitch, _ = find_rows(json.loads(raw), rgb.size[0], gray)
            merged = [_merge(r, block, pitch) for r in rows]
            page = np.asarray(rgb)
            reads = []
            for base, outline in merged:
                above, below = band_of(base, outline)
                reads.append(read_line(net, chars, Image.fromarray(cut_line(page, base, above, below)))[0].replace(" ", ""))
            have[image] = {"key": key, "size": list(rgb.size), "reads": reads,
                           "rows": [[[list(map(float, q)) for q in b], [list(map(float, q)) for q in o]] for b, o in merged]}
        out[image] = have[image]
        if len(out) % 100 == 0:
            if cache:
                cache.write_text(json.dumps(have, ensure_ascii=False), encoding="utf-8")
            if not quiet:
                print(f"  read {k}/{len(images)} pages", flush=True)
    if cache:
        cache.write_text(json.dumps(have, ensure_ascii=False), encoding="utf-8")
    return out


def pair(ms_id, pages):
    """pages: read_pages' result. -> [{"image", "size", "lines": [(index, baseline, outline, text, reading)],
    "faults": [(index, why)], "total", "whole": why the page is left out, or None}]"""
    import collections
    from . import gt
    truth = collections.defaultdict(list)
    for r in gt.load(ms_id):
        truth[r["image"]].append(r["text"])
    gaps = gap_lines(ms_id)
    out = []
    for image in sorted(truth):
        texts, p = truth[image], pages.get(image)
        if p is None:
            out.append({"image": image, "size": None, "lines": [], "faults": [], "total": len(texts), "whole": "not segmented"})
            continue
        matched = {j: (i, dist) for i, j, dist in in_order(p["reads"], texts)}
        kept, faults = [], []
        for j, text in enumerate(texts):
            if not text:
                faults.append((j, "an empty line"))
            elif (image, j) in gaps:
                faults.append((j, "an illegible stretch (<gap/>) in the transcription"))
            elif j not in matched:
                faults.append((j, "no row reads as this line"))
            else:
                i = matched[j][0]
                over = end_excess(p["reads"][i], text)
                if over >= END:
                    faults.append((j, "an end of the line and of its row do not agree"))
                else:
                    kept.append((j, p["rows"][i][0], p["rows"][i][1], text, p["reads"][i]))
        whole = None
        if len(kept) < MIN_PART * len(texts):
            whole = f"{len(kept)} of {len(texts)} lines pair ({len(p['rows'])} rows)"
        out.append({"image": image, "size": p["size"], "lines": kept, "faults": faults, "total": len(texts), "whole": whole})
    return out
