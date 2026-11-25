"""Step 1: turn the TEI transcription into one plain-text string per manuscript line.

The recogniser must learn "what ink is on this line", so editorial layers are resolved
towards what the scribe wrote in the main text block:

    <choice><orig>X</orig><corr>Y</corr></choice>  -> X   (scribe's spelling, not the correction)
    <surplus>X</surplus>                           -> X   (on the page, even if superfluous)
    <unclear>X</unclear>                           -> X
    <add>X</add>                                   -> X   (scribe's own addition)
    <del>X</del>                                   -> dropped (struck out / lightened)
    <supplied>X</supplied>                         -> dropped (editor's, not on the page)
    <note ...>X</note>                             -> dropped (margins, editorial remarks)
    <space quantity="n"/>                          -> one space
    <hi>, <quote>, <lg>, <div>                     -> transparent

अ followed by ा, ो or ौ becomes आ, ओ, औ: the scribe turns अ into आ by adding the stroke above the
line, which the TEI rightly records as अ<add place="supralinear">ा</add>, but अ + ा is not a
Unicode letter (and the recogniser would learn it so).

The manuscript has no word spacing; the spaces in the TEI are the editor's. `text` has
them removed (used for training), `text_spaced` keeps them.
"""
import collections
import json
import re
import unicodedata

from lxml import etree

from .config import MANUSCRIPTS, ms_dir

DROP = {"note", "del", "corr", "supplied"}
JOIN_A = {"ा": "आ", "ो": "ओ", "ौ": "औ"}     # अ with a vowel sign added to it -> the vowel


def _local(el):
    return etree.QName(el).localname if isinstance(el.tag, str) else None


def extract(ms_id: str, verbose: bool = True):
    ms = MANUSCRIPTS[ms_id]
    tree = etree.parse(str(ms["tei"]), etree.XMLParser(remove_comments=True, recover=True))
    body = next(e for e in tree.iter() if _local(e) == "body")

    lines = []                      # dicts in document order
    dropped = collections.Counter()
    seen_pages = collections.Counter()
    cur = {"page": None, "line": None}

    def emit(text):
        if text and cur["line"] is not None:
            cur["line"]["buf"].append(text)

    def walk(el):
        tag = _local(el)
        if tag == "pb":
            pb = el.get("n")
            seen_pages[pb] += 1
            if seen_pages[pb] == 2 and pb.endswith("_1"):
                # the TEI repeats one page number; the second is the verso
                fixed = pb[:-1] + "2"
                print(f"  ! duplicate <pb n=\"{pb}\"/> - treating the second one as {fixed}")
                pb = fixed
            cur["page"], cur["line"] = pb, None
        elif tag == "lb":
            cur["line"] = {"page": cur["page"], "n": int(el.get("n")), "buf": []}
            lines.append(cur["line"])
        elif tag == "space":
            emit(" ")
        elif tag in DROP:
            dropped[tag] += 1
        elif tag is not None:
            emit(el.text)
            for child in el:
                walk(child)
                emit(child.tail)

    walk(body)

    out = []
    for ln in lines:
        spaced = unicodedata.normalize("NFC", re.sub(r"\s+", " ", "".join(ln["buf"])).strip())
        spaced = re.sub("अ([ाोौ])", lambda m: JOIN_A[m.group(1)], spaced)
        out.append({
            "page": ln["page"],
            "image": ms["image_name"](ln["page"]),
            "n": ln["n"],
            "text": spaced.replace(" ", ""),
            "text_spaced": spaced,
        })

    path = ms_dir(ms_id) / "gt_lines.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if verbose:
        pages = collections.Counter(r["page"] for r in out)
        missing = [r["image"] for r in out if not (ms["images"] / r["image"]).exists()]
        chars = collections.Counter(c for r in out for c in r["text"])
        lens = sorted(len(r["text"]) for r in out)
        print(f"[gt] {ms_id}")
        print(f"  TEI file            : {ms['tei'].name}")
        print(f"  pages / lines       : {len(pages)} / {len(out)}")
        print(f"  lines per page      : {dict(collections.Counter(pages.values()))}  (lines: number of pages)")
        print(f"  page images missing : {len(set(missing))}")
        print(f"  elements dropped    : {dict(dropped)}")
        print(f"  characters          : {sum(chars.values()):,} total, {len(chars)} distinct")
        print(f"  line length (chars) : min {lens[0]}, median {lens[len(lens)//2]}, max {lens[-1]}")
        print(f"  empty lines         : {sum(1 for r in out if not r['text'])}")
        print("  most common chars   : " + " ".join(f"{c}:{n}" for c, n in chars.most_common(15)))
        rare = [(c, n) for c, n in chars.items() if n < 5]
        print(f"  rare chars (<5 uses): {len(rare)}  " + " ".join(f"{c}(U+{ord(c):04X})x{n}" for c, n in sorted(rare, key=lambda x: x[1])))
        print(f"  sample              : p{out[40]['page']} l{out[40]['n']}: {out[40]['text'][:70]}...")
        print(f"  wrote               : {path}")
    return out


def load(ms_id: str):
    with open(ms_dir(ms_id) / "gt_lines.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]
