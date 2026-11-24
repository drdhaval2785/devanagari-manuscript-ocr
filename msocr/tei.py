"""Read and rewrite a TEI transcription page by page, touching nothing but the pages asked for.

The file is handled as text, not re-serialised, so the editor's header, divisions, comments
and spacing survive untouched. It is cut at every <pb .../> into chunks; each chunk is the
<pb> tag plus everything up to the next <pb> (the last chunk runs to the end of the file).

A manuscript with commentary blocks (tripāṭha, pañcapāṭha) is written by Manuscribe as two
streams, <div type="base-text"> and <div type="commentary">, each holding every folio side
with its own <pb/>: such a page has one chunk in each stream, and the commentary chunk marks
its blocks with <cb n="top"/>, <cb n="right"/>, <cb n="left"/>, <cb n="bottom"/>. A chunk knows
its stream from the <div> it stands in, and resolves its lines by block. A linear file has one
chunk per page and no streams.

A chunk is "plain" when it holds nothing but <lb .../> tags each followed by text, with <cb/>
tags between the blocks: such a chunk can be rewritten safely. Anything else inside (<hi>,
<note>, a </div> between two lines, ...) makes it "marked up", and it is then kept as it is.

Tags may come back from an editor in any of the forms <pb n="x"/>, <pb n='x' /> or
<pb n="x"></pb>, with attributes in any order; all are accepted.

An editor may mark the proofreading state on the <pb> as level="#ocr" (machine text, not yet
proofread) or level="#proof1" and so on; such a page counts as proofread. The attribute is kept
as it is when a page is rewritten, so a machine page stays level="#ocr".

Line text is resolved as in gt.py: the scribe's text, with note/del/corr/supplied dropped.
"""
import collections
import re
import unicodedata
from xml.sax.saxutils import escape

from lxml import etree

from .gt import DROP, _local
from .pagetext import BOXES

OCR_RESP = "#ocr"
OCR_LEVEL = "#ocr"      # the editor's level="..." on a <pb> that has not been proofread; anything else means it has
MARK = f'resp="{OCR_RESP}"'
PB = re.compile(r"<pb\b[^>]*?(?:/>|>\s*</pb>)")
DIV = re.compile(r"<div\b[^>]*>|</div\s*>")
PLAIN = re.compile(r"(?:\s*(?:<cb\b[^>]*?/>\s*)?<lb\b[^>]*?(?:/>|>\s*</lb>)[^<]*)+")
_MARK = re.compile(r"""\s*resp=["']""" + re.escape(OCR_RESP) + """["']""")
STREAM_BOXES = {"main": ("main",), "commentary": ("top", "right", "left", "bottom"), "gloss": ("gloss",)}


def stream_of(div_type):
    """Manuscribe's stream for a <div type="...">: base-text -> main, commentary, interlinear-commentary -> gloss."""
    t = div_type or ""
    return "gloss" if re.search(r"interlin|gloss", t, re.I) else "commentary" if re.search(r"comment", t, re.I) else "main"


class Page:
    """One <pb/> chunk: a page in a linear file, or a page's part in one stream of a two-stream file."""

    def __init__(self, pb, body, stream=None):
        self.pb, self.body, self.stream = pb, body, stream
        m = re.search(r"""\bn=["']([^"']+)["']""", pb)
        self.pid = m.group(1) if m else None
        self.marked = bool(_MARK.search(pb))
        m = re.search(r"""\blevel=["']([^"']*)["']""", pb)
        self.level = m.group(1) if m else None                 # the editor's proofreading level, e.g. #ocr, #proof1
        self.proofread = self.level is not None and self.level != OCR_LEVEL
        m = PLAIN.match(body)
        self.plain = bool(m) and "<lb" not in body[m.end():]
        self.rest = body[m.end():] if self.plain else body        # what follows the lines
        lines_part = body[:m.end()] if self.plain else ""
        self.lead_ws = re.match(r"\s*", lines_part).group(0) or "\n"       # whitespace before the first <lb>
        self.tail_ws = re.search(r"\s*$", lines_part).group(0) or "\n"     # whitespace after the last line
        self.boxes = resolve_boxes(body, stream)                           # {box: [lines]}
        self.lines = [t for b in self.boxes for t in self.boxes[b]]

    def rewrite(self, boxes, mark=False):
        """This chunk with new lines, {box: [lines]} (a plain list is the main text); spacing and the
        tail (if any) kept. Only the boxes of this chunk's stream are written, each block after its
        <cb/> when the stream has several. mark=True puts resp="#ocr" on the <pb>; otherwise any such
        mark is removed."""
        if isinstance(boxes, list):
            boxes = {"main": boxes}
        pb = _MARK.sub("", self.pb)
        if mark:
            attrs = re.match(r"<pb\b(.*?)\s*/?>", pb, re.S).group(1)
            pb = f"<pb{attrs} {MARK}/>"
        mine = STREAM_BOXES.get(self.stream or "main", ("main",))
        parts = []
        for box in mine:
            lines = boxes.get(box) or []
            if not lines:
                continue
            if len(mine) > 1:
                parts.append(f'<cb n="{box}"/>')
            parts.extend(f'<lb n="{i}" break="no"/>{escape(t)}' for i, t in enumerate(lines, 1))
        return pb + self.lead_ws + "\n".join(parts) + self.tail_ws + self.rest

    def unmarked(self):
        """The chunk as it is, without the machine-read mark."""
        return _MARK.sub("", self.pb) + self.body


def resolve_boxes(fragment, stream=None):
    """-> {box: [lines]}: the lines of the chunk by block, following its <cb n="..."/> tags.
    Without a <cb/> the lines belong to the stream's first box (main; top for a commentary)."""
    root = etree.fromstring(f"<r>{fragment}</r>", etree.XMLParser(remove_comments=True, recover=True))
    boxes = collections.OrderedDict()
    box = STREAM_BOXES.get(stream or "main", ("main",))[0]
    cur = None

    def emit(t):
        if t and cur is not None:
            cur.append(t)

    def walk(el):
        nonlocal cur, box
        tag = _local(el)
        if tag == "cb":
            box = el.get("n") or box
        elif tag == "lb":
            cur = []
            boxes.setdefault(box, []).append(cur)
        elif tag == "space":
            emit(" ")
        elif tag in DROP:
            pass
        elif tag is not None:
            emit(el.text)
            for ch in el:
                walk(ch)
                emit(ch.tail)
    walk(root)
    clean = lambda b: unicodedata.normalize("NFC", re.sub(r"\s+", " ", "".join(b)).strip())
    out = {b: [clean(l) for l in boxes[b]] for b in BOXES if b in boxes}
    out.update({b: [clean(l) for l in ls] for b, ls in boxes.items() if b not in BOXES})
    return out


def resolve_lines(fragment):
    """The chunk's lines in block order, as one list."""
    boxes = resolve_boxes(fragment)
    return [t for b in boxes for t in boxes[b]]


def read(path):
    """-> (head, [Page, ...]) in file order; head is everything before the first <pb>.
    A two-stream file gives two Pages per folio side, one in each stream (see by_page)."""
    text = path.read_text(encoding="utf-8")
    tags = list(PB.finditer(text))
    if not tags:
        return text, []
    head = text[:tags[0].start()]
    # the stream each <pb> stands in: the innermost open <div> with a type
    events = sorted([(m.start(), m.group(0)) for m in DIV.finditer(text)] + [(m.start(), None) for m in tags])
    stack, stream_at, typed = [], {}, 0
    for pos, tag in events:
        if tag is None:
            stream_at[pos] = stream_of(stack[-1]) if stack else None
        elif tag.startswith("</"):
            if stack:
                stack.pop()
        else:
            m = re.search(r"""\btype=["']([^"']*)["']""", tag)
            stack.append(m.group(1) if m else "")
            typed += bool(m)
    pages = []
    for i, m in enumerate(tags):
        end = tags[i + 1].start() if i + 1 < len(tags) else len(text)
        pages.append(Page(m.group(0), text[m.end():end], stream_at[m.start()] if typed else None))
    return head, pages


def by_page(pages):
    """-> OrderedDict pid -> [chunks of that page, in file order]."""
    out = collections.OrderedDict()
    for p in pages:
        out.setdefault(p.pid, []).append(p)
    return out


def page_boxes(chunks):
    """The lines of a page from all its chunks, {box: [lines]} in block order."""
    boxes = {}
    for c in chunks:
        for b, lines in c.boxes.items():
            boxes.setdefault(b, []).extend(lines)
    return {b: boxes[b] for b in BOXES if b in boxes} | {b: v for b, v in boxes.items() if b not in BOXES}


RESP_TEXT = "machine reading, uncorrected"


def with_resp_stmt(head, who="devanagari-manuscript-ocr recogniser"):
    """Add the respStmt that resp="#ocr" points to, once, if the header has a titleStmt."""
    if 'xml:id="ocr"' in head or "</titleStmt>" not in head:
        return head
    m = re.search(r"([ \t]*)</titleStmt>", head)
    ind = m.group(1)
    stmt = (f"{ind}\t<respStmt>\n{ind}\t\t<resp>{RESP_TEXT}</resp>\n"
            f'{ind}\t\t<name xml:id="ocr">{escape(who)}</name>\n{ind}\t</respStmt>\n')
    return head[:m.start()] + stmt + head[m.start():]


def without_resp_stmt(head):
    """Remove the respStmt that with_resp_stmt added, if present; nothing else."""
    return re.sub(r"[ \t]*<respStmt>\s*<resp>" + re.escape(RESP_TEXT) + r"</resp>\s*<name xml:id=\"ocr\">[^<]*</name>\s*</respStmt>[ \t]*\n?",
                  "", head, count=1)


def norm(lines):
    return [re.sub(r"\s+", "", unicodedata.normalize("NFC", t)) for t in lines]
