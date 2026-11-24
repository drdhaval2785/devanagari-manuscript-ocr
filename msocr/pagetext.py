"""The text of one page as a file: one line per manuscript line, and, on a page with several
blocks, a marker line before each block naming it as Manuscribe names its boxes:

    ;box{main}
    ...lines of the main text...
    ;box{top}
    ...lines of the commentary above...

A page with the main text only has no marker. Blocks come in the fixed order main, top, right,
left, bottom, which is also the order of the line images (<page>_01.png for the main text,
<page>_top_01.png for the top block, and so on), so that the k-th line of a block always goes
with the k-th image of that block. drafts/, to_correct/, corrected/ and machine/ all use this.
"""
import re

BOXES = ("main", "top", "right", "left", "bottom")
_MARK = re.compile(r"\s*;box\{(\w+)\}\s*")


def parse(text):
    """-> {box: [lines]}; a text without markers is the main block. Every line counts, an empty one
    too (a manuscript line the model read as nothing, an ornament), so that line k of a block
    always goes with image k; only the file's final newline is not a line."""
    boxes, cur = {}, "main"
    for raw in text.splitlines():
        m = _MARK.fullmatch(raw)
        if m:
            cur = m.group(1)
            boxes.setdefault(cur, [])
        else:
            boxes.setdefault(cur, []).append(raw.rstrip())
    return {b: boxes[b] for b in BOXES if b in boxes} | {b: v for b, v in boxes.items() if b not in BOXES}


def format(boxes):
    """{box: [lines]} -> the file's text; markers only when there is more than the main block."""
    boxes = {b: boxes[b] for b in BOXES if b in boxes and boxes[b]}
    if list(boxes) == ["main"] or not boxes:
        return "\n".join(boxes.get("main", [])) + "\n"
    return "".join(f";box{{{b}}}\n" + "\n".join(lines) + "\n" for b, lines in boxes.items())


def flat(boxes):
    """All lines in block order: the order of the line images and of training."""
    return [t for b in BOXES if b in boxes for t in boxes[b]]


def read(path):
    return parse(path.read_text(encoding="utf-8"))


def write(path, boxes):
    path.write_text(format(boxes), encoding="utf-8")


def image_name(stem, box, k):
    """The k-th (1-based) line image of a block: <page>_03.png for the main text, <page>_top_03.png for a block."""
    return f"{stem}_{k:02d}.png" if box == "main" else f"{stem}_{box}_{k:02d}.png"


def image_pattern(stem, box):
    """A regex matching this block's line images and nothing else."""
    return re.compile(rf"^{re.escape(stem)}_{'' if box == 'main' else re.escape(box) + '_'}\d+\.png$")
