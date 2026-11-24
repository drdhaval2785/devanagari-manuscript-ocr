"""Checks on contributed lines before the recogniser may learn from them: that each line is an image of writing, that
its text is text the recogniser reads, that the two fit each other, that it is not a copy of a line already taken, and
that the image says what the text says. A package (msocr.contributions) whose lines fail too often is held back whole
for the maintainer to look at (quarantined); the lines that fail are left out in any case.

Each line:
  image     a PNG that opens; 16-160 px high (the developer's lines, cut from pages at the scale Manuscribe renders a
            side at: 30-59), at least 48 px wide; not blank and not solid: ink (grey < 128) on 1-90 % of it (the
            developer's: 26-61 %), a grey spread (standard deviation) of at least 12 (theirs: 41-81)
  text      NFC, whitespace taken out (the recogniser reads unspaced text; the word spacer puts the spaces back); only
            Devanagari (U+0900-097F, the Extended blocks U+A8E0-A8FF and U+11B00-11B5F, Vedic U+1CD0-1CFF) and the
            hyphen; at least 2 letters, at most 400 characters
  fit       8-24 px of image a character once scaled to the recogniser's 48 px (the developer's: 9.4-21.7, median
            15.6), and wide enough for CTC to place every character (one output column per 4 px, one more between
            two equal characters)
  copy      not the same image (its pixels) as a line taken before, in this package or another
  reading   the judge (a recogniser of this pipeline) reads the image, and its reading differs from the text by at most
            60 % of the text's characters. Measured 24 November 2026 with ft4 on the corrected pages of four
            manuscripts: the true text differs by 0-23 % (99th percentile), another line's text by 76 % or more (1st)
  as read   where the judge is the very model that read the line in Manuscribe (the package names its pack), its
            reading now is what Manuscribe proposed then (within 10 %): the image is the one Manuscribe cut
A package is quarantined where more than 25 % of its lines fail, where its median reading difference exceeds 35 %, where
more than 20 % of the lines the judge can check do not read as Manuscribe read them, or where more than 20 % of its
lines (5 or more) have one and the same text. The maintainer may accept a quarantined package after looking at it: its
lines that failed only on the reading (a hand the judge reads poorly) are then taken; the others stay out.
"""
import hashlib
import io
import statistics
import unicodedata

import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein

from .params import P

HEIGHT, STRIDE = P["recogniser"]["height"], P["recogniser"]["stride"]
LIMITS = {"height": (16, 160), "width_min": 48, "ink": (0.01, 0.90), "grey_std_min": 12.0, "letters_min": 2, "chars_max": 400,
          "px_per_char": (8.0, 24.0), "reading_max": 0.60, "as_read_max": 0.10,
          "package_failed_max": 0.25, "package_reading_median_max": 0.35, "package_not_as_read_max": 0.20, "package_same_text_max": 0.20}
RANGES = ((0x0900, 0x097F), (0xA8E0, 0xA8FF), (0x11B00, 0x11B5F), (0x1CD0, 0x1CFF))
EXTRA = set("-")
SOFT = "reading"          # the one reason a maintainer's acceptance overrules


def allowed_char(c):
    o = ord(c)
    return c in EXTRA or any(a <= o <= b for a, b in RANGES)


def is_letter(c):
    return unicodedata.category(c) == "Lo"


def normalise(text):
    return "".join(unicodedata.normalize("NFC", text or "").split())


def check_text(text):
    """-> (the text as the recogniser learns it, reasons)"""
    t = normalise(text)
    why = []
    bad = sorted({c for c in t if not allowed_char(c)})
    if bad:
        why.append("text: characters the recogniser does not read: " + " ".join(f"{c!r} U+{ord(c):04X}" for c in bad[:8]))
    if sum(map(is_letter, t)) < LIMITS["letters_min"]:
        why.append(f"text: fewer than {LIMITS['letters_min']} letters")
    if len(t) > LIMITS["chars_max"]:
        why.append(f"text: longer than {LIMITS['chars_max']} characters")
    return t, why


def check_image(data):
    """-> (grey image or None, stats, reasons)"""
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        if im.format != "PNG":
            return None, {}, [f"image: a {im.format}, not a PNG"]
        g = im.convert("L")
    except Exception as e:
        return None, {}, [f"image: does not open ({e.__class__.__name__})"]
    w, h = g.size
    a = np.asarray(g, dtype=np.float32)
    st = {"width": w, "height": h, "ink": round(float((a < 128).mean()), 3), "grey_std": round(float(a.std()), 1),
          "pixels": hashlib.sha256(np.asarray(g).tobytes() + f"{w}x{h}".encode()).hexdigest()}
    why = []
    lo, hi = LIMITS["height"]
    if not lo <= h <= hi:
        why.append(f"image: {h} px high, not {lo}-{hi}")
    if w < LIMITS["width_min"]:
        why.append(f"image: {w} px wide, under {LIMITS['width_min']}")
    lo, hi = LIMITS["ink"]
    if not lo <= st["ink"] <= hi:
        why.append(f"image: ink on {100 * st['ink']:.0f} % of it, not {100 * lo:.0f}-{100 * hi:.0f} % (blank or solid)")
    if st["grey_std"] < LIMITS["grey_std_min"]:
        why.append(f"image: hardly any contrast (grey spread {st['grey_std']})")
    return g, st, why


def check_fit(stats, text):
    """the text against the image's width once scaled to the recogniser's height"""
    if not stats or not text:
        return []
    w48 = max(STRIDE * 2, round(stats["width"] * HEIGHT / stats["height"]))
    ppc = w48 / len(text)
    stats["px_per_char"] = round(ppc, 2)
    why = []
    lo, hi = LIMITS["px_per_char"]
    if not lo <= ppc <= hi:
        why.append(f"fit: {ppc:.1f} px of line a character, not {lo:g}-{hi:g} (the text is not this line's, or not all of it)")
    repeats = sum(1 for a, b in zip(text, text[1:]) if a == b)
    if w48 // STRIDE < len(text) + repeats:
        why.append("fit: the image is too narrow for its text")
    return why


def cer(a, b):
    return Levenshtein.distance(a, b) / max(1, len(b))


def judge_readings(net, chars, images):
    """the judge's reading of each grey image (recognizer.to_tensor, greedy CTC)"""
    import torch
    from . import recognizer as R
    samples = [(R.to_tensor(im), "", str(k)) for k, im in enumerate(images)]
    out = [None] * len(samples)
    with torch.no_grad():
        for idx in R.batches(samples, 16):
            x, lens = R.collate(samples, idx)
            lp = net(x)
            for k, i in enumerate(idx):
                out[i] = R.decode(lp[k, :lens[k]], chars)
    return out


def verdict(lines, accepted=False):
    """lines: [{reasons, reading, as_read, text}] -> (package verdict 'ok' | 'quarantined', reasons, summary); marks
    each line's 'usable'"""
    n = len(lines)
    for l in lines:
        hard = [r for r in l["reasons"] if not r.startswith(SOFT)]
        l["usable"] = not hard and (accepted or not l["reasons"])
    failed = sum(1 for l in lines if l["reasons"])
    readings = [l["reading"] for l in lines if l.get("reading") is not None]
    checked = [l for l in lines if l.get("as_read") is not None]
    not_as_read = sum(1 for l in checked if l["as_read"] > LIMITS["as_read_max"])
    texts = [l["text"] for l in lines if l["text"]]
    same = max((texts.count(t) for t in set(texts)), default=0)
    med = statistics.median(readings) if readings else None
    why = []
    if n and failed / n > LIMITS["package_failed_max"]:
        why.append(f"{failed} of {n} lines fail")
    if med is not None and med > LIMITS["package_reading_median_max"]:
        why.append(f"the judge's reading differs from the texts by {100 * med:.0f} % at the median (a hand it reads poorly, or texts not of these lines)")
    if checked and not_as_read / len(checked) > LIMITS["package_not_as_read_max"]:
        why.append(f"{not_as_read} of {len(checked)} lines do not read as Manuscribe read them (not the images it cut)")
    if same >= 5 and same / n > LIMITS["package_same_text_max"]:
        why.append(f"{same} of {n} lines have one and the same text")
    summary = {"lines": n, "failed": failed, "usable": sum(l["usable"] for l in lines),
               "reading_median": None if med is None else round(med, 3), "as_read_checked": len(checked), "not_as_read": not_as_read}
    return ("quarantined" if why else "ok"), why, summary
