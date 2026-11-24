"""Cutting a line's text into shorter pieces for the warm-up stage of training.

Why: a full line here holds ~143 characters. A blank model cannot work out which stretch of
such a long image belongs to which character, and training stalls. Short pieces (~35
characters) make that discovery easy. The model warmed up on pieces is then trained on
full lines.

How: nobody has marked where each character sits, so the cut is estimated. The script has
no word spaces and fairly even letter widths, so the text is divided in proportion to an
estimated width per character, always between whole akṣaras. Cut points are therefore a
character or two off at piece edges; that is acceptable for a warm-up, and is the reason
the final training uses full lines, whose text is exact.

recognizer.py cuts the line image at the same fractions that split_text() returns.
"""
import unicodedata

PIECES = 4
VIRAMA = "्"


def _width(ch):
    """Rough relative width of a character as written in this hand."""
    if ch == VIRAMA:
        return -0.4                      # the consonant before it shrinks to a half form
    cat = unicodedata.category(ch)
    if ch in "ािीोौ":
        return 0.55 if ch in "ािी" else 0.9
    if ch in "ेै":
        return 0.35                      # pṛṣṭhamātrā: a stroke before the consonant
    if cat in ("Mn", "Me"):
        return 0.0                       # marks above or below: ं ु ू ृ ँ
    if ch == "ः":
        return 0.45
    if ch == "।":
        return 0.4
    if ch == "॥":
        return 0.6
    if ch == "ऽ":
        return 0.8
    return 1.0


def _aksharas(text):
    """Split into units that must not be cut: consonant cluster + its vowel signs and marks."""
    units, cur = [], ""
    for ch in text:
        cat = unicodedata.category(ch)
        joins = cat in ("Mn", "Mc", "Me") or (cur.endswith(VIRAMA) and cat == "Lo")
        if cur and not joins:
            units.append(cur)
            cur = ""
        cur += ch
    if cur:
        units.append(cur)
    return units


def split_text(text, pieces=PIECES):
    units = _aksharas(text)
    widths = [max(0.2, sum(_width(c) for c in u)) for u in units]
    total = sum(widths)
    out, cuts, acc, cur, k = [], [], 0.0, "", 1
    for u, w in zip(units, widths):
        if k < pieces and acc + w / 2 > total * k / pieces and cur:
            out.append(cur)
            cuts.append(acc / total)
            cur, k = "", k + 1
        cur += u
        acc += w
    out.append(cur)
    return out, cuts
