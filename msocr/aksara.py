"""Akṣaras, and the grapheme error rate (GER) counted in them.

An akṣara here is what Manuscribe's editor counts as one (its caret, its status bar): a consonant with the consonants
joined to it by virama (a conjunct, क्ष, स्त्र, with nukta and ZWJ/ZWNJ), its vowel sign and marks (anusvāra, visarga,
candrabindu, a final virama, Vedic accents); an independent vowel with its marks; any other character (daṇḍa, digit,
avagraha) with the marks after it. Spaces are not akṣaras (training texts have none).

CER counts the edits that turn a reading into the truth per 100 characters of the truth; GER counts them per 100
akṣaras, each akṣara one unit: an akṣara read wrong in any of its parts is one error, however many characters it has.
It is the rate a reader meets, akṣara by akṣara, and does not grow with the number of signs in a conjunct. In these
manuscripts an akṣara holds 1.8-2.2 characters, and GER has run at 1.2-2.0 times CER (1.5 on average: ft6 and pool1 on
six held-out sets, 2026-11-25), less than the akṣara's length because a misread akṣara often has several characters wrong.
"""
from rapidfuzz.distance import Levenshtein

_R = lambda a, b: {chr(c) for c in range(a, b + 1)}                     # noqa: E731
CONS = _R(0x915, 0x939) | _R(0x958, 0x95F) | _R(0x978, 0x97F)
MARK = _R(0x900, 0x903) | _R(0x93A, 0x93C) | _R(0x93E, 0x94C) | _R(0x94E, 0x94F) | _R(0x951, 0x957) | _R(0x962, 0x963) \
    | _R(0x1CD0, 0x1CFF) | _R(0xA8E0, 0xA8F1)
VIRAMA, NUKTA, JOIN = "\u094d", "\u093c", {"\u200c", "\u200d"}


def aksharas(s):
    """a line split into akṣaras (spaces left out)"""
    out, i, n = [], 0, len(s)
    while i < n:
        if s[i].isspace():
            i += 1
            continue
        j = i + 1
        if s[i] in CONS:
            while True:
                while j < n and s[j] == NUKTA:
                    j += 1
                if j < n and s[j] == VIRAMA:
                    k = j + 1
                    while k < n and s[k] in JOIN:
                        k += 1
                    if k < n and s[k] in CONS:
                        j = k + 1
                        continue
                    j = k
                break
        while j < n and (s[j] in MARK or s[j] == VIRAMA):
            j += 1
        out.append(s[i:j])
        i = j
    return out


def _encode(*seqs):
    """lists of akṣaras as strings of one private-use character each, the same akṣara the same character"""
    vocab = {}
    return ["".join(chr(0xF0000 + vocab.setdefault(x, len(vocab))) for x in s) for s in seqs]


def errors(read, truth):
    """-> (akṣara edits turning the reading into the truth, akṣaras of the truth)"""
    a, b = _encode(aksharas(read), aksharas(truth))
    return Levenshtein.distance(a, b), len(b)


def ger(pairs):
    """GER in % over [(reading, truth, ...)]: akṣara edits per 100 akṣaras of the truth"""
    e = n = 0
    for p in pairs:
        de, dn = errors(p[0], p[1])
        e, n = e + de, n + dn
    return 100.0 * e / max(1, n)
