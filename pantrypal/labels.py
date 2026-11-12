"""Entity types and BIO label set.

TASTEset-2.0 has 15 entity types. Three are too rare to learn (EXCLUSIVE 1, EXCLUDED 7,
POSSIBLE_SUBSTITUTE 13 training examples), so they are merged into FOOD, leaving 12 types
and 25 BIO labels. Cue words ("or", "substitute") are handled in decoding instead.
"""
TYPES = ["FOOD", "QUANTITY", "UNIT", "PROCESS", "PHYSICAL_QUALITY", "COLOR", "TASTE",
         "PURPOSE", "PART", "TRADE_NAME", "DIET", "EXAMPLE"]
MERGE = {"EXCLUSIVE": "FOOD", "EXCLUDED": "FOOD", "POSSIBLE_SUBSTITUTE": "FOOD"}
LABELS = ["O"] + [f"{p}-{t}" for t in TYPES for p in ("B", "I")]


def merge_tag(tag: str) -> str:
    """Map a 15-type BIO tag to the 12-type set."""
    if tag == "O":
        return tag
    prefix, typ = tag.split("-", 1)
    return f"{prefix}-{MERGE.get(typ, typ)}"


def repair(tags):
    """Make a predicted BIO sequence valid: an I- that does not continue its type becomes B-."""
    out, prev = [], "O"
    for t in tags:
        if t.startswith("I-") and prev[2:] != t[2:]:
            t = "B-" + t[2:]
        out.append(t)
        prev = t
    return out


def spans(tags):
    """BIO tags -> list of (type, start, end) with end exclusive."""
    res, start, typ = [], None, None
    for i, t in enumerate(list(tags) + ["O"]):
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and t[2:] != typ):
            if typ is not None:
                res.append((typ, start, i))
            start, typ = (i, t[2:]) if t != "O" else (None, None)
    return res
