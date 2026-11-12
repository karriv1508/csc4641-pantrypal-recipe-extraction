"""Turn tagged tokens into the JSON record PantryPal stores for one ingredient line.

The tagger finds spans; these rules decide what they mean:
  * the first FOOD span is the ingredient; COLOR/TASTE/DIET words right before it and a PART
    right after it belong to its name ("brown sugar", "chicken breasts")
  * other FOOD spans joined by "or" are alternatives, by "and"/"+" are extra items
  * the first QUANTITY+UNIT outside parentheses is the amount; "2 14 ounce cans" gives
    2 cans with a 14-ounce package size; amounts in parentheses are kept as notes
  * "to taste" or a missing quantity marks the line as not scalable
"""
import re
from fractions import Fraction

from .labels import repair, spans
from .textproc import detok

NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
             "half": 0.5, "dozen": 12, "couple": 2}
UNITS = {
    "teaspoon": ["teaspoon", "teaspoons", "tsp", "tsps", "t"],
    "tablespoon": ["tablespoon", "tablespoons", "tbsp", "tbsps", "tbs", "tbl", "T"],
    "cup": ["cup", "cups", "c"],
    "ounce": ["ounce", "ounces", "oz", "ozs"],
    "fluid ounce": ["fl oz", "fluid ounce", "fluid ounces"],
    "pound": ["pound", "pounds", "lb", "lbs"],
    "gram": ["gram", "grams", "g", "gr"],
    "kilogram": ["kilogram", "kilograms", "kg"],
    "milliliter": ["milliliter", "milliliters", "ml"],
    "liter": ["liter", "liters", "litre", "litres", "l"],
    "pint": ["pint", "pints", "pt"], "quart": ["quart", "quarts", "qt"],
    "clove": ["clove", "cloves"], "can": ["can", "cans"], "package": ["package", "packages", "pkg"],
    "packet": ["packet", "packets"], "jar": ["jar", "jars"], "bottle": ["bottle", "bottles"],
    "pinch": ["pinch", "pinches"], "dash": ["dash", "dashes"], "slice": ["slice", "slices"],
    "piece": ["piece", "pieces"], "stick": ["stick", "sticks"], "bunch": ["bunch", "bunches"],
    "head": ["head", "heads"], "sprig": ["sprig", "sprigs"], "stalk": ["stalk", "stalks"],
    "handful": ["handful", "handfuls"], "bag": ["bag", "bags"], "box": ["box", "boxes"],
    "inch": ["inch", "inches", "in"],
}
UNIT_OF = {}
for canon, forms in UNITS.items():
    for f in forms:
        UNIT_OF[f if f in ("T", "t") else f.lower()] = canon
CONTAINERS = {"can", "package", "packet", "jar", "bottle", "bag", "box", "stick"}
NAME_MODIFIERS = {"COLOR", "TASTE", "DIET"}


def normalize_unit(text):
    t = text.strip().rstrip(".").strip()
    if t in ("T", "t"):
        return UNIT_OF[t]
    t = t.lower().replace(".", "")
    return UNIT_OF.get(t, t)


def parse_number(s):
    s = s.strip().lower()
    if s in NUM_WORDS:
        return float(NUM_WORDS[s])
    parts = s.split()
    if len(parts) > 2 or (len(parts) == 2 and "/" not in parts[1]):
        return None  # "2 14" is two numbers, not a mixed fraction
    try:
        return float(sum(Fraction(p) for p in parts))  # "1 1/2" -> 1.5
    except (ValueError, ZeroDivisionError):
        return None


def parse_quantity(text):
    """'1 1/2' -> {value 1.5}; '2 - 3' / '2 to 3' -> {min 2, max 3}; 'to taste' -> not scalable."""
    t = text.lower().strip()
    if re.search(r"\btaste\b|\bneeded\b|\bdesired\b", t):
        return {"text": text, "value": None, "min": None, "max": None, "scalable": False}
    t = re.sub(r"^(about|approx\.?|approximately|around|scant|heaping)\s+", "", t)
    parts = re.split(r"\s*(?:-|–|\bto\b|\bor\b)\s*", t)
    vals = [parse_number(p) for p in parts if p]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"text": text, "value": None, "min": None, "max": None, "scalable": False}
    lo, hi = min(vals), max(vals)
    return {"text": text, "value": (lo + hi) / 2 if lo != hi else lo, "min": lo, "max": hi,
            "scalable": True}


def fmt_number(x):
    """1.5 -> '1 1/2', 0.333 -> '1/3' (nearest 1/8 or 1/3)."""
    if x is None:
        return None
    best = min((Fraction(round(x * d), d) for d in (1, 2, 3, 4, 8)), key=lambda f: abs(float(f) - x))
    whole, frac = divmod(best, 1)
    if frac == 0:
        return str(int(whole))
    return f"{int(whole)} {frac}" if whole else str(frac)


def _paren_depth(words):
    depth, out = 0, []
    for w in words:
        if w == "(":
            depth += 1
        out.append(depth > 0)
        if w == ")":
            depth = max(0, depth - 1)
    return out


def decode(words, tags, probs=None, text=None):
    """words, BIO tags (and optional per-token confidence) -> PantryPal record."""
    tags = repair(tags)
    sp = spans(tags)
    in_par = _paren_depth(words)
    seg = lambda a, b: detok(words[a:b])  # noqa: E731
    by_type = lambda typ: [s for s in sp if s[0] == typ]  # noqa: E731

    rec = {"text": text if text is not None else detok(words), "ingredient": None, "food": None,
           "quantity": None, "unit": None, "unit_text": None, "package": None,
           "preparation": [], "qualities": [], "purpose": None, "brand": [], "alternatives": [],
           "also": [], "notes": [], "scalable": False, "confidence": None, "needs_review": False}

    foods = by_type("FOOD")
    main = next((s for s in foods if not in_par[s[1]]), foods[0] if foods else None)
    if main:
        a, b = main[1], main[2]
        prev = {e: t for t, s, e in [(x[0], x[1], x[2]) for x in sp]}  # end index -> type
        while a > 0 and prev.get(a) in NAME_MODIFIERS:
            a = next(s[1] for s in sp if s[2] == a)
        nxt = next((s for s in sp if s[1] == b and s[0] == "PART"), None)
        if nxt:
            b = nxt[2]
        rec["food"] = seg(main[1], main[2])
        rec["ingredient"] = seg(a, b)
        for s in foods:
            if s is main:
                continue
            between = [w.lower() for w in words[main[2]:s[1]]] if s[1] > main[2] else []
            name = seg(s[1], s[2])
            if "or" in between or "/" in between:
                rec["alternatives"].append(name)
            elif in_par[s[1]]:
                rec["notes"].append(name)
            else:
                rec["also"].append(name)

    # "2 14 ounce cans" tagged as one QUANTITY span: split it into the count and the package size
    for s in [s for s in sp if s[0] == "QUANTITY" and s[2] - s[1] == 2]:
        if re.fullmatch(r"\d+ \d+(?:\.\d+)?", seg(s[1], s[2])):
            k = sp.index(s)
            sp[k:k + 1] = [("QUANTITY", s[1], s[1] + 1), ("QUANTITY", s[1] + 1, s[2])]
    # amount: the first QUANTITY outside parentheses that reads as an amount ("Juice of 1 lemon"
    # skips a mis-tagged "Juice"); a following QUANTITY UNIT UNIT is a package size
    qs = [s for s in by_type("QUANTITY") if not in_par[s[1]]] or by_type("QUANTITY")
    readable = [s for s in qs if parse_quantity(seg(s[1], s[2]))["value"] is not None
                or re.search(r"\b(taste|needed|desired)\b", seg(s[1], s[2]).lower())]
    units = by_type("UNIT")
    if qs:
        q = (readable or qs)[0]
        rec["quantity"] = parse_quantity(seg(q[1], q[2]))
        rec["scalable"] = rec["quantity"]["scalable"]
        after = [s for s in sp if s[1] >= q[2]]
        # "2 14 ounce cans" / "1 (15 oz) can": QUANTITY QUANTITY UNIT UNIT
        if len(after) >= 3 and after[0][0] == "QUANTITY" and after[1][0] == "UNIT" and after[2][0] == "UNIT" \
                and normalize_unit(seg(after[2][1], after[2][2])) in CONTAINERS:
            pq = parse_quantity(seg(after[0][1], after[0][2]))
            rec["package"] = {"quantity": pq["value"], "unit": normalize_unit(seg(after[1][1], after[1][2]))}
            u = after[2]
        elif len(after) >= 2 and after[0][0] == "UNIT" and after[1][0] == "UNIT" \
                and normalize_unit(seg(after[1][1], after[1][2])) in CONTAINERS:
            # "6-ounce package": the number belongs to the package size, so one package
            rec["package"] = {"quantity": rec["quantity"]["value"], "unit": normalize_unit(seg(after[0][1], after[0][2]))}
            rec["quantity"] = parse_quantity("1")
            rec["quantity"]["text"] = None
            u = after[1]
        else:
            u = after[0] if after and after[0][0] == "UNIT" else None
        if u:
            rec["unit_text"] = seg(u[1], u[2])
            rec["unit"] = normalize_unit(rec["unit_text"])
        # other amounts in parentheses, e.g. "(about 1 cup)", "(470 grams)"
        for s in by_type("QUANTITY"):
            if s is not q and in_par[s[1]] and not (rec["package"] and s[1] < (u[1] if u else 0)):
                nu = next((x for x in units if x[1] == s[2]), None)
                rec["notes"].append(seg(s[1], nu[2] if nu else s[2]))
    elif units:  # "pinch of salt", "a squeeze of lime juice": one unit, unscaled amount
        u = units[0]
        rec["unit_text"] = seg(u[1], u[2])
        rec["unit"] = normalize_unit(rec["unit_text"])

    rec["preparation"] = [seg(s[1], s[2]) for s in by_type("PROCESS")]
    rec["qualities"] = [seg(s[1], s[2]) for s in sp if s[0] == "PHYSICAL_QUALITY"]
    p = by_type("PURPOSE")
    rec["purpose"] = seg(p[0][1], p[0][2]) if p else None
    rec["brand"] = [seg(s[1], s[2]) for s in by_type("TRADE_NAME")]
    if probs is not None and len(probs):
        ent = [p for p, t in zip(probs, tags) if t != "O"] or list(probs)
        rec["confidence"] = round(float(min(ent)), 3)
    rec["needs_review"] = rec["ingredient"] is None or (rec["confidence"] is not None and rec["confidence"] < 0.6)
    return rec


def scale(rec, factor):
    """Return the amount scaled by factor, as display text ('3 cups'); None if not scalable."""
    q = rec.get("quantity")
    if not q or not q["scalable"]:
        return None
    if q["min"] != q["max"]:
        amt = f"{fmt_number(q['min'] * factor)}-{fmt_number(q['max'] * factor)}"
    else:
        amt = fmt_number(q["value"] * factor)
    unit = rec.get("unit")
    if unit and (q["max"] or 0) * factor > 1 and not unit.endswith("s"):
        unit = {"inch": "inches", "pinch": "pinches", "dash": "dashes", "box": "boxes", "bunch": "bunches"}.get(unit, unit + "s")
    return f"{amt} {unit}" if unit else amt
