"""
CSC 4641 - Recipe Information Extraction (PantryPal)
Data cleaning and preparation pipeline.

Inputs (in the project folder):
  TASTEset.csv                               - 1,000 recipes with gold entity spans
  nyt-ingredients-snapshot-2015.csv          - ~179k NYT ingredient lines (name/qty/unit/comment)
  RecipeNLG_dataset.csv/RecipeNLG_dataset.csv - ~2.2M unlabeled recipes

Outputs (in data_processed/):
  tasteset_bio.jsonl     one ingredient line per record, tokens + BIO tags + train/val/test split
  nyt_clean.jsonl        deduplicated, HTML-stripped NYT lines with structured fields + coarse BIO tags
  recipenlg_subset.jsonl cleaned, deduplicated 25k-recipe subset, tokenized, ready for silver labeling
  stats.json             counts reported in the Data Collection report

Usage:
  python scripts/clean_data.py --root "<project folder>" [--recipenlg-target 25000] [--max-rows N]
"""
import argparse, csv, html, json, random, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
SEED = 4641

# ---------------------------------------------------------------- text utils
FRACTIONS = {"½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4", "¾": "3/4", "⅕": "1/5",
             "⅖": "2/5", "⅗": "3/5", "⅘": "4/5", "⅙": "1/6", "⅚": "5/6", "⅛": "1/8",
             "⅜": "3/8", "⅝": "5/8", "⅞": "7/8"}
FRAC_CHARS = "".join(FRACTIONS)

# numbers, fractions (1/2, 1½), words with internal - or ', or any single symbol
TOKEN_RE = re.compile(rf"\d*[{FRAC_CHARS}]|\d+(?:[./]\d+)?|\w+(?:[-'’]\w+)*|[^\w\s]")


def norm_token(tok: str) -> str:
    """ASCII-normalize one token: '1½' -> '1 1/2', '½' -> '1/2', curly quotes -> straight."""
    m = re.fullmatch(rf"(\d*)([{FRAC_CHARS}])", tok)
    if m:
        whole, frac = m.groups()
        return f"{whole} {FRACTIONS[frac]}" if whole else FRACTIONS[frac]
    return tok.replace("’", "'")


def tokenize(text):
    """Return [(token, start, end)] with character offsets into text."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


TAG_RE = re.compile(r"<[^>]+>")
SEE_RECIPE_RE = re.compile(r"\(\s*<a [^>]*>.*?</a>\s*\)|<a [^>]*>.*?</a>", re.I)


def clean_line(s: str) -> str:
    """Cleaning for scraped/unlabeled ingredient lines (NOT used on gold TASTEset text)."""
    s = html.unescape(s)
    s = SEE_RECIPE_RE.sub("", s)          # NYT "(see recipe)" cross-links
    s = TAG_RE.sub("", s)                  # any remaining HTML tags
    s = unicodedata.normalize("NFC", s)
    s = s.replace("⁄", "/")           # fraction slash
    s = re.sub(rf"(\d)([{FRAC_CHARS}])", lambda m: m.group(1) + " " + FRACTIONS[m.group(2)], s)
    s = re.sub(rf"[{FRAC_CHARS}]", lambda m: FRACTIONS[m.group()], s)
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[®™©]", "", s)
    s = re.sub(r"^\s*[-•*▢□]+\s*", "", s)  # bullet characters
    s = re.sub(r"\s+", " ", s).strip(" ,;")
    return s


# ---------------------------------------------------------------- TASTEset
SPAN_RE = re.compile(r"\((\d+),\s*(\d+)\)")


def process_tasteset(path: Path, out: Path, stats: dict):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    ids = list(range(len(rows)))
    random.Random(SEED).shuffle(ids)
    n = len(ids)
    split_of = {}
    for k, rid in enumerate(ids):  # recipe-level split so no recipe straddles splits
        split_of[rid] = "train" if k < 0.70 * n else ("val" if k < 0.85 * n else "test")

    ent_counts, split_counts = Counter(), Counter()
    span_mismatch = lines_total = tokens_total = price_noise = 0
    line_index = defaultdict(set)  # normalized line -> recipe ids (for leakage check)
    recipe_lines = {}

    with open(out, "w", encoding="utf-8") as f:
        for rid, r in enumerate(rows):
            text = r["ingredients"]
            ents = json.loads(r["ingredients_entities"])
            # character-level labels
            char_lab = [None] * len(text)
            for e in ents:
                spans = sorted((int(a), int(b)) for a, b in SPAN_RE.findall(e["span"]))
                merged = []  # merge fragments separated only by whitespace
                for a, b in spans:
                    if merged and text[merged[-1][1]:a].strip() == "":
                        merged[-1] = (merged[-1][0], b)
                    else:
                        merged.append((a, b))
                spans = merged
                ent_counts[e["type"]] += 1
                joined = " ".join(text[a:b] for a, b in spans)
                if joined.split() != e["entity"].split():
                    span_mismatch += 1
                for a, b in spans:  # discontinuous entities: each fragment starts with B-
                    for c in range(a, min(b, len(text))):
                        char_lab[c] = ("B" if c == a else "I", e["type"])

            # split into lines, keep offsets
            pos, lines = 0, []
            for ln in text.split("\n"):
                if ln.strip():
                    lines.append((ln, pos))
                pos += len(ln) + 1
            recipe_lines[rid] = [l.lower().strip() for l, _ in lines]

            for li, (ln, off) in enumerate(lines):
                toks = tokenize(ln)
                tags = []
                for tok, a, b in toks:
                    lab = char_lab[off + a]
                    if lab is None:
                        tags.append("O")
                    else:
                        # token starting mid-entity continues it unless it starts the span
                        prev = char_lab[off + a - 1] if off + a > 0 else None
                        bio = "B" if lab[0] == "B" or prev is None or prev[1] != lab[1] else "I"
                        tags.append(f"{bio}-{lab[1]}")
                if "$" in ln:
                    price_noise += 1
                rec = {
                    "id": f"tasteset-{rid:04d}-{li:02d}",
                    "source": "TASTEset-2.0",
                    "recipe_id": rid,
                    "line_no": li,
                    "text": ln,
                    "tokens": [t for t, _, _ in toks],
                    "norm_tokens": [norm_token(t) for t, _, _ in toks],
                    "offsets": [[a, b] for _, a, b in toks],
                    "tags": tags,
                    "split": split_of[rid],
                    "label_type": "gold",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                lines_total += 1
                tokens_total += len(toks)
                split_counts[split_of[rid]] += 1
                line_index[ln.lower().strip()].add(rid)

    stats["tasteset"] = {
        "recipes": len(rows), "ingredient_lines": lines_total, "tokens": tokens_total,
        "entities_by_type": dict(ent_counts.most_common()),
        "lines_by_split": dict(split_counts),
        "span_text_mismatches": span_mismatch,
        "lines_with_price_or_promo_text": price_noise,
    }
    return line_index, recipe_lines


# ---------------------------------------------------------------- NYT
UNIT_FORMS = None


def tag_nyt(tokens, name, unit, qty):
    """Coarse BIO tags aligned from NYT's structured fields (for the rule-based baseline)."""
    name_toks = {t.lower() for t in TOKEN_RE.findall(name or "")}
    unit_l = (unit or "").lower()
    tags, prev = [], "O"
    for i, t in enumerate(tokens):
        tl = t.lower()
        lab = "O"
        if re.fullmatch(r"[\d/.]+|\d+ \d+/\d+", t) and i < 4:
            lab = "QUANTITY"
        elif unit_l and (tl == unit_l or tl.rstrip("s") == unit_l or tl == unit_l + "es"):
            lab = "UNIT"
        elif tl in name_toks and tl not in {",", "(", ")", "or", "and"}:
            lab = "FOOD"
        if lab == "O":
            tags.append("O")
        else:
            tags.append(("I-" if prev == lab else "B-") + lab)
        prev = lab
    return tags


def process_nyt(path: Path, out: Path, stats: dict):
    seen = set()
    total = missing = html_fixed = dups = kept = 0
    with open(path, encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as f:
        for r in csv.DictReader(fin):
            total += 1
            raw = (r.get("input") or "").strip()
            if not raw or not (r.get("name") or "").strip():
                missing += 1
                continue
            text = clean_line(raw)
            if text != raw and "<" in raw:
                html_fixed += 1
            key = text.lower()
            if key in seen:
                dups += 1
                continue
            seen.add(key)
            toks = [norm_token(t) for t in TOKEN_RE.findall(text)]
            unit = (r.get("unit") or "").strip().lower() or None
            rec = {
                "id": f"nyt-{r['index']}",
                "source": "NYT-ingredient-phrase-tagger",
                "raw": raw,
                "text": text,
                "name": r["name"].strip(),
                "qty": float(r["qty"]) if r.get("qty") else None,
                "range_end": float(r["range_end"]) if r.get("range_end") not in (None, "", "0.0") else None,
                "unit": unit,
                "comment": (r.get("comment") or "").strip() or None,
                "tokens": toks,
                "tags": tag_nyt(toks, r["name"], unit, r.get("qty")),
                "label_type": "aligned-from-fields",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
    stats["nyt"] = {"rows_in": total, "dropped_missing_text_or_name": missing,
                    "html_cleaned": html_fixed, "exact_duplicates_removed": dups,
                    "rows_out": kept}


# ---------------------------------------------------------------- RecipeNLG
def process_recipenlg(path: Path, out: Path, stats: dict, target: int, max_rows,
                      taste_index, taste_lines):
    rng = random.Random(SEED)
    # RecipeNLG has ~2.23M rows, ordered by source (Gathered first, Recipes1M later), so
    # sample uniformly across the WHOLE file rather than taking the first N recipes.
    keep_prob = min(1.0, target * 1.03 / 2_231_142)
    seen = set()
    c = Counter()
    kept = 0
    total_lines = total_tokens = 0
    src_counts = Counter()
    with open(path, encoding="utf-8", newline="") as fin, open(out, "w", encoding="utf-8") as f:
        reader = csv.reader(fin)
        header = next(reader)
        col = {h: i for i, h in enumerate(header)}
        for row in reader:
            c["rows_scanned"] += 1
            if max_rows and c["rows_scanned"] > max_rows:
                break
            if rng.random() > keep_prob:
                continue
            c["sampled"] += 1
            try:
                raw_ings = json.loads(row[col["ingredients"]])
                ner = json.loads(row[col["NER"]])
            except (json.JSONDecodeError, IndexError):
                c["dropped_bad_json"] += 1
                continue
            ings = [clean_line(x) for x in raw_ings]
            ings = [x for x in ings if x and re.search(r"[A-Za-z]", x)]
            if len(ings) < 2 or len(ings) > 40:
                c["dropped_too_few_or_many_ingredients"] += 1
                continue
            title = re.sub(r"\s*\((?:servings|serves|makes)[^)]*\)", "", clean_line(row[col["title"]]), flags=re.I)
            key = (title.lower(), tuple(x.lower() for x in ings))
            if key in seen:
                c["dropped_duplicate"] += 1
                continue
            seen.add(key)
            # leakage guard: drop if >=80% of lines match a single TASTEset recipe
            hits = Counter()
            for x in ings:
                for rid in taste_index.get(x.lower(), ()):
                    hits[rid] += 1
            if hits and max(hits.values()) / len(ings) >= 0.8:
                c["dropped_overlaps_tasteset"] += 1
                continue
            toks = [[norm_token(t) for t in TOKEN_RE.findall(x)] for x in ings]
            rec = {
                "id": f"recipenlg-{row[0]}",
                "source": "RecipeNLG",
                "origin": row[col["source"]],  # Gathered / Recipes1M
                "link": row[col["link"]],
                "title": title,
                "raw_ingredients": raw_ings,
                "ingredients": ings,
                "tokens": toks,
                "ner_foods": ner,
                "tags": None,  # filled later by TASTEset-trained model (silver labels)
                "label_type": "unlabeled",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            src_counts[row[col["source"]]] += 1
            total_lines += len(ings)
            total_tokens += sum(len(t) for t in toks)
            if kept >= target:
                break
    stats["recipenlg"] = {**c, "recipes_out": kept, "ingredient_lines": total_lines,
                          "tokens": total_tokens, "by_origin": dict(src_counts),
                          "sample_rate": keep_prob, "seed": SEED}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--recipenlg-target", type=int, default=25000)
    ap.add_argument("--max-rows", type=int, default=None, help="limit RecipeNLG rows scanned (testing)")
    ap.add_argument("--skip-recipenlg", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    outdir = root / "data_processed"
    outdir.mkdir(exist_ok=True)
    stats = {}

    print("TASTEset ...", flush=True)
    idx, tl = process_tasteset(root / "TASTEset.csv", outdir / "tasteset_bio.jsonl", stats)
    print("NYT ...", flush=True)
    process_nyt(root / "nyt-ingredients-snapshot-2015.csv", outdir / "nyt_clean.jsonl", stats)
    if not a.skip_recipenlg:
        print("RecipeNLG ...", flush=True)
        p = root / "RecipeNLG_dataset.csv"
        if p.is_dir():
            p = p / "RecipeNLG_dataset.csv"
        process_recipenlg(p, outdir / "recipenlg_subset.jsonl", stats,
                          a.recipenlg_target, a.max_rows, idx, tl)
    json.dump(stats, open(outdir / "stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
