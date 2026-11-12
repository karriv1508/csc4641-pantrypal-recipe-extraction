"""Loading the processed datasets."""
import json
from pathlib import Path

from .labels import merge_tag
from .textproc import gold_words

ROOT = Path(__file__).resolve().parent.parent
TASTESET = ROOT / "data_processed" / "tasteset_bio.jsonl"
SCRAPED_GOLD = ROOT / "data" / "eval" / "scraped_gold.jsonl"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def tasteset(split=None, path=TASTESET):
    """Gold TASTEset lines as dicts with words, 12-type tags, text, id, split."""
    out = []
    for r in read_jsonl(path):
        if split and r["split"] != split:
            continue
        w, t = gold_words(r)
        out.append({"id": r["id"], "recipe_id": r["recipe_id"], "split": r["split"],
                    "text": r["text"], "words": w, "tags": [merge_tag(x) for x in t]})
    return out


def scraped_gold(path=SCRAPED_GOLD):
    """Hand-labeled cross-site test lines (already in the 12-type tag set)."""
    return read_jsonl(path)
