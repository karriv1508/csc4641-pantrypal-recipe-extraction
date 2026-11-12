"""Mine the rule baseline's word lists from the NYT ingredient data (data_processed/nyt_clean.jsonl).

NYT's labels are fields, not token tags, so they are too noisy to train on (Section 3 of our
data report), but they are reliable for word lists: which words are units, which appear in
ingredient names, and which appear in preparation comments. The CRF also uses these lists as
features.

  python scripts/build_lexicon.py --nyt data_processed/nyt_clean.jsonl --out models/rules/nyt_lexicon.json
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pantrypal.textproc import TOKEN_RE  # noqa: E402

# NYT spells units out, so abbreviations common on blogs are added by hand
ABBREVIATIONS = ["tbsp", "tbs", "tbl", "tsp", "c", "oz", "lb", "lbs", "g", "gr", "kg", "ml", "l",
                 "qt", "pt", "pkg", "T", "t"]
NOT_PROCESS = {"approximately", "been", "between", "called", "often", "only", "even"}


def words(s):
    return [w.lower() for w in TOKEN_RE.findall(s or "") if re.fullmatch(r"[A-Za-z][A-Za-z'-]*", w)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nyt", default=str(ROOT / "data_processed/nyt_clean.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "models/rules/nyt_lexicon.json"))
    ap.add_argument("--min-count", type=int, default=5)
    a = ap.parse_args()

    unit_tok, in_name, in_comment, total = Counter(), Counter(), Counter(), Counter()
    n = 0
    for line in open(a.nyt, encoding="utf-8"):
        r = json.loads(line)
        n += 1
        for t, g in zip(r["tokens"], r["tags"]):
            if g.endswith("UNIT") and re.fullmatch(r"[A-Za-z]+", t):
                unit_tok[t.lower()] += 1
        total.update(words(r["text"]))
        in_name.update(set(words(r["name"])))
        in_comment.update(set(words(r["comment"])))

    m = a.min_count
    units = sorted({u for u, c in unit_tok.items() if c >= m} | set(ABBREVIATIONS))
    food = sorted(w for w, c in in_name.items() if c >= m and c / total[w] >= 0.5)
    # preparation words: mostly in comments, and verb or adverb shaped
    process = sorted(w for w, c in in_comment.items()
                     if c >= m and c / total[w] >= 0.5 and w not in food and w not in NOT_PROCESS
                     and re.search(r"(ed|en|ly)$", w))
    lex = {"source": "NYT ingredient-phrase-tagger snapshot, cleaned (nyt_clean.jsonl)",
           "lines": n, "min_count": m, "units": units, "food_words": food, "process_words": process}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(lex, indent=1), encoding="utf-8")
    print(f"{n} NYT lines -> {len(units)} unit forms, {len(food)} food words, {len(process)} process words")
    print("units:", units[:40])
    print("process:", process[:40])


if __name__ == "__main__":
    main()
