"""Turn the hand-labeled scraped lines (data/eval/scraped_annotations.txt) into token-level gold.

Checks that every annotated line, with its markup removed, is exactly the cleaned line that
scrape_recipes.py wrote for that recipe, so the gold matches what the system will see.

  python scripts/build_scraped_eval.py \
      --scraped data_processed/scraped_clean.jsonl --out data/eval/scraped_gold.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pantrypal.textproc import TOKEN_RE, norm_token  # noqa: E402

CODES = {"F": "FOOD", "Q": "QUANTITY", "U": "UNIT", "P": "PROCESS", "PQ": "PHYSICAL_QUALITY",
         "C": "COLOR", "T": "TASTE", "PU": "PURPOSE", "PA": "PART", "TN": "TRADE_NAME",
         "D": "DIET", "E": "EXAMPLE"}
SPAN = re.compile(r"\{([^{}|]+)\|([A-Z]+)\}")


def parse(annot):
    """'{2|Q} {cups|U} oats' -> ('2 cups oats', [(0,1,'QUANTITY'), (2,6,'UNIT')])."""
    text, spans, pos = "", [], 0
    for m in SPAN.finditer(annot):
        text += annot[pos:m.start()]
        a = len(text)
        text += m.group(1)
        spans.append((a, len(text), CODES[m.group(2)]))
        pos = m.end()
    return text + annot[pos:], spans


def to_bio(text, spans):
    words, tags = [], []
    for m in TOKEN_RE.finditer(text):
        a, b = m.span()
        lab = next((s for s in spans if s[0] <= a and b <= s[1]), None)
        if lab is None and any(s[0] < b and a < s[1] for s in spans):
            raise ValueError(f"span boundary inside token {m.group()!r}: {text}")
        for k, w in enumerate(norm_token(m.group()).split(" ")):
            words.append(w)
            if lab is None:
                tags.append("O")
            else:
                first = a == lab[0] and k == 0
                tags.append(("B-" if first else "I-") + lab[2])
    return words, tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default=str(ROOT / "data/eval/scraped_annotations.txt"))
    ap.add_argument("--scraped", default=str(ROOT / "data_processed/scraped_clean.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/eval/scraped_gold.jsonl"))
    a = ap.parse_args()

    recipes = {}
    if Path(a.scraped).exists():
        recipes = {r["id"]: r for r in map(json.loads, open(a.scraped, encoding="utf-8"))}
    else:
        print(f"note: {a.scraped} not found; skipping the check against the scraped file")

    groups, cur = {}, None
    for raw in open(a.annotations, encoding="utf-8"):
        line = raw.rstrip("\n")
        if line.startswith("## "):
            cur = line[3:].strip()
            groups[cur] = []
        elif line.strip() and not line.startswith("#"):
            groups[cur].append(line)

    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for rid, lines in groups.items():
            rec = recipes.get(rid)
            if rec is not None and len(rec["ingredients"]) != len(lines):
                sys.exit(f"{rid}: {len(lines)} annotated lines, {len(rec['ingredients'])} scraped")
            for i, annot in enumerate(lines):
                text, spans = parse(annot)
                if rec is not None and text != rec["ingredients"][i]:
                    sys.exit(f"{rid} line {i}: text mismatch\n  {text!r}\n  {rec['ingredients'][i]!r}")
                words, tags = to_bio(text, spans)
                f.write(json.dumps({"id": f"{rid}-{i:02d}", "recipe": rid, "line_no": i,
                                    "link": rec["link"] if rec else None, "text": text,
                                    "words": words, "tags": tags}, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} lines from {len(groups)} recipes to {a.out}")


if __name__ == "__main__":
    main()
