"""PantryPal ingredient-line extractor: raw ingredient lines in, tagged spans and JSON records out.

Examples
  python predict.py "1 1/2 cups whole milk" "2 cloves garlic, minced"
  python predict.py --file my_recipe.txt --scale 2          # one ingredient per line
  python predict.py --model crf "1 (15 oz) can black beans, drained"
  echo "salt and pepper to taste" | python predict.py --json
  python predict.py --recipes data_processed/scraped_clean.jsonl --out results/12types/scraped_predictions.jsonl

Models: roberta (default, fine-tuned RoBERTa-base), crf (baseline), rules (baseline).
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pantrypal.decode import decode, scale  # noqa: E402
from pantrypal.labels import repair, spans  # noqa: E402
from pantrypal.taggers import load_tagger  # noqa: E402
from pantrypal.textproc import detok, prepare  # noqa: E402

SHORT = {"QUANTITY": "QTY", "PHYSICAL_QUALITY": "QUALITY", "TRADE_NAME": "BRAND"}


def run(tagger, lines):
    prepped = [prepare(x) for x in lines]
    outs = tagger.predict([w for _, w, _ in prepped])
    recs = []
    for raw, (clean, words, offsets), (tags, conf) in zip(lines, prepped, outs):
        tags = repair(tags)
        rec = decode(words, tags, probs=conf, text=clean)
        rec["entities"] = [{"type": t, "text": detok(words[a:b]),
                            "start": offsets[a][0], "end": offsets[b - 1][1]} for t, a, b in spans(tags)]
        rec["raw"] = raw
        recs.append((rec, words, tags))
    return recs


def show(rec, words, tags, factor):
    marked, i = [], 0
    for t, a, b in spans(tags) + [(None, len(words), len(words))]:
        marked += words[i:a]
        if t:
            marked.append(f"[{detok(words[a:b])}]{SHORT.get(t, t)}")
        i = b
    print(rec["raw"])
    print("  tags  : " + " ".join(marked))
    q = rec["quantity"]
    amount = "-" if not q else ("to taste / as needed" if not q["scalable"] else
                                (q["text"] or "1") + (f"  (= {q['min']:g}-{q['max']:g})" if q["min"] != q["max"]
                                                      else f"  (= {q['value']:g})"))
    print(f"  item  : {rec['ingredient'] or '(none found)'}")
    print(f"  amount: {amount}   unit: {rec['unit'] or '-'}"
          + (f"   package: {rec['package']['quantity']:g} {rec['package']['unit']}" if rec["package"] else ""))
    extra = [("prep", ", ".join(rec["preparation"])), ("quality", ", ".join(rec["qualities"])),
             ("purpose", rec["purpose"]), ("brand", ", ".join(rec["brand"])),
             ("or", ", ".join(rec["alternatives"])), ("also", ", ".join(rec["also"])),
             ("notes", "; ".join(rec["notes"]))]
    extra = [f"{k}: {v}" for k, v in extra if v]
    if extra:
        print("  other : " + " | ".join(extra))
    if factor != 1:
        print(f"  x{factor:g}    : {scale(rec, factor) or 'not scaled (no fixed amount)'}")
    conf = rec["confidence"]
    print(f"  conf. : {conf if conf is not None else 'n/a'}" + ("   ** needs review **" if rec["needs_review"] else ""))
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lines", nargs="*", help="ingredient lines (reads stdin if none and no --file)")
    ap.add_argument("--file", help="text file, one ingredient line per line")
    ap.add_argument("--recipes", help="JSONL of recipes with an 'ingredients' list (e.g. scraped_clean.jsonl)")
    ap.add_argument("--out", help="with --recipes: where to write the JSONL of records")
    ap.add_argument("--model", default="roberta", choices=["roberta", "crf", "rules"])
    ap.add_argument("--model-path", help="override the saved model location")
    ap.add_argument("--json", action="store_true", help="print one JSON record per line instead of a table")
    ap.add_argument("--scale", type=float, default=1.0, help="also show amounts scaled by this factor")
    a = ap.parse_args()

    t0 = time.time()
    tagger = load_tagger(a.model, a.model_path)
    load_s = time.time() - t0

    if a.recipes:
        recipes = [json.loads(l) for l in open(a.recipes, encoding="utf-8") if l.strip()]
        lines = [(r["id"], x) for r in recipes for x in r["ingredients"]]
        t = time.time()
        recs = run(tagger, [x for _, x in lines])
        secs = time.time() - t
        out = Path(a.out or ROOT / "results" / "12types" / f"{Path(a.recipes).stem}_{a.model}.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        stats = Counter()
        with open(out, "w", encoding="utf-8") as f:
            for (rid, _), (rec, _, _) in zip(lines, recs):
                f.write(json.dumps({"recipe": rid, **rec}, ensure_ascii=False) + "\n")
                stats["ingredient found"] += rec["ingredient"] is not None
                stats["scalable amount"] += rec["scalable"]
                stats["unit found"] += rec["unit"] is not None
                stats["needs review"] += rec["needs_review"]
        n = len(recs)
        print(f"{len(recipes)} recipes, {n} lines -> {out}  ({1000 * secs / n:.1f} ms/line, model load {load_s:.1f}s)")
        for k, v in stats.items():
            print(f"  {k:17s} {v:5d}  ({100 * v / n:.1f}%)")
        return

    if a.file:
        lines = [l.rstrip("\n") for l in open(a.file, encoding="utf-8")]
    elif a.lines:
        lines = a.lines
    else:
        lines = [l.rstrip("\n") for l in sys.stdin]
    lines = [l for l in lines if l.strip()]
    for rec, words, tags in run(tagger, lines):
        if a.json:
            if a.scale != 1:
                rec["scaled"] = scale(rec, a.scale)
            print(json.dumps(rec, ensure_ascii=False))
        else:
            show(rec, words, tags, a.scale)


if __name__ == "__main__":
    main()
