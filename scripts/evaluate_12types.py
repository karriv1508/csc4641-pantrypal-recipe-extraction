"""Evaluate the taggers on the held-out sets and write tables, examples, and per-line predictions.

Task: tag each token of an ingredient line with one of 12 entity types (BIO), then decode the
spans into PantryPal's record (quantity, unit, ingredient, ...).
Metrics:
  * entity-level exact-match precision / recall / F1 (micro over all entities; macro over types)
  * line accuracy: every tag on the line correct
  * PantryPal field accuracy: decoded quantity, unit, and ingredient all match the gold decoding
Test sets (none used for training or model selection):
  tasteset-test    824 lines from 150 TASTEset recipes held out by the team's recipe-level split
  unseen           the 600 of those lines whose text never appears in the training split
  scraped          151 hand-labeled lines from 12 held-out Pinch of Yum recipes (cross-site)

  python scripts/evaluate_12types.py --models rules crf roberta --sets tasteset-test unseen scraped
"""
import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pantrypal.data import scraped_gold, tasteset  # noqa: E402
from pantrypal.decode import decode  # noqa: E402
from pantrypal.labels import TYPES, repair, spans  # noqa: E402
from pantrypal.taggers import load_tagger  # noqa: E402
from pantrypal.textproc import detok  # noqa: E402


def norm_text(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def load_sets(names):
    train_text = {norm_text(r["text"]) for r in tasteset("train")}
    sets = {}
    for name in names:
        if name == "val":
            sets[name] = tasteset("val")
        elif name == "tasteset-test":
            sets[name] = tasteset("test")
        elif name == "unseen":
            sets[name] = [r for r in tasteset("test") if norm_text(r["text"]) not in train_text]
        elif name == "scraped":
            sets[name] = scraped_gold()
    overlap = {k: sum(norm_text(r["text"]) in train_text for r in v) for k, v in sets.items()}
    return sets, overlap


def entity_scores(gold_tags, pred_tags):
    by_type = defaultdict(Counter)
    for g, p in zip(gold_tags, pred_tags):
        G, P = set(spans(g)), set(spans(repair(p)))
        for s in G & P:
            by_type[s[0]]["tp"] += 1
        for s in P - G:
            by_type[s[0]]["fp"] += 1
        for s in G - P:
            by_type[s[0]]["fn"] += 1

    def prf(c):
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    total = sum(by_type.values(), Counter())
    per = {}
    for t in TYPES:
        c = by_type.get(t, Counter())
        if c["tp"] + c["fn"] + c["fp"]:
            p, r, f = prf(c)
            per[t] = {"p": p, "r": r, "f1": f, "support": c["tp"] + c["fn"]}
    p, r, f = prf(total)
    supported = [v["f1"] for v in per.values() if v["support"] > 0]
    return {"p": p, "r": r, "f1": f, "macro_f1": sum(supported) / len(supported) if supported else 0.0,
            "per_type": per}


def fields(rec):
    q = rec["quantity"]
    qv = None if not q else (q["min"], q["max"])
    ing = (rec["ingredient"] or "").lower() or None
    return qv, rec["unit"], ing


def phenomena(text):
    t = text.lower()
    cats = []
    if re.search(r"\d\s*(-|to)\s*\d", t):
        cats.append("range")
    if "(" in t:
        cats.append("parenthetical note")
    if re.search(r"\bor\b", t):
        cats.append("alternative (or)")
    if not re.search(r"\d|\b(a|an|half|one|two|three)\b", t):
        cats.append("no quantity")
    if re.search(r"\d+\s*-?\s*(ounce|oz)\s+(can|package|jar)|\(\s*\d+ ?(ounce|oz)", t):
        cats.append("package size")
    if len(t.split()) > 10:
        cats.append("long / chatty (>10 words)")
    if re.search(r"\b(pinch|squeeze|swish|sprinkle|bit)\b", t):
        cats.append("vague amount")
    return cats or ["plain"]


def bootstrap_diff(gold, pa, pb, n=1000, seed=4641):
    """Paired bootstrap over lines: 95% interval for F1(a) - F1(b)."""
    rng = random.Random(seed)
    idx = list(range(len(gold)))
    diffs = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        fa = entity_scores([gold[i] for i in s], [pa[i] for i in s])["f1"]
        fb = entity_scores([gold[i] for i in s], [pb[i] for i in s])["f1"]
        diffs.append(fa - fb)
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["rules", "crf", "roberta"])
    ap.add_argument("--sets", nargs="+", default=["tasteset-test", "unseen", "scraped"])
    ap.add_argument("--out", default=str(ROOT / "results" / "12types"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    sets, overlap = load_sets(a.sets)
    results = {"sets": {k: {"lines": len(v), "lines_also_in_train": overlap[k]} for k, v in sets.items()},
               "models": {}}
    preds = {}
    for m in a.models:
        tagger = load_tagger(m)
        results["models"][m] = {}
        for name, recs in sets.items():
            t = time.time()
            out_ = tagger.predict([r["words"] for r in recs])
            secs = time.time() - t
            pt = [repair(p[0]) for p in out_]
            preds[(m, name)] = pt
            gold = [r["tags"] for r in recs]
            sc = entity_scores(gold, pt)
            line_acc = sum(g == p for g, p in zip(gold, pt)) / len(recs)
            fok, rows = 0, []
            cat_tot, cat_ok = Counter(), Counter()
            for r, p, (_, conf) in zip(recs, pt, out_):
                gd = decode(r["words"], r["tags"], text=r["text"])
                pd = decode(r["words"], p, probs=conf, text=r["text"])
                ok = fields(gd) == fields(pd)
                fok += ok
                for c in phenomena(r["text"]):
                    cat_tot[c] += 1
                    cat_ok[c] += ok
                rows.append({"id": r["id"], "text": r["text"], "words": r["words"], "gold": r["tags"],
                             "pred": p, "tags_correct": r["tags"] == p, "fields_correct": ok,
                             "gold_fields": dict(zip(("quantity", "unit", "ingredient"), fields(gd))),
                             "pred_fields": dict(zip(("quantity", "unit", "ingredient"), fields(pd))),
                             "confidence": pd["confidence"]})
            with open(out / f"predictions_{m}_{name}.jsonl", "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            results["models"][m][name] = {
                **{k: round(v, 4) for k, v in sc.items() if k != "per_type"},
                "line_acc": round(line_acc, 4), "field_acc": round(fok / len(recs), 4),
                "per_type": {t: {k: round(v, 4) for k, v in d.items()} for t, d in sc["per_type"].items()},
                "by_phenomenon": {c: {"lines": cat_tot[c], "field_acc": round(cat_ok[c] / cat_tot[c], 4)}
                                  for c in sorted(cat_tot)},
                "ms_per_line": round(1000 * secs / len(recs), 2)}
            r_ = results["models"][m][name]
            print(f"{m:8s} {name:14s} P={r_['p']:.3f} R={r_['r']:.3f} F1={r_['f1']:.3f} "
                  f"macroF1={r_['macro_f1']:.3f} line={r_['line_acc']:.3f} fields={r_['field_acc']:.3f}", flush=True)

    if "roberta" in a.models and "crf" in a.models:
        results["bootstrap_f1_diff_roberta_minus_crf"] = {}
        for name, recs in sets.items():
            lo, hi = bootstrap_diff([r["tags"] for r in recs], preds[("roberta", name)], preds[("crf", name)])
            results["bootstrap_f1_diff_roberta_minus_crf"][name] = [round(lo, 4), round(hi, 4)]
            print(f"bootstrap 95% CI, F1(roberta) - F1(crf) on {name}: [{lo:+.3f}, {hi:+.3f}]")

    (out / "metrics.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    write_markdown(results, out / "metrics.md")
    print(f"wrote {out / 'metrics.json'} and {out / 'metrics.md'}")


def write_markdown(res, path):
    L = ["# Evaluation results", "", "Entity-level exact match. Field accuracy = decoded quantity, unit, "
         "and ingredient all equal the gold decoding.", ""]
    L += ["| Model | Test set | Lines | P | R | F1 | Macro F1 | Line acc. | Field acc. |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m, d in res["models"].items():
        for s, r in d.items():
            L.append(f"| {m} | {s} | {res['sets'][s]['lines']} | {r['p']:.3f} | {r['r']:.3f} | **{r['f1']:.3f}** | "
                     f"{r['macro_f1']:.3f} | {r['line_acc']:.3f} | {r['field_acc']:.3f} |")
    if "bootstrap_f1_diff_roberta_minus_crf" in res:
        L += ["", "Paired bootstrap (1,000 resamples of lines), 95% interval for F1(RoBERTa) - F1(CRF): " +
              "; ".join(f"{k} [{v[0]:+.3f}, {v[1]:+.3f}]" for k, v in res["bootstrap_f1_diff_roberta_minus_crf"].items())]
    for s in res["sets"]:
        L += ["", f"## Per-type F1 on {s}", "", "| Type | Support | " + " | ".join(res["models"]) + " |",
              "|---|---|" + "---|" * len(res["models"])]
        types = {t for m in res["models"].values() for t in m.get(s, {}).get("per_type", {})}
        for t in TYPES:
            if t not in types:
                continue
            sup = max(m[s]["per_type"].get(t, {}).get("support", 0) for m in res["models"].values())
            L.append(f"| {t} | {sup} | " + " | ".join(
                f"{m[s]['per_type'].get(t, {}).get('f1', 0):.3f}" for m in res["models"].values()) + " |")
        L += ["", f"Field accuracy by phenomenon on {s}:", "", "| Phenomenon | Lines | " +
              " | ".join(res["models"]) + " |", "|---|---|" + "---|" * len(res["models"])]
        first = next(iter(res["models"].values()))[s]["by_phenomenon"]
        for c, v in first.items():
            L.append(f"| {c} | {v['lines']} | " + " | ".join(
                f"{m[s]['by_phenomenon'][c]['field_acc']:.2f}" for m in res["models"].values()) + " |")
    L += ["", "Overlap check: " + ", ".join(f"{k}: {v['lines_also_in_train']} of {v['lines']} lines also appear "
                                            f"verbatim in the training split" for k, v in res["sets"].items())]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
