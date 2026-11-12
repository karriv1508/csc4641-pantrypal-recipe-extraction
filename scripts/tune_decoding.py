"""Estimate BIO tag-bigram scores from the training split and choose their weight for the RoBERTa
tagger's Viterbi decoding on the validation split (the test sets are not read).

  python scripts/tune_decoding.py --out models/bio_transitions.json
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pantrypal.data import tasteset  # noqa: E402
from pantrypal.labels import LABELS  # noqa: E402
from pantrypal.spacy_scorer import bio_prf  # noqa: E402


def estimate(train):
    trans, start = Counter(), Counter()
    for r in train:
        start[r["tags"][0]] += 1
        for a, b in zip(r["tags"], r["tags"][1:]):
            trans[(a, b)] += 1
    log_trans, log_start = [], []
    for a in LABELS:
        tot = sum(trans[(a, b)] for b in LABELS) + 0.5 * len(LABELS)
        row = []
        for b in LABELS:
            invalid = b.startswith("I-") and a[2:] != b[2:]
            row.append(-1e4 if invalid else math.log((trans[(a, b)] + 0.5) / tot))
        log_trans.append(row)
    tot = sum(start.values()) + 0.5 * len(LABELS)
    log_start = [-1e4 if b.startswith("I-") else math.log((start[b] + 0.5) / tot) for b in LABELS]
    return log_trans, log_start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "models/bio_transitions.json"))
    ap.add_argument("--model", default=str(ROOT / "models/roberta/model-best"))
    a = ap.parse_args()
    log_trans, log_start = estimate(tasteset("train"))
    out = Path(a.out)
    val = tasteset("val")
    from pantrypal.taggers import RobertaTagger
    results = {}
    for w in (0.0, 0.25, 0.5, 1.0):
        out.write_text(json.dumps({"labels": LABELS, "weight": w, "log_trans": log_trans,
                                   "log_start": log_start}), encoding="utf-8")
        tagger = RobertaTagger(a.model, transitions=out)
        pred = tagger.predict([r["words"] for r in val])
        results[w] = bio_prf([(r["tags"], p) for r, (p, _) in zip(val, pred)])
        print(f"weight {w}: val P={results[w][0]:.4f} R={results[w][1]:.4f} F1={results[w][2]:.4f}", flush=True)
    best = max(results, key=lambda w: results[w][2])
    out.write_text(json.dumps({"labels": LABELS, "weight": best, "log_trans": log_trans,
                               "log_start": log_start,
                               "val_f1_by_weight": {str(k): round(v[2], 4) for k, v in results.items()}}),
                   encoding="utf-8")
    print(f"chose weight {best} -> {out}")


if __name__ == "__main__":
    main()
