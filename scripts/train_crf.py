"""Train the CRF baseline on the TASTEset training split.

The c1/c2 regularization pair is chosen on the validation split; the test split is not read.
With --cv K it also reports K-fold recipe-level cross-validation on train+val (spread of F1).

  python scripts/train_crf.py --out models/crf/crf.pkl [--cv 5]
"""
import argparse
import json
import pickle
import random
import statistics
import sys
import time
from pathlib import Path

import sklearn_crfsuite

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pantrypal.data import tasteset  # noqa: E402
from pantrypal.spacy_scorer import bio_prf  # noqa: E402
from pantrypal.taggers import crf_features  # noqa: E402


def fit(recs, lex, c1, c2):
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=150,
                               all_possible_transitions=True)
    crf.fit([crf_features(r["words"], lex) for r in recs], [r["tags"] for r in recs])
    return crf


def f1(crf, recs, lex):
    pred = crf.predict([crf_features(r["words"], lex) for r in recs])
    return bio_prf([(r["tags"], list(p)) for r, p in zip(recs, pred)])[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "models/crf/crf.pkl"))
    ap.add_argument("--lexicon", default=str(ROOT / "models/rules/nyt_lexicon.json"))
    ap.add_argument("--cv", type=int, default=0)
    a = ap.parse_args()
    lx = json.loads(Path(a.lexicon).read_text(encoding="utf-8"))
    lex = (set(lx["units"]), set(lx["food_words"]), set(lx["process_words"]))
    train, val = tasteset("train"), tasteset("val")

    grid = {}
    for c1 in (0.05, 0.1, 0.3):
        for c2 in (0.01, 0.05, 0.1):
            t = time.time()
            grid[(c1, c2)] = f1(fit(train, lex, c1, c2), val, lex)
            print(f"c1={c1} c2={c2}  val F1={grid[(c1, c2)]:.4f}  ({time.time() - t:.0f}s)", flush=True)
    c1, c2 = max(grid, key=grid.get)
    crf = fit(train, lex, c1, c2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(crf, f)
    info = {"c1": c1, "c2": c2, "val_f1": round(grid[(c1, c2)], 4), "train_lines": len(train),
            "grid": {f"{k[0]},{k[1]}": round(v, 4) for k, v in grid.items()}}

    if a.cv:
        pool = train + val
        ids = sorted({r["recipe_id"] for r in pool})
        random.Random(4641).shuffle(ids)
        folds = [set(ids[i::a.cv]) for i in range(a.cv)]
        scores = []
        for k, fold in enumerate(folds):
            tr = [r for r in pool if r["recipe_id"] not in fold]
            te = [r for r in pool if r["recipe_id"] in fold]
            scores.append(f1(fit(tr, lex, c1, c2), te, lex))
            print(f"fold {k + 1}/{a.cv}: F1={scores[-1]:.4f} ({len(te)} lines)", flush=True)
        info["cv"] = {"k": a.cv, "f1": [round(s, 4) for s in scores],
                      "mean": round(statistics.mean(scores), 4), "sd": round(statistics.stdev(scores), 4)}
    Path(a.out).with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
