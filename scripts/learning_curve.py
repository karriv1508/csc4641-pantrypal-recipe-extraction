"""Learning curve for the CRF: validation F1 when training on 25%, 50%, and 100% of the training
recipes. Tells us whether more labeled (or silver) data is likely to help.

  python scripts/learning_curve.py
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pantrypal.data import tasteset  # noqa: E402
from scripts.train_crf import f1, fit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/12types/learning_curve.json"))
    a = ap.parse_args()
    lx = json.loads((ROOT / "models/rules/nyt_lexicon.json").read_text(encoding="utf-8"))
    lex = (set(lx["units"]), set(lx["food_words"]), set(lx["process_words"]))
    info = json.loads((ROOT / "models/crf/crf.json").read_text())
    train, val = tasteset("train"), tasteset("val")
    ids = sorted({r["recipe_id"] for r in train})
    res = {}
    for frac in (0.25, 0.5, 1.0):
        keep = set(random.Random(4641).sample(ids, round(len(ids) * frac)))  # same subsets as prepare_corpus.py
        sub = [r for r in train if r["recipe_id"] in keep]
        res[str(frac)] = {"lines": len(sub), "crf_val_f1": round(f1(fit(sub, lex, info["c1"], info["c2"]), val, lex), 4)}
        print(frac, res[str(frac)], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
