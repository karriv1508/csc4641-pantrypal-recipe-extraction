"""Run the whole prototype pipeline in order (cross-platform; uses the current Python).

  python run_pipeline.py                 # data -> lexicon -> CRF -> evaluate -> scraped batch demo
  python run_pipeline.py --train-roberta # also retrain RoBERTa (about 1 hour on a 2-core CPU;
                                         # needs: pip install -r requirements-train.txt)

Without --train-roberta it uses the saved model in models/roberta-tasteset (see PROTOTYPE.md).
Steps whose inputs are missing are skipped with a message instead of failing.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def step(title, args, need=()):
    missing = [str(p) for p in need if not (ROOT / p).exists()]
    if missing:
        print(f"\n== {title}: SKIPPED, missing {', '.join(missing)}")
        return False
    print(f"\n== {title}\n   $ python {' '.join(args)}", flush=True)
    subprocess.run([PY, "-W", "ignore", *args], cwd=ROOT, check=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-roberta", action="store_true")
    a = ap.parse_args()

    step("1. Download TASTEset-2.0 and NYT (checksum-pinned)", ["scripts/get_data.py"])
    step("2. Clean and split (team script, seed 4641)",
         ["scripts/clean_data.py", "--root", ".", "--skip-recipenlg"])
    step("3. NYT word lists for the rule baseline and CRF features", ["scripts/build_lexicon.py"],
         need=["data_processed/nyt_clean.jsonl"])
    step("4. Scraped cross-site test set from the hand labels", ["scripts/build_scraped_eval.py"])
    step("5. Train the CRF baseline (c1/c2 chosen on validation, 5-fold CV)",
         ["scripts/train_crf.py", "--cv", "5"])
    if a.train_roberta:
        step("6a. spaCy corpus files", ["scripts/prepare_corpus.py"])
        step("6b. RoBERTa stage 1: train the new output layer only",
             ["-m", "spacy", "train", "configs/roberta_stage1.cfg", "--code", "pantrypal/spacy_scorer.py",
              "--output", "models/roberta_stage1"])
        step("6b2. Let the transformer receive gradients again",
             ["scripts/unfreeze_stage1.py", "models/roberta_stage1/model-last", "models/roberta_stage1/model-unfrozen"])
        step("6c. RoBERTa stage 2: fine-tune everything (about 45 minutes on a laptop CPU)",
             ["-m", "spacy", "train", "configs/roberta_stage2.cfg", "--code", "pantrypal/spacy_scorer.py",
              "--output", "models/roberta"])
        step("6d. Tag-bigram weight for Viterbi decoding, chosen on validation", ["scripts/tune_decoding.py"])
    have_roberta = any((ROOT / p).exists() for p in ("models/roberta-tasteset/model", "models/roberta/model-best"))
    models = ["rules", "crf"] + (["roberta"] if have_roberta else [])
    if "roberta" not in models:
        print("\n(no models/roberta-tasteset: evaluating the baselines only; see PROTOTYPE.md to get the model)")
    step("7. Evaluate on TASTEset test, unseen-line subset, and scraped cross-site set",
         ["scripts/evaluate_12types.py", "--models", *models])
    step("8. Run the best available model over all scraped recipes",
         ["predict.py", "--model", models[-1], "--recipes", "data_processed/scraped_clean.jsonl",
          "--out", f"results/12types/scraped_predictions_{models[-1]}.jsonl"],
         need=["data_processed/scraped_clean.jsonl"])
    print("\nDone. Tables: results/12types/metrics.md")


if __name__ == "__main__":
    main()
