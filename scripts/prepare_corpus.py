"""Write the TASTEset splits as spaCy DocBin files for training the RoBERTa tagger.

Each token's BIO label (12 types) is stored as its tag. Words come from the same
tokenizer that new input goes through (pantrypal.textproc), so training and inference match.

  python scripts/prepare_corpus.py --out corpus [--train-fraction 0.25]
"""
import argparse
import random
import sys
from pathlib import Path

import spacy
from spacy.tokens import Doc, DocBin

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pantrypal.data import tasteset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "corpus"))
    ap.add_argument("--train-fraction", type=float, default=1.0,
                    help="keep this fraction of training recipes (learning curve)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vocab = spacy.blank("en").vocab
    for split in ("train", "val", "test"):
        recs = tasteset(split)
        if split == "train" and a.train_fraction < 1.0:
            ids = sorted({r["recipe_id"] for r in recs})
            keep = set(random.Random(4641).sample(ids, round(len(ids) * a.train_fraction)))
            recs = [r for r in recs if r["recipe_id"] in keep]
        db = DocBin(attrs=["TAG"], store_user_data=False)
        for r in recs:
            doc = Doc(vocab, words=r["words"], spaces=[True] * len(r["words"]))
            for tok, tag in zip(doc, r["tags"]):
                tok.tag_ = tag
            db.add(doc)
        name = f"train_{round(a.train_fraction * 100)}" if split == "train" and a.train_fraction < 1.0 else split
        db.to_disk(out / f"{name}.spacy")
        print(f"{split}: {len(recs)} lines -> {out / (name + '.spacy')}")


if __name__ == "__main__":
    main()
