"""Between the two RoBERTa training stages: copy the stage-1 pipeline with the tagger's listener
switched from grad_factor 0 (transformer receives no gradient) to 1 (fine-tune everything).
The factor is stored with the model weights, so the config alone cannot change it.

  python scripts/unfreeze_stage1.py models/roberta_stage1/model-last models/roberta_stage1/model-unfrozen
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import spacy  # noqa: E402

import pantrypal.spacy_scorer  # noqa: E402,F401  registers the scorer named in the config


def main():
    src, dst = sys.argv[1], sys.argv[2]
    nlp = spacy.load(src)
    changed = 0
    for node in nlp.get_pipe("bio_tagger").model.walk():
        if "grad_factor" in node.attrs:
            node.attrs["grad_factor"] = 1.0
            changed += 1
    nlp.config["components"]["bio_tagger"]["model"]["tok2vec"]["grad_factor"] = 1.0
    nlp.to_disk(dst)
    check = spacy.load(dst)
    vals = [n.attrs["grad_factor"] for n in check.get_pipe("bio_tagger").model.walk() if "grad_factor" in n.attrs]
    print(f"set grad_factor=1.0 on {changed} listener node(s); reloaded values: {vals} -> {dst}")


if __name__ == "__main__":
    main()
