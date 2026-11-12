#!/usr/bin/env bash
# Two-stage fine-tuning of RoBERTa-base as a BIO token classifier (CPU is fine).
# Needs: pip install -r requirements-train.txt. On Windows, run the lines below one by one
# (or use: python run_pipeline.py --train-roberta).
set -euo pipefail
cd "$(dirname "$0")/.."
python -W ignore scripts/prepare_corpus.py
python -W ignore -m spacy train configs/roberta_stage1.cfg --code pantrypal/spacy_scorer.py --output models/roberta_stage1
python -W ignore scripts/unfreeze_stage1.py models/roberta_stage1/model-last models/roberta_stage1/model-unfrozen
python -W ignore -m spacy train configs/roberta_stage2.cfg --code pantrypal/spacy_scorer.py --output models/roberta
python -W ignore scripts/tune_decoding.py
