"""Fine-tune a Hugging Face encoder (DistilBERT by default) as a BIO token classifier on exactly the
same data as our RoBERTa model, so the two can be compared fairly:

  * same file and split: data_processed/tasteset_bio.jsonl, split == "train" (3,721 lines), recipe-level
    70/15/15 split with seed 4641; validation (829) picks the epoch; test (824) is scored once at the end
  * same input words: norm_tokens, with tokens like "1 1/2" split in two (pantrypal.textproc.gold_words)
  * same 12 entity types (EXCLUSIVE, EXCLUDED, POSSIBLE_SUBSTITUTE merged into FOOD), same seed 4641
  * same metric: entity-level exact-match micro F1 (identical to seqeval's)
Only each word's first sub-word piece is labeled and scored, the usual setup for BERT-style NER.

Needs internet access to huggingface.co the first time (to download the base model), plus:
  pip install "transformers>=4.40"
Run from the repository root (a CPU is fine; DistilBERT takes roughly 15-30 minutes on a laptop):
  python scripts/train_hf_tagger.py --model distilbert-base-uncased --out models/distilbert-tasteset
  python scripts/train_hf_tagger.py --model roberta-base --out models/roberta-base-hf   # optional
Writes the best checkpoint, metrics.json (val F1 per epoch, test F1, settings), and
predictions_test.jsonl (gold_tags_12 / pred_tags per line, the format predict_tokens.py writes).
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pantrypal.data import tasteset  # noqa: E402
from pantrypal.labels import LABELS, repair  # noqa: E402
from pantrypal.spacy_scorer import bio_prf  # noqa: E402

L2I = {l: i for i, l in enumerate(LABELS)}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode(tok, recs, max_len):
    out = []
    for r in recs:
        enc = tok(r["words"], is_split_into_words=True, truncation=True, max_length=max_len)
        labels, prev = [], None
        for wid in enc.word_ids():
            labels.append(-100 if wid is None or wid == prev else L2I[r["tags"][wid]])
            prev = wid
        out.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                    "labels": labels, "word_ids": enc.word_ids()})
    return out


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    pad = lambda xs, v: [x + [v] * (n - len(x)) for x in xs]  # noqa: E731
    return {"input_ids": torch.tensor(pad([b["input_ids"] for b in batch], pad_id)),
            "attention_mask": torch.tensor(pad([b["attention_mask"] for b in batch], 0)),
            "labels": torch.tensor(pad([b["labels"] for b in batch], -100))}


@torch.no_grad()
def predict(model, data, recs, pad_id, bs=64):
    model.eval()
    tags = []
    for k in range(0, len(data), bs):
        chunk = data[k:k + bs]
        batch = collate(chunk, pad_id)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        pred = logits.argmax(-1).tolist()
        for d, p in zip(chunk, pred):
            seq, prev = [], None
            for pos, wid in enumerate(d["word_ids"]):
                if wid is not None and wid != prev:
                    seq.append(LABELS[p[pos]])
                prev = wid
            tags.append(seq)
    # a line cut by truncation gets "O" for its missing words
    return [repair(t + ["O"] * (len(r["words"]) - len(t))) for t, r in zip(tags, recs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--out", default=str(ROOT / "models/distilbert-tasteset"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=32, help="lines per step")
    ap.add_argument("--warmup", type=float, default=0.1, help="share of steps for linear warm-up")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=4641)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only this many lines per split")
    a = ap.parse_args()

    from transformers import AutoModelForTokenClassification, AutoTokenizer
    set_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = {s: tasteset(s) for s in ("train", "val", "test")}
    if a.limit:
        splits = {s: v[:a.limit] for s, v in splits.items()}

    kw = {"add_prefix_space": True} if "roberta" in a.model.lower() else {}
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True, **kw)
    model = AutoModelForTokenClassification.from_pretrained(
        a.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I,
        ignore_mismatched_sizes=True)
    data = {s: encode(tok, v, a.max_len) for s, v in splits.items()}
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    steps_per_epoch = math.ceil(len(data["train"]) / a.batch)
    total = steps_per_epoch * a.epochs
    warm = int(a.warmup * total)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)],
               "weight_decay": a.weight_decay},
              {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)],
               "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))

    history, best_f1, t0 = [], -1.0, time.time()
    rng = random.Random(a.seed)
    for epoch in range(1, a.epochs + 1):
        model.train()
        order = list(range(len(data["train"])))
        rng.shuffle(order)
        loss_sum = 0.0
        for k in range(0, len(order), a.batch):
            batch = collate([data["train"][i] for i in order[k:k + a.batch]], pad_id)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            loss_sum += loss.item()
        pred = predict(model, data["val"], splits["val"], pad_id)
        p, r, f = bio_prf([(x["tags"], y) for x, y in zip(splits["val"], pred)])
        history.append({"epoch": epoch, "train_loss": round(loss_sum / steps_per_epoch, 4), "val_f1": round(f, 4),
                        "minutes": round((time.time() - t0) / 60, 1)})
        print(history[-1], flush=True)
        if f > best_f1:
            best_f1 = f
            model.save_pretrained(out)
            tok.save_pretrained(out)

    # test once, with the checkpoint that scored best on validation
    model = AutoModelForTokenClassification.from_pretrained(out)
    pred = predict(model, data["test"], splits["test"], pad_id)
    p, r, f = bio_prf([(x["tags"], y) for x, y in zip(splits["test"], pred)])
    with open(out / "predictions_test.jsonl", "w", encoding="utf-8") as fh:
        for x, y in zip(splits["test"], pred):
            fh.write(json.dumps({"id": x["id"], "text": x["text"], "words": x["words"],
                                 "gold_tags_12": x["tags"], "pred_tags": y}, ensure_ascii=False) + "\n")
    metrics = {"base_model": a.model, "seed": a.seed, "epochs": a.epochs, "lr": a.lr, "batch_lines": a.batch,
               "warmup_share": a.warmup, "weight_decay": a.weight_decay, "train_lines": len(splits["train"]),
               "best_epoch": max(history, key=lambda h: h["val_f1"])["epoch"], "history": history,
               "test": {"p": round(p, 4), "r": round(r, 4), "f1": round(f, 4), "lines": len(splits["test"])},
               "train_minutes": history[-1]["minutes"]}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(f"test P={p:.4f} R={r:.4f} F1={f:.4f} -> {out}")


if __name__ == "__main__":
    main()
