"""Entity-level scorer for the spaCy BIO tagger, so training keeps the checkpoint with the best
validation F1 instead of the best token accuracy. Passed to `spacy train` with --code."""
import spacy

from pantrypal.labels import repair, spans


def bio_prf(pairs):
    """pairs: iterable of (gold_tags, pred_tags). Exact-match entity micro P/R/F1."""
    tp = fp = fn = 0
    for gold, pred in pairs:
        g, p = set(spans(gold)), set(spans(repair(pred)))
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f


@spacy.registry.scorers("pantrypal.bio_scorer.v1")
def make_bio_scorer():
    def score(examples, **kwargs):
        pairs, right, total = [], 0, 0
        for eg in examples:
            gold = [t.tag_ for t in eg.reference]
            pred = [t.tag_ for t in eg.predicted]
            pairs.append((gold, pred))
            right += sum(g == p for g, p in zip(gold, pred))
            total += len(gold)
        p, r, f = bio_prf(pairs)
        return {"tag_acc": right / total if total else 0.0, "ents_p": p, "ents_r": r, "ents_f": f}
    return score
