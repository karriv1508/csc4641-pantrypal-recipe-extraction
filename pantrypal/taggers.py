"""Three taggers with one interface: predict(list of word lists) -> list of (tags, confidences).

  RuleTagger     baseline 1: hand-written rules + NYT-derived word lists (no training)
  CRFTagger      baseline 2: linear-chain CRF on TASTEset train, NYT word lists as features
  RobertaTagger  main model: RoBERTa-base fine-tuned as a BIO token classifier (spaCy)
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
NUM_RE = re.compile(r"^(\d+([./]\d+)?|\d+ \d+/\d+)$")
NUM_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
             "eleven", "twelve", "half", "dozen"}
SIZE = {"large", "small", "medium", "big", "jumbo", "extra-large", "fresh", "frozen", "whole",
        "boneless", "skinless", "dried", "dry", "ripe", "warm", "cold", "hot", "uncooked", "raw",
        "cooked", "room", "temperature", "packed", "heaping", "level", "thick", "thin"}
COLORS = {"red", "green", "yellow", "white", "black", "brown", "orange", "purple", "dark", "golden"}
TASTES = {"sweet", "unsalted", "salted", "unsweetened", "sweetened", "bittersweet", "semisweet",
          "mild", "spicy", "sour", "smoked"}
STOP = {"of", "a", "an", "the", "or", "and", "to", "for", "with", "about", "plus", "if", "you",
        "i", "my", "your", "more", "optional", "use", "used", "like", "such", "as", "other", "any",
        "is", "are", "be", "it", "them", "but", "not", "can", "will", "into", "in", "on", "at",
        "see", "notes", "note", "each", "extra", "some", "favorite", "we", "this", "that", "from"}


def _bio(labels, starts):
    """Per-token entity types (or None) and span-start flags -> BIO tags."""
    tags, prev = [], None
    for lab, st in zip(labels, starts):
        tags.append("O" if lab is None else ("I-" if lab == prev and not st else "B-") + lab)
        prev = lab
    return tags


class RuleTagger:
    name = "rules"

    def __init__(self, path=MODELS / "rules" / "nyt_lexicon.json"):
        lex = json.loads(Path(path).read_text(encoding="utf-8"))
        self.units = set(lex["units"])
        self.food = set(lex["food_words"])
        self.process = set(lex["process_words"])

    def tag(self, words):
        n, low = len(words), [w.lower() for w in words]
        lab, st = [None] * n, [False] * n
        i = 0
        while i < n:  # quantities, ranges ("1 - 2", "2 to 3") and "to taste"
            w = low[i]
            if low[i:i + 2] == ["to", "taste"]:
                lab[i] = lab[i + 1] = "QUANTITY"; st[i] = True; i += 2; continue  # noqa: E702
            if NUM_RE.match(w) or (w in NUM_WORDS and i + 1 < n):
                j = i + 1
                while j + 1 < n and low[j] in ("-", "to") and NUM_RE.match(low[j + 1]):
                    j += 2
                for k in range(i, j):
                    lab[k] = "QUANTITY"
                st[i] = True
                i = j; continue  # noqa: E702
            i += 1
        for i, w in enumerate(words):  # units: right after a quantity, "(", "-", or "a"
            if lab[i] is None and (w in self.units or w.lower() in self.units):
                prev = low[i - 1] if i else ""
                container = w.lower() in ("can", "cans", "package", "packages", "jar", "jars", "box", "bag")
                if i and (lab[i - 1] == "QUANTITY" or prev in ("a", "-", "heaping") or
                          (container and (lab[i - 1] == "UNIT" or prev == ")"))):
                    lab[i], st[i] = "UNIT", True
                    if i + 1 < n and words[i + 1] == "." and len(w) <= 4:
                        lab[i + 1] = "UNIT"
        i = 0
        while i < n:  # "for serving", "for topping" -> PURPOSE up to punctuation
            if low[i] == "for" and i + 1 < n and re.match(r"[a-z]", low[i + 1]):
                j = i + 1
                while j < n and re.match(r"[a-z]", low[j]) and lab[j] is None:
                    j += 1
                for k in range(i, j):
                    lab[k] = "PURPOSE"
                st[i] = True
                i = j; continue  # noqa: E702
            i += 1
        for i, w in enumerate(low):  # preparation words; "thinly sliced", "cut into chunks"
            if lab[i] is None and (w in self.process or w in ("cut", "minced", "chopped", "sliced", "diced")):
                lab[i], st[i] = "PROCESS", True
                if i and low[i - 1].endswith("ly") and lab[i - 1] in (None, "PROCESS"):
                    if lab[i - 1] is None:
                        lab[i - 1], st[i - 1] = "PROCESS", True
                    st[i] = False  # "thinly sliced" is one span
                if w == "cut":
                    j = i + 1
                    while j < n and re.match(r"[a-z0-9/]", low[j]) and lab[j] is None and low[j] not in ("and", "or"):
                        lab[j] = "PROCESS"; j += 1  # noqa: E702
        for i, w in enumerate(low):
            if lab[i] is None and w in SIZE:
                lab[i], st[i] = "PHYSICAL_QUALITY", True
        # food: content words before the first comma/parenthesis; later only NYT food words
        main_end = next((i for i, w in enumerate(words) if w in (",", "(", ";")), n)
        for i, w in enumerate(low):
            if lab[i] is not None or not re.match(r"[a-z]", w) or w in STOP:
                continue
            if w in COLORS and i + 1 < n and low[i + 1] not in STOP and re.match(r"[a-z]", low[i + 1]):
                lab[i], st[i] = "COLOR", True
            elif w in TASTES and i + 1 < n and re.match(r"[a-z]", low[i + 1]):
                lab[i], st[i] = "TASTE", True
            elif i < main_end or w in self.food:
                lab[i] = "FOOD"
        return _bio(lab, st)

    def predict(self, batch):
        return [(self.tag(w), None) for w in batch]


def crf_features(words, lex):
    units, food, process = lex
    low = [w.lower() for w in words]
    depth, in_par, after_comma, seen = 0, [], [], False
    for w in words:
        depth += w == "("
        in_par.append(depth > 0)
        depth -= w == ")" and depth > 0
        after_comma.append(seen)
        seen = seen or w == ","
    feats = []
    for i, w in enumerate(words):
        lw = low[i]
        f = {"bias": 1.0, "w": lw, "suf3": lw[-3:], "suf2": lw[-2:], "pre3": lw[:3],
             "num": bool(NUM_RE.match(lw)), "numword": lw in NUM_WORDS, "title": w.istitle(),
             "upper": w.isupper(), "punct": not any(c.isalnum() for c in w),
             "shape": re.sub(r"[a-z]+", "a", re.sub(r"[A-Z]+", "A", re.sub(r"\d+", "0", w))),
             "unit": w in units or lw in units, "foodlex": lw in food, "proclex": lw in process,
             "pos": min(i, 4), "rpos": min(len(words) - 1 - i, 4), "paren": in_par[i],
             "aftercomma": after_comma[i]}
        for d in (-2, -1, 1, 2):
            j = i + d
            if 0 <= j < len(words):
                f[f"{d}:w"] = low[j]
                f[f"{d}:num"] = bool(NUM_RE.match(low[j]))
                f[f"{d}:unit"] = words[j] in units or low[j] in units
                f[f"{d}:suf2"] = low[j][-2:]
            else:
                f[f"{d}:edge"] = True
        if i > 0:
            f["-1|0"] = low[i - 1] + "|" + lw
        if i + 1 < len(words):
            f["0|+1"] = lw + "|" + low[i + 1]
        feats.append(f)
    return feats


class CRFTagger:
    name = "crf"

    def __init__(self, path=MODELS / "crf" / "crf.pkl", lexicon=MODELS / "rules" / "nyt_lexicon.json"):
        import pickle
        lex = json.loads(Path(lexicon).read_text(encoding="utf-8"))
        self.lex = (set(lex["units"]), set(lex["food_words"]), set(lex["process_words"]))
        with open(path, "rb") as f:
            self.crf = pickle.load(f)

    def predict(self, batch):
        X = [crf_features(w, self.lex) for w in batch]
        tags = self.crf.predict(X)
        margs = self.crf.predict_marginals(X)
        return [(list(t), [m[k][lab] for k, lab in enumerate(t)]) for t, m in zip(tags, margs)]


def viterbi(logp, trans, start):
    """Best label path for one line given per-token log-probs (n x L) and log transition scores."""
    import numpy as np
    score, back = start + logp[0], []
    for i in range(1, len(logp)):
        cand = score[:, None] + trans
        back.append(cand.argmax(axis=0))
        score = cand.max(axis=0) + logp[i]
    path = [int(score.argmax())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    return np.array(path[::-1])


def roberta_dir():
    """The team's model folder (models/roberta-tasteset/model), else a model you retrained yourself."""
    for p in (MODELS / "roberta-tasteset" / "model", MODELS / "roberta" / "model-best"):
        if p.exists():
            return p
    return MODELS / "roberta-tasteset" / "model"


class RobertaTagger:
    """Per-token softmax over BIO labels, then Viterbi decoding with tag-bigram scores from the
    training split (models/bio_transitions.json; weight chosen on validation). The softmax alone
    can split one entity into two (B-FOOD B-FOOD); the bigram scores make that unlikely."""
    name = "roberta"

    def __init__(self, path=None, transitions=MODELS / "bio_transitions.json"):
        path = path or roberta_dir()
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        import numpy as np
        import spacy
        from spacy.tokens import Doc
        from . import spacy_scorer  # noqa: F401  registers the scorer named in the config
        self.np, self.Doc = np, Doc
        self.nlp = spacy.load(path)
        self.trf = self.nlp.get_pipe("transformer")
        self.tagger = self.nlp.get_pipe("bio_tagger")
        self.labels = list(self.tagger.labels)
        self.trans = None
        if transitions and Path(transitions).exists():
            tr = json.loads(Path(transitions).read_text(encoding="utf-8"))
            order = [tr["labels"].index(l) for l in self.labels]
            w = tr["weight"]
            self.trans = w * np.array(tr["log_trans"])[np.ix_(order, order)]
            self.start = w * np.array(tr["log_start"])[order]

    def predict(self, batch, batch_size=64):
        np, out = self.np, []
        docs = [self.Doc(self.nlp.vocab, words=w) for w in batch if w]
        docs = list(self.trf.pipe(docs, batch_size=batch_size))
        scores = []
        for k in range(0, len(docs), batch_size):
            scores.extend(self.tagger.model.predict(docs[k:k + batch_size]))
        it = iter(zip(docs, scores))
        for w in batch:
            if not w:
                out.append(([], []))
                continue
            _, s = next(it)
            s = np.asarray(s, dtype="float64")
            p = np.exp(s - s.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)  # the layer returns logits at inference
            best = p.argmax(axis=1) if self.trans is None else viterbi(np.log(p + 1e-12), self.trans, self.start)
            out.append(([self.labels[j] for j in best], [float(p[i, j]) for i, j in enumerate(best)]))
        return out


def load_tagger(name, path=None):
    cls = {"rules": RuleTagger, "crf": CRFTagger, "roberta": RobertaTagger}[name]
    return cls(path) if path else cls()
