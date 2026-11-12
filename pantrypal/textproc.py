"""One tokenization for training data and for new input.

The team's clean_data.py defines the tokenizer (TOKEN_RE), per-token ASCII normalization
(norm_token) and line cleaning (clean_line). Gold TASTEset tokens keep the original text, so
a token such as "1½" normalizes to "1 1/2"; here such tokens are split in two so that gold
lines and cleaned new input produce the same word sequence.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from clean_data import TOKEN_RE, clean_line, norm_token  # noqa: E402,F401


def prepare(text: str):
    """Clean a raw ingredient line and tokenize it.

    Returns (clean_text, words, offsets): offsets index into clean_text.
    """
    clean = clean_line(text)
    words, offsets = [], []
    for m in TOKEN_RE.finditer(clean):
        words.append(norm_token(m.group()))
        offsets.append((m.start(), m.end()))
    return clean, words, offsets


def gold_words(rec):
    """Words and BIO tags for a gold record from clean_data.py, splitting tokens like '1 1/2'."""
    words, tags = [], []
    for tok, tag in zip(rec["norm_tokens"], rec["tags"]):
        parts = tok.split(" ")
        for k, p in enumerate(parts):
            words.append(p)
            tags.append(tag if k == 0 or tag == "O" else "I-" + tag[2:])
    return words, tags


def detok(words):
    """Join words back into readable text (no space before , . ) or after ( )."""
    s = " ".join(words)
    s = re.sub(r"\s+([,.;:)%])", r"\1", s)
    s = re.sub(r"([($])\s+", r"\1", s)
    s = re.sub(r"\s*-\s*", "-", s)
    return s
