"""Download the two public raw datasets the prototype needs, pinned to exact commits and checked
by SHA-256, into the project root (where clean_data.py looks for them).

  python scripts/get_data.py

RecipeNLG and the scraped Pinch of Yum pages are not public downloads; they are in the team
OneDrive folder (see README). The prototype only needs data_processed/scraped_clean.jsonl from it.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = {
    "TASTEset.csv": (
        "https://raw.githubusercontent.com/taisti/TASTEset-2.0/"
        "adf9da58737c891559e779658a7ce2b98f805657/data/TASTEset.csv",
        "59c797745c8e7e9cb8044122ee5e31198cd47fb31498dfb1d298ce2270a55992"),
    "nyt-ingredients-snapshot-2015.csv": (
        "https://raw.githubusercontent.com/nytimes/ingredient-phrase-tagger/"
        "e414c2ca279f23c99c8338ceba00653d88d40dfe/nyt-ingredients-snapshot-2015.csv",
        "7e029d8d00acd86933bd92dbec81d04f4b9fe4e60b3422325bac1cd5a049f56d"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ok = True
    for name, (url, digest) in FILES.items():
        dest = ROOT / name
        if dest.exists() and sha256(dest) == digest:
            print(f"{name}: already present, checksum OK")
            continue
        print(f"{name}: downloading ...", flush=True)
        urllib.request.urlretrieve(url, dest)
        if sha256(dest) != digest:
            print(f"{name}: CHECKSUM MISMATCH (got {sha256(dest)})")
            ok = False
        else:
            print(f"{name}: OK ({dest.stat().st_size:,} bytes)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
