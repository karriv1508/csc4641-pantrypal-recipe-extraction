# PantryPal: Recipe Information Extraction

CSC 4641 (Natural Language Processing), MSOE, Fall 2026
Team: Venkata Reddy Karri, Kazuya Rexer

Extract structured information (quantity, unit, food, preparation, ...) from free-text recipe
ingredient lines, as the ingredient-normalization component of **PantryPal**, a pantry-inventory
and cooking app for students and people cooking alone.

## Repository layout
| Path | Contents |
|---|---|
| `scripts/scrape_recipes.py` | Self-collected data: scrapes recent recipes with `recipe-scrapers` |
| `scripts/clean_data.py` | Cleans TASTEset, NYT, and RecipeNLG into tokenized / BIO-tagged JSONL |
| `samples/` | Raw → processed examples for each data source |
| `DATA_README.md` | Data sources, licenses, counts, and every cleaning step |
| `requirements.txt` | Python dependencies |

Full datasets are not stored here (size and licensing). They are shared separately via OneDrive.

## Quick start
    pip install -r requirements.txt
    python scripts/scrape_recipes.py --per-site 120
    python scripts/clean_data.py --root .

## Milestones
- [x] Proposal
- [x] Background research / technical roadmap
- [x] Data collection and preparation
- [ ] Prototype with cross-validation
- [ ] Draft paper
- [ ] Final paper and presentation
