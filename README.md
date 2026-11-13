# PantryPal: Recipe Information Extraction

CSC 4641 (Natural Language Processing), MSOE, Fall 2026
Team: Venkata Reddy Karri, Kazuya Rexer

Extract structured information (quantity, unit, food, preparation, ...) from free-text recipe
ingredient lines, as the ingredient-normalization component of **PantryPal**, a pantry-inventory
and cooking app for students and people cooking alone.

## Prototype (Milestone 4): start here
The working prototype, its evaluation, and full instructions are in **[PROTOTYPE.md](PROTOTYPE.md)**:
requirements, setup, run commands, example inputs and outputs, evaluation results, and known limitations.

Quick start (Python 3.12 or 3.13, CPU only, no API keys):

    pip install -r requirements-prototype.txt
    python scripts/get_data.py
    python scripts/clean_data.py --root . --skip-recipenlg
    # copy models/roberta-tasteset from the team OneDrive folder (link in PROTOTYPE.md), then:
    python predict.py "2 (15 oz) cans chickpeas, drained and rinsed"
    python predict.py --model crf "1 1/2 cups whole milk"     # baselines work without the RoBERTa download

## Repository layout (data collection, Milestone 3)
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
- [x] Prototype and initial evaluation (see PROTOTYPE.md)
- [ ] Draft paper
- [ ] Final paper and presentation
