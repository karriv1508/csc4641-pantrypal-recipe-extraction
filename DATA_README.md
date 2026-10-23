# Recipe Information Extraction (PantryPal) - Data Preparation

## Run
    python scripts/clean_data.py --root .
Needs Python 3 only (standard library). About 4 minutes, mostly scanning the 2.3 GB RecipeNLG file.
Seeded (4641), so re-running gives the same subset and splits. Log: data_clean_log.txt.

## Inputs (raw, not redistributed)
| File | Source / creator | Format | License |
|---|---|---|---|
| TASTEset.csv | TASTEset-2.0, Wróblewska et al. (github.com/taisti/TASTEset) | CSV: recipe text + JSON entity spans | MIT |
| nyt-ingredients-snapshot-2015.csv | NYT Ingredient Phrase Tagger (github.com/nytimes/ingredient-phrase-tagger) | CSV: input, name, qty, range_end, unit, comment | Apache 2.0 |
| RecipeNLG_dataset.csv | RecipeNLG, Bień et al. 2020 (recipenlg.cs.put.poznan.pl) | CSV: title, ingredients (JSON list), directions, link, source, NER | Non-commercial research use |

## Outputs (data_processed/)
| File | Records | What it is | Used for |
|---|---|---|---|
| tasteset_bio.jsonl | 5,374 lines / 1,000 recipes | tokens + gold BIO tags, recipe-level 70/15/15 split | train the tagger; held-out test set |
| nyt_clean.jsonl | 85,008 lines | deduplicated, HTML-free; structured fields + coarse BIO tags | rule-based / CRF baseline |
| recipenlg_subset.jsonl | 25,000 recipes / 218,694 lines | cleaned, tokenized, unlabeled (`tags: null`) | silver-labeled by the TASTEset model, then main training data |
| stats.json | - | all counts | report numbers |

## Cleaning steps (all implemented)
**TASTEset (gold, text left unchanged so offsets stay valid)**
1. Parse span strings `"[(0, 1), ]"`, merge discontinuous fragments separated only by whitespace.
2. Split each recipe into ingredient lines; tokenize with character offsets.
3. Character-level labels become token-level BIO tags (15 entity types).
4. `norm_tokens` adds an ASCII form (½ -> 1/2, 1½ -> 1 1/2) without changing the original text.
5. Recipe-level train/val/test split (3,721 / 829 / 824 lines), so lines from one recipe never appear in two splits.
6. Checks: 6 entity spans whose text does not match the `entity` field; 6 lines contain store/price promo text (e.g. "Save-A-Lot - $0.99").

**NYT**
1. Drop 539 rows with missing input or name.
2. Strip HTML `(<a href=...>see recipe</a>)` links (923 lines).
3. Normalize unicode fractions, quotes, ®/™, whitespace.
4. Remove 93,660 exact duplicates (52% of the file).
5. Align `name`/`unit`/`qty` fields to tokens to get coarse QUANTITY/UNIT/FOOD tags.

**RecipeNLG**
1. Uniform random sample (rate ≈1.15%, seed 4641) across the whole file. The file is ordered by source, so taking the first N would give only cookbooks.com recipes. Result: 18,954 Gathered + 6,046 Recipes1M.
2. Parse JSON ingredient lists; same line cleaning as NYT plus bullet removal; remove "(Servings: 4)" from titles.
3. Drop recipes with fewer than 2 or more than 40 ingredients (47) and exact title+ingredient duplicates (1).
4. **Leakage guard:** drop any recipe where 80% or more of its lines match one TASTEset recipe (3 dropped), so evaluation data never shows up in silver training data.
5. Keep raw and cleaned lines, source link, and RecipeNLG's NER food list as metadata.

## Planned (not yet done)
- Silver-label recipenlg_subset with a model trained on the TASTEset train split.
- Clean self-scraped recipes (recipe-scrapers) with the same `clean_line` function and schema.
- Hand-check about 150 scraped lines as an extra test set.

## Samples (samples/) - raw and processed versions of the same records
- tasteset_raw.csv -> tasteset_processed.jsonl
- recipenlg_raw.csv -> recipenlg_processed.json
- nyt_raw_vs_processed.jsonl (raw and text fields side by side)
