# Prototype (Milestone 4): RoBERTa tagger, baselines, and 12-type evaluation

This part of the repository is the code and data behind the numbers in our evaluation summary:
RoBERTa against the rule and CRF baselines, scored on all 12 entity types, plus the unseen-line
subset, the Pinch of Yum cross-site set, and the bootstrap intervals. Everything here can be rerun.

The repository has two evaluation scripts. `scripts/evaluate_12types.py` (this document) scores all
12 types and produced the summary's tables. `scripts/evaluate.py` and `scripts/extract.py` (Venkata)
score the 6 core types (QUANTITY, UNIT, FOOD, PROCESS, PHYSICAL_QUALITY, COLOR) and write
`results/summary.csv`. Both use the same model folder, `models/roberta-tasteset`.

## How it works

```
"1 1/2 lbs. boneless chicken breasts, cut into strips"
   │  clean_line() + team tokenizer         (scripts/clean_data.py: same code as the training data)
   ▼
 1 | 1/2 | lbs | . | boneless | chicken | breasts | , | cut | into | strips
   │  tagger: RoBERTa-base (default) or CRF / rules baselines
   ▼
 B-QUANTITY I-QUANTITY B-UNIT O B-PHYSICAL_QUALITY B-FOOD B-PART O B-PROCESS I-PROCESS I-PROCESS
   │  decoding rules                        (pantrypal/decode.py)
   ▼
 {"ingredient": "chicken breasts", "quantity": {"value": 1.5, ...}, "unit": "pound",
  "preparation": ["cut into strips"], "qualities": ["boneless"], "scalable": true, "confidence": 0.924, ...}
```

* **Entity types (12):** FOOD, QUANTITY, UNIT, PROCESS, PHYSICAL_QUALITY, COLOR, TASTE, PURPOSE,
  PART, TRADE_NAME, DIET, EXAMPLE. TASTEset-2.0's three rarest types (EXCLUSIVE, EXCLUDED,
  POSSIBLE_SUBSTITUTE: 1, 7, and 13 training examples) are merged into FOOD.
* **Main model:** RoBERTa-base, fine-tuned as a BIO token classifier (a softmax layer over 25 labels
  on the last transformer layer) with spaCy 3.8. The pretrained weights come from spaCy's
  `en_core_web_trf` 3.8.0 package, whose RoBERTa-base weights spaCy had already fine-tuned on
  OntoNotes 5 (tagging, parsing, NER). Training has two stages: the new output layer first, then all
  12 layers (`configs/roberta_stage1.cfg`, `configs/roberta_stage2.cfg`). At prediction time a
  Viterbi pass with tag-bigram scores from the training data keeps spans well-formed.
* **Baseline 1, rules:** numbers, unit words, "for ..." purposes, and "to taste", using word lists
  mined from the NYT ingredient data (`models/rules/nyt_lexicon.json`). No training.
* **Baseline 2, CRF:** a linear-chain CRF (sklearn-crfsuite) on the same training lines, with word
  shape, context, and the NYT word lists as features.
* **Decoding:** the first FOOD outside parentheses is the ingredient, with adjacent COLOR, TASTE,
  or DIET words and a following PART ("brown sugar", "chicken breasts"). Other foods become
  alternatives ("or") or extra items ("and"). Quantities become numbers or ranges. "2 14-ounce cans"
  becomes 2 cans with a 14-ounce package size. Units are normalized ("lbs." → pound). "To taste" lines
  are marked as not scalable. Lines with low confidence or no food are flagged `needs_review`.

## Repository layout

| Path | Contents |
|---|---|
| `predict.py` | **Command-line prototype**: lines, a text file, stdin, or a recipe JSONL in; tags + JSON out |
| `run_pipeline.py` | Runs every step below in order (data → baselines → evaluation → batch demo) |
| `pantrypal/` | Library: tokenization (`textproc.py`), labels, taggers, decoding to JSON, scorer |
| `scripts/get_data.py` | Downloads TASTEset-2.0 and NYT, pinned to exact commits, checks SHA-256 |
| `scripts/clean_data.py` | Team data pipeline (Milestone 3), now with the fraction-slash fix (2 lines; the model was trained on fixed data) |
| `scripts/scrape_recipes.py` | Our recipe scraper (Milestone 3) |
| `scripts/build_lexicon.py` | NYT word lists for the rule baseline and CRF features |
| `scripts/build_scraped_eval.py` | Hand labels → token-level gold for the cross-site test set |
| `scripts/train_crf.py` | CRF baseline: c1/c2 chosen on validation, optional k-fold CV |
| `scripts/prepare_corpus.py`, `scripts/train_roberta.sh`, `scripts/unfreeze_stage1.py`, `configs/` | RoBERTa training (two stages) |
| `scripts/tune_decoding.py` | Tag-bigram weight for Viterbi decoding, chosen on validation |
| `scripts/train_hf_tagger.py` | DistilBERT (or `roberta-base`) through Hugging Face on the same split, words, labels, and seed; needs Hugging Face access |
| `scripts/evaluate_12types.py` | All 12-type metrics, tables, and per-line predictions in `results/12types/` |
| `scripts/learning_curve.py` | CRF validation F1 at 25 / 50 / 100% of the training recipes |
| `examples/new_lines.txt` | New ingredient lines for trying the prototype |
| `requirements-prototype.txt`, `requirements-train.txt` | Packages to run / to retrain RoBERTa |
| `data/eval/` | Cross-site test set: `scraped_annotations.txt` (readable) and `scraped_gold.jsonl` |
| `eval/pinchofyum_gold.jsonl` | The same 151 lines in the format of `tasteset_bio.jsonl`, for `scripts/evaluate.py --sets pinchofyum` |
| `models/rules/`, `models/crf/`, `models/bio_transitions.json` | Small saved models (in Git) |
| `models/roberta-tasteset/` | Fine-tuned RoBERTa (476 MB, **not in Git**, see below) |
| `results/12types/` | `metrics.md`, `metrics.json`, per-line predictions, learning curve, batch output on scraped recipes |
| `samples/`, `DATA_README.md` | Raw → processed data samples and data documentation (Milestone 3) |

## Requirements

* Python 3.12 or 3.13 (tested on 3.13.16, Linux). Every package has Windows wheels for both versions.
* About 4 GB of RAM and 3 GB of disk (mostly PyTorch). No GPU, API keys, or accounts are needed.

## Setup

macOS / Linux:

```bash
git clone https://github.com/karriv1508/csc4641-pantrypal-recipe-extraction.git
cd csc4641-pantrypal-recipe-extraction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-prototype.txt
python scripts/get_data.py                           # TASTEset-2.0 + NYT, checksum-verified
python scripts/clean_data.py --root . --skip-recipenlg
```

Windows (PowerShell): the same, but activate with `.venv\Scripts\Activate.ps1` and use `python`.

**Saved RoBERTa model (needed for the default model; the baselines work without it).** Copy the
`models/roberta-tasteset` folder from the team OneDrive folder
([link](https://msoe365-my.sharepoint.com/:f:/g/personal/karriv_msoe_edu/IgCuADrTwMLRRbzmPFo1rIRVAar7drefwAtpJr_zZKZx-FE?e=2P1jda),
MSOE login) into `models/`, so that `models/roberta-tasteset/model/config.cfg` exists. The weights file
`model/transformer/model` has SHA-256 `c3d8fad588de2a9aea1aba51cb9639fd9b69b748d56eb927f5be11ae23661b59`.
A model you retrain yourself lands in `models/roberta/model-best`; the scripts use it when
`models/roberta-tasteset` is missing.

**Scraped recipes (optional, for the batch demo).** Copy `data_processed/scraped_clean.jsonl` from the
same OneDrive folder. The cross-site test set itself is already in `data/eval/`.

## Run

```bash
# 1. Tag new lines (default model: RoBERTa). --scale 2 also prints doubled amounts.
python predict.py "1 1/2 cups whole milk" "2 cloves garlic, minced" --scale 2
python predict.py --file my_recipe.txt               # one ingredient per line
python predict.py --json "salt and pepper to taste"  # JSON record per line
python predict.py --model crf "..."                  # or --model rules

# 2. Evaluate all three models on the three test sets -> results/12types/metrics.md
python scripts/evaluate_12types.py

# 3. Batch: every line of the 120 scraped recipes -> results/12types/scraped_predictions_roberta.jsonl
python predict.py --recipes data_processed/scraped_clean.jsonl --out results/12types/scraped_predictions_roberta.jsonl

# Everything in order (uses the saved RoBERTa model if present):
python run_pipeline.py
# Retrain RoBERTa too (about 40 minutes on a 2-core CPU):
pip install -r requirements-train.txt
python run_pipeline.py --train-roberta
```

## Example inputs and outputs

These lines are new; none of them is in any of our datasets. The output is real `predict.py` output.

**Example 1: a package size.** `python predict.py --scale 2 "2 (15 oz) cans chickpeas, drained and rinsed"`

```
2 (15 oz) cans chickpeas, drained and rinsed
  tags  : [2]QTY ( [15]QTY [oz]UNIT ) [cans]UNIT [chickpeas]FOOD , [drained]PROCESS and [rinsed]PROCESS
  item  : chickpeas
  amount: 2  (= 2)   unit: can   package: 15 ounce
  other : prep: drained, rinsed
  x2    : 4 cans
  conf. : 0.999
```

The tagger finds two quantities and two units. The decoding rules read them as 2 cans of 15 ounces each,
not 2 ounces or 15 cans, so doubling the recipe gives 4 cans. PantryPal can then match the line to a
pantry item "chickpeas" and subtract 2 cans. With `--json`, the same line comes back as a record with
character offsets for every entity.

**Example 2: two foods and an amount that must not scale.**
`python predict.py --scale 2 "salt and freshly ground black pepper, to taste"`

```
salt and freshly ground black pepper, to taste
  tags  : [salt]FOOD and [freshly ground]PROCESS [black]COLOR [pepper]FOOD , [to taste]QTY
  item  : salt
  amount: to taste / as needed   unit: -
  other : prep: freshly ground | also: pepper
  x2    : not scaled (no fixed amount)
  conf. : 0.983
```

"to taste" is tagged as a quantity, and decoding marks the line as not scalable, so doubling the recipe
leaves it alone instead of inventing an amount. The COLOR word joins its food ("black pepper" is the
name, not "pepper"), and the second food is kept under `also`, so PantryPal can check both items.

**Example 3: a mistake.** `python predict.py "1½ cups unsweetened almond milk (or any non-dairy milk)"`

```
1½ cups unsweetened almond milk (or any non-dairy milk)
  tags  : [1 1/2]QTY [cups]UNIT [unsweetened]TASTE [almond milk]FOOD ( or any [non-dairy]FOOD milk )
  item  : unsweetened almond milk
  amount: 1 1/2  (= 1.5)   unit: cup
  other : or: non-dairy
  conf. : 0.896
```

`½` becomes `1/2` and the main fields are right, but the substitute is tagged as "non-dairy" instead of
"non-dairy milk". The confidence (0.896) is above the 0.6 review threshold, so this one is not flagged.

More new lines are in `examples/new_lines.txt`: `python predict.py --file examples/new_lines.txt`.

## Evaluation (Milestone 4)

Entity-level exact-match F1 over the 12 types, plus *field accuracy*: the share of lines whose decoded
quantity, unit, and ingredient all equal the gold decoding. Full tables are in `results/12types/metrics.md`.

| Model | TASTEset test F1 (824 lines) | Unseen lines F1 (600) | Scraped F1 (151) | Scraped field acc. |
|---|---|---|---|---|
| Rules (NYT word lists) | 0.818 | 0.796 | 0.853 | 0.775 |
| CRF | 0.931 | 0.914 | 0.850 | 0.848 |
| **RoBERTa-base** | **0.937** | **0.922** | **0.882** | **0.881** |

* *Unseen lines* = the test lines whose text never appears in training (224 of 824 do, such as
  "1 teaspoon salt").
* *Scraped* = 151 labeled lines from 12 Pinch of Yum recipes drawn at random (seed 4641) from our 120
  scraped recipes: a different site, updated in the last five years, with chatty notes. None of the
  scraped recipes were used for training or tuning. The 95% paired-bootstrap interval for F1(RoBERTa) − F1(CRF) is
  [+0.009, +0.061] here, but [−0.002, +0.014] on the TASTEset test split. RoBERTa's advantage shows on new
  writing, not on TASTEset-style lines.
* Model choices (CRF regularization, RoBERTa checkpoint, Viterbi weight) used the validation split only.
  The test sets were scored once, and the system was not changed after seeing the results.
* Lines with confidence under 0.6 or no food found are flagged `needs_review`. On the scraped set the
  flag marks 14% of lines and catches 11 of the 18 lines with a wrong quantity, unit, or ingredient.
* Learning curve (CRF, validation F1 at 25 / 50 / 100% of the training recipes): 0.903 / 0.923 / 0.938.
  It is still rising, so more labeled or silver data should help (`results/12types/learning_curve.json`).

## Data and artifacts

| Item | Where | Size | Source / license |
|---|---|---|---|
| TASTEset-2.0 (gold, 1,000 recipes) | downloaded by `scripts/get_data.py` | 1.7 MB | Wróblewska et al., MIT |
| NYT ingredient snapshot (word lists only) | downloaded by `scripts/get_data.py` | 14 MB | NYT, Apache 2.0 |
| `data_processed/tasteset_bio.jsonl` (splits 3,721 / 829 / 824 lines) | made by `clean_data.py` (seed 4641) | 2.3 MB | derived |
| Scraped Pinch of Yum recipes (120 recipes, 1,515 lines) | team OneDrive, `data_processed/scraped_clean.jsonl` | 0.2 MB | our scraper; not redistributed |
| Cross-site test set (151 lines, 569 entities) | `data/eval/`, `eval/` (in Git) | 140 KB | labeled with Claude (see limitations) |
| Rule lexicon, CRF model, Viterbi bigrams | `models/` (in Git) | 1.2 MB | trained here |
| Fine-tuned RoBERTa | team OneDrive `models/roberta-tasteset` | 476 MB | trained here; base weights MIT (spaCy) |
| Metrics, per-line predictions, batch output | `results/12types/` (in Git) | 3.5 MB | `evaluate_12types.py`, `predict.py` |

RecipeNLG (Milestone 3) is not used by this prototype. It is reserved for silver labeling if the
learning curve shows that more training data would help.

## Known limitations

* **One annotation pass.** The 151 scraped test lines were labeled in one pass with Claude (an AI
  assistant), following the conventions at the top of `data/eval/scraped_annotations.txt`. No second
  annotator has checked them yet, so agreement is not measured. Some "errors" are convention choices. In "a pinch of garlic powder", for example, the
  model reads "a" as quantity 1, while our labels leave it out. TASTEset is inconsistent on such cases too.
* **Brands and rare types.** TRADE_NAME (F1 0.57 on scraped), DIET, and EXAMPLE have few training
  examples. "Better Than Bouillon beef flavor base" is tagged as one long food name.
* **One site.** The cross-site test is Pinch of Yum only. Other sites, handwritten lists, and non-US
  units are untested.
* **Decoding rules are simple.** A container unit the tagger misses ("1 28-ounce can …" without
  "can") loses the package reading. An amount given only in parentheses after a vague one ("teeny tiny bit
  of salt (let's say 1/8 teaspoon)") is not used. Lines with two ingredients ("salt and pepper") keep
  the first as the ingredient and list the other under `also`.
* **BIO limits.** Discontinuous entities ("chicken … broth" in "chicken or vegetable broth") cannot be
  represented, as noted in Milestone 3.
* **Speed and size.** RoBERTa needs about 21 ms per line on a 2-core CPU and a 476 MB model; the CRF is
  1 MB and about 70 times faster (0.3 ms per line). Retraining takes about 40 minutes on the same CPU.

## Change from the technical roadmap

The roadmap planned DistilBERT through Hugging Face. We use RoBERTa-base through spaCy instead,
the model family that scored best on TASTEset in Goel et al. Its weights install as a Python
package from GitHub, so the prototype runs fully offline with no Hugging Face access. The task,
labels, splits, and metrics are unchanged.
