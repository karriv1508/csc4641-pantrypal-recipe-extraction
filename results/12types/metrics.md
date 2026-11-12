# Evaluation results

Entity-level exact match. Field accuracy = decoded quantity, unit, and ingredient all equal the gold decoding.

| Model | Test set | Lines | P | R | F1 | Macro F1 | Line acc. | Field acc. |
|---|---|---|---|---|---|---|---|---|
| rules | tasteset-test | 824 | 0.811 | 0.826 | **0.818** | 0.472 | 0.570 | 0.772 |
| rules | unseen | 600 | 0.787 | 0.804 | **0.796** | 0.460 | 0.480 | 0.737 |
| rules | scraped | 151 | 0.854 | 0.852 | **0.853** | 0.582 | 0.642 | 0.775 |
| crf | tasteset-test | 824 | 0.934 | 0.928 | **0.931** | 0.764 | 0.809 | 0.936 |
| crf | unseen | 600 | 0.918 | 0.910 | **0.914** | 0.747 | 0.747 | 0.918 |
| crf | scraped | 151 | 0.883 | 0.819 | **0.850** | 0.712 | 0.682 | 0.848 |
| roberta | tasteset-test | 824 | 0.930 | 0.944 | **0.937** | 0.769 | 0.829 | 0.934 |
| roberta | unseen | 600 | 0.914 | 0.931 | **0.922** | 0.751 | 0.775 | 0.918 |
| roberta | scraped | 151 | 0.897 | 0.868 | **0.882** | 0.749 | 0.715 | 0.881 |

Paired bootstrap (1,000 resamples of lines), 95% interval for F1(RoBERTa) - F1(CRF): tasteset-test [-0.002, +0.014]; unseen [-0.001, +0.018]; scraped [+0.009, +0.061]

## Per-type F1 on tasteset-test

| Type | Support | rules | crf | roberta |
|---|---|---|---|---|
| FOOD | 890 | 0.794 | 0.906 | 0.924 |
| QUANTITY | 832 | 0.870 | 0.989 | 0.992 |
| UNIT | 692 | 0.959 | 0.986 | 0.988 |
| PROCESS | 194 | 0.682 | 0.909 | 0.922 |
| PHYSICAL_QUALITY | 175 | 0.455 | 0.790 | 0.774 |
| COLOR | 47 | 0.800 | 0.891 | 0.870 |
| TASTE | 22 | 0.550 | 0.909 | 0.837 |
| PURPOSE | 16 | 0.552 | 0.839 | 0.833 |
| PART | 11 | 0.000 | 0.783 | 0.783 |
| TRADE_NAME | 46 | 0.000 | 0.427 | 0.644 |
| DIET | 15 | 0.000 | 0.522 | 0.500 |
| EXAMPLE | 7 | 0.000 | 0.222 | 0.160 |

Field accuracy by phenomenon on tasteset-test:

| Phenomenon | Lines | rules | crf | roberta |
|---|---|---|---|---|
| alternative (or) | 64 | 0.78 | 0.95 | 0.94 |
| long / chatty (>10 words) | 42 | 0.64 | 0.95 | 0.93 |
| no quantity | 108 | 0.87 | 0.93 | 0.93 |
| package size | 19 | 0.58 | 0.84 | 0.79 |
| parenthetical note | 216 | 0.77 | 0.94 | 0.93 |
| plain | 479 | 0.75 | 0.93 | 0.94 |
| range | 15 | 0.80 | 1.00 | 1.00 |
| vague amount | 8 | 0.88 | 1.00 | 1.00 |

## Per-type F1 on unseen

| Type | Support | rules | crf | roberta |
|---|---|---|---|---|
| FOOD | 665 | 0.763 | 0.878 | 0.903 |
| QUANTITY | 637 | 0.853 | 0.985 | 0.989 |
| UNIT | 538 | 0.953 | 0.981 | 0.984 |
| PROCESS | 178 | 0.681 | 0.902 | 0.915 |
| PHYSICAL_QUALITY | 154 | 0.446 | 0.767 | 0.751 |
| COLOR | 42 | 0.787 | 0.878 | 0.854 |
| TASTE | 20 | 0.556 | 0.900 | 0.821 |
| PURPOSE | 14 | 0.480 | 0.815 | 0.812 |
| PART | 9 | 0.000 | 0.737 | 0.737 |
| TRADE_NAME | 46 | 0.000 | 0.427 | 0.644 |
| DIET | 11 | 0.000 | 0.471 | 0.444 |
| EXAMPLE | 7 | 0.000 | 0.222 | 0.160 |

Field accuracy by phenomenon on unseen:

| Phenomenon | Lines | rules | crf | roberta |
|---|---|---|---|---|
| alternative (or) | 64 | 0.78 | 0.95 | 0.94 |
| long / chatty (>10 words) | 42 | 0.64 | 0.95 | 0.93 |
| no quantity | 61 | 0.85 | 0.90 | 0.92 |
| package size | 18 | 0.61 | 0.83 | 0.78 |
| parenthetical note | 208 | 0.78 | 0.94 | 0.93 |
| plain | 313 | 0.69 | 0.90 | 0.91 |
| range | 15 | 0.80 | 1.00 | 1.00 |
| vague amount | 4 | 0.75 | 1.00 | 1.00 |

## Per-type F1 on scraped

| Type | Support | rules | crf | roberta |
|---|---|---|---|---|
| FOOD | 180 | 0.848 | 0.807 | 0.906 |
| QUANTITY | 147 | 0.902 | 0.890 | 0.872 |
| UNIT | 129 | 0.942 | 0.930 | 0.930 |
| PROCESS | 49 | 0.780 | 0.907 | 0.968 |
| PHYSICAL_QUALITY | 22 | 0.375 | 0.667 | 0.744 |
| COLOR | 9 | 0.900 | 0.947 | 0.824 |
| TASTE | 6 | 0.800 | 0.800 | 0.800 |
| PURPOSE | 11 | 0.857 | 0.762 | 0.769 |
| PART | 4 | 0.000 | 0.857 | 0.857 |
| TRADE_NAME | 10 | 0.000 | 0.267 | 0.571 |
| DIET | 2 | 0.000 | 0.000 | 0.000 |
| EXAMPLE | 0 | 0.000 | 0.000 | 0.000 |

Field accuracy by phenomenon on scraped:

| Phenomenon | Lines | rules | crf | roberta |
|---|---|---|---|---|
| alternative (or) | 11 | 0.82 | 0.64 | 0.82 |
| long / chatty (>10 words) | 14 | 0.79 | 0.64 | 0.86 |
| no quantity | 23 | 0.83 | 0.70 | 0.83 |
| package size | 5 | 0.60 | 0.40 | 0.60 |
| parenthetical note | 34 | 0.79 | 0.68 | 0.76 |
| plain | 85 | 0.78 | 0.98 | 0.96 |
| range | 7 | 1.00 | 0.86 | 1.00 |
| vague amount | 7 | 0.14 | 0.00 | 0.14 |

Overlap check: tasteset-test: 224 of 824 lines also appear verbatim in the training split, unseen: 0 of 600 lines also appear verbatim in the training split, scraped: 25 of 151 lines also appear verbatim in the training split
