# 🔍 Evaluation hotwords

This folder contains the two-stage lexical analysis of model-generated
justifications for the adjective-selection task. It focuses on stable matched
cases in which Standard Mandarin receives a favorable adjective and regional
speech receives an unfavorable adjective.

Run the commands in this README from `evaluation_hotwords/`.

## 📄 Files

| File | Description |
| --- | --- |
| `extract_hotwords.py` | Selects stable favorable-Standard/unfavorable-regional cases and writes their original justifications and intermediate counts. |
| `create_tables.py` | Computes unigram, bigram, and trigram comparisons and writes ranked CSV, Markdown, and LaTeX tables. |

## 🔗 Expected input layout

Both adjective-result roots must contain the same model and variety structure:

```text
results_individual/
└── <model>/
    └── <variety>/
        └── <variety>_<adjective-pair>.jsonl

results_individual_reverse/
└── <model>/
    └── <variety>/
        └── <variety>_<adjective-pair>.jsonl
```

`extract_hotwords.py` receives these two roots in normal-then-reversed order.
`create_tables.py` then receives one model's extracted justification directory,
such as `hotwords_output/justifications/Qwen3-Omni`.

## 1. 🧲 Extract justifications

Provide the normal-order adjective results first and the reversed-order results
second. Both roots must follow the `MODEL/VARIETY/*.jsonl` layout.

```bash
python extract_hotwords.py \
  /path/to/results_individual \
  /path/to/results_individual_reverse \
  --output-dir hotwords_output
```

The script writes intermediate CSV files, LaTeX tables, and original
justifications. The latter are stored in
`output-dir/justifications/<model>/`.

## 2. 📊 Compute characteristic n-grams

Select the justifications directory for one model generated in the first step.
For example, for Qwen3-Omni:

```bash
python create_tables.py \
  --input-dir hotwords_output/justifications/Qwen3-Omni \
  --output-dir top_words_qwen3_omni
```

For another model, replace `Qwen3-Omni` with the relevant directory name under
`hotwords_output/justifications/`. The script writes CSV, Markdown, and LaTeX
results, as well as `scores.csv` and `metadata.json`.

## 📦 Output structure

The extraction step writes original justifications and per-group term counts:

```text
hotwords_output/
├── justifications/
│   └── <model>/<adjective-pair>/justifications_<condition>.jsonl
└── hotwords/
    └── <model>/<condition>/<adjective-pair>/<variety>.csv
```

The table-generation step writes the complete scored terms and two comparison
tables in CSV, Markdown, and LaTeX formats:

```text
top_palabras_qwen3_omni/
├── scores.csv
├── comparison_negative.csv
├── comparison_negative.md
├── comparison_negative.tex
├── comparison_positive.csv
├── comparison_positive.md
├── comparison_positive.tex
└── metadata.json
```

`metadata.json` records the selected source files, filtering parameters,
per-dimension diagnostics, and the direction of the descriptive z scores.

## ⚙️ Optional n-gram parameters

`create_tables.py` accepts the following optional arguments:

| Argument | Default | Meaning |
| --- | ---: | --- |
| `--n` | `10` | Maximum number of ranked terms shown in each table cell. |
| `--field` | `both` | Justification source: `normal_justification`, `reverse_justification`, or `both`. |
| `--min-docs` | `5` | Minimum combined response-document frequency for a term to be scored. |
| `--prior-scale` | `0.1` | Positive smoothing prior used by the descriptive log-odds calculation. |

## 💡 Interpretation

The extraction step requires four valid, order-consistent responses for every
matched pair: normal and reversed answers for both the regional and Standard
Mandarin recordings. It then retains cases where the Standard Mandarin audio
is assigned the favorable adjective and the regional audio the unfavorable one.

The table-generation step pools the retained cases across all eight regional
varieties and both prompt orders. It compares terms used in the Standard and
regional justifications. Positive and negative association scores identify
terms that are comparatively characteristic of one condition; consult the
generated table captions for the direction used in each table.
