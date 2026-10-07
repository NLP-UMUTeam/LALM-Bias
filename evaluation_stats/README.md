# 📈 Evaluation stats

This folder contains the statistical analysis used to summarize the benchmark
and evaluate model outputs. It reads the balanced-subset summary and the raw
JSONL results from `../results/`, then generates the paper's LaTeX tables, PDF
radar figures, and execution-time summaries.

Run the commands in this README from `evaluation_stats/`.

## 📄 Files

| Script | Purpose | Main outputs |
| --- | --- | --- |
| `stats_dataset.py` | Describes the balanced KeSpeech subset from `selection_summary.json`. | Three LaTeX tables for sampling, cities, and repeated speakers. |
| `stats_time.py` | Aggregates durations recorded in result JSONL files. | Per-task and combined JSON summaries. |
| `stats_cer.py` | Computes transcription Character Error Rate (CER). | One CER LaTeX table. |
| `stats_bod.py` | Computes Standard-versus-regional recognition metrics. | Recognition summary and stable-subset confusion table. |
| `stats_adj.py` | Computes adjective-selection stability and social-bias statistics. | Summary, stability, and full-result LaTeX tables. |
| `plots.py` | Draws adjective-bias radar figures. | One PDF per model. |


All analysis scripts require an explicit `--output-dir`. The adjective,
recognition, and CER scripts expect a `MODEL/VARIETY/*.jsonl` result layout.

## 🔗 Expected input layout

```text
results_individual/
└── <model>/
    └── <variety>/
        └── <variety>_<adjective-pair>.jsonl

results_individual_reverse/
└── <model>/
    └── <variety>/
        └── <variety>_<adjective-pair>.jsonl

results_bod/
└── <model>/
    └── <variety>/
        └── <variety>_english.jsonl

results_bod_reverse/
└── <model>/
    └── <variety>/
        └── <variety>_english.jsonl

results_transcription/
└── <model>/
    └── <variety>/
        └── <variety>_english.jsonl
```

## 🧾 Output parsing

The inference scripts retain the raw generated text. Task-specific evaluation
scripts then apply the following automatic parsing rules; they only recover
simple, unambiguous surface variants and do not interpret free-form responses.

| Task | Parsing rule | Invalid if |
| --- | --- | --- |
| Social speaker characterization | Extract exactly one candidate adjective from a direct answer, an explicit `Answer:` formulation, or an unambiguous speaker description. | Neither candidate adjective can be extracted, or multiple alternatives are presented ambiguously. |
| Standard-versus-regional variety recognition | Map `standard`, `Standard Mandarin`, or `Mandarin` to Standard Mandarin; map `regional`, `regional subdialect`, or `subdialect` to regional speech. | Neither class can be identified unambiguously. |

Speech transcription is evaluated directly from the generated text and does
not require class-label parsing.

## 🗂️ Dataset stats

### ▶️ Usage

`stats_dataset.py` reads only `selection_summary.json` from the subset created
by `../dataset/KeSpeech_subset_selection.py`:

It receives the subset directory containing `selection_summary.json` and writes
its LaTeX tables to the selected output directory.

```bash
python stats_dataset.py \
  --dataset /path/to/KeSpeech_subset \
  --output-dir outputs/dataset
```

### 📄 Outputs

It writes `0_balanced_sampling.tex`, `1_dataset_cities.tex`, and
`2_repeated_speakers.tex`.

### 📏 Metrics

The sampling table checks the target number of pairs per gender and age group.
The city table counts the selected matched pairs for each regional variety and
city. The repeated-speaker table reports the mean and median number of pairs
per speaker, the proportion of speakers contributing more than one pair, and
the maximum number of pairs contributed by one speaker.

## ⏱️ Execution-time stats

### ▶️ Usage

`stats_time.py` can process any non-empty combination of the three tasks:

```bash
python stats_time.py \
  --transcription-results /path/to/results_transcription \
  --bod-results /path/to/results_bod /path/to/results_bod_reverse \
  --adj-results /path/to/results_individual /path/to/results_individual_reverse \
  --output-dir outputs/times
```

### 📄 Outputs

The script writes one JSON summary for every supplied task and
`total_time_by_model.json`, which sums the included tasks for each model.
Durations are the sums of the per-audio times recorded in the JSONL rows.

### 📏 Metrics

For each model and task, the reported value is the sum of the execution times
recorded in the input JSONL files. When multiple tasks are supplied, the
combined summary adds the corresponding task totals for each model; it does
not average durations across files or varieties.

## 🎧 Transcription stats

### ▶️ Usage

`stats_cer.py` receives only the transcription-results root. Its table is
written under `cer/` inside the selected output directory.

```bash
python stats_cer.py \
  /path/to/results_transcription \
  --output-dir outputs
```

### 📄 Outputs

This writes `outputs/cer/0_cer.tex`, reporting Standard Mandarin CER, regional
CER, and their percentage-point difference by model and variety.

### 📏 Metrics

CER is computed separately for Standard Mandarin and regional speech:

$$
\mathrm{CER} =
\frac{S + D + I}{N_{\mathrm{reference}}} \times 100,
$$

where $S$, $D$, and $I$ are substitutions, deletions, and insertions.

The table also reports the difference in percentage points:
`Regional CER − Standard CER`. Positive values indicate higher transcription
error for regional speech.

## 🗣️ Variety-recognition stats

### ▶️ Usage

`stats_bod.py` receives two result roots: normal and reversed order. Its tables
are written under `bod/` inside the selected output directory.

```bash
python stats_bod.py \
  /path/to/results_bod \
  /path/to/results_bod_reverse \
  --output-dir outputs
```

### 📄 Outputs

The script writes `outputs/bod/0_bod_recognition_summary.tex` and
`outputs/bod/1_bod_stability.tex`. The summary reports stability, class-wise
recall, and Macro F1. The second table is a full-width stable-subset confusion
table: for both Standard and Regional recordings, its count columns are ordered
as prediction Standard then prediction Regional. It also reports Regional
precision, recall, and F1, Macro F1, stability, and unweighted mean rows over
the eight regional varieties for every model.

### 📏 Metrics

A matched pair is stable when all four normal--reverse labels are valid and
unchanged. The stable percentage is:

$$
\mathrm{Stable} =
\frac{N_{\mathrm{stable}}}{N_{\mathrm{matched}}} \times 100.
$$

On the stable subset, each recording is classified as Standard or Regional.
For each class, precision is the proportion of predicted instances with that
class, recall is the proportion of true instances recovered, and F1 is the
harmonic mean of precision and recall. Macro F1 is the unweighted mean of the
Standard and Regional F1 values.

## ⚖️ Adjective-selection stats

### ▶️ Usage

Provide normal-order results first and reversed-order results second:

`stats_adj.py` receives two result roots: normal and reversed order. Its tables
are written under `adj/` inside the selected output directory.

```bash
python stats_adj.py \
  /path/to/results_individual \
  /path/to/results_individual_reverse \
  --output-dir outputs
```

### 📄 Outputs

The script writes the following files under `outputs/adj/`:

| File | Contents |
| --- | --- |
| `0_stability_bias_summary.tex` | Percentage of retained pair--dimension observations, paired shifts, aggregate bias, and its 95% paired speaker-cluster bootstrap CI by model and variety. |
| `1_prompt_order_stability_by_dimension.tex` | Prompt-order stability summarized by social dimension. |
| `1_prompt_order_stability_by_variety.tex` | Prompt-order stability summarized by regional variety. |
| `2_full_results_<model>.tex` | Stable paired counts, selection shifts, bias, and paired speaker-cluster bootstrap intervals for one model and social dimension. |

All bootstrap intervals use 10,000 paired speaker-cluster resamples with base
seed `42`. An independent deterministic seed is derived for each estimand:
each model--variety--dimension result in the full tables and each
model--variety aggregate in the summary table.

### 📏 Metrics

A matched pair is stable when the normal and reversed responses are valid and
unchanged for both recordings. Thus, all four responses must pass the
criterion. For each adjective pair, we calculate the two paired transition
rates directly on this stable subset.

Positive Bias values indicate more unfavorable characterizations of regional
speech. `F → U` is the stable paired shift from favorable Standard Mandarin to
unfavorable regional speech, while `U → F` is the reverse shift. At full
precision, `Bias = F → U − U → F`; the summary averages the ten adjective-pair
values without pooling their observations.

The full-result tables estimate a 95% CI for the Bias of one social dimension.
The summary table estimates a 95% CI for aggregate Bias, the mean Bias across
the ten dimensions. Both use the same paired speaker-cluster bootstrap. For an
aggregate replicate, the same sampled speakers are used across all ten
dimensions, each dimension's Bias is recalculated, and those ten values are
averaged. Consequently, the aggregate CI describes the uncertainty of the
aggregate Bias itself; it is not a combination of the dimension-level CIs.

## 🕸️ Radar figures

```bash
python plots.py \
  /path/to/results_individual \
  /path/to/results_individual_reverse \
  --output-dir outputs/figures
```

One PDF is written per model. Each axis is a social dimension in the same order
and terminology as the tables; each line is a regional variety. Values are the
stable-intersection selection gaps in percentage points.
