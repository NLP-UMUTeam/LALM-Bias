# 📦 Results

This folder stores the raw JSONL outputs produced by the inference scripts.
Each task has a separate result root; tasks with prompt-order reversal have a
normal-order root and a reversed-order root.

## 📁 Directories

| Directory | Task | Output contents |
| --- | --- | --- |
| `results_transcription/` | Transcription of one audio recording. | One JSONL file per regional variety. |
| `results_bod/` | Standard Mandarin versus regional-variety recognition. | One JSONL file per regional variety. |
| `results_bod_reverse/` | The same recognition task with label order reversed. | One JSONL file per regional variety. |
| `results_individual/` | Social characterization using ten adjective pairs. | One JSONL file per adjective pair and regional variety. |
| `results_individual_reverse/` | The same characterization task with every pair reversed. | One JSONL file per adjective pair and regional variety. |

## 🗂️ Directory structure

```text
results/
├── results_transcription/
│   └── <model>/<variety>/<variety>_english.jsonl
├── results_bod/
│   └── <model>/<variety>/<variety>_english.jsonl
├── results_bod_reverse/
│   └── <model>/<variety>/<variety>_english.jsonl
├── results_individual/
│   └── <model>/<variety>/<variety>_<adjective-pair>.jsonl
└── results_individual_reverse/
    └── <model>/<variety>/<variety>_<adjective-pair>.jsonl
```

`<model>` is one of the inference model names and `<variety>` is one of the
eight regional Mandarin varieties. Every model--variety directory also contains
`metadata.json`, which records aggregate run and prompt timings.

## ⚙️ Evaluation settings

| Result root | Evaluation setting | Prompt configuration |
| --- | --- | --- |
| `results_transcription/` | Chinese speech transcription. | No order-reversal counterpart. |
| `results_bod/` | Standard Mandarin versus regional-variety recognition. | `standard` listed first. |
| `results_bod_reverse/` | The same recognition decisions. | Label order reversed. |
| `results_individual/` | Social speaker characterization over ten adjective pairs. | Unfavorable adjective listed first. |
| `results_individual_reverse/` | The same characterization decisions. | Adjective order reversed. |

Normal and reversed files for a task contain the same matched audio pairs. The
stability analyses retain only pairs whose valid answer is unchanged under the
two prompt orders for both recordings.

## 🧾 JSONL format

Each line is one matched regional--Standard Mandarin audio pair. The exact
response fields can vary by model, but result rows generally include:

| Field | Description |
| --- | --- |
| `speaker_id` | Shared speaker identifier for the two recordings. |
| `text` / `normalized_text` | Original text and normalized matching text. |
| `dialect_audio` / `mandarin_audio` | Paths to the paired regional and Standard Mandarin recordings. |
| `dialect_response` / `mandarin_response` | Raw response for each audio condition. |
| `dialect_token_probabilities` / `mandarin_token_probabilities` | Token probabilities when provided by the model. |
| `dialect_inference_time_sec` / `mandarin_inference_time_sec` | Model-inference durations. |
| `dialect_total_time_sec` / `mandarin_total_time_sec` | Total durations recorded for the two requests. |

The retained paths document inference provenance and can be absolute paths from
the environment in which the experiments were run. They are not portable audio
locations. Reconstruct the balanced KeSpeech subset as described in
`../dataset/README.md` before rerunning inference.

Use `../evaluation_stats/` to produce tables, figures, and time summaries, and
`../evaluation_hotwords/` to perform the lexical justification analysis.
