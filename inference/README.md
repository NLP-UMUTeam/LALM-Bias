# 🧠 Inference

This folder contains the zero-shot inference code used to evaluate
audio-language models on the regional--Standard Mandarin pairs in the KeSpeech
subset. Model weights are downloaded from Hugging Face on first use.

## 📄 Files

| File | Description |
| --- | --- |
| `ALMBase.py` | Shared loader for the three supported models and adapter for their processors. |
| `ZeroShotAudio.py` | Runs an audio--prompt interaction and returns the response, token probabilities, and timings. |
| `main.py` | Reads per-variety manifests and writes JSONL results. |
| `utils.py` | Command-line arguments, random seeds, speaker validation, and output paths. |

## 🤖 Supported models

| `--model_name` value | Hugging Face identifier | Intended environment |
| --- | --- | --- |
| `Qwen3-Omni-30B-A3B-Instruct` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | `qwen3` |
| `Phi-4-multimodal-instruct` | `microsoft/Phi-4-multimodal-instruct` | `phi_kimi` |
| `Kimi-Audio-7B-Instruct` | `moonshotai/Kimi-Audio-7B-Instruct` | `phi_kimi` |

For Kimi-Audio, also install the official package from
[MoonshotAI/Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio):

```bash
python -m pip install git+https://github.com/MoonshotAI/Kimi-Audio.git@349251e1d8f4
```

## 🔬 Experimental inference configuration

The reported experiments use the following model checkpoints and revisions.
No model is fine-tuned.

| Model | Checkpoint ID | Revision |
| --- | --- | --- |
| Qwen3-Omni | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | `26291f793822fb6be9555850f06dfe95f2d7e695` |
| Phi-4 | `microsoft/Phi-4-multimodal-instruct` | `93f923e1a7727d1c4f446756212d9d3e8fcc5d81` |
| Kimi-Audio | `moonshotai/Kimi-Audio-7B-Instruct` | `9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b` |

All models receive audio before the task-specific textual instruction, with no
system prompt. Shared settings are seed `42` and `max_new_tokens=512`.

| Setting | Qwen3-Omni | Phi-4 | Kimi-Audio |
| --- | --- | --- | --- |
| Decoding | Sampling disabled | Sampling disabled | Temperature `0` |
| Output | Text only (Thinker) | Text only | Text only |
| Audio generation | Talker disabled | Not applicable | Disabled through text-only output |
| Input interface | Official chat template | Native multimodal format | Official message format |
| Framework | Transformers `5.6.2` | Transformers `4.48.2` | Official Kimi-Audio inference code |
| Precision | `bfloat16` | `bfloat16` | Model default |

> **Reproducibility note:** the Qwen3-Omni and Phi-4 loaders pin the revisions
> listed above. `KimiAudio` does not expose a revision parameter. To reproduce
> the Kimi-Audio checkpoint exactly, download the reported snapshot and replace
> the existing `KimiAudio(...)` initialization in `ALMBase.py` with the
> following code:

```python
from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="moonshotai/Kimi-Audio-7B-Instruct",
    revision="9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b",
)

self.model = KimiAudio(
    model_path=model_path,
    load_detokenizer=False,
)
```

## 📥 Expected input

Before running inference, build the KeSpeech subset with the script in
`../dataset/`. `--dataset_dir` must point to the output directory containing:

```text
KeSpeech_subset/
├── Audio/
└── parallel_pairs_by_dialect/
    ├── Beijing.jsonl
    ├── Ji-Lu.jsonl
    └── ...
```

Each manifest requires `speaker_id`, `text`, `normalized_text`, `subdialect`,
`dialect_audio`, and `mandarin_audio`. Audio paths can be relative to the
subset or absolute.

## ⚙️ Installation

Model weights are downloaded from Hugging Face on first use. Kimi-Audio also
requires the official inference package listed below.

The repository includes the environment snapshots used for inference:

```bash
# Qwen3-Omni
conda activate qwen3
python -m pip install -r ../requirements_inference_qwen.txt

# Phi-4 and Kimi-Audio
conda activate phi_kimi
python -m pip install -r ../requirements_inference_phi_kimi.txt
```

Install the official Kimi-Audio package in the `phi_kimi` environment before
running that model:

```bash
python -m pip install git+https://github.com/MoonshotAI/Kimi-Audio.git@349251e1d8f4
```

## 🚀 Basic usage

Run the following commands from `inference/`. `--results_dir`, `--prompts_dir`,
`--dataset_dir`, and `--model_name` are required.

```bash
python main.py \
  --results_dir /path/to/results \
  --prompts_dir ../prompts/prompts_individual_reason \
  --dataset_dir /path/to/KeSpeech_subset \
  --model_name Qwen3-Omni-30B-A3B-Instruct
```

Change the model name to run Phi-4 or Kimi-Audio:

```bash
python main.py \
  --results_dir /path/to/results \
  --prompts_dir ../prompts/prompts_individual_reason \
  --dataset_dir /path/to/KeSpeech_subset \
  --model_name Phi-4-multimodal-instruct
```

```bash
python main.py \
  --results_dir /path/to/results \
  --prompts_dir ../prompts/prompts_individual_reason \
  --dataset_dir /path/to/KeSpeech_subset \
  --model_name Kimi-Audio-7B-Instruct
```

Use the relevant directories under `../prompts/` for each task:

| Task | Normal prompt | Reversed prompt |
| --- | --- | --- |
| Transcription | `prompts_transcription/` | Not applicable. |
| Variety recognition | `prompts_base_or_dialect/` | `prompts_base_or_dialect_reverse/` |
| Social characterization | `prompts_individual_reason/` | `prompts_individual_reason_reverse/` |

## 🗣️ One or more varieties

If `--dialects` is omitted, every manifest in
`<dataset_dir>/parallel_pairs_by_dialect/` is processed. To run only Beijing:

```bash
python main.py \
  --results_dir /path/to/results \
  --prompts_dir ../prompts/prompts_individual_reason \
  --dataset_dir /path/to/KeSpeech_subset \
  --dialects Beijing \
  --model_name Qwen3-Omni-30B-A3B-Instruct
```

For multiple varieties, add their names after `--dialects`:

```bash
python main.py \
  --results_dir /path/to/results \
  --prompts_dir ../prompts/prompts_individual_reason \
  --dataset_dir /path/to/KeSpeech_subset \
  --dialects Beijing Ji-Lu Jiang-Huai \
  --model_name Qwen3-Omni-30B-A3B-Instruct
```

## 📦 Output

Results are written to:

```text
<results_dir>/
└── <model>/
    └── <variety>/
        ├── <variety>_<prompt>.jsonl
        └── metadata.json
```

Each JSONL row retains the pair metadata, paths for both recordings, responses
for the regional and Standard Mandarin audio, available token probabilities,
and inference and execution times. When the same prompt is run again, pairs
whose two audio paths already occur in the JSONL file are skipped to avoid
duplicates.
