# 🗂️ Dataset

This folder contains the script that reconstructs the balanced KeSpeech subset
used for evaluation. The subset pairs a regional-variety recording and a
Standard Mandarin recording from the same speaker reading the same normalized
text.

## 📄 Files

| File | Description |
| --- | --- |
| `KeSpeech_subset_selection.py` | Builds the subset, copies the selected audio files, and writes JSONL manifests and a selection summary. |

## 🎧 KeSpeech files

Download KeSpeech under its terms of use. `--metadata-dir` must point to its
`Metadata/` directory; the parent directory of `Metadata/` must contain
`Audio/`.

The following metadata files are required:

```text
Metadata/
├── phase1.text
├── phase1.wav.scp
├── phase1.utt2style
├── phase1.utt2subdialect
├── phase2.text
├── phase2.wav.scp
├── utt2spk
├── spk2age
├── spk2city
└── spk2gender
```

## ⚙️ Balanced subset construction

Run this command from this folder:

```bash
python KeSpeech_subset_selection.py \
  --metadata-dir /path/to/KeSpeech/Metadata \
  --output-dir KeSpeech_subset
```

`KeSpeech_subset` must not exist before running the script. For example, if
the original `Metadata/` directory is at `/data/KeSpeech/Metadata`:

```bash
python KeSpeech_subset_selection.py \
  --metadata-dir /data/KeSpeech/Metadata \
  --output-dir ../KeSpeech_subset
```

### 🔎 Selection procedure

The script:

1. retains Phase 1 regional recordings labelled `dialect`;
2. matches them to Phase 2 recordings using the same speaker identifier and
   text after Unicode, punctuation, and whitespace normalization;
3. verifies that both audio files are located under the same speaker directory;
4. uses Beijing as the reference variety for the target gender and age
   distribution;
5. samples the remaining seven varieties to reproduce those targets, allocating
   each stratum across cities in proportion to its available matched pairs; and
6. copies only the selected audio files and writes one manifest per variety.

Sampling is deterministic and uses seed `42`. The included varieties are
Beijing, Ji-Lu, Jiang-Huai, Jiao-Liao, Lan-Yin, Northeastern, Southwestern,
and Zhongyuan.

## ⚖️ Sampling targets

The following targets are reproduced independently for each of the eight
regional varieties:

| Sampling target | Pairs per variety |
| --- | ---: |
| Female speakers | 423 |
| Male speakers | 423 |
| Age 18--29 | 262 |
| Age 30--39 | 408 |
| Age 40+ | 176 |
| **Total** | **846** |

## 📦 Output structure

```text
KeSpeech_subset/
├── Audio/
│   └── <speaker_id>/
│       ├── phase1/
│       └── phase2/
├── parallel_pairs_by_dialect/
│   ├── Beijing.jsonl
│   ├── Ji-Lu.jsonl
│   └── ...
└── selection_summary.json
```

## 🧾 Manifest fields

Each manifest row contains, among others, the following fields:

| Field | Description |
| --- | --- |
| `speaker_id` | Identifier shared by both recordings. |
| `text` / `normalized_text` | Original transcription and the version used for matching. |
| `subdialect` | Regional variety of the Phase 1 recording. |
| `dialect_audio` | Relative path to the Phase 1 regional audio. |
| `mandarin_audio` | Relative path to the Phase 2 Standard Mandarin audio. |
| `dialect_utt_id` / `mandarin_utt_id` | Original identifiers of the two utterances. |

`selection_summary.json` records the seed, pair counts by variety, gender, age,
and city, as well as repeated-speaker statistics. Pass the output directory
directly to `../evaluation_stats/stats_dataset.py` to generate descriptive tables.

## 📊 Final subset statistics

Pair construction initially yields 75,017 matched regional--Standard Mandarin
pairs from 9,124 unique speakers after speaker-identifier validation. The
balanced evaluation subset contains 6,768 matched pairs, or 846 pairs for each
of the eight regional varieties. This corresponds to 13,536 audio recordings
from 3,892 unique speakers.

The eight regional varieties are Beijing, Ji-Lu, Jiang-Huai, Jiao-Liao, Lan-Yin, Northeastern, Southwestern,
and Zhongyuan.

### 🏙️ Geographic composition

The selected pairs are distributed across cities according to the availability
of matched pairs within each gender--age stratum.

| Variety | City | Pairs |
| --- | --- | ---: |
| Beijing | Langfang | 661 |
| Beijing | Chaoyang | 63 |
| Beijing | Chengde | 46 |
| Beijing | Beijing | 40 |
| Beijing | Chifeng | 36 |
| Ji-Lu | Shijiazhuang | 571 |
| Ji-Lu | Jinan | 122 |
| Ji-Lu | Baoding | 60 |
| Ji-Lu | Tangshan | 47 |
| Ji-Lu | Xingtai | 46 |
| Jiang-Huai | Hefei | 493 |
| Jiang-Huai | Nanjing | 337 |
| Jiang-Huai | Yangzhou | 16 |
| Jiao-Liao | Qingdao | 561 |
| Jiao-Liao | Dalian | 141 |
| Jiao-Liao | Yantai | 108 |
| Jiao-Liao | Yingkou | 18 |
| Jiao-Liao | Weihai | 16 |
| Jiao-Liao | Dandong | 2 |
| Lan-Yin | Yinchuan | 415 |
| Lan-Yin | Lanzhou | 291 |
| Lan-Yin | Zhongwei | 61 |
| Lan-Yin | Wuzhong | 57 |
| Lan-Yin | Shizuishan | 22 |
| Northeastern | Shenyang | 320 |
| Northeastern | Changchun | 312 |
| Northeastern | Harbin | 214 |
| Southwestern | Chengdu | 486 |
| Southwestern | Chongqing | 127 |
| Southwestern | Kunming | 90 |
| Southwestern | Guilin | 76 |
| Southwestern | Guiyang | 67 |
| Zhongyuan | Xi'an | 430 |
| Zhongyuan | Zhengzhou | 416 |

### 🗣️ Repeated speakers

Sampling is performed at the pair level, so a speaker can contribute more than
one matched pair. The final subset has the following distribution:

| Statistic | Value |
| --- | ---: |
| Mean pairs per speaker | 1.74 |
| Median pairs per speaker | 1 |
| Speakers with more than one pair | 38.4% |
| Maximum pairs from one speaker | 18 |

`../evaluation_stats/stats_dataset.py` regenerates the corresponding LaTeX
tables directly from `selection_summary.json`.

## 🔁 Reproducibility

To reconstruct the subset, download KeSpeech, provide the required metadata
and audio files, and run `KeSpeech_subset_selection.py` with the command above.
The fixed seed, matching procedure, sampling targets, and selected counts are
recorded in `selection_summary.json`. Exact copied-file paths depend on the
location of the local KeSpeech installation.
