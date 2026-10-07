import argparse
import json
import random
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


DIALECTS = (
    "Beijing",
    "Ji-Lu",
    "Jiang-Huai",
    "Jiao-Liao",
    "Lan-Yin",
    "Northeastern",
    "Southwestern",
    "Zhongyuan",
)
AGE_GROUPS = ("18-29", "30-39", "40+", "Unknown")
SEED = 42


def read_mapping(path):
    mapping = {}

    with path.open(encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line:
                key, value = line.split(maxsplit=1)
                mapping[key] = value

    return mapping


def normalize_text(text):
    normalized = unicodedata.normalize("NFKC", text).lower()

    return "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith("P")
        and not character.isspace()
    )


def resolve_audio_path(raw_path, metadata_dir):
    if raw_path.endswith("|"):
        return None

    path = Path(raw_path)
    if path.is_absolute():
        return path

    candidates = (metadata_dir.parent / path, metadata_dir / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (metadata_dir.parent / path).resolve()


def get_age_group(age):
    if age is None or age < 18:
        return "Unknown"
    if age <= 29:
        return "18-29"
    if age <= 39:
        return "30-39"
    return "40+"


def get_audio_speaker_id(audio_path, audio_root):
    try:
        return audio_path.relative_to(audio_root).parts[0]
    except ValueError:
        return None


def build_pairs(metadata_dir):
    phase1_text = read_mapping(metadata_dir / "phase1.text")
    phase1_wav = read_mapping(metadata_dir / "phase1.wav.scp")
    phase1_style = read_mapping(metadata_dir / "phase1.utt2style")
    phase1_subdialect = read_mapping(metadata_dir / "phase1.utt2subdialect")
    phase2_text = read_mapping(metadata_dir / "phase2.text")
    phase2_wav = read_mapping(metadata_dir / "phase2.wav.scp")
    utt2spk = read_mapping(metadata_dir / "utt2spk")
    phase2_index = defaultdict(list)

    for utt_id, text in phase2_text.items():
        speaker_id = utt2spk.get(utt_id)
        audio_path = phase2_wav.get(utt_id)
        if speaker_id is not None and audio_path is not None:
            phase2_index[speaker_id, normalize_text(text)].append((
                utt_id,
                text,
                resolve_audio_path(audio_path, metadata_dir),
            ))

    pairs = []
    for utt_id, text in phase1_text.items():
        if phase1_style.get(utt_id, "").strip().lower() != "dialect":
            continue

        speaker_id = utt2spk.get(utt_id)
        subdialect = phase1_subdialect.get(utt_id)
        audio_path = phase1_wav.get(utt_id)
        if (
            speaker_id is None
            or subdialect not in DIALECTS
            or audio_path is None
        ):
            continue

        normalized_text = normalize_text(text)
        dialect_audio = resolve_audio_path(audio_path, metadata_dir)
        for mandarin_utt_id, mandarin_text, mandarin_audio in phase2_index[
            speaker_id,
            normalized_text
        ]:
            if dialect_audio is None or mandarin_audio is None:
                continue

            pairs.append({
                "speaker_id": speaker_id,
                "text": text,
                "normalized_text": normalized_text,
                "subdialect": subdialect,
                "dialect_utt_id": utt_id,
                "dialect_audio": dialect_audio,
                "mandarin_utt_id": mandarin_utt_id,
                "mandarin_audio": mandarin_audio,
                "phase1_text": text,
                "phase2_text": mandarin_text,
            })

    return pairs


def filter_valid_pairs(records, audio_root):
    return [
        record
        for record in records
        if (
            str(record["speaker_id"])
            == get_audio_speaker_id(record["dialect_audio"], audio_root)
            == get_audio_speaker_id(record["mandarin_audio"], audio_root)
        )
    ]


def get_record_info(record, speaker_to_age, speaker_to_city, speaker_to_gender):
    speaker_id = str(record["speaker_id"])
    age_text = speaker_to_age.get(speaker_id)
    age = int(age_text) if age_text is not None else None

    return (
        speaker_to_gender.get(speaker_id, "Unknown"),
        get_age_group(age),
        speaker_to_city.get(speaker_id, "Unknown"),
    )


def get_reference_records(records, speaker_to_age, speaker_to_city, speaker_to_gender):
    records_by_gender = {"Female": [], "Male": []}

    for record in records:
        gender, _, _ = get_record_info(
            record,
            speaker_to_age,
            speaker_to_city,
            speaker_to_gender,
        )
        if gender in records_by_gender:
            records_by_gender[gender].append(record)

    pairs_per_gender = min(map(len, records_by_gender.values()))
    selected_records = []

    for gender, gender_records in records_by_gender.items():
        random.Random(f"{SEED}:{gender}").shuffle(gender_records)
        selected_records.extend(gender_records[:pairs_per_gender])

    random.Random(f"{SEED}:Beijing").shuffle(selected_records)

    return selected_records


def get_target_age_counts(records, speaker_to_age, speaker_to_city, speaker_to_gender):
    counts = Counter()

    for record in records:
        gender, age_group, _ = get_record_info(
            record,
            speaker_to_age,
            speaker_to_city,
            speaker_to_gender,
        )
        counts[gender, age_group] += 1

    return counts


def get_age_counts(target_counts, available_counts, pairs_per_gender):
    selected_counts = Counter()

    for gender, target_count in pairs_per_gender.items():
        available_pairs = sum(
            available_counts[gender, age_group]
            for age_group in AGE_GROUPS
        )
        if available_pairs < target_count:
            raise ValueError(
                f"Not enough valid {gender} pairs: "
                f"{available_pairs} available, {target_count} needed"
            )

        for _ in range(target_count):
            age_group = min(
                (
                    age_group
                    for age_group in AGE_GROUPS
                    if selected_counts[gender, age_group]
                    < available_counts[gender, age_group]
                ),
                key=lambda item: (
                    (selected_counts[gender, item] + 1 - target_counts[gender, item]) ** 2
                    - (selected_counts[gender, item] - target_counts[gender, item]) ** 2,
                    -target_counts[gender, item],
                    item,
                ),
            )
            selected_counts[gender, age_group] += 1

    return selected_counts


def get_city_counts(total, available_counts):
    total_available = sum(available_counts.values())
    counts = {
        city: total * count // total_available
        for city, count in available_counts.items()
    }
    remaining = total - sum(counts.values())
    cities = sorted(
        available_counts,
        key=lambda city: (
            -(total * available_counts[city] % total_available),
            city,
        ),
    )

    for city in cities[:remaining]:
        counts[city] += 1

    return counts


def select_records(
    records,
    target_age_counts,
    pairs_per_gender,
    speaker_to_age,
    speaker_to_city,
    speaker_to_gender,
    dialect,
):
    records_by_group = defaultdict(list)
    available_counts = Counter()

    for record in records:
        gender, age_group, city = get_record_info(
            record,
            speaker_to_age,
            speaker_to_city,
            speaker_to_gender,
        )
        if gender in pairs_per_gender:
            records_by_group[gender, age_group, city].append(record)
            available_counts[gender, age_group] += 1

    selected_age_counts = get_age_counts(
        target_age_counts,
        available_counts,
        pairs_per_gender,
    )
    selected_records = []

    for (gender, age_group), target_count in selected_age_counts.items():
        if not target_count:
            continue

        city_records = {
            city: records_by_group[gender, age_group, city]
            for group_gender, group_age, city in records_by_group
            if group_gender == gender and group_age == age_group
        }
        city_counts = get_city_counts(
            target_count,
            {city: len(city_records[city]) for city in city_records},
        )

        for city, count in city_counts.items():
            group_records = city_records[city]
            random.Random(
                f"{SEED}:{dialect}:{gender}:{age_group}:{city}"
            ).shuffle(group_records)
            selected_records.extend(group_records[:count])

    random.Random(f"{SEED}:{dialect}").shuffle(selected_records)

    return selected_records


def validate_audio_paths(records, dataset_root):
    audio_paths = set()

    for record in records:
        for key in ("dialect_audio", "mandarin_audio"):
            audio_path = record[key]
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            try:
                audio_path.relative_to(dataset_root)
            except ValueError as error:
                raise ValueError(
                    f"Audio file is outside the KeSpeech directory: {audio_path}"
                ) from error
            audio_paths.add(audio_path)

    return audio_paths


def copy_subset(selected_by_dialect, dataset_root, output_dir):
    records = [
        record
        for dialect_records in selected_by_dialect.values()
        for record in dialect_records
    ]
    audio_paths = validate_audio_paths(records, dataset_root)
    output_dir.mkdir(parents=True)

    for audio_path in sorted(audio_paths):
        destination = output_dir / audio_path.relative_to(dataset_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audio_path, destination)

    pairs_dir = output_dir / "parallel_pairs_by_dialect"
    pairs_dir.mkdir()

    for dialect, records in sorted(selected_by_dialect.items()):
        with (pairs_dir / f"{dialect}.jsonl").open("w", encoding="utf-8") as output:
            for record in records:
                copied_record = dict(record)
                copied_record["dialect_audio"] = str(
                    record["dialect_audio"].relative_to(dataset_root)
                )
                copied_record["mandarin_audio"] = str(
                    record["mandarin_audio"].relative_to(dataset_root)
                )
                output.write(json.dumps(copied_record, ensure_ascii=False) + "\n")

    return len(audio_paths)


def get_city_counts_by_dialect(selected_by_dialect, speaker_to_age, speaker_to_city, speaker_to_gender):
    counts_by_dialect = {}

    for dialect in DIALECTS:
        counts = Counter()
        for record in selected_by_dialect[dialect]:
            _, _, city = get_record_info(
                record,
                speaker_to_age,
                speaker_to_city,
                speaker_to_gender,
            )
            counts[city] += 1
        counts_by_dialect[dialect] = dict(counts)

    return counts_by_dialect


def get_repeated_speaker_statistics(selected_by_dialect):
    pairs_by_speaker = Counter(
        str(record["speaker_id"])
        for records in selected_by_dialect.values()
        for record in records
    )
    pair_counts = list(pairs_by_speaker.values())

    return {
        "mean_pairs_per_speaker": sum(pair_counts) / len(pair_counts),
        "median_pairs_per_speaker": median(pair_counts),
        "speakers_with_multiple_pairs_percentage": (
            100 * sum(count > 1 for count in pair_counts) / len(pair_counts)
        ),
        "maximum_pairs_per_speaker": max(pair_counts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("KeSpeech_subset"))
    args = parser.parse_args()

    metadata_dir = args.metadata_dir.resolve()
    dataset_root = metadata_dir.parent
    audio_root = dataset_root / "Audio"

    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")

    speaker_to_age = read_mapping(metadata_dir / "spk2age")
    speaker_to_city = read_mapping(metadata_dir / "spk2city")
    speaker_to_gender = read_mapping(metadata_dir / "spk2gender")
    pairs = filter_valid_pairs(build_pairs(metadata_dir), audio_root)
    records_by_dialect = defaultdict(list)

    for record in pairs:
        records_by_dialect[record["subdialect"]].append(record)

    missing_dialects = set(DIALECTS) - set(records_by_dialect)
    if missing_dialects:
        raise ValueError(f"No valid pairs for: {', '.join(sorted(missing_dialects))}")

    reference_records = get_reference_records(
        records_by_dialect["Beijing"],
        speaker_to_age,
        speaker_to_city,
        speaker_to_gender,
    )
    pairs_per_gender = Counter(
        get_record_info(
            record,
            speaker_to_age,
            speaker_to_city,
            speaker_to_gender,
        )[0]
        for record in reference_records
    )
    target_age_counts = get_target_age_counts(
        reference_records,
        speaker_to_age,
        speaker_to_city,
        speaker_to_gender,
    )
    selected_by_dialect = {"Beijing": reference_records}

    for dialect in DIALECTS:
        if dialect != "Beijing":
            selected_by_dialect[dialect] = select_records(
                records_by_dialect[dialect],
                target_age_counts,
                pairs_per_gender,
                speaker_to_age,
                speaker_to_city,
                speaker_to_gender,
                dialect,
            )

    audio_count = copy_subset(selected_by_dialect, dataset_root, args.output_dir)
    summary = {
        "seed": SEED,
        "pairs_per_dialect": {
            dialect: len(records)
            for dialect, records in selected_by_dialect.items()
        },
        "pairs_per_gender": dict(pairs_per_gender),
        "pairs_per_gender_and_age": {
            f"{gender}_{age_group}": target_age_counts[gender, age_group]
            for gender in ("Female", "Male")
            for age_group in AGE_GROUPS
        },
        "pairs_per_city": get_city_counts_by_dialect(
            selected_by_dialect,
            speaker_to_age,
            speaker_to_city,
            speaker_to_gender,
        ),
        "repeated_speaker_statistics": get_repeated_speaker_statistics(
            selected_by_dialect
        ),
        "unique_audio_files": audio_count,
    }

    with (args.output_dir / "selection_summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)

    for dialect in DIALECTS:
        print(f"{dialect}: {len(selected_by_dialect[dialect])} pairs")
    print(f"Unique audio files: {audio_count}")


if __name__ == "__main__":
    main()
