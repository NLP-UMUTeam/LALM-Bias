#!/usr/bin/env python3

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path


NEGATIVE_ADJECTIVE_PAIRS = (
    ("careless / conscientious", "careless"),
    ("closed-minded / open-minded", "closed-minded"),
    ("competent / incompetent", "incompetent"),
    ("friendly / unfriendly", "unfriendly"),
    ("hardworking / lazy", "lazy"),
    ("rural / urban", "rural"),
    ("temperamental / calm", "temperamental"),
    ("trustworthy / untrustworthy", "untrustworthy"),
    ("uneducated / educated", "uneducated"),
    ("warm / cold", "cold"),
)
UNFAVORABLE_BY_PAIR = dict(NEGATIVE_ADJECTIVE_PAIRS)
FAVORABLE_BY_PAIR = {
    pair: next(word for word in pair.split(" / ") if word != negative)
    for pair, negative in NEGATIVE_ADJECTIVE_PAIRS
}
SELECTION = "stable_standard_favorable_dialect_unfavorable"
PAIR_BY_ADJECTIVES = {
    frozenset(pair_name.split(" / ")): pair_name
    for pair_name, _ in NEGATIVE_ADJECTIVE_PAIRS
}
PAIR_ORDER = {
    pair_name: index for index, (pair_name, _) in enumerate(NEGATIVE_ADJECTIVE_PAIRS)
}
DIALECT_DISPLAY_ORDER = (
    "Beijing",
    "Ji-Lu",
    "Jiang-Huai",
    "Jiao-Liao",
    "Lan-Yin",
    "Northeastern",
    "Southwestern",
    "Zhongyuan",
)
DIALECT_ORDER = {dialect: index for index, dialect in enumerate(DIALECT_DISPLAY_ORDER)}
MODEL_DISPLAY_NAMES = {
    "Kimi-Audio-7B-Instruct": "Kimi-Audio",
    "Qwen3-Omni-30B-A3B-Instruct": "Qwen3-Omni",
    "Phi-4-multimodal-instruct": "Phi-4",
}

BARE_ANSWER_RE = re.compile(
    r"^\s*(?:[\"'`*<]\s*)?"
    r"(?P<answer>[a-z]+(?:-[a-z]+)?)"
    r"(?:\s*[\"'`*>.,!])?\s*$",
    re.IGNORECASE,
)
LABELLED_ANSWER_RE = re.compile(
    r"(?:^|\n)\s*answer\s*[:：]\s*(?:<\s*)?"
    r"(?P<answer>[a-z]+(?:-[a-z]+)?)(?:\s*>)?",
    re.IGNORECASE,
)
SPEAKER_ANSWER_RE = re.compile(
    r"\b(?:the\s+)?speaker\s+(?:is\s+)?"
    r"(?:better\s+described\s+as\s+|more\s+)?"
    r"['\"]?(?P<answer>[a-z]+(?:-[a-z]+)?)['\"]?\b",
    re.IGNORECASE,
)
REASONING_RE = re.compile(
    r"(?:^|\n)\s*reasoning\s*[:：]\s*(?P<reasoning>.+)\s*$",
    re.IGNORECASE | re.DOTALL,
)
STOPWORDS = {
    "a", "about", "after", "all", "also", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "being", "both", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "has",
    "have", "he", "her", "here", "him", "his", "how", "if", "in", "into",
    "is", "it", "its", "just", "likely", "may", "might", "more", "most",
    "much", "of", "on", "or", "other", "our", "out", "over", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "there", "these", "they", "this", "those", "to", "too", "under", "very",
    "was", "we", "were", "what", "when", "which", "while", "who", "with",
    "would", "you", "your", "answer", "reasoning", "audio", "based", "content",
    "dialect", "information", "input", "language", "option", "response", "speaker",
    "spoken", "speech", "therefore", "however", "appears", "appearing", "indicate",
    "indicates", "suggests", "standard", "regional", "mandarin", "chinese",
    "themselves", "where", "often", "typically", "commonly",
    "speaker's", "speaker’s",
    "seem", "seems", "seemed", "seeming", "suggest", "suggesting",
    "suggested", "indicating", "indicated", "implies", "imply", "implying",
    "shows", "show", "shown", "appeared", "appear", "impression",
    "describe", "describes", "described", "describing", "description",
    "choose", "chooses", "choosing", "chosen", "choice", "choices",
    "provide", "provides", "providing", "provided",
    "use", "uses", "using", "used", "giving", "given",
    "adjective", "adjectives", "pair", "pairs", "options", "alternative",
    "alternatives", "trait", "traits", "descriptor", "descriptors",
    "label", "labels", "respond", "responses",
    "question", "questions", "task", "tasks", "term", "terms",
    "judgment", "judgments", "judgement", "judgements",
    "mention", "mentions", "mentioned", "mentioning", "reference",
    "references", "refer", "refers", "referred", "referring",
    "including", "rather", "towards",
    "explanation", "explanations", "reason", "reasons",
    "explain", "explains", "explained", "explaining",
    "select", "selects", "selected", "selecting", "selection", "selections",
    "classify", "classifies", "classified", "classifying", "classification",
    "infer", "infers", "inferred", "inferring", "inference",
    "conclude", "concludes", "concluded", "concluding", "conclusion",
    "conclusions", "assess", "assesses", "assessed", "assessing", "assessment",
    "evaluate", "evaluates", "evaluated", "evaluating", "evaluation",
    "prompt", "prompts", "recording", "recordings", "clip", "clips",
    "sample", "samples", "utterance", "utterances",
    "according", "thus", "hence", "overall", "generally", "somewhat",
    "perhaps", "possibly", "probably", "certainly", "indeed",
    "additionally", "furthermore", "moreover", "nevertheless", "thereby",
    "appearing", "appearances", "implied", "suggestion", "suggestions",
    "indication", "indications", "descriptions", "responds", "responded",
    "responding", "provided", "descriptors", "speaker's", "speakers",
    "person", "people", "individual", "individuals", "someone", "others",
    "self", "context", "situation", "source", "fact", "matter",
    "possibility", "certain", "specific", "different", "relevant",
    "associated", "associate", "associates", "association", "associations",
    "align", "aligns", "aligned", "aligning", "due",
    "make", "makes", "making", "made", "take", "takes", "taking", "taken",
    "put", "puts", "putting", "allow", "allows", "allowed", "allowing",
    "require", "requires", "required", "requiring", "found", "trying",
    "discuss", "discusses", "discussed", "discussing",
    "express", "expresses", "expressed", "expressing",
    "convey", "conveys", "conveyed", "conveying",
    "carefully", "formally", "clearly", "actively", "effectively",
    "calmly", "steadily", "deliberately", "confidently", "professionally",
    "precisely", "accurately", "correctly", "properly", "fluently",
    "articulately", "conscientiously", "diligently", "thoughtfully",
    "attentively", "politely", "warmly", "coldly", "honestly", "sincerely",
    "reliably", "responsibly", "competently", "incompetently", "lazily",
    "carelessly", "emotionally", "unemotionally", "neutrally",
    "monotonously", "conversationally", "gently", "firmly", "slowly",
    "rapidly", "quietly", "loudly", "strongly", "positively", "negatively",
    "highly", "particularly", "especially", "relatively", "largely",
    "mainly", "mostly", "simply", "merely", "really", "actually",
    "essentially", "basically", "apparently", "presumably", "potentially",
    "necessarily", "explicitly", "implicitly", "directly", "indirectly",
    "consistently", "frequently", "occasionally", "usually", "normally",
    "completely", "entirely", "fully", "slightly", "quite", "somehow",
    "either", "neither", "each", "every", "another", "anything",
    "something", "whatever", "whom", "whose", "whether", "although",
    "though", "unless", "since", "then", "thereof", "whereas", "focused",
    "level", "levels", "sense", "nature", "characteristic", "characteristics",
    "approach", "approaches", "manner", "way", "ways", "quality", "qualities",
    "better", "best", "good", "well", "common", "typical", "new",
    "time", "times", "life", "own", "lot", "lots", "thing", "things",
    "key", "main", "general", "particular", "certainly", "prone",
    "importance", "important", "meaningful", "valuable", "fitting",
    "related", "relate", "relates", "relating", "regarding", "concerning",
    "demonstrate", "demonstrates", "demonstrated", "demonstrating",
    "reflect", "reflects", "reflected", "reflecting",
    "improve", "improves", "improved", "improving", "improvement",
    "ask", "asks", "asking", "asked", "offer", "offers", "offering", "offered",
    "emphasize", "emphasizes", "emphasized", "emphasizing",
    "deliver", "delivers", "delivered", "delivering",
    "discussions", "discussion", "decision", "decisions", "claim", "claims",
    "example", "examples", "instance", "instances", "aspect", "aspects",
    "factor", "factors", "element", "elements", "point", "points",
    "word", "words", "topic", "topics", "subject", "subjects",
    "statement", "statements", "phrase", "phrases", "message", "messages",
    "high", "higher", "highest", "low", "lower", "lowest", "less", "least",
    "minded", "thorough",
    "action", "actions", "field", "fields", "state", "states",
    "present", "presents", "presented", "presenting", "look", "looking",
    "have", "having", "one", "two", "between", "among", "across",
    "ensure", "ensures", "ensuring", "indicative", "major", "global",
    "suitable", "prevalent", "audience",
    "answering", "answered", "answers", "attribute", "attributes",
    "mean", "means", "meant", "seen", "see", "seeing",
    "possess", "possesses", "possessing", "role", "roles",
    "maintain", "maintains", "maintaining", "maintained",
}


@dataclass(frozen=True)
class Record:
    mandarin_answer: str | None
    dialect_answer: str | None
    mandarin_response: str
    dialect_response: str


def display_model_name(model_name):
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def filename_label(value):
    return re.sub(r"[^A-Za-z0-9.-]+", "-", value).strip("-")


def sort_pair(pair_name):
    return (PAIR_ORDER.get(pair_name, len(PAIR_ORDER)), pair_name)


def sort_dialect(dialect):
    return (DIALECT_ORDER.get(dialect, len(DIALECT_ORDER)), dialect)


def latex_escape(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def pair_from_directory_name(directory_name):
    for pair_name in PAIR_ORDER:
        if filename_label(pair_name) == directory_name:
            return pair_name
    return directory_name.replace("-", " ")


def dialect_from_filename(filename):
    stem = Path(filename).stem
    for dialect in DIALECT_ORDER:
        if filename_label(dialect) == stem:
            return dialect
    return stem


def parse_answer(response, adjectives):
    """Return one unambiguous adjective answer, or None when it is invalid."""
    if not isinstance(response, str):
        return None
    for pattern in (BARE_ANSWER_RE, LABELLED_ANSWER_RE, SPEAKER_ANSWER_RE):
        match = pattern.search(response)
        if match is None:
            continue
        answer = match.group("answer").lower()
        if re.match(r"\s*(?:or|and|/)\b", response[match.end() :], re.IGNORECASE):
            continue
        if answer in adjectives:
            return answer
    return None


def extract_reasoning(response):
    if not isinstance(response, str):
        return ""
    match = REASONING_RE.search(response)
    return match.group("reasoning").strip() if match else ""


def tokenize_reasoning(reasoning):
    tokens = re.findall(
        r"[^\W\d_]+(?:['’][^\W\d_]+)?",
        unicodedata.normalize("NFKC", reasoning).lower(),
        flags=re.UNICODE,
    )
    return [
        token
        for token in (value.strip("'’-" ) for value in tokens)
        if len(token) >= 2 and token not in STOPWORDS
    ]


def dataset_directories(input_path):
    if not input_path.is_dir():
        raise ValueError(f"Input directory does not exist: {input_path}")

    child_dirs = [path for path in input_path.iterdir() if path.is_dir()]
    if any(list(path.glob("*.jsonl")) for path in child_dirs):
        yield input_path.name, input_path
        return

    found = False
    for model_path in sorted(child_dirs):
        if any(
            list(dialect_path.glob("*.jsonl"))
            for dialect_path in model_path.iterdir()
            if dialect_path.is_dir()
        ):
            found = True
            yield model_path.name, model_path
    if not found:
        raise ValueError(
            f"{input_path} does not contain MODEL/DIALECT JSONL results."
        )


def pair_from_filename(jsonl_path, dialect):
    prefix = f"{dialect}_"
    if not jsonl_path.stem.startswith(prefix):
        return None
    labels = tuple(jsonl_path.stem[len(prefix) :].lower().split("_"))
    if len(labels) != 2:
        return None
    pair_name = PAIR_BY_ADJECTIVES.get(frozenset(labels))
    return None if pair_name is None else (pair_name, frozenset(labels))


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path}:{line_number}.")
            yield line_number, row


def sample_identifier(row, path, line_number):
    required = ("speaker_id", "mandarin_audio", "dialect_audio")
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"{path}:{line_number} is missing alignment fields: {missing}")
    return tuple(str(row[field]) for field in required)


def collect_records(input_path):
    records = {}
    observed_groups = set()
    for model_name, dataset_path in dataset_directories(input_path):
        for dialect_path in sorted(path for path in dataset_path.iterdir() if path.is_dir()):
            dialect = dialect_path.name
            for jsonl_path in sorted(dialect_path.glob("*.jsonl")):
                pair_definition = pair_from_filename(jsonl_path, dialect)
                if pair_definition is None:
                    continue
                pair_name, adjectives = pair_definition
                observed_groups.add((model_name, pair_name, dialect))
                for line_number, row in read_jsonl(jsonl_path):
                    required = ("mandarin_response", "dialect_response")
                    missing = [field for field in required if field not in row]
                    if missing:
                        raise ValueError(f"{jsonl_path}:{line_number} is missing fields: {missing}")
                    sample_id = sample_identifier(row, jsonl_path, line_number)
                    key = (model_name, pair_name, dialect, sample_id)
                    if key in records:
                        raise ValueError(f"Duplicate aligned record: {jsonl_path}:{line_number}.")
                    records[key] = Record(
                        mandarin_answer=parse_answer(row["mandarin_response"], adjectives),
                        dialect_answer=parse_answer(row["dialect_response"], adjectives),
                        mandarin_response=row["mandarin_response"],
                        dialect_response=row["dialect_response"],
                    )
    return records, observed_groups


def selected_pairs(normal_records, reverse_records):
    """Yield the identical F->U subset for token counts and raw text exports."""
    for key, normal in normal_records.items():
        reverse = reverse_records.get(key)
        if reverse is None:
            continue
        pair_name = key[1]
        favorable = FAVORABLE_BY_PAIR[pair_name]
        unfavorable = UNFAVORABLE_BY_PAIR[pair_name]
        if (
            normal.mandarin_answer == reverse.mandarin_answer == favorable
            and normal.dialect_answer == reverse.dialect_answer == unfavorable
        ):
            yield key, normal, reverse


def write_justification_files(output_dir, normal_records, reverse_records, groups=None):
    """Export positive/negative files per model and adjective pair, across dialects."""
    model_pairs = {(key[0], key[1]) for key in normal_records.keys() | reverse_records.keys()}
    if groups is not None:
        model_pairs.update((group[0], group[1]) for group in groups)
    paths = []
    targets = {}
    with ExitStack() as stack:
        for model, pair in sorted(model_pairs, key=lambda item: (item[0], sort_pair(item[1]))):
            directory = output_dir / "justifications" / filename_label(display_model_name(model)) / filename_label(pair)
            directory.mkdir(parents=True, exist_ok=True)
            for condition, suffix in (("mandarin", "positive"), ("dialect", "negative")):
                path = directory / f"justifications_{suffix}.jsonl"
                targets[(model, pair, condition)] = stack.enter_context(path.open("w", encoding="utf-8"))
                paths.append(path)
        for key, normal, reverse in selected_pairs(normal_records, reverse_records):
            model, pair, dialect, sample_id = key
            speaker_id, mandarin_audio, dialect_audio = sample_id
            for condition, polarity in (
                ("mandarin", "favorable"),
                ("dialect", "unfavorable"),
            ):
                target = targets[(model, pair, condition)]
                normal_response = getattr(normal, f"{condition}_response")
                reverse_response = getattr(reverse, f"{condition}_response")
                normal_reasoning = extract_reasoning(normal_response)
                reverse_reasoning = extract_reasoning(reverse_response)
                row = {
                    "selection": SELECTION,
                    "model": display_model_name(model), "model_id": model,
                    "adjective_pair": pair, "dialect": dialect,
                    "speaker_id": speaker_id,
                    "mandarin_audio": mandarin_audio, "dialect_audio": dialect_audio,
                    "condition": condition, "polarity": polarity,
                    "normal_answer": getattr(normal, f"{condition}_answer"),
                    "reverse_answer": getattr(reverse, f"{condition}_answer"),
                    "normal_justification": normal_reasoning,
                    "reverse_justification": reverse_reasoning,
                    "normal_reasoning_present": bool(normal_reasoning),
                    "reverse_reasoning_present": bool(reverse_reasoning),
                    "normal_response": normal_response, "reverse_response": reverse_response,
                }
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
    return paths


def collect_stable_hotwords(normal_records, reverse_records, groups, min_frequency, top_n):
    counters = defaultdict(Counter)
    total_tokens = Counter()
    stable_examples = Counter()
    reasoning_responses = Counter()

    for key, normal, reverse in selected_pairs(normal_records, reverse_records):
        model_name, pair_name, dialect, _ = key
        group = (model_name, pair_name, dialect)
        for condition, responses in (
            ("mandarin", (normal.mandarin_response, reverse.mandarin_response)),
            ("dialect", (normal.dialect_response, reverse.dialect_response)),
        ):
            condition_group = (*group, condition)
            stable_examples[condition_group] += 1
            for response in responses:
                reasoning = extract_reasoning(response)
                if not reasoning:
                    continue
                tokens = tokenize_reasoning(reasoning)
                counters[condition_group].update(tokens)
                total_tokens[condition_group] += len(tokens)
                reasoning_responses[condition_group] += 1

    long_rows = []
    condition_groups = {(*group, condition) for group in groups for condition in ("mandarin", "dialect")}
    for group in sorted(condition_groups, key=lambda item: (item[0], item[3], sort_pair(item[1]), sort_dialect(item[2]))):
        model_name, pair_name, dialect, condition = group
        ranked_tokens = [
            (word, count)
            for word, count in counters[group].most_common()
            if count >= min_frequency
        ][:top_n]
        for rank, (word, count) in enumerate(ranked_tokens, start=1):
            long_rows.append(
                {
                    "model": display_model_name(model_name),
                    "selection": SELECTION,
                    "condition": condition,
                    "adjective_pair": pair_name,
                    "dialect": dialect,
                    "rank": rank,
                    "word": word,
                    "count": count,
                    "per_1000_tokens": (
                        1000 * count / total_tokens[group] if total_tokens[group] else 0.0
                    ),
                    "total_retained_tokens": total_tokens[group],
                    "stable_examples": stable_examples[group],
                    "reasoning_responses": reasoning_responses[group],
                    "response_slots": 2 * stable_examples[group],
                }
            )
    return long_rows, counters, total_tokens, stable_examples, reasoning_responses


def write_group_files(output_dir, groups, long_rows):
    fields = [
        "model", "selection", "condition", "adjective_pair", "dialect", "rank", "word", "count",
        "per_1000_tokens", "total_retained_tokens", "stable_examples",
        "reasoning_responses", "response_slots",
    ]
    output_paths = []
    condition_groups = {(*group, condition) for group in groups for condition in ("mandarin", "dialect")}
    for model_name, pair_name, dialect, condition in sorted(
        condition_groups, key=lambda item: (display_model_name(item[0]), item[3], sort_pair(item[1]), sort_dialect(item[2]))
    ):
        output_path = (
            output_dir
            / filename_label(display_model_name(model_name))
            / condition
            / filename_label(pair_name)
            / f"{filename_label(dialect)}.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                row
                for row in long_rows
                if row["model"] == display_model_name(model_name)
                and row["condition"] == condition
                and row["adjective_pair"] == pair_name
                and row["dialect"] == dialect
            )
        output_paths.append(output_path)

    return output_paths


def read_hotwords_from_csvs(hotwords_dir):
    """Read the per-dialect CSVs without revisiting the raw JSONL results."""
    if not hotwords_dir.is_dir():
        raise ValueError(f"Hotword CSV directory does not exist: {hotwords_dir}")

    tables = defaultdict(lambda: defaultdict(dict))
    csv_paths = sorted(hotwords_dir.glob("*/*/*/*.csv"))
    if not csv_paths:
        raise ValueError("No separated condition CSVs found. Regenerate from the normal/reverse JSONL files; pooled CSVs cannot be split.")
    for csv_path in csv_paths:
        with csv_path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            required_fields = {"model", "selection", "condition", "adjective_pair", "dialect", "rank", "word"}
            if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
                raise ValueError(f"{csv_path} lacks F->U selection metadata. Regenerate from the original JSONL files.")
            rows = list(reader)
            if any(row["selection"] != SELECTION for row in rows):
                raise ValueError(f"{csv_path} contains a different selection. Regenerate F->U CSVs from JSONL.")

        model_name = rows[0]["model"] if rows else csv_path.parents[2].name
        condition = csv_path.parents[1].name
        if condition not in ("mandarin", "dialect") or any(row["condition"] != condition for row in rows):
            raise ValueError(f"Invalid or mixed condition in {csv_path}")
        pair_name = (
            rows[0]["adjective_pair"]
            if rows
            else pair_from_directory_name(csv_path.parent.name)
        )
        dialect = rows[0]["dialect"] if rows else dialect_from_filename(csv_path.name)
        rows.sort(key=lambda row: int(row["rank"]))
        tables[(model_name, condition)][pair_name][dialect] = [row["word"] for row in rows]
    return tables


def ranked_latex_word(word, rank, top_n):
    """Return plain text; rank is conveyed by the order within each cell."""
    if not 1 <= rank <= top_n:
        raise ValueError("Rank must be between 1 and top_n.")
    return latex_escape(word)


def write_latex_tables_from_csvs(hotwords_dir, table_top_n=10):
    """Create one Mandarin and one dialect LaTeX table per model."""
    tables = read_hotwords_from_csvs(hotwords_dir)
    output_paths = []
    for model_name, condition in sorted(tables, key=lambda key: (display_model_name(key[0]), key[1])):
        pairs_by_dialect = tables[(model_name, condition)]
        pairs = sorted(pairs_by_dialect, key=sort_pair)
        dialects = sorted(
            {dialect for values in pairs_by_dialect.values() for dialect in values},
            key=sort_dialect,
        )
        if not pairs or not dialects:
            continue

        column_specification = (
            r"@{}l" + "*{" + str(len(dialects))
            + r"}{>{\raggedright\arraybackslash}X}@{}"
        )
        lines = [
            r"% Requires \usepackage{tabularx,booktabs}. Tokens are ordered by rank.",
            r"\begin{table*}[!ht]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2pt}",
            r"\renewcommand{\arraystretch}{1.15}",
            rf"\begin{{tabularx}}{{\textwidth}}{{{column_specification}}}",
            r"\toprule",
            r"Adjectives & " + " & ".join(latex_escape(dialect) for dialect in dialects) + r" \\",
            r"\midrule",
        ]
        for pair_index, pair_name in enumerate(pairs):
            cells = []
            for dialect in dialects:
                words = pairs_by_dialect[pair_name].get(dialect, [])[:table_top_n]
                cells.append(
                    ", ".join(ranked_latex_word(word, rank, table_top_n)
                              for rank, word in enumerate(words, start=1))
                    if words
                    else "--"
                )
            adjectives = pair_name.split(" / ", 1)
            pair_label = (
                r"\begin{tabular}[t]{@{}l@{}}" + latex_escape(adjectives[0]) + r" /\\"
                + latex_escape(adjectives[1]) + r"\end{tabular}"
                if len(adjectives) == 2 else latex_escape(pair_name)
            )
            lines.append(pair_label + " & " + " & ".join(cells) + r" \\")
            if pair_index < len(pairs) - 1:
                lines.append(r"\midrule")
        model_label = filename_label(model_name).lower()
        speech = "Standard Mandarin" if condition == "mandarin" else "regional speech"
        matching_note = (" Columns identify the regional varieties matched to the Standard Mandarin recordings."
                         if condition == "mandarin" else "")
        lines.extend(
            (
                r"\bottomrule",
                r"\end{tabularx}",
                "\\caption{Most frequent reasoning tokens in stable adjective decisions for "
                + latex_escape(model_name)
                + " on " + speech
                + ". Only matched pairs with favorable Standard Mandarin and unfavorable regional judgments in both prompt orders are retained. Each cell pools normal and reversed explanations for this condition only. Tokens are ordered by decreasing frequency; -- denotes no qualifying tokens."
                + matching_note + "}",
                rf"\label{{tab:stable-hotwords-f-to-u-{model_label}-{condition}}}",
                r"\end{table*}",
                "",
            )
        )
        output_path = hotwords_dir / filename_label(model_name) / f"hotwords_table_{condition}.tex"
        output_path.write_text("\n".join(lines), encoding="utf-8")
        output_paths.append(output_path)
    return output_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("normal_results", type=Path)
    parser.add_argument("reverse_results", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for F->U CSVs, LaTeX, and original justifications.",
    )
    parser.add_argument("--min-frequency", type=int, default=3, help="Minimum token count (default: 3).")
    parser.add_argument("--top-n", type=int, default=10, help="Maximum ranked words per CSV (default: 10).")
    parser.add_argument("--table-top-n", type=int, default=10, help="Words per LaTeX cell (default: 10).")
    args = parser.parse_args()

    if args.min_frequency < 1 or args.top_n < 1 or args.table_top_n < 1:
        parser.error("--min-frequency, --top-n, and --table-top-n must be positive integers.")

    try:
        normal_records, normal_groups = collect_records(args.normal_results)
        reverse_records, reverse_groups = collect_records(args.reverse_results)
        groups = normal_groups | reverse_groups
        if not normal_records:
            raise ValueError("No recognised individual-adjective JSONL files were found.")
        long_rows, counters, total_tokens, stable_examples, reasoning_responses = (
            collect_stable_hotwords(
                normal_records, reverse_records, groups, args.min_frequency, args.top_n
            )
        )
        hotwords_dir = args.output_dir / "hotwords"
        output_paths = write_group_files(hotwords_dir, groups, long_rows)
        output_paths.extend(write_justification_files(args.output_dir, normal_records, reverse_records, groups))
        output_paths.extend(write_latex_tables_from_csvs(hotwords_dir, args.table_top_n))
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for output_path in output_paths:
        print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
