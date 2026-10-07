#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

DIMENSIONS = (
    ('Competence', 'incompetent', 'competent'),
    ('Conscientiousness', 'careless', 'conscientious'),
    ('Diligence', 'lazy', 'hardworking'),
    ('Educational attainment', 'uneducated', 'educated'),
    ('Friendliness', 'unfriendly', 'friendly'),
    ('Open-mindedness', 'closed-minded', 'open-minded'),
    ('Rural/urban identity', 'rural', 'urban'),
    ('Temperament', 'temperamental', 'calm'),
    ('Trustworthiness', 'untrustworthy', 'trustworthy'),
    ('Warmth', 'cold', 'warm'),
)
BY_PAIR = {frozenset((neg, pos)): dim for dim, neg, pos in DIMENSIONS}
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
TOKEN = re.compile(r"[a-z]+(?:[-'][a-z]+)*")


def read_rows(path):
    with path.open(encoding='utf-8-sig') as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f'{path}: expected JSON objects')
    return rows


def load_groups(input_dir):
    groups = {dim: {'negative': [], 'positive': []} for dim, _, _ in DIMENSIONS}
    sources = []

    def add(path, side=None, expected_dim=None):
        sources.append(str(path))
        for row in read_rows(path):
            pair = frozenset(w.strip().lower() for w in row.get('adjective_pair', '').split('/'))
            dim = BY_PAIR.get(pair)
            if dim is None or (expected_dim and dim != expected_dim):
                raise ValueError(f'{path}: unrecognized/mismatched adjective_pair')
            condition = side or {'unfavorable': 'negative', 'favorable': 'positive'}.get(row.get('polarity'))
            if condition is None:
                raise ValueError(f'{path}: combined input needs favorable/unfavorable polarity')
            polarity = 'unfavorable' if condition == 'negative' else 'favorable'
            if row.get('polarity', polarity) != polarity:
                raise ValueError(f'{path}: incorrect polarity for {condition} file')
            groups[dim][condition].append(row)

    if not input_dir.is_dir():
        raise ValueError(f'Input directory does not exist: {input_dir}')
    for dim, neg, pos in DIMENSIONS:
        folders = [input_dir / name for name in (f'{neg}-{pos}', f'{pos}-{neg}') if (input_dir / name).is_dir()]
        if len(folders) != 1:
            raise ValueError(f'{dim}: expected exactly one adjective folder in {input_dir}')
        for side in ('negative', 'positive'):
            add(folders[0] / f'justifications_{side}.jsonl', side, dim)
    return groups, sources


def texts(rows, field):
    keys = ('normal_justification', 'reverse_justification') if field == 'both' else (field,)
    result = []
    for row in rows:
        for key in keys:
            value = row.get(key)
            if value is not None and not isinstance(value, str):
                raise ValueError(f'{key} must contain text')
            if value and value.strip():
                result.append(value)
    return result


def ngrams(text, size, excluded):
    text = unicodedata.normalize('NFKC', text).lower().replace('’', "'")
    for segment in re.split(r"[^a-z \t\-']+", text):
        tokens = TOKEN.findall(segment)
        for start in range(len(tokens) - size + 1):
            window = tokens[start:start + size]
            if any(word in excluded for word in window):
                continue
            if size > 1 and (window[0] in STOPWORDS or window[-1] in STOPWORDS):
                continue
            if any(len(word) > 1 and word not in STOPWORDS for word in window):
                yield ' '.join(window)


def score_ngrams(negative, positive, size, excluded, min_docs, prior_scale):
    counts, docs = [], []
    for corpus in (negative, positive):
        count, freq = Counter(), Counter()
        for text in corpus:
            grams = list(ngrams(text, size, excluded))
            count.update(grams)
            freq.update(set(grams))
        counts.append(count)
        docs.append(freq)
    cn, cp = counts
    dn, dp = docs
    vocab = sorted(w for w in cn.keys() | cp.keys() if dn[w] + dp[w] >= min_docs)
    if not negative or not positive:
        return [], 'missing_justifications_in_one_or_both_conditions'
    if len(vocab) < 2:
        return [], 'fewer_than_two_eligible_terms'
    # Original weighted log-odds estimator: pooled informative Dirichlet prior.
    alpha = {w: prior_scale * (cn[w] + cp[w]) for w in vocab}
    a0 = sum(alpha.values())
    nn, np = sum(cn[w] for w in vocab), sum(cp[w] for w in vocab)
    result = []
    for word in vocab:
        an, ap = cn[word] + alpha[word], cp[word] + alpha[word]
        bn, bp = nn + a0 - an, np + a0 - ap
        if min(an, ap, bn, bp) <= 0:
            continue
        delta = math.log(an / bn) - math.log(ap / bp)
        z = delta / math.sqrt(1 / an + 1 / ap)
        result.append({'term': word, 'z': z, 'negative_count': cn[word], 'positive_count': cp[word],
                       'negative_doc_pct': 100 * dn[word] / len(negative),
                       'positive_doc_pct': 100 * dp[word] / len(positive)})
    return result, 'ok' if result else 'undefined_odds'


def latex(value):
    escape = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
              '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(escape.get(c, c) for c in str(value))


def colored_terms(items, command, top_n):
    if not items:
        return '--'
    pieces = []
    rank = 0
    previous = None
    for index, item in enumerate(items):
        magnitude = abs(item['z'])
        if magnitude != previous:
            rank = index
        intensity = round(50 - 42 * rank / max(1, top_n - 1))
        pieces.append(
            r'\colorbox{' + command + '!' + str(intensity)
            + r'}{\textcolor{black}{' + latex(item['term']) + '}}'
        )
        previous = magnitude
    return ', '.join(pieces)


def write_outputs(output, scores, top_n, metadata):
    output.mkdir(parents=True, exist_ok=True)
    fields = ['dimension', 'adjective_pair', 'ngram_size', 'term', 'z', 'negative_count', 'positive_count',
              'negative_doc_pct', 'positive_doc_pct']
    with (output / 'scores.csv').open('w', encoding='utf-8', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scores)
    columns = ['Dimension', 'Unigram', 'Bigram', 'Trigram']
    note = ('Positive z favors unfavorable dialect explanations; negative z favors favorable Standard Mandarin explanations. '
            'Terms in each cell are ordered by decreasing absolute z within the indicated sign. '
            'Darker shading indicates a higher rank, not the magnitude of z; tied scores share a shade. '
            '-- denotes no eligible result (see metadata.json). Scores are descriptive, not significance tests.')
    for side in ('negative', 'positive'):
        command = 'unfavorable' if side == 'negative' else 'favorable'
        if side == 'negative':
            caption = (
                'Qwen3-Omni n-grams associated with unfavorable regional '
                'justifications ($z>0$) in stable cases where Standard Mandarin '
                'receives the favorable adjective. Each cell lists up to ten '
                'terms by decreasing $|z|$, pooling all eight regional varieties '
                'and both prompt orders. Darker orange indicates higher rank '
                'within a cell.'
            )
        else:
            caption = (
                'Qwen3-Omni n-grams associated with favorable Standard Mandarin '
                'justifications ($z<0$) for the same cases as '
                'Table~\\ref{tab:ngrams-negative}. Each cell lists up to ten '
                'terms by decreasing $|z|$. Darker favorable indicates higher '
                'rank within a cell.'
            )
        rows, latex_rows = [], []
        for dim, neg, pos in DIMENSIONS:
            ranked = {}
            for size in (1, 2, 3):
                selected = [r for r in scores if r['dimension'] == dim and r['ngram_size'] == size
                            and (r['z'] > 0 if side == 'negative' else r['z'] < 0)]
                ranked[size] = sorted(selected, key=lambda r: (-abs(r['z']), r['term']))[:top_n]
            rows.append([dim] + [', '.join(item['term'] for item in ranked[size]) or '--' for size in (1, 2, 3)])
            latex_rows.append([latex(dim)] + [colored_terms(ranked[size], command, top_n) for size in (1, 2, 3)])
        base = output / f'comparison_{side}'
        with base.with_suffix('.csv').open('w', encoding='utf-8', newline='') as target:
            writer = csv.writer(target)
            writer.writerow(columns)
            writer.writerows(rows)
        md = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * 4) + ' |']
        md.extend('| ' + ' | '.join(row) + ' |' for row in rows)
        base.with_suffix('.md').write_text(note + '\n\n' + '\n'.join(md) + '\n', encoding='utf-8')
        tex = [r'\begin{table*}[t]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{2pt}',
               r'\setlength{\fboxsep}{0.5pt}',
               r'\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.16\textwidth}*{3}{>{\raggedright\arraybackslash}X}@{}}',
               r'\toprule', ' & '.join(map(latex, columns)) + r' \\', r'\midrule']
        for index, row in enumerate(latex_rows):
            if index:
                tex.append(r'\midrule')
            tex.append(' & '.join(row) + r' \\')
        tex.extend([r'\bottomrule', r'\end{tabularx}',
                    r'\caption{' + caption + '}',
                    r'\label{tab:ngrams-' + side + '}', r'\end{table*}'])
        base.with_suffix('.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--field', default='both', choices=['normal_justification', 'reverse_justification', 'both'])
    parser.add_argument('--min-docs', type=int, default=5, help='Minimum combined response-document frequency')
    parser.add_argument('--prior-scale', type=float, default=0.1)
    args = parser.parse_args()
    if args.n < 1 or args.min_docs < 1 or not math.isfinite(args.prior_scale) or args.prior_scale <= 0:
        parser.error('n, min-docs and prior-scale must be positive and finite')
    try:
        groups, sources = load_groups(args.input_dir)
        scores, diagnostics = [], {}
        for dim, neg, pos in DIMENSIONS:
            group = groups[dim]
            negative, positive = texts(group['negative'], args.field), texts(group['positive'], args.field)
            diagnostics[dim] = {'negative_records': len(group['negative']), 'positive_records': len(group['positive']),
                                'negative_responses': len(negative), 'positive_responses': len(positive), 'ngram_status': {}}
            for size in (1, 2, 3):
                result, status = score_ngrams(negative, positive, size, {neg, pos}, args.min_docs, args.prior_scale)
                diagnostics[dim]['ngram_status'][size] = status
                scores.extend({'dimension': dim, 'adjective_pair': f'{neg} / {pos}', 'ngram_size': size, **r} for r in result)
        write_outputs(args.output_dir, scores, args.n,
                      {'sources': sources, 'field': args.field, 'min_docs': args.min_docs, 'prior_scale': args.prior_scale,
                       'top_n': args.n, 'dimensions': diagnostics,
                       'z_direction': 'unfavorable dialect minus favorable standard',
                       'inference': 'Descriptive z scores; paired/repeated explanations are dependent; no p-values.',
                       'tokenization': 'Contiguous English n-grams, punctuation barriers, target adjectives excluded; bigrams/trigrams cannot start or end with stopwords, but internal stopwords are allowed.'})
    except (OSError, ValueError) as error:
        parser.exit(1, f'Error: {error}\n')
    print(f'Wrote two four-column comparison tables (CSV, Markdown, LaTeX), scores.csv and metadata.json to {args.output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
