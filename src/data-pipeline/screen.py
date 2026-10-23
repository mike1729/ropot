"""Static APPS screening; passing this module does not qualify a task yet."""

import hashlib
import io
import json
import re
import tokenize


INTEGER = re.compile(r"-?(0|[1-9][0-9]*)", re.ASCII)
RUSSIAN_HEADER = re.compile(
    r"^[ \t-]*(?:Входные данные|Выходные данные|Примеры?|Ввод|Вывод)[ \t-]*$",
    re.MULTILINE | re.IGNORECASE,
)
NONUNIQUE = re.compile(
    r"\b(?:print|output|return)\s+(?:any\b|one\s+of\b)"
    r"|\bany\s+(?:valid|correct|suitable)\s+answer\b"
    r"|\bany\s+of\s+them\s+will\s+be\s+accepted\b",
    re.IGNORECASE,
)
OUTPUT_SECTION = re.compile(
    r"^-----Outputs?-----[ \t]*\n(.*?)(?=^-----|\Z)", re.MULTILINE | re.DOTALL
)
TEXT_OUTPUT = re.compile(
    r"\b(?:print|output|return)[ \t]+((?:[^\s.]+[ \t]+){0,3}?)(?:strings?|characters?)\b",
    re.IGNORECASE,
)
NUMERIC_TARGET = re.compile(r"\b(?:integers?|numbers?|length|count|sum|amount|size)\b", re.I)


class Rejected(ValueError):
    """A task exclusion reason; callers must catch this type specifically."""


def integer_answer(output):
    """Validate the stored token without changing its spelling or parsing an int."""
    answer = output.strip()
    if INTEGER.fullmatch(answer) is None:
        raise Rejected("noncanonical_integer_output")
    return answer


def output_matches(stdout, answer):
    """Only surrounding whitespace may differ from the stored answer token."""
    return stdout.split() == [answer]


def token_fingerprint(source):
    """Ignore comments/formatting; retain significant newlines and block structure."""
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.NL, tokenize.ENDMARKER):
            continue
        value = "" if token.type in (tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE) else token.string
        tokens.append((tokenize.tok_name[token.type], value))
    if not tokens:
        return None
    return hashlib.sha256(json.dumps(tokens, ensure_ascii=True).encode()).hexdigest()


def decode_field(row, name, expected_type):
    raw = row.get(name)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise Rejected("missing_cases" if name == "input_output" else f"missing_{name}")
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        raise Rejected(f"invalid_{name}") from None
    if not isinstance(value, expected_type):
        raise Rejected(f"invalid_{name}")
    return value


def screen_task(row, split):
    """Screen a raw APPS row and return a decoded reference-validation candidate.

    Checks performed, in order:
    1. Validate the split and integer task ID; keep introductory/interview tasks.
    2. Require a nonempty statement and exclude Russian headings, explicit
       arbitrary-answer wording and instructions to output strings/characters.
    3. Decode input_output, exclude function-call and line-list tasks, and require
       nonempty, equally sized input/output lists containing only strings.
    4. Require each stripped output to be one canonical decimal integer token,
       preserving its spelling without numeric conversion.
    5. Require a nonempty dedicated Output/Outputs section after validating cases.
    6. Decode supplied solutions, skip blank/comment-only/untokenizable entries,
       and collect all token-distinct candidates in dataset order, requiring
       at least three.

    Return original metadata/sources, cases with raw input/output and exact answer
    tokens, candidate source indices/token hashes, and unusable source indices.
    Raise ValueError for an invalid split/ID or Rejected with an exclusion reason
    when a screening check fails. Catch Rejected specifically for task exclusions.

    Statement wording checks are heuristics, not semantic proofs. Example mapping,
    the nonvisible-case gate and reference execution belong to later steps.
    """
    if split not in ("train", "test") or type(row.get("id")) is not int:
        raise ValueError("Invalid split or task ID")
    if row.get("difficulty") not in ("introductory", "interview"):
        raise Rejected("unsupported_difficulty")
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise Rejected("invalid_question")
    if RUSSIAN_HEADER.search(question):
        raise Rejected("non_english_statement")
    if NONUNIQUE.search(question):
        raise Rejected("nonunique_answer_wording")
    section = OUTPUT_SECTION.search(question)
    if section and any(
        not NUMERIC_TARGET.search(match[1])
        for match in TEXT_OUTPUT.finditer(section[1])
    ):
        raise Rejected("string_or_character_output")

    input_output = decode_field(row, "input_output", dict)
    if input_output.get("fn_name"):
        raise Rejected("function_call_task")
    inputs, outputs = input_output.get("inputs"), input_output.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise Rejected("invalid_cases")
    if not inputs or len(inputs) != len(outputs):
        raise Rejected("invalid_cases")
    if any(isinstance(case, list) for case in inputs + outputs):
        raise Rejected("unsupported_line_list_cases")
    if not all(isinstance(case, str) for case in inputs + outputs):
        raise Rejected("invalid_cases")
    cases = [
        {"index": index, "input": stdin, "output": output, "answer": integer_answer(output)}
        for index, (stdin, output) in enumerate(zip(inputs, outputs))
    ]
    if section is None or not section[1].strip():
        raise Rejected("missing_output_section")

    solutions = decode_field(row, "solutions", list)
    if not all(isinstance(source, str) for source in solutions):
        raise Rejected("invalid_solutions")
    references, seen, unusable = [], set(), []
    for index, source in enumerate(solutions):
        if not source.strip():
            unusable.append(index)
            continue
        try:
            fingerprint = token_fingerprint(source)
        except (tokenize.TokenError, SyntaxError):
            unusable.append(index)
            continue
        if fingerprint is None:
            unusable.append(index)
            continue
        if fingerprint not in seen:
            seen.add(fingerprint)
            references.append({"index": index, "token_sha256": fingerprint})
    if len(references) < 3:
        raise Rejected("insufficient_distinct_solutions")

    return {
        "split": split,
        "id": row["id"],
        "question": question,
        "difficulty": row["difficulty"],
        "url": row.get("url", ""),
        "starter_code": row.get("starter_code", ""),
        "cases": cases,
        "solutions": solutions,
        "reference_candidates": references,
        "unusable_solution_indices": unusable,
    }
