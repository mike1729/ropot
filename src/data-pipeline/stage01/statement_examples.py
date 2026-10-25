"""Map supported English statement examples onto statically screened APPS cases."""

import re

from .screen import Rejected, integer_answer


SECTION = re.compile(r"^-----([^\r\n]+?)-----([^\r\n]*)\r?$", re.MULTILINE)
CF_MARKER = re.compile(r"^[ \t]*(Input|Output)[ \t]*\r?$", re.MULTILINE)
SAMPLE = re.compile(r"Sample (Input|Output)(?: ([1-9][0-9]*))?")
PARAGRAPH_BREAK = re.compile(r"\r?\n[ \t]*\r?\n")
EXAMPLE_LABEL = re.compile(
    r"^[ \t]*(?:Sample[ \t]+)?(?:Input|Output)"
    r"(?:[ \t]+(?:examples?[ \t]+)?#?[0-9]+)?[ \t]*(?::[^\r\n]*)?[ \t]*\r?$",
    re.MULTILINE | re.IGNORECASE,
)


def example_answer(output):
    try:
        return integer_answer(output)
    except Rejected:
        raise Rejected("invalid_example_output") from None


def parse_statement_examples(question):
    """Return (format, input/output pairs); reject incomplete or mixed formats.

    Codeforces uses one Example(s) section with alternating Input/Output lines.
    AtCoder uses one unnumbered pair or pairs with unique matching numbers.
    Its output is the first paragraph; prose after a blank line is explanation.
    A further numeric paragraph is ambiguous and is rejected. Sample input/output
    slices retain their original whitespace; explanation remains in the question.
    """
    headers = list(SECTION.finditer(question))
    if EXAMPLE_LABEL.search(question[:headers[0].start()] if headers else question):
        raise Rejected("unsupported_example_format")
    relevant = []
    for i, header in enumerate(headers):
        name = header[1]
        end = headers[i + 1].start() if i + 1 < len(headers) else len(question)
        body = question[header.end():end]
        if re.search(r"sample|example", name, re.I):
            if header[2].strip() or (name not in ("Example", "Examples") and not SAMPLE.fullmatch(name)):
                raise Rejected("unsupported_example_format")
            relevant.append((name, body))
        elif EXAMPLE_LABEL.search(body) or (
            re.match(r"(?:Inputs?|Outputs?)\b", name, re.I)
            and (relevant or name not in ("Input", "Inputs", "Output", "Outputs"))
        ):
            raise Rejected("unsupported_example_format")
    if not relevant:
        raise Rejected("missing_examples")
    codeforces = [(name, body) for name, body in relevant if name in ("Example", "Examples")]
    if codeforces:
        if len(relevant) != 1:
            raise Rejected("ambiguous_examples")
        body = codeforces[0][1]
        markers = list(CF_MARKER.finditer(body))
        if not markers:
            raise Rejected("unsupported_example_format")
        if body[:markers[0].start()].strip() or len(markers) % 2:
            raise Rejected("ambiguous_examples")
        pairs = []
        for i in range(0, len(markers), 2):
            first, second = markers[i:i + 2]
            if (first[1], second[1]) != ("Input", "Output"):
                raise Rejected("ambiguous_examples")
            end = markers[i + 2].start() if i + 2 < len(markers) else len(body)
            pairs.append((body[first.end():second.start()], body[second.end():end]))
        return "codeforces", pairs

    if len(relevant) % 2:
        raise Rejected("ambiguous_examples")
    pairs, labels = [], set()
    for i in range(0, len(relevant), 2):
        input_name, stdin = relevant[i]
        output_name, stdout = relevant[i + 1]
        first, second = SAMPLE.fullmatch(input_name), SAMPLE.fullmatch(output_name)
        if (first[1], second[1]) != ("Input", "Output") or first[2] != second[2]:
            raise Rejected("ambiguous_examples")
        label = first[2]
        if label in labels or (labels and (label is None or None in labels)):
            raise Rejected("ambiguous_examples")
        labels.add(label)
        separator = PARAGRAPH_BREAK.search(stdout, len(stdout) - len(stdout.lstrip()))
        explanation = stdout[separator.end():] if separator else ""
        for paragraph in PARAGRAPH_BREAK.split(explanation):
            if EXAMPLE_LABEL.search(paragraph):
                raise Rejected("unsupported_example_format")
            if re.fullmatch(r"[\s+\-.]*[0-9][\s+\-0-9.]*", paragraph):
                raise Rejected("ambiguous_examples")
        pairs.append((stdin, stdout[:separator.end()] if separator else stdout))
    return "atcoder", pairs


def map_statement_examples_to_cases(task):
    """Enrich a screen_task result without mutating it or qualifying the task.

    Match inputs by whitespace tokens and answers by exact canonical spelling.
    Every equivalent stored input becomes visible, including duplicates. Reject
    contradictory stored answers even for inputs absent from the statement.
    Original case content and indices are retained for later reference execution.
    Unchanged metadata and solution lists are shared with the input candidate.
    """
    stored = {}
    for case in task["cases"]:
        key = " ".join(case["input"].split())
        matches = stored.setdefault(key, [])
        if matches and matches[0]["answer"] != case["answer"]:
            raise Rejected("conflicting_case_answers")
        matches.append(case)

    format_name, pairs = parse_statement_examples(task["question"])
    mappings, visible = [], set()
    for index, (stdin, stdout) in enumerate(pairs):
        if not stdin.strip():
            raise Rejected("ambiguous_examples")
        answer = example_answer(stdout)
        matches = stored.get(" ".join(stdin.split()))
        if matches is None:
            raise Rejected("example_input_not_stored")
        if matches[0]["answer"] != answer:
            raise Rejected("example_answer_mismatch")
        indices = [case["index"] for case in matches]
        visible.update(indices)
        mappings.append({"index": index, "input": stdin, "output": stdout,
                         "answer": answer, "case_indices": indices})

    if len(visible) == len(task["cases"]):
        raise Rejected("no_nonvisible_cases")
    return {
        **task,
        "cases": [{**case, "visible": case["index"] in visible} for case in task["cases"]],
        "example_format": format_name,
        "example_mappings": mappings,
        "visible_case_count": len(visible),
        "nonvisible_case_count": len(task["cases"]) - len(visible),
    }
