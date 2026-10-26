"""Deterministic, byte-bounded display of the verifier's public projection."""
import json

MARKER = "[truncated]"
FIELDS = ("input", "answer", "stdout", "stderr", "diagnostic")
LONGEST_SUMMARY = "status=failed hidden=not_run\n"


def validate_limits(field_bytes, total_bytes):
    if (type(field_bytes) is not int or field_bytes < len(MARKER)
            or type(total_bytes) is not int or total_bytes < len(LONGEST_SUMMARY + MARKER)):
        raise ValueError("Feedback caps must accommodate the summary and truncation marker")


def clip(text, cap):
    if type(cap) is not int or cap < len(MARKER):
        raise ValueError("Byte cap must accommodate the truncation marker")
    raw = text.encode("utf-8")
    return text if len(raw) <= cap else raw[:cap-len(MARKER)].decode("utf-8", errors="ignore") + MARKER


def render(feedback, field_bytes, total_bytes):
    """Bound public display text, preserving statuses/nulls and private evidence.

    A trailing marker denotes omitted visible detail/entries. Cache each case's
    JSON bytes and update the total while fitting; encode a case at most twice.
    The final cut is display text, not necessarily valid JSON.
    """
    validate_limits(field_bytes, total_bytes)
    summary = f"status={feedback['status']} hidden={feedback['hidden']}\n"
    cases, capped, stripped, omitted = [], 0, 0, 0
    for original in feedback["visible"]:
        case = {k: original[k] for k in ("status", "returncode", *FIELDS) if k in original}
        for key in FIELDS:
            if isinstance(case.get(key), str):
                text = clip(case[key], field_bytes)
                capped += text != case[key]
                case[key] = text
        cases.append(case)
    cases.sort(key=lambda c: c["status"] == "passed")

    def encode(case):
        return json.dumps(case, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    encoded = [encode(case) for case in cases]
    size = 2 + max(0, len(encoded) - 1) + sum(map(len, encoded))
    suffix = ("\n" + MARKER).encode("utf-8")
    available = total_bytes - len(summary.encode("utf-8"))

    def oversized():
        return size + (len(suffix) if stripped or omitted else 0) > available

    for index in range(len(cases) - 1, -1, -1):
        if not oversized():
            break
        case = cases[index]
        keys = [key for key in FIELDS if isinstance(case.get(key), str)]
        if case["status"] == "passed" and keys:
            for key in keys:
                del case[key]
            compact = encode(case)
            size += len(compact) - len(encoded[index])
            encoded[index] = compact
            stripped += 1
    while cases and cases[-1]["status"] == "passed" and oversized():
        size -= len(encoded.pop()) + (1 if len(cases) > 1 else 0)
        cases.pop()
        omitted += 1
    text = (b"[" + b",".join(encoded) + b"]" + (suffix if stripped or omitted else b"")).decode("utf-8")
    bounded = clip(text, available)
    return dict(text=summary + bounded, fields_truncated=capped, passing_details_removed=stripped,
                visible_omitted=omitted, text_truncated=bounded != text)
