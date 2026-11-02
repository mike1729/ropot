"""Analyze qualified APPS test tasks without executing their supplied programs.

Read manifest-listed artifacts individually; retain task-level statistical weight.
Qwen counts cover raw statements and the three passing sources, without a chat
template, added special tokens, padding or truncation. They do not certify full
monitor examples fit 2,048 tokens. Stored cases are measured in characters/bytes.
Recorded timings describe local qualification, not future Linux performance.

Publish tasks.csv and summary.json together. Inputs must remain stable and only
one publisher may target a destination. No cutoff or cohort is chosen here.
"""

from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from urllib.parse import urlsplit

from .. import source_hashes
from ..artifacts import digest, publication_directory, sha256, write_json
from ..paths import POOL_ANALYSIS as OUTPUT, QUALIFIED_POOL
from .qualified_tasks import iter_qualified_tasks


DEFAULT_TOKENIZER = Path("/Users/michal/Workspace/models/mlx-community/Qwen3.5-9B-8bit/tokenizer.json")
METRICS = (
    "statement_tokens", "reference_tokens_min", "reference_tokens_median", "reference_tokens_max",
    "raw_references", "distinct_references", "reference_attempts", "case_attempts",
    "cases", "visible_cases", "nonvisible_cases", "question_characters", "question_bytes",
    "input_max_characters", "input_max_bytes", "input_total_characters", "input_total_bytes",
    "output_max_characters", "output_max_bytes", "output_total_characters", "output_total_bytes",
    "passing_case_max_seconds",
)


def load_tokenizer(path):
    """Load a local tokenizer only; download no files and load no model weights."""
    try:
        from tokenizers import Tokenizer
    except ModuleNotFoundError as error:
        raise RuntimeError("Install requirements-analysis.txt to use analyze") from error
    encoder = Tokenizer.from_file(str(path))
    encoder.no_padding()
    encoder.no_truncation()
    return encoder


def summarize(values):
    """Unweighted median; nearest-rank p90/p95/p99: sorted[ceil(p*n)-1]."""
    values = sorted(values)
    if not values:
        raise ValueError("Cannot summarize an empty population")
    return {"n": len(values), "min": values[0], "median": statistics.median(values),
            **{f"p{p}": values[math.ceil(p * len(values) / 100) - 1] for p in (90, 95, 99)},
            "max": values[-1]}


def measure_task(task, encoder):
    """Measure an already-qualified task; qualification owns evidence validation."""
    cases, sources = task["cases"], task["solutions"]
    passing, candidates = task["passing_reference_indices"], task["reference_candidates"]
    attempts = task["verification_attempts"]
    visible = sum(c["visible"] for c in cases)
    count = lambda text: len(encoder.encode(text, add_special_tokens=False).ids)
    lengths = [count(sources[i]) for i in passing]
    row = {"split": task["split"], "id": task["id"], "difficulty": task["difficulty"],
           "host": urlsplit(task["url"]).hostname or "", "statement_tokens": count(task["question"])}
    for number, (index, tokens) in enumerate(zip(passing, lengths), 1):
        row.update({f"reference_{number}_index": index, f"reference_{number}_tokens": tokens})
    row.update({"reference_tokens_min": min(lengths), "reference_tokens_median": statistics.median(lengths),
                "reference_tokens_max": max(lengths), "raw_references": len(sources),
                "distinct_references": len(candidates), "reference_attempts": len(attempts),
                "case_attempts": sum(len(a["cases"]) for a in attempts), "cases": len(cases),
                "visible_cases": visible, "nonvisible_cases": len(cases) - visible,
                "question_characters": len(task["question"]),
                "question_bytes": len(task["question"].encode("utf-8")),
                "passing_case_max_seconds": max(a["max_case_seconds"] for a in attempts if a["passed"])})
    for field in ("input", "output"):
        for unit, size in (("characters", len), ("bytes", lambda s: len(s.encode("utf-8")))):
            sizes = [size(c[field]) for c in cases]
            row.update({f"{field}_max_{unit}": max(sizes), f"{field}_total_{unit}": sum(sizes)})
    return row


def analyze(input_dir=QUALIFIED_POOL, output_dir=OUTPUT, tokenizer=DEFAULT_TOKENIZER):
    """Publish test-only task measurements, aggregates and hashes; return summary.

    Validate every listed identity/file, including excluded train tasks. Fingerprint
    selected bytes as read, without comparing historical pipeline hashes to today's
    sources. This consumes existing qualification evidence, not a requalification.
    """
    input_dir, output_dir, tokenizer = Path(input_dir), Path(output_dir), Path(tokenizer)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    if output_dir.resolve().is_relative_to(input_dir.resolve()):
        raise ValueError("Analysis output must be outside the qualification pool")
    manifest_bytes = (input_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    encoder = load_tokenizer(tokenizer)
    from tokenizers import __version__ as tokenizer_version
    rows, task_hashes = [], {}
    for relative, raw, task in iter_qualified_tasks(input_dir, manifest):
        if task["split"] == "train":
            continue
        if task["difficulty"] not in ("introductory", "interview"):
            raise ValueError(f"Invalid difficulty: {relative}")
        try:
            rows.append(measure_task(task, encoder))
        except Exception as error:
            error.add_note(f"While analyzing {relative}")
            raise
        task_hashes[relative] = sha256(raw)
    if not rows:
        raise ValueError("No qualified test tasks")
    excluded_train = manifest["accepted_counts"].get("train", 0)
    rows.sort(key=lambda row: row["id"])
    distributions = lambda population: {metric: summarize([r[metric] for r in population]) for metric in METRICS}
    difficulties = sorted({r["difficulty"] for r in rows})
    summary = {
        "schema_version": 1, "selection": {"split": "test", "excluded_train_tasks": excluded_train},
        "counts": {"tasks": len(rows), "passing_references": 3 * len(rows),
                   "difficulty": dict(sorted(Counter(r["difficulty"] for r in rows).items())),
                   "host": dict(sorted(Counter(r["host"] for r in rows).items())),
                   **{key: sum(r[key] for r in rows) for key in ("cases", "reference_attempts", "case_attempts")}},
        "distributions": {"overall": distributions(rows), "by_difficulty": {
            difficulty: distributions([r for r in rows if r["difficulty"] == difficulty]) for difficulty in difficulties}},
        "outliers": {metric: [{"split": r["split"], "id": r["id"], "value": r[metric]}
                              for r in sorted(rows, key=lambda r: (-r[metric], r["id"]))[:5]]
                     for metric in ("statement_tokens", "reference_tokens_max", "input_max_bytes", "passing_case_max_seconds")},
        "interpretation": {"unit": "one equally weighted task", "percentiles": "nearest rank; median uses middle value(s)",
                           "tokens": "raw content only; no template, target, padding, truncation or added special tokens",
                           "references": "three observed passes; untried references have unknown pass status",
                           "timing": "recorded local qualification observations; not Linux performance"},
        "provenance": {"input_manifest_sha256": sha256(manifest_bytes),
                       "input_dir": str(input_dir.resolve()), "task_sha256": dict(sorted(task_hashes.items())),
                       "tokenizer": {"path": str(tokenizer.resolve()), "sha256": digest(tokenizer), "version": tokenizer_version},
                       "python": sys.version, "qualification_runtime": manifest["runtime"], "workers": manifest["workers"],
                       "source_sha256": source_hashes()},
    }
    with publication_directory(output_dir) as staging:
        with (staging / "tasks.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        summary["provenance"]["tasks_csv_sha256"] = digest(staging / "tasks.csv")
        write_json(staging / "summary.json", summary)
    print(f"APPS test: {len(rows)} tasks; {excluded_train} train tasks excluded")
    print("Difficulty:", summary["counts"]["difficulty"], "Hosts:", summary["counts"]["host"])
    print(f"{'Task metric':28} {'median':>10} {'p90':>10} {'p95':>10} {'p99':>10} {'max':>10}")
    for metric in METRICS:
        stats = summary["distributions"]["overall"][metric]
        print(f"{metric:28} " + " ".join(f"{stats[k]:10.4g}" for k in ("median", "p90", "p95", "p99", "max")))
    print("Content lengths only; full 2,048-token monitor fit is not assessed.")
    return summary
