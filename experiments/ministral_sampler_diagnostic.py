"""Observe the pinned failed greedy qualification without changing its sampler."""
import base64
from contextlib import contextmanager
from hashlib import sha256
import heapq
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys

BASELINE = Path(__file__).resolve().parents[1] / "data/03_throughput/ministral-diagnostic-20261025/baseline.json"
BASELINE_SHA256 = "9414fbaf7e7ebf38ed28afa7b79891d39dea98ed2d475f80f4917428e6261cd7"
OUTPUT, ORIGINAL_SOURCE = Path("diagnostic.jsonl"), Path("diagnostic-original-worker.py")
LIMIT, CANDIDATES = 8 * 1024**2, (13207, 2951)

def load_baseline():
    raw = BASELINE.read_bytes()
    if sha256(raw).hexdigest() != BASELINE_SHA256: raise RuntimeError("Diagnostic baseline changed")
    return json.loads(raw)

def load_original(baseline):
    original = baseline["original_module"]
    path = Path(original["path"])
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != original["sha256"]: raise RuntimeError("Original worker changed before execution")
    with ORIGINAL_SOURCE.open("xb") as output:
        output.write(raw); output.flush(); os.fsync(output.fileno())
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location("ministral_diagnostic_original", path))
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module

def append(record):
    raw = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if (OUTPUT.stat().st_size if OUTPUT.exists() else 0) + len(raw) > LIMIT: raise RuntimeError("Diagnostic output limit")
    with OUTPUT.open("ab") as output:
        output.write(raw); output.flush(); os.fsync(output.fileno())

def encode_scores(scores):
    return dict(dtype="<f4", count=len(scores), data_base64=base64.b64encode(struct.pack(f"<{len(scores)}f", *scores)).decode("ascii"))

def observe(window, row, index, scores, result):
    record = dict(window=window, row_index=row["row_index"], fixture_id=row["fixture_id"], index=index,
                  prefix=list(row["prefix"]), native_dtype=row["native_dtype"])
    if index != len(row["prefix"]) or index >= len(row["expected_tokens"]) or row["prefix"] != row["expected_tokens"][:index]:
        append(dict(record, event="mismatch", reason="Prefix/sample index differs"))
        raise RuntimeError("Diagnostic prefix/sample index differs")
    nonfinite = dict(nan=sum(math.isnan(x) for x in scores), posinf=sum(x == math.inf for x in scores), neginf=sum(x == -math.inf for x in scores))
    record["nonfinite"] = nonfinite
    if index == 10: record.update(prompt_ids=row["prompt_ids"], vector=encode_scores(scores))
    if any(nonfinite.values()):
        append(dict(record, event="nonfinite"))
        raise RuntimeError("Nonfinite normalized sampler scores")
    top = heapq.nlargest(2, range(len(scores)), key=scores.__getitem__)
    reason = "Argmax/sampler differs" if result != top[0] else "Retained branch differs" if result != row["expected_tokens"][index] else None
    record.update(event="mismatch" if reason else "scores", argmax=top[0], returned_token=result,
                  top2=[dict(token=t, score=scores[t], prob=math.exp(scores[t])) for t in top],
                  margin=scores[top[0]] - scores[top[1]], candidates={str(t): scores[t] for t in CANDIDATES})
    if reason: record["reason"] = reason
    append(record)
    if reason: raise RuntimeError(reason)
    row["prefix"].append(result)

@contextmanager
def install_hooks(module, baseline):
    original_window, original_factory = module.run_window, module.sampler_for
    current = {}
    def sampler_for(row, eos, safe):
        original = original_factory(row, eos, safe)
        context = current.get(id(row))
        if context is None or row["sampling"] != "greedy": return original
        index = 0
        def sampler(scores):
            nonlocal index
            result = original(scores)
            if index < 11:
                context["native_dtype"] = str(scores.dtype)
                observe(context["window"], context, index, scores.tolist()[0], result.item())
            index += 1
            return result
        return sampler
    def run_window(model, tokenizer, fixtures, window, emit, **kwargs):
        nonlocal current
        previous, current = current, {}
        captured = window["id"] in baseline["expected"]
        try:
            if captured:
                if any(fixtures[f["id"]] != f for f in baseline["fixtures"]): raise RuntimeError("Rendered fixture differs")
                current = {id(row): dict(window=window["id"], row_index=i, fixture_id=row["fixture_id"],
                           prompt_ids=fixtures[row["fixture_id"]]["prompt_ids"], expected_tokens=baseline["expected"][window["id"]][i]["tokens"], prefix=[])
                           for i, row in enumerate(window["rows"])}
            result = original_window(model, tokenizer, fixtures, window, emit, **kwargs)
            if captured and result != baseline["expected"][window["id"]]:
                append(dict(event="window_mismatch", window=window["id"]))
                raise RuntimeError("Retained full window differs")
            return result
        finally:
            current = previous
    module.run_window, module.sampler_for = run_window, sampler_for
    try: yield module
    finally: module.run_window, module.sampler_for = original_window, original_factory

def main():
    baseline = load_baseline()
    raw = Path(sys.argv[1]).read_bytes()
    job, helper = json.loads(raw), Path(__file__).resolve()
    helper_sha = sha256(helper.read_bytes()).hexdigest()
    if len(job["source_hashes"]) != 6 or job["source_hashes"].get(str(helper)) != helper_sha: raise RuntimeError("Diagnostic helper source binding differs")
    module = load_original(baseline)
    append(dict(event="provenance", job_sha256=sha256(raw).hexdigest(), helper_sha256=helper_sha,
                baseline_sha256=BASELINE_SHA256, original_job_sha256=baseline["original_job_sha256"],
                original_module=baseline["original_module"], source_hashes=job["source_hashes"]))
    with install_hooks(module, baseline): return module.main()

if __name__ == "__main__":
    raise SystemExit(main())
