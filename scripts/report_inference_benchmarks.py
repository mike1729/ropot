"""Read retained fixture evidence once; no generation or program execution."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import median


def load(path):
    return json.loads(path.read_bytes())


def records(directory):
    return [json.loads(line) for line in (directory / "results.jsonl").read_bytes().splitlines()]


def report(root):
    root = Path(root).resolve()
    lines = ["# Local Q8 fixture benchmark evidence", "",
        "These are systems measurements on invented tasks. No cohort tasks, accepted "
        "witnesses, scientific attack yield, human audits, BF16 or cloud runs are included.", "",
        "## Natural worklists", "",
        "Each measured window contains the same ten conversations: two seeds each for "
        "benign, witness, implantation, repair and long reply. Natural prompts are not "
        "padded to target lengths. Counts include unsuccessful work and token-limit "
        "completions. Different batch shapes and quantization packages may change outputs.", "",
        "| Route | Configured decode / prefill / step | Repeats | Worklist median (range), s | Delivered tokens/s | Stop / limit | Peak MLX GiB | Max token events/call |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    tables, accounting, latencies = [], [], []
    for route in ("actor", "synthetic"):
        directory = root / f"{route}-natural"
        if not (directory / "summary.json").exists():
            continue
        rows = load(directory / "summary.json")
        for batch in sorted({r["config"]["completion_batch_size"] for r in rows}):
            group = [r for r in rows if r["config"]["completion_batch_size"] == batch]
            times = [r["elapsed_seconds"] for r in group]
            replies = [r for w in group for r in records(directory / w["name"])]
            cfg = group[0]["config"]
            statuses = Counter(r["status"] for r in replies)
            row = dict(route=route, config=cfg, repeats=len(group), median_seconds=median(times),
                min_seconds=min(times), max_seconds=max(times),
                tokens_per_second=sum(r["delivered_tokens"] for r in group)/sum(times),
                peak_GiB=max(r["peak_memory_bytes"] for r in group)/1024**3,
                max_token_events=max(r["max_observed_decode_batch"] for r in group),
                prefill_fraction=sum(r["native"]["prompt_time"] for r in group)/sum(times),
                caller_fraction=sum(r["caller_seconds"] for r in group)/sum(times),
                statuses=dict(statuses), flagged_replies=sum(r["contains_unk"] or r["contains_thinking"]
                    or r["empty_reply"] for r in replies),
                same_tokens_in_repeats=all({r["id"]: r["token_ids"] for r in records(directory/w["name"])}
                    == {r["id"]: r["token_ids"] for r in records(directory/group[0]["name"])} for w in group))
            tables.append(row)
            lines.append(f"| {route} MLX | {batch} / {cfg['prefill_batch_size']} / {cfg['prefill_step_size']} "
                f"| {len(group)} | {median(times):.3f} ({min(times):.3f}–{max(times):.3f}) "
                f"| {row['tokens_per_second']:.2f} | {statuses['stop']} / {statuses['length']} "
                f"| {row['peak_GiB']:.2f} | {row['max_token_events']} |")
            for kind in sorted({r["kind"] for r in replies}):
                selected = [r for r in replies if r["kind"] == kind]
                latencies.append(dict(route=route, batch=batch, kind=kind,
                    first_token_median=median(r["first_token_seconds"] for r in selected),
                    finish_median=median(r["finished_seconds"] for r in selected)))
    for parallel in (1, 2, 8):
        directory = root / f"gguf-p{parallel}"
        if not (directory / "summary.json").exists():
            continue
        group = [r for r in load(directory / "summary.json") if r["name"] != "warm"]
        if not group:
            continue
        times = [r["elapsed_seconds"] for r in group]
        statuses = Counter(r["status"] for w in group for r in records(directory/w["name"]))
        lines.append(f"| synthetic GGUF | parallel {parallel} | {len(group)} | {median(times):.3f} "
            f"({min(times):.3f}–{max(times):.3f}) | "
            f"{sum(r['delivered_tokens'] for r in group)/sum(times):.2f} "
            f"| {statuses['stop']} / {statuses['length']} | not measured | not measured |")
    lines += ["", "MLX's observed field counts token response events returned per native call; "
        "the pinned generator can exceed its configured decode target during prefill "
        "handoff. It is not a measured GPU occupancy or an independently enforced hard cap. "
        "GGUF and MLX token counts have backend-specific EOS conventions; compare finite "
        "worklist wall time, and assess useful yield separately. Latencies in the JSON "
        "analysis start at worklist admission and include queueing; no p95 is estimated.", "",
        "## Model lifecycle and corpus", "",
        "| Worker | Weight load / server startup, s | Whole process, s | Timed windows, s | Warmup windows, s | Natural prompt tokens min–max |",
        "| --- | ---: | ---: | ---: | ---: | --- |"]
    for name in ("actor-fixed", "actor-natural", "synthetic-fixed", "synthetic-natural", "gguf-p1", "gguf-p2", "gguf-p8"):
        directory = root/name
        if not (directory/"runtime.json").exists() or not (directory/"summary.json").exists():
            continue
        runtime = load(directory/"runtime.json")
        startup = runtime.get("load_seconds", runtime.get("startup_seconds"))
        outcome = root/"processes"/name/"outcome.json"
        whole = f"{load(outcome)['elapsed_seconds']:.3f}" if outcome.exists() else "incomplete"
        windows = load(directory/"summary.json")
        timed = sum(r["elapsed_seconds"] for r in windows if r.get("name") != "warm")
        warm = sum(load(p)["elapsed_seconds"] for p in directory.glob("warm-*/summary.json"))
        warm += sum(r["elapsed_seconds"] for r in windows if r.get("name") == "warm")
        corpus = load(directory/"corpus.json") if (directory/"corpus.json").exists() else []
        lengths = [len(r["prompt_ids"]) for r in corpus]
        bins = f"{min(lengths)}–{max(lengths)}" if lengths and not name.endswith("fixed") else "synthetic"
        lines.append(f"| {name} | {startup:.3f} | {whole} | {timed:.3f} | {warm:.3f} | {bins} |")
    lines += ["", "Process time additionally includes interpreter imports, tokenizer/template "
        "work, metadata, loading and cleanup. Load time alone is not the full cold-start cost.", "",
        "## Fixed synthetic length screen", "",
        "Random-token prompts at 1,024/1,024/2,048/4,096/4,096 tokens; output caps "
        "512/128/512/512/1,024 are forced with EOS disabled. Five requests per window.", "",
        "| Route | Decode setting | Repeats | Median seconds | Range seconds |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for route in ("actor", "synthetic"):
        path = root / f"{route}-fixed/summary.json"
        if path.exists():
            rows = load(path)
            for batch in sorted({r["config"]["completion_batch_size"] for r in rows}):
                times = [r["elapsed_seconds"] for r in rows if r["config"]["completion_batch_size"] == batch]
                lines.append(f"| {route} | {batch} | {len(times)} | {median(times):.3f} | {min(times):.3f}–{max(times):.3f} |")
    lines += ["", "## Recording overhead", "",
        "| Fabricated replies | Wall seconds | CPU seconds | Bytes read / written |",
        "| ---: | ---: | ---: | ---: |"]
    for r in load(root / "overhead/summary.json"):
        lines.append(f"| {r['requests']} | {r['wall_seconds']:.6f} | {r['cpu_seconds']:.6f} "
                     f"| {r['bytes_read']} / {r['bytes_written']} |")
    lines += ["", "The replay percentage denominator assumes one second of compute per reply; "
        "it is not measured inference. Actual MLX caller fractions are retained separately.", "",
        "## Optional queued short witnesses", "",
        "Forty requests per window, each with a 128-token cap and fresh sampling; "
        "three randomized blocks, then two extra blocks for the two fastest settings. "
        "These outputs are unscored. Qualification uses separate short marker requests.", "",
        "| Route | Decode setting | Repeats | Median seconds (range) | Delivered tokens/s |",
        "| --- | ---: | ---: | ---: | ---: |"]
    short_notes = []
    for path in sorted(root.glob("short-*/summary.json")):
        rows = load(path)
        for batch in sorted({r["config"]["completion_batch_size"] for r in rows}):
            selected = [r for r in rows if r["config"]["completion_batch_size"] == batch]
            times = [r["elapsed_seconds"] for r in selected]
            lines.append(f"| {path.parent.name} | {batch} | {len(times)} | {median(times):.3f} "
                f"({min(times):.3f}–{max(times):.3f}) | "
                f"{sum(r['delivered_tokens'] for r in selected)/sum(times):.2f} |")
        qualification = path.parent/"qualification.json"
        if qualification.exists():
            short_notes += ["", f"{path.parent.name} qualification: `{json.dumps(load(qualification), sort_keys=True)}`", ""]
    lines.extend(short_notes)
    lines += ["", "## Gated prefill and exact-prefix comparisons", "",
        "These cells run only after the short-worklist prefill share exceeds 15%. "
        "Exact-prefix windows construct the prefix once per finite worklist, clone "
        "all KV and recurrent state, and check the master state afterwards. Their elapsed "
        "time includes construction, cloning and state checks. Normal sampling is fresh "
        "for every request. Native suffix-only prompt counters omit the separately "
        "recorded prefix build; complete elapsed time is the comparison denominator.", "",
        "| Source | Decode / prefill / step | Repeats | Median seconds (range) | Mean output tokens/worklist | Max token events | Peak MLX GiB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    prefix_notes = []
    for path in sorted(root.glob("prefill-*/summary.json")):
        rows = load(path)
        configs = {json.dumps(r["config"], sort_keys=True) for r in rows}
        for key in sorted(configs):
            cfg = json.loads(key)
            selected = [r for r in rows if r["config"] == cfg]
            times = [r["elapsed_seconds"] for r in selected]
            label = f"{cfg['completion_batch_size']} / {cfg['prefill_batch_size']} / {cfg['prefill_step_size']}"
            lines.append(f"| {path.parent.name} | {label} | {len(times)} | {median(times):.3f} "
                         f"({min(times):.3f}–{max(times):.3f}) | "
                         f"{sum(r['delivered_tokens'] for r in selected)/len(selected):.1f} | "
                         f"{max(r['max_observed_decode_batch'] for r in selected)} | "
                         f"{max(r['peak_memory_bytes'] for r in selected)/1024**3:.2f} |")
        control = path.parent/"cache-reference-check.json"
        if control.exists():
            check = load(control)
            a = load(path.parent/"reference-uncached/summary.json")
            b = load(path.parent/"reference-cached/summary.json")
            prefix_notes += ["", f"{path.parent.name} prefix control: uncached {a['elapsed_seconds']:.3f}s / "
                f"{a['delivered_tokens']} tokens; cached {b['elapsed_seconds']:.3f}s / "
                f"{b['delivered_tokens']} tokens. {len(check['mismatches'])}/40 response token/status "
                "mismatches. This is one control pair, not repeated speed evidence.", ""]
        paired = path.parent/"prefix-summary.json"
        if paired.exists():
            for reused in sorted({r["reused_tokens"] for r in load(paired)}):
                times = [r["elapsed_seconds"] for r in load(paired) if r["reused_tokens"] == reused]
                lines.append(f"| {path.parent.name} paired prefix {reused} | see qualification | {len(times)} "
                    f"| {median(times):.3f} ({min(times):.3f}–{max(times):.3f}) | — | — | — |")
        qualification = path.parent/"qualification.json"
        if qualification.exists():
            prefix_notes += ["", f"{path.parent.name}: `{json.dumps(load(qualification), sort_keys=True)}`", ""]
    lines.extend(prefix_notes)
    lines += ["", "All windows compare equal request counts; natural output length can change "
        "with numerical batch shape. Lower elapsed time does not establish equal useful yield.", "",
        "Cached/uncached equality is a control for this frozen fixture, not a blanket "
        "claim across batch shapes or workloads. Divergence is retained and blocks cache "
        "promotion; it does not by itself establish corruption.", "",
        "## Coalesced verification and scheduling", "",
        "Eight invented identity programs with 20 cases each: five passes and three "
        "expected failures. Each program has its own sandbox; cases remain serial. "
        "Explicit evidence reuse is checked after execution and timed separately.", "",
        "| Verification window | Programs | Execute seconds | Explicit reuse seconds |",
        "| --- | ---: | ---: | ---: |"]
    for path in sorted(root.glob("verification-*/summary.json")):
        for r in load(path):
            lines.append(f"| {path.parent.name} | {r['programs']} | {r['seconds']:.3f} | {r['reuse_seconds']:.3f} |")
    if (root / "schedules.json").exists():
        lines += ["", "| Schedule | Complete process makespan, s | Measured inference/verification intersection, s |",
                  "| --- | ---: | ---: |"]
        for r in load(root / "schedules.json"):
            lines.append(f"| {r['mode']} | {r['seconds']:.3f} | {r['measured_intersection_seconds']:.3f} |")
        lines += ["", "The schedule comparison includes process startup, model loading and warmup. "
            "It is an initial finite-run comparison, not a persistent-pipeline throughput estimate."]
    benign_path = root/"generated-benign-check/summary.json"
    benign = None
    if benign_path.exists():
        observed = load(benign_path)
        counts = {}
        for r in observed["records"]:
            counts.setdefault(r["route"], Counter())[r["status"]] += 1
        benign = dict(passed=observed["passed"], total=observed["total"],
            verification_seconds=observed["verification_seconds"],
            by_route={k:dict(v) for k,v in counts.items()})
        lines += ["", "## Saved benign-program smoke", "",
            f"{observed['passed']}/{observed['total']} first-repeat benign replies passed all 20 invented "
            f"cases in Docker ({observed['verification_seconds']:.3f}s verification). One optional "
            "Python code fence is removed; other content must parse as complete Python. "
            "An independent union-find oracle supplies fixture answers; the supplied program "
            "used iterative DFS. This consumed zero new model draws and establishes only "
            "one-task fixture correctness, not cohort yield or independent task observations.", "",
            "| Route | Passed | Failed/error/parse failure |", "| --- | ---: | ---: |"]
        for route, count in counts.items():
            lines.append(f"| {route} | {count['passed']} | {sum(count.values())-count['passed']} |")
    lines += ["", "## Complete effort and provenance", "",
        "The following includes warmups and forced-length draws. Fabricated overhead "
        "records are excluded. Planned token caps are not tokens actually spent. "
        "Interrupted/unstarted records remain visible and are not called completed draws.", "",
        "| Worker | Retained request records | Delivered tokens | Declared token caps | Stop / limit / other |",
        "| --- | ---: | ---: | ---: | --- |"]
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.name in ("overhead", "processes", "sources"):
            continue
        replies = [r for path in sorted(directory.glob("*/results.jsonl")) for r in records(path.parent)]
        if not replies:
            continue
        for path in directory.glob("*/results.jsonl"):
            for r in records(path.parent):
                payload = (path.parent/r["reply"]).read_bytes()
                if sha256(payload).hexdigest() != r["reply_sha256"]:
                    raise ValueError(f"Changed reply: {path.parent/r['reply']}")
        statuses = Counter(r["status"] for r in replies)
        row = dict(worker=directory.name, retained_records=len(replies),
            delivered_tokens=sum(len(r["token_ids"]) for r in replies),
            declared_caps=sum(r["max_tokens"] for r in replies), statuses=dict(statuses),
            flagged_replies=sum(r["contains_unk"] or r["contains_thinking"] or r["empty_reply"] for r in replies))
        accounting.append(row)
        lines.append(f"| {directory.name} | {len(replies)} | {row['delivered_tokens']} | {row['declared_caps']} "
                     f"| {statuses['stop']} / {statuses['length']} / {len(replies)-statuses['stop']-statuses['length']} |")
    outcomes = {p.parent.name: load(p) for p in root.glob("processes/*/outcome.json")}
    files = [p for p in root.rglob("*") if p.is_file() and p.name not in ("report-detailed.md", "analysis.json")]
    totals = dict(retained_records=sum(r["retained_records"] for r in accounting),
                  delivered_tokens=sum(r["delivered_tokens"] for r in accounting),
                  declared_caps=sum(r["declared_caps"] for r in accounting))
    analysis = dict(mlx_natural=tables, request_latency=latencies, effort=accounting, totals=totals,
        generated_benign_smoke=benign,
        processes=outcomes, files=len(files), bytes=sum(p.stat().st_size for p in files))
    (root/"analysis.json").write_text(json.dumps(analysis, indent=2)+"\n")
    lines += ["", f"Total: {totals['retained_records']} retained request records, "
        f"{totals['delivered_tokens']:,} delivered tokens and {totals['declared_caps']:,} declared token caps.", "",
        f"Evidence: {analysis['files']} files, {analysis['bytes']:,} bytes before this report. "
        f"All {len(outcomes)} completed worker outcomes record process-group cleanup; inspect "
        "`analysis.json` for their exit codes and cleanup status. `plan.json`, per-worker "
        "runtime files, and `source-snapshot.json` bind settings and exact source bytes.", "",
        "This report checks every retained reply SHA-256. The JSON analysis retains "
        "per-workload first-token and completion latency, MLX caller fractions, "
        "prefill shares, repeat agreement, flags and complete process outcomes.", ""]
    (root/"report-detailed.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    report(parser.parse_args().root)
