"""Finite fabricated qualification; run only in the pinned MLX Python."""
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from time import monotonic
import unittest
from unittest import mock

from tests.fixture_verification import verify_saved_programs

g = importlib.import_module("src.data-pipeline.stage03.local_generation")
MODELS = Path("/Users/michal/Workspace/models")


def fixture(offset):
    square = offset % 2 == 0
    cases = [("2\n", str(4 + offset)), ("-3\n", str(9 + offset))] if square else [
        ("5\n1 2 2 1 3\n", str(3 + offset)), ("0\n\n", str(offset))]
    task = {"question": ("Read one integer and print its square" if square else
            "Read n followed by n integers and print the number of distinct integers") + f" plus {offset}.",
            "cases": [dict(index=i, input=a, answer=b, visible=i == 0) for i, (a, b) in enumerate(cases)]}
    messages = [dict(role="system", content="Return only a complete Python 3 program."),
                dict(role="user", content=task["question"] +
                     " Use separate parse_input and solve functions, and an if __name__ main guard. Read stdin; print the answer.")]
    return task, dict(messages=messages, seed=17 if offset == 0 else 100 + offset,
                      temperature=0 if offset == 1 else .2, top_p=.95, max_tokens=256)


def worklist(size):
    offsets = [0, 0] + (list(range(1, 7)) if size == 8 else list(range(6, 0, -1)) + list(range(7, 31)))
    tasks, requests = map(list, zip(*(fixture(i) for i in offsets)))
    requests[-1]["messages"] += [dict(role="assistant", content="print(0)"), dict(role="user",
        content=f"Visible case returned 0; expected {tasks[-1]['cases'][0]['answer']}. Replace the program.")]
    cap = copy.deepcopy(requests[0])
    cap.update(max_tokens=1, seed=991)
    cap["messages"][-1]["content"] = "List all integers from one to one thousand."
    return tasks, requests + [cap]


@unittest.skipUnless(os.environ.get("ROPOT_MLX_BATCH_TESTS") == "1", "set ROPOT_MLX_BATCH_TESTS=1 in the MLX Python")
class LiveBatchQualification(unittest.TestCase):
    def test_finite_qualification(self):
        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler
        from transformers import AutoTokenizer

        native = importlib.import_module("mlx_lm.generate")
        Path("data/03_generation_checks").mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="batch-06-", dir="data/03_generation_checks")).resolve()
        sources = [Path(g.__file__), Path(__file__), Path(__file__).with_name("fixture_verification.py")]
        source_hashes = {str(path.resolve()): g.file_hash(path) for path in sources}
        summary = {"status": "running", "runs": [], "phase": "sampler", "root": str(root),
                   "supplied_request_limit": 90, "output_token_allowance": 22020,
                   "test_command": [sys.executable, "-m", "unittest", "tests.test_batched_generation_live", "-v"],
                   "test_environment": {"ROPOT_MLX_BATCH_TESTS": "1"},
                   "source_sha256_before": source_hashes}
        print("Retained Batch 6 qualification:", root, flush=True)
        def gate(condition, message):
            if not condition:
                raise AssertionError(message)
        try:
            # Flat logits make every token plausible, so this cannot pass through argmax alone.
            sequences = []
            for order in ([s for s in (17, 18, 19) for _ in range(32)], [17, 18, 19] * 32, [19, 17, 18] * 32):
                samplers = {s: make_sampler(temp=.7, top_p=.95, seed=s) for s in (17, 18, 19)}
                draws = {s: [] for s in samplers}
                for seed in order:
                    draws[seed].append(int(samplers[seed](mx.zeros((1, 256))).item()))
                sequences.append(draws)
            g.save(root / "sampler.json", sequences)
            gate(sequences[0] == sequences[1] == sequences[2], "Seeded sampler sequence changed with schedule")
            gate(len({tuple(x) for x in sequences[0].values()}) == 3, "Different seeds did not produce different draws")

            specs = [("qwen", "Qwen3.5-9B-8bit", "16daa4818c54ce5f5436f929d52542eb65bbed9d", 256, (4, 16)),
                     ("ministral", "Ministral-3-8B-Instruct-2512-8bit", "f333d6bcb3df97264eddcf985523b69b747f5385", 2048, (2, 8))]
            for name, slug, revision, chunk, prefills in specs:
                model = MODELS / "mlx-community" / slug
                manifest = MODELS / "metadata" / f"{slug}-{revision}.json"
                summary["phase"] = f"{name}:template"
                options = dict(add_generation_prompt=True, enable_thinking=False, return_dict=False)
                tokenizers = [AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False,
                              **({"fix_mistral_regex": True} if name == "ministral" else {}))]
                if name == "ministral":
                    tokenizers.append(AutoTokenizer.from_pretrained(MODELS / "mistralai/Ministral-3-8B-Instruct-2512/5b26027e7b19eeb4b7352e1fed3926375dd2cb4d",
                        local_files_only=True, trust_remote_code=False, fix_mistral_regex=True))
                histories = [worklist(8)[1][i]["messages"] for i in (0, 7)]
                checks = [[dict(prompt=t.apply_chat_template(h, tokenize=False, **options),
                           ids=t.apply_chat_template(h, tokenize=True, **options), eos=t.eos_token_id)
                           for t in tokenizers] for h in histories]
                g.save(root / f"{name}-templates.json", {"histories": histories, "checks": checks})
                for check in checks:
                    if name == "ministral":
                        gate(check[0] == check[1] and check[0]["ids"][0] == 1 and check[0]["ids"].count(1) == 1
                             and check[0]["eos"] == 2, "Official/local Ministral template, regex, BOS or EOS mismatch")
                    else:
                        gate(check[0]["prompt"].endswith("<think>\n\n</think>\n\n"), "Qwen thinking-off template mismatch")
                layouts = []
                for size, prefill in zip((8, 32), prefills):
                    directory = root / f"{name}-{size}"
                    tasks, requests = worklist(size)
                    g.save(root / f"{name}-{size}-fixtures.json", {"tasks": tasks, "requests": requests})
                    summary["phase"] = f"{name}-{size}:runtime"
                    result = g.generate_batched(model, requests, directory, model_manifest=manifest,
                        completion_batch_size=size, prefill_batch_size=prefill, prefill_step_size=chunk)
                    record = {"run": directory.name, "status": result["status"], "failures": []}
                    summary["runs"].append(record)
                    gate(result["status"] == "success", f"Runtime failure: {directory}")
                    rows = result["requests"]
                    gate(len(rows) == size + 1 and [r["status"] for r in rows] == ["stop"] * size + ["length"], "Completion/exhaustion mismatch")
                    gate(result["max_observed_decode_batch"] == size, "Target decode concurrency was not observed")
                    gate(rows[0]["tokens"] == rows[1]["tokens"], "Same-group replay mismatch; stop for review")
                    for req, row in zip(requests, rows):
                        gate(row["prompt"] == tokenizers[-1].apply_chat_template(req["messages"], tokenize=False, **options)
                             and row["prompt_ids"] == tokenizers[-1].apply_chat_template(req["messages"], tokenize=True, **options), "Prompt rendering consistency mismatch")
                        gate(row["prompt_tokens"] == len(row["prompt_ids"]) and row["generation_tokens"] == len(row["tokens"]), "Token accounting mismatch")
                    record["prompt_rendering_consistent"] = True
                    layout = {json.dumps(req, sort_keys=True): row["tokens"] for req, row in zip(requests[:-1], rows[:-1])}
                    layouts.append(layout)
                    record["same_group_equal"] = True
                    record["cross_layout_equal"] = {str(i): layouts[0][key] == row["tokens"]
                        for i, (req, row) in enumerate(zip(requests[:-1], rows[:-1]))
                        if (key := json.dumps(req, sort_keys=True)) in layouts[0]}
                    gate(all(row["tokens"][-1] in result["eos_token_ids"] for row in rows[:-1]), "Natural completion omitted terminal EOS accounting")
                    summary["phase"] = f"{name}-{size}:extraction"
                    for index, row in enumerate(rows[:-1]):
                        g.save(directory / f"task-{index}.json", tasks[index])
                        try:
                            (directory / f"candidate-{index}.py").write_bytes(g.extract_program(row))
                        except (ValueError, SyntaxError, MemoryError, RecursionError) as error:
                            category = ("compiler_resource" if isinstance(error, (MemoryError, RecursionError))
                                        else "syntax" if isinstance(error, SyntaxError) else "format")
                            record["failures"].append({"request": index, "category": category, "error": repr(error)})
                    gate(not record["failures"], f"Extraction failed: {directory}")
                    summary["phase"] = f"{name}-{size}:verification"
                    began = monotonic()
                    verify_saved_programs(directory, [{"task": i, "program": f"candidate-{i}.py"}
                                                      for i in range(size)])
                    record["verification_seconds"] = monotonic() - began
                    verdicts = json.loads((directory / "verification.json").read_text())
                    record["failures"] += [{"request": i, "category": "verification", "status": v["status"]}
                                           for i, v in enumerate(verdicts) if v["status"] != "passed"]
                    gate(len(verdicts) == size and not record["failures"], f"Verification failed: {directory}")
                    g.save(root / "qualification.json", summary)

                summary["phase"] = f"{name}:interrupt"
                directory = root / f"{name}-interrupt"
                req = fixture(0)[1]
                req["messages"][-1]["content"] = "Enumerate all integers from one to one million, one per line, without abbreviation."
                requests = [copy.deepcopy(req) for _ in range(3)]
                original_limit = mx.set_cache_limit(4 * 1024 ** 3)
                mx.set_cache_limit(original_limit)
                cleanup = {"uids": [], "delivered": [], "removed": [], "closed": False,
                           "native_close_stream_synchronized": False, "adapter_stream_synchronized": False}
                streams = []
                in_close = False
                synchronize = mx.synchronize
                def observe_sync(*args, **kwargs):
                    value = synchronize(*args, **kwargs)
                    if args and any(args[0] is stream for stream in streams):
                        cleanup["native_close_stream_synchronized" if in_close else "adapter_stream_synchronized"] = True
                    return value
                base = native.BatchGenerator
                class Observed(base):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        streams.append(self.stream)
                    def insert(self, *args, **kwargs):
                        uids = super().insert(*args, **kwargs)
                        cleanup["uids"] += uids
                        return uids
                    def next(self):
                        if len(set(cleanup["delivered"])) == 2:
                            raise KeyboardInterrupt("Batch 6 deterministic post-delivery probe")
                        events = super().next()
                        cleanup["delivered"] += [r.uid for r in events[1]]
                        return events
                    def remove(self, uids, *args, **kwargs):
                        value = super().remove(uids, *args, **kwargs)
                        cleanup["removed"] += list(uids)
                        cleanup["cache_empty"] = not self.extract_cache(cleanup["uids"])
                        return value
                    def close(self):
                        nonlocal in_close
                        in_close = True
                        try:
                            super().close()
                            cleanup["closed"] = True
                        finally:
                            in_close = False
                propagated = False
                with mock.patch.object(native, "BatchGenerator", Observed), \
                        mock.patch.object(mx, "synchronize", side_effect=observe_sync):
                    try:
                        g.generate_batched(model, requests, directory, model_manifest=manifest,
                            completion_batch_size=2, prefill_batch_size=1, prefill_step_size=chunk)
                    except KeyboardInterrupt:
                        propagated = True
                mx.synchronize()
                observed_limit = mx.set_cache_limit(original_limit)
                cleanup.update(synchronized=True, allocator_restored=observed_limit == original_limit,
                               interruption_propagated=propagated,
                               all_uids_removed=set(cleanup["uids"]) <= set(cleanup["removed"]))
                g.save(directory / "cleanup-checks.json", cleanup)
                result = json.loads((directory / "outcome.json").read_text())
                record = {"run": directory.name, "status": result["status"], "cleanup": cleanup}
                summary["runs"].append(record)
                rows = result["requests"]
                gate([r["status"] for r in rows] == ["cancelled", "cancelled", "not_run"] and result["status"] == "cancelled",
                     "Interruption probe failed or was inconclusive")
                gate(cleanup["closed"] and cleanup["all_uids_removed"] and cleanup.get("cache_empty") and cleanup["allocator_restored"]
                     and cleanup["adapter_stream_synchronized"], "Abort cleanup check failed")
                gate(propagated and all(r["tokens"] for r in rows[:2]), "Interruption did not retain both partial replies")
                g.save(root / "qualification.json", summary)
            summary["status"] = "passed"
        except BaseException as error:
            category = summary["phase"].rsplit(":", 1)[-1]
            summary.update(status="review_stop", error=repr(error), error_category=(
                category if category in ("extraction", "verification") else "runtime/protocol"))
            raise
        finally:
            summary["source_sha256_after"] = {str(path.resolve()): g.file_hash(path) for path in sources}
            summary["source_unchanged"] = summary["source_sha256_after"] == source_hashes
            if not summary["source_unchanged"]:
                summary.update(status="review_stop")
                summary.setdefault("error", "Qualification sources changed during the run")
                summary.setdefault("error_category", "qualification_source")
            g.save(root / "qualification.json", summary)
            if not summary["source_unchanged"] and sys.exc_info()[0] is None:
                raise AssertionError("Qualification sources changed during the run")
