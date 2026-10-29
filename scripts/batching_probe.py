"""Screen selected batching profiles; experimental sweeps require explicit flags."""
import argparse
import gc
import importlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
g = importlib.import_module("data-pipeline.stage03.local_generation")
GIB = 1024**3
PROFILES = ROOT / "configs/inference/v2-batching.json"
BATCH_KEYS = ("completion_batch_size", "prefill_batch_size", "prefill_step_size")
BASE = ROOT / "data/05_attack_experiments/batch-03-qwen36-capability-20261027T190000Z"
MODELS = {
    "qwen": (Path("/Users/michal/Models/mlx/Qwen3.6-27B-8bit"), BASE / "model-manifest.json"),
    "gemma": (Path("/Users/michal/Workspace/models/mlx-community/gemma-4-31b-it-8bit"),
              Path("/Users/michal/Workspace/models/metadata/gemma-4-31b-it-8bit-f5f3dc92ab4af76724c36c21eb6bedadb3a851be.json")),
}


def selected_profile(args):
    if args.sweep or args.minimal or args.capacity or args.chunk_only or args.probe_settings:
        return None
    return "gemma-short" if args.model == "gemma" and args.short_only else args.model


def system_memory():
    swap = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
    pressure = subprocess.check_output(["sysctl", "kern.memorystatus_vm_pressure_level"], text=True)
    return dict(swap_used_mib=float(re.search(r"used = ([\d.]+)M", swap)[1]),
                pressure=int(pressure.split(":")[1]))


def freeze(directory):
    """Only previously exposed requests and generated candidates; no private cases."""
    requests, sources = [], []
    for source in ("actor",):
        folder = BASE / source / "generation-0000"
        inputs = json.loads((folder / "inputs.json").read_text())
        outcome = json.loads((folder / "outcome.json").read_text())
        sources.append(dict(path=str(folder), inputs_sha256=g.file_hash(folder / "inputs.json"),
                            outcome_sha256=g.file_hash(folder / "outcome.json")))
        for i, request in enumerate(inputs["requests"]):
            request = json.loads(json.dumps(request))
            request.update(seed=41 + len(requests), temperature=0.2, top_p=0.95, max_tokens=8192,
                           enable_thinking=True)
            requests.append(request)
    long_path = ROOT / "data/03_throughput/q8-followups-20261025-v1/actor-long/7k-calibration/inputs.json"
    long = json.loads(long_path.read_text())
    sources.append(dict(path=str(long_path), sha256=g.file_hash(long_path),
                        note="Invented historical graph-program repair fixtures, first eight requests"))
    for old in long["requests"][:8]:
        # Flatten saved public conversation into a fresh single-turn request.
        content = "\n\n".join(f"{m['role']}:\n{m['content']}" for m in old["messages"])
        requests.append(dict(messages=[dict(role="user", content=content)],
            seed=41 + len(requests), temperature=0.2, top_p=0.95, max_tokens=8192, enable_thinking=True))
    g.save(directory / "workload.json", dict(requests=requests, sources=sources,
        note="8 historical initial prompts + 8 invented long repair fixtures; thinking on; no execution or yield scoring"))
    return requests


def run(args):
    profile = selected_profile(args)
    settings = json.loads(PROFILES.read_text())["profiles"][profile] if profile else None
    args.capacity = args.capacity or (profile is not None and args.model == "gemma")
    import mlx.core as mx
    from mlx.utils import tree_flatten
    native = importlib.import_module("mlx_lm.generate")

    directory = Path(args.output).absolute() / args.model
    directory.mkdir(parents=True, exist_ok=False)
    model_path, manifest_path = MODELS[args.model]
    requests = freeze(directory)
    if args.short_only:
        requests = requests[:8]
        g.save(directory / "minimal-workload.json", requests)
    elif args.minimal or profile == "gemma":
        requests = requests[:4] + requests[8:12]
        g.save(directory / "minimal-workload.json", requests)
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("files", [])
    if not entries:
        raise ValueError("Manifest has no files")
    for entry in entries:
        path = model_path / entry["path"]
        if path.stat().st_size != entry["bytes"] or g.file_hash(path) != entry["sha256"]:
            raise ValueError(f"Checkpoint differs from manifest: {path}")
    config = json.loads((model_path / "config.json").read_text())
    start_memory = system_memory()
    if start_memory["pressure"] != 1:
        raise RuntimeError("Require normal initial memory pressure")
    mx.set_cache_limit(4 * GIB)
    mx.set_memory_limit(44 * GIB)  # Advisory only, not a hard allocation bound.
    loaded = directory / "load"
    loaded.mkdir()
    model, tokenizer = g.load_model(model_path, loaded, config,
        extra_modules=(f"mlx_lm.models.{config['model_type']}",
                       "mlx_lm.models.gemma4_text" if args.model == "gemma" else "mlx_lm.models.qwen3_5",
                       "mlx_lm.models.cache", "mlx_lm.generate_utils"))
    weights = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
    text = config["text_config"]
    max_prompt = 0

    rendered = [g.render(tokenizer, r["messages"], enable_thinking=r["enable_thinking"]) for r in requests]
    max_prompt = max(r["prompt_tokens"] for r in rendered)
    g.save(directory / "rendered.json", rendered)
    g.save(directory / "contract.json", dict(model=str(model_path), manifest_sha256=g.file_hash(manifest_path),
        runner_sha256=g.file_hash(__file__), adapter_sha256=g.file_hash(g.__file__),
        weight_bytes=weights, device=mx.device_info(), initial_memory=start_memory,
        screening_tokens=256, confirmation_ceiling=8192, thinking=True, request_count=len(requests),
        selected_profile=profile, profiles_sha256=g.file_hash(PROFILES) if profile else None,
        max_prompt_tokens=max_prompt, peak_stop_bytes=44 * GIB,
        admission=("weights + full KV + chunk-bounded rotating KV + >=2 GiB measured workspace <=44 GiB"
                   if args.capacity else "weights + 1.5 * estimated KV/state + 4 GiB workspace <= 44 GiB; advisory estimate"),
        caveat="Benchmark overrides resident load reuse and screening EOS; no production adapter changes"))
    runtime = json.loads((loaded / "runtime.json").read_text())
    # Warning pressure is diagnostic: earlier batch-8 Qwen runs succeeded.
    # Stop on critical pressure or new swap, rather than warning alone.
    for _ in range(2):
        state = system_memory()
        with (directory / "load-memory.jsonl").open("a") as stream:
            stream.write(json.dumps(state) + "\n")
        if state["swap_used_mib"] > start_memory["swap_used_mib"] + 256 or state["pressure"] >= 4:
            raise RuntimeError("Resource stop during model load")
        time.sleep(2)

    def resident_load(path, output, config, **kwargs):
        g.save(output / "runtime.json", dict(runtime, load_seconds=0, resident_model_reused=True))
        return model, tokenizer

    g.load_model = resident_load
    original_generator = native.BatchGenerator
    screening = False
    current = None
    last_check = 0
    emitted = 0
    window_start = 0

    class ObservedGenerator(original_generator):
        def __init__(self, *a, **kw):
            if screening:
                kw["stop_tokens"] = []
            super().__init__(*a, **kw)

        def next(self):
            nonlocal last_check, emitted
            now = time.monotonic()
            if now - last_check >= 2:
                state = dict(seconds=now - window_start, **system_memory(),
                             active_bytes=mx.get_active_memory(), peak_bytes=mx.get_peak_memory(),
                             emitted_tokens=emitted)
                with (current / "memory.jsonl").open("a") as stream:
                    stream.write(json.dumps(state) + "\n")
                last_check = now
                if (state["pressure"] >= 4 or state["swap_used_mib"] > start_memory["swap_used_mib"] + 256
                        or state["peak_bytes"] > 44 * GIB):
                    raise RuntimeError("Resource stop: pressure, swap growth or live MLX peak")
            result = super().next()
            emitted += len(result[1])
            return result

    native.BatchGenerator = ObservedGenerator
    records = []
    workspace = 2 * GIB

    def capacity_kv(c, ceiling, step):
        length = max_prompt + ceiling + 256  # Full-cache allocation rounding.
        full = text["layer_types"].count("full_attention")
        per_request = full * text["num_global_key_value_heads"] * text["global_head_dim"] * 4 * length
        per_request += (text["num_hidden_layers"] - full) * text["num_key_value_heads"] * text["head_dim"] * 4 * (text["sliding_window"] + step - 1)
        return per_request * c

    def estimate(c, ceiling):
        length = max_prompt + ceiling + 256
        full = text["layer_types"].count("full_attention")
        heads = text.get("num_global_key_value_heads", text["num_key_value_heads"])
        kv = full * heads * text.get("global_head_dim", text["head_dim"]) * 4 * length
        if args.model == "gemma":
            kv += (text["num_hidden_layers"] - full) * text["num_key_value_heads"] * text["head_dim"] * 4 * (text["sliding_window"] + 1024)
        else:
            kv += 0.1 * GIB  # Recurrent state and convolution cache per sequence.
        return weights + 1.5 * kv * c + 4 * GIB

    def window(label, settings, short=False, warm=False):
        nonlocal screening, current, last_check, emitted, window_start
        ceiling = 16 if warm else 256 if short else 8192
        predicted = (weights + capacity_kv(settings[0], ceiling, settings[2]) + workspace
                     if args.capacity else estimate(settings[0], ceiling))
        if predicted > 44 * GIB:
            record = dict(label=label, settings=settings, status="skipped_memory_estimate", estimated_bytes=predicted)
            records.append(record)
            g.save(directory / "progress.json", records)
            print(json.dumps(record), flush=True)
            return None
        screening, current, last_check, emitted = short or warm, directory / label, 0, 0
        reqs = [dict(r, max_tokens=ceiling) for r in (requests[:2] if warm else requests)]
        gc.collect()
        mx.clear_cache()
        window_start = time.monotonic()
        print(json.dumps(dict(start=label, settings=settings, estimated_gib=predicted / GIB)), flush=True)
        result = g.generate_batched(model_path, reqs, current, model_manifest=manifest_path,
            **dict(zip(BATCH_KEYS, settings)))
        state = system_memory()
        record = dict(label=label, settings=settings, status=result["status"],
            seconds=result["elapsed_seconds"], tokens=sum(r["generation_tokens"] for r in result["requests"]),
            stops=sum(r["status"] == "stop" for r in result["requests"]),
            lengths=sum(r["status"] == "length" for r in result["requests"]),
            peak_gib=result["peak_memory_bytes"] / GIB, max_observed_batch=result["max_observed_decode_batch"],
            error=result["error"], memory_after=state, estimated_bytes=predicted)
        records.append(record)
        g.save(directory / "progress.json", records)
        print(json.dumps(record), flush=True)
        if result["status"] != "success":
            raise RuntimeError(f"Stopped after {label}: {result['error']}")
        if short and record["tokens"] != len(reqs) * 256:
            raise RuntimeError("Screening did not deliver exactly 256 tokens per request")
        if state["pressure"] >= 4 or state["swap_used_mib"] > start_memory["swap_used_mib"] + 256:
            raise RuntimeError("Resource stop after window")
        return record

    window("warmup", (2, 1, 256), warm=True)
    if profile:
        result = window("selected", tuple(settings[k] for k in BATCH_KEYS), short=True)
        g.save(directory / "summary.json", dict(profile=profile, selected=result,
            limitation="One fixed-256-token window; provisional profile, natural completions unqualified"))
        return
    if args.probe_settings:
        result = window("single-probe", tuple(args.probe_settings), short=True)
        g.save(directory / "summary.json", dict(probe=result,
            limitation="Single fixed-length setting; no natural completion confirmation"))
        return
    if args.capacity:
        if args.model != "gemma" or not args.minimal:
            raise ValueError("Capacity probe requires Gemma --minimal")
        b = window("capacity-b4", (4, 1, 256), short=True)
        if b is None:
            raise RuntimeError("Batch 4 exceeds allocation estimate")
        workspace = max(workspace, b["peak_gib"] * GIB - weights - capacity_kv(4, 256, 256) + GIB)
        step = next((s for s in (256, 128, 64) if weights + capacity_kv(8, 256, s) + workspace <= 44 * GIB), None)
        if step is not None:
            window("capacity-b8", (8, 1, step), short=True)
        g.save(directory / "summary.json", dict(windows=records, calibrated_workspace_bytes=workspace,
            limitation="Allocation-aware estimates, not hard bounds; batch-8 chunk may differ; eight mixed prompts only"))
        return
    if args.chunk_only:
        result = window("screen-E", (8, 2, 1024), short=True)
        g.save(directory / "summary.json", dict(chunk_comparison=result,
            limitation="Single window; compare with retained full 16-request Qwen sweep"))
        return
    a = window("screen-A", (2, 1, 256), short=True)
    b = window("screen-B", (4, 1, 256), short=True)
    decode = min([r for r in (a, b) if r], key=lambda r: r["seconds"])
    if b and b["seconds"] <= a["seconds"] * 0.9:
        c = window("screen-C", (8, 1, 256), short=True)
        if c and c["seconds"] < decode["seconds"]:
            decode = c
    # Full confirmation needs long-reply headroom; minimal mode screens only.
    adoption_ceiling = 256 if args.minimal else 8192
    decode = min([r for r in records if r["label"] in ("screen-A", "screen-B", "screen-C")
                  and r["status"] == "success" and estimate(r["settings"][0], adoption_ceiling) <= 44 * GIB],
                 key=lambda r: r["seconds"])
    d = window("screen-D", (decode["settings"][0], 2, 256), short=True)
    prefill = min([r for r in (decode, d) if r], key=lambda r: r["seconds"])
    window("screen-E", (*prefill["settings"][:2], 1024), short=True)
    if args.minimal:
        candidates = [r for r in records if r["label"].startswith("screen-") and r["status"] == "success"]
        fastest = min(r["seconds"] for r in candidates)
        choice = min([r for r in candidates if r["seconds"] <= fastest * 1.05], key=lambda r: r["settings"])
        g.save(directory / "summary.json", dict(selected=choice,
            limitation="Single fixed-length screening windows only; provisional, no natural completion confirmation"))
        return
    candidates = sorted([r for r in records if r["label"].startswith("screen-") and r["status"] == "success"
                         and estimate(r["settings"][0], 8192) <= 44 * GIB], key=lambda r: r["seconds"])[:2]
    if len(candidates) != 2:
        raise RuntimeError("Need two memory-admissible finalists")
    for repeat in range(3):
        for i in (range(2) if repeat % 2 == 0 else reversed(range(2))):
            window(f"confirm-{i}-{repeat}", candidates[i]["settings"])
    finalists = []
    for i, candidate in enumerate(candidates):
        runs = [r for r in records if r["label"].startswith(f"confirm-{i}-")]
        finalists.append(dict(settings=candidate["settings"], median_seconds=statistics.median(r["seconds"] for r in runs),
            range_seconds=[min(r["seconds"] for r in runs), max(r["seconds"] for r in runs)],
            tokens=[r["tokens"] for r in runs], lengths=[r["lengths"] for r in runs]))
    fastest = min(r["median_seconds"] for r in finalists)
    choice = min([r for r in finalists if r["median_seconds"] <= fastest * 1.05], key=lambda r: r["settings"])
    g.save(directory / "summary.json", dict(finalists=finalists, selected=choice,
        limitation="Timing selection only; no quality qualification or longer-context capacity claim"))
    print(json.dumps(dict(selected=choice)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("output")
    parser.add_argument("--sweep", action="store_true", help="Run the original adaptive sweep and full confirmation")
    parser.add_argument("--show-settings", action="store_true", help="Print the selected profile without loading a model")
    parser.add_argument("--minimal", action="store_true", help="Eight-request screening only; no confirmation")
    parser.add_argument("--chunk-only", action="store_true", help="Finish the interrupted Qwen 8/2/1024 comparison")
    parser.add_argument("--capacity", action="store_true", help="Gemma batch 4, then batch 8 if cache/workspace estimate fits")
    parser.add_argument("--probe-settings", nargs=3, type=int, metavar=("C", "P", "S"), help="Run one explicit screening setting")
    parser.add_argument("--short-only", action="store_true", help="Use the eight original short prompts for capacity qualification")
    args = parser.parse_args()
    if args.show_settings:
        profile = selected_profile(args)
        print(json.dumps(dict(profile=profile, settings=json.loads(PROFILES.read_text())["profiles"].get(profile)), indent=2))
    else:
        run(args)
