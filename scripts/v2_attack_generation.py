"""Bounded attack generation worker with resource guards; not a qualification fixture.

Called by generation_loop with one round JSON. The acquisition's resource baseline is
shared across every source/chain/round, so swap growth is never reset per worker.
Only this process patches the native generator; the shared backend stays unchanged.
"""
import importlib
import json
from pathlib import Path
import sys
from time import monotonic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(round_file):
    attacks = importlib.import_module("src.data-pipeline.stage05.attacks")
    generation = attacks.generation
    specification = attacks.generation_loop.read(round_file)
    output = Path(specification["output_dir"])
    baseline = attacks.generation_loop.read(output.parent.parent / "resource-baseline.json")
    trace = output.with_name(output.name + "-memory.jsonl")
    initial = attacks.memory_state()
    generation.save(output.with_name(output.name + "-guard.json"), dict(
        baseline=baseline, initial=initial, driver_sha256=generation.file_hash(__file__),
        peak_stop_bytes=44 * 1024**3, swap_growth_stop_mib=baseline.get('swap_growth_stop_mib', 256), critical_pressure=4))
    attacks.check_resources(initial, baseline)
    import mlx.core as mx
    native = importlib.import_module("mlx_lm.generate")
    original = native.BatchGenerator

    def check():
        state = dict(attacks.memory_state(), peak_bytes=mx.get_peak_memory(), monotonic_seconds=monotonic())
        with trace.open("a") as stream:
            stream.write(json.dumps(state) + "\n")
        attacks.check_resources(state, baseline)

    class Guarded(original):
        def insert(self, *args, **kwargs):
            check()
            return super().insert(*args, **kwargs)

        def next(self):
            check()
            return super().next()

    native.BatchGenerator = Guarded
    try:
        result = generation.generate_batched(**specification)
    finally:
        native.BatchGenerator = original
    if result["status"] == "success":
        check()  # Also catch a limit crossed during the final native step.
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
