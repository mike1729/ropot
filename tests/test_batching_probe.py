"""Probe workload integration with the shared thinking-aware renderer."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests.test_thinking_generation import ThinkingTokenizer

probe = importlib.import_module("scripts.batching_probe")


class ProbeTests(unittest.TestCase):
    def test_frozen_requests_enable_thinking_without_a_render_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "capability/actor/generation-0000"
            source.mkdir(parents=True)
            initial = [dict(messages=[dict(role="user", content=f"Initial {i}")],
                            seed=i, enable_thinking=False) for i in range(8)]
            probe.g.save(source / "inputs.json", dict(requests=initial))
            probe.g.save(source / "outcome.json", {})
            long_path = root / "data/03_throughput/q8-followups-20261025-v1/actor-long/7k-calibration/inputs.json"
            long_path.parent.mkdir(parents=True)
            probe.g.save(long_path, dict(requests=[dict(messages=[dict(role="user", content="Repair")])] * 8))
            with mock.patch.object(probe, "ROOT", root), mock.patch.object(probe, "BASE", root / "capability"):
                requests = probe.freeze(root)
            self.assertEqual(json.loads((root / "workload.json").read_text())["requests"], requests)
            self.assertEqual([r["seed"] for r in requests], list(range(41, 57)))
            self.assertTrue(all(r["enable_thinking"] is True for r in requests))
            self.assertEqual([r["messages"] for r in requests[:8]], [r["messages"] for r in initial])
            self.assertEqual(requests[8]["messages"], [dict(role="user", content="user:\nRepair")])
            tokenizer = ThinkingTokenizer("qwen")
            row = probe.g.render(tokenizer, requests[0]["messages"], enable_thinking=requests[0]["enable_thinking"])
            row.update(text="Private reasoning.\n</think>" + tokenizer.pieces[12], tokens=[10, 31, 12, 99])
            probe.g.split_answer(tokenizer, row)
            self.assertEqual(row["answer_text"], tokenizer.pieces[12])
