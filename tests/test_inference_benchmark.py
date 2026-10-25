from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

from experiments import inference_benchmark as b


@dataclass
class Stats:
    generation_tokens: int = 0
    prompt_time: float = .01


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_recorder_replay_detects_corruption_duplicate_ids_and_overbudget(self):
        rec = b.Recorder(self.root/"record", {"fixture": True})
        row = {"id": 0, "max_tokens": 2, "token_ids": [1, 9]}
        rec.append(row, "exact\r\nbytes\n")
        rows, _ = b.replay(rec.directory)
        self.assertEqual(rows[0]["token_ids"], [1, 9])
        with self.assertRaises(ValueError):
            rec.append(row, "again")
        with self.assertRaises(ValueError):
            rec.append({**row, "id": 1, "token_ids": [1, 2, 3]}, "over cap")
        (rec.directory/"reply-0000.txt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            b.replay(rec.directory)
        with self.assertRaises(FileExistsError):
            b.Recorder(rec.directory, {})

    def fake_runtime(self, fail=False, cleanup=False):
        owner = self
        class Generator:
            def __init__(self, model, **kwargs):
                owner.options = kwargs
                self.calls, self.stats_row = 0, Stats()
                owner.generator = self
            def insert(self, prompts, caps, samplers, **options):
                owner.samplers, owner.caps = samplers, caps
                owner.insertion_options = options
                return [42, 7, 99]
            @contextmanager
            def stats(self):
                yield self.stats_row
            def next(self):
                self.calls += 1
                if fail and self.calls == 2:
                    raise RuntimeError("injected native failure")
                if self.calls == 1:
                    responses = [NS(uid=42, token=1, finish_reason=None),
                                 NS(uid=7, token=9, finish_reason="stop")]
                elif self.calls == 2:
                    responses = [NS(uid=42, token=9, finish_reason="stop"),
                                 NS(uid=99, token=2, finish_reason=None)]
                else:
                    responses = [NS(uid=99, token=3, finish_reason="length")]
                self.stats_row.generation_tokens += len(responses)
                return [NS(uid=r.uid) for r in responses], responses
            def close(self):
                owner.closed = True
                if cleanup:
                    raise RuntimeError("injected cleanup failure")
        mx = NS(synchronize=lambda: None, reset_peak_memory=lambda: None, get_peak_memory=lambda: 123)
        return {"mlx": NS(core=mx), "mlx.core": mx, "mlx_lm": NS(),
                "mlx_lm.generate": NS(BatchGenerator=Generator),
                "mlx_lm.sample_utils": NS(make_sampler=lambda **kw: NS(**kw))}

    def invoke(self, name="window", prefix=0, max_peak_bytes=None, **failures):
        rows = [{"id": i, "kind": "fixture", "max_tokens": 2, "seed": 17+i,
                 "temperature": .2, "top_p": .95, "prompt_ids": [5+i]} for i in range(3)]
        with mock.patch.dict("sys.modules", self.fake_runtime(**failures)), \
             mock.patch.object(b, "resources", return_value={}):
            return b.window(object(), NS(eos_token_ids=[9], decode=lambda ids: str(ids)), rows,
                dict(completion_batch_size=2, prefill_batch_size=2, prefill_step_size=256), self.root/name,
                reuse_prefix_tokens=prefix, max_peak_bytes=max_peak_bytes)

    def test_peak_memory_abort_preserves_partial_rows_and_closes_generator(self):
        with self.assertRaisesRegex(MemoryError, "peak-memory"):
            self.invoke(max_peak_bytes=122)
        rows, _ = b.replay(self.root/"window")
        self.assertEqual({r['id']:r['status'] for r in rows}, {0:'interrupted',1:'stop',2:'not_run'})
        self.assertTrue(self.closed)

    def test_prefix_options_are_forwarded_and_shared_state_is_checked_after_cleanup(self):
        prefix = NS(build_seconds=0., insertion=mock.Mock(return_value=([[5], [6], [7]],
            {"caches": ["one", "two", "three"], "all_tokens": [[4], [4], [4]]})),
            check=mock.Mock(), summary=mock.Mock(return_value={"tokens": 1}))
        with mock.patch("experiments.prefix_cache.ExactPrefixState", return_value=prefix):
            result = self.invoke(prefix=1)
        self.assertEqual(self.insertion_options["caches"], ["one", "two", "three"])
        self.assertEqual(self.insertion_options["all_tokens"], [[4], [4], [4]])
        self.assertEqual(result["prefix"], {"tokens": 1})
        prefix.check.assert_called_once()
        self.assertTrue(self.closed)

    def test_prefix_state_mutation_rejects_an_otherwise_completed_window(self):
        prefix = NS(build_seconds=0., insertion=lambda requests: ([r["prompt_ids"] for r in requests], {}),
            check=mock.Mock(side_effect=RuntimeError("shared cache mutated")), summary=lambda: {"tokens": 1})
        with mock.patch("experiments.prefix_cache.ExactPrefixState", return_value=prefix):
            with self.assertRaisesRegex(RuntimeError, "mutated"):
                self.invoke(prefix=1)
        result = json.loads((self.root/"window/summary.json").read_text())
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["completed"], 3)
        self.assertEqual(len(b.replay(self.root/"window")[0]), 3)
        self.assertTrue(self.closed)

    def test_uid_association_eos_counts_capacities_and_seed_ownership(self):
        result = self.invoke()
        rows, _ = b.replay(self.root/"window")
        rows.sort(key=lambda r: r["id"])
        self.assertEqual([r["token_ids"] for r in rows], [[1, 9], [9], [2, 3]])
        self.assertEqual([r["status"] for r in rows], ["stop", "stop", "length"])
        self.assertEqual(result["delivered_tokens"], 5)
        self.assertEqual(result["max_observed_decode_batch"], 2)
        self.assertEqual(self.options["prefill_batch_size"], 2)
        self.assertEqual(self.options["completion_batch_size"], 2)
        self.assertEqual([s.seed for s in self.samplers], [17, 18, 19])
        self.assertTrue(self.closed)
        # A downstream consumer can fail, then read every saved reply without a redraw.
        calls = self.generator.calls
        with self.assertRaises(RuntimeError):
            for row in rows:
                raise RuntimeError("injected downstream processing failure")
        self.assertEqual(len(b.replay(self.root/"window")[0]), 3)
        self.assertEqual(self.generator.calls, calls)

    def test_native_failure_preserves_settled_partial_and_unstarted_rows(self):
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            self.invoke(fail=True)
        rows, _ = b.replay(self.root/"window")
        rows.sort(key=lambda r: r["id"])
        self.assertEqual([r["status"] for r in rows], ["interrupted", "stop", "not_run"])
        self.assertEqual(rows[0]["token_ids"], [1])
        self.assertTrue(self.closed)
        self.assertEqual(json.loads((self.root/"window/summary.json").read_text())["status"], "error")

    def test_cleanup_failure_is_not_reported_as_success(self):
        with self.assertRaisesRegex(RuntimeError, "cleanup"):
            self.invoke(cleanup=True)
        summary = json.loads((self.root/"window/summary.json").read_text())
        self.assertEqual(summary["status"], "error")

    def test_overhead_run_has_exact_work_sizes_and_explicit_modeled_denominator(self):
        measurements = b.overhead(self.root/"overhead")
        self.assertEqual([r["requests"] for r in measurements], [8, 32, 128])
        self.assertEqual([r["validation_calls"] for r in measurements], [16, 64, 256])
        self.assertEqual([r["assumed_compute_seconds"] for r in measurements], [8, 32, 128])

    def test_every_conversation_alternates_after_its_system_message(self):
        for row in b.corpus():
            roles = [m["role"] for m in row["messages"]][1:]
            self.assertEqual(roles, ["user" if i % 2 == 0 else "assistant" for i in range(len(roles))])
            self.assertEqual(roles[-1], "user")


@unittest.skipUnless(os.environ.get("ROPOT_TOKENIZER_TESTS") == "1", "requires pinned local tokenizers")
class TokenizerTests(unittest.TestCase):
    def test_actual_pinned_templates_accept_all_fabricated_conversations(self):
        from transformers import AutoTokenizer
        root = Path("/Users/michal/Workspace/models/mlx-community")
        for model in ("Qwen3.5-9B-8bit", "Ministral-3-8B-Instruct-2512-8bit"):
            tokenizer = AutoTokenizer.from_pretrained(root/model, local_files_only=True, trust_remote_code=False)
            for row in b.corpus():
                ids = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False, return_dict=False)
                self.assertTrue(ids)
                if tokenizer.bos_token_id is not None:
                    self.assertEqual(ids[0], tokenizer.bos_token_id)


if __name__ == "__main__":
    unittest.main()
