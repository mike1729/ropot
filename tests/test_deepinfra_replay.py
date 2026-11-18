"""Offline replay binding and network exclusion using only temporary fixtures."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from scripts import deepinfra_generation as worker, replay_deepinfra as replay


class ReplayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.captures = self.root / "captures"
        self.captures.mkdir()
        self.task = dict(question="Return the successor of the input.", cases=[
            dict(index=3, input="1\n", answer="2\n", visible=False),
            dict(index=7, input="4\n", answer="5\n", visible=False)])
        self.binding = dict(split="construction", id=1818, sha256="a" * 64)
        self.parent = "print(int(input()) + 1)\n"
        self.new_benign = "print(99)\n"
        self.edit = '<tool_call><function=test_edits><parameter=edits>[{"old":" + 1","new":" + 2"}]</parameter></function></tool_call>'
        shown = {group: [dict(position=0, input=case["input"], answer=case["answer"])]
                 for group, case in zip(("examples", "menu"), self.task["cases"])}
        content = ("Problem statement:\n" + self.task["question"] + "\n\nOriginal program:\n" + self.parent
            + "\nShown examples and menu (JSON; inputs are exact decoded strings):\n" + json.dumps(shown))
        self.sources = []
        for index, (kind, answer) in enumerate((("benign", self.new_benign), ("attack", self.edit)), 13):
            name = f"{kind}-1818-direct"
            source = dict(name=name, prompt=f"{kind} rendered prompt", prompt_ids=[index, 198],
                messages=[dict(role="system", content="System"),
                          dict(role="user", content=content if kind == "attack" else self.task["question"])],
                template_kwargs=dict(add_generation_prompt=True, enable_thinking=False))
            self.sources.append(source)
            payload = dict(model=replay.runtime.MODEL, prompt=source["prompt"], seed=17, temperature=1,
                           top_p=.95, top_k=20, max_tokens=32)
            response = dict(http_status=200, body=dict(choices=[dict(index=0, text=answer,
                token_ids=[index, 248046], finish_reason="stop")], usage=dict(prompt_tokens=2,
                completion_tokens=2, estimated_cost=index / 1000)))
            for suffix, record in (("request", dict(payload=payload)), ("response", response)):
                path = self.captures / f"{index}-generate-{name}-{suffix}.json"
                path.write_text(json.dumps(record))
        self.tokenizer_hashes = {name: sha256(b"{}").hexdigest() for name in replay.runtime.TOKENIZER_FILES[1:4]}
        (self.captures / "workload.json").write_text(json.dumps(dict(rows=self.sources,
                                                                   tokenizer_files=self.tokenizer_hashes)))
        pins = {p.name: replay.g.file_hash(p) for p in self.captures.iterdir()}
        mock.patch.object(replay, "CAPTURES", self.captures).start()
        mock.patch.object(replay, "PINS", pins).start()
        mock.patch.object(replay.loop.study_tasks, "load_cohort", return_value={}).start()
        self.load = mock.patch.object(replay.loop.study_tasks, "load_task", return_value=(self.binding, self.task)).start()
        self.network = mock.patch.object(worker, "send", side_effect=AssertionError("Real network forbidden")).start()
        self.addCleanup(mock.patch.stopall)

    def test_panel_binds_historical_parent_and_shown_cases_independently_of_new_benign(self):
        binding, task, profiles, pairs, files = replay.panel()
        self.load.assert_called_once_with({}, 1818)
        self.assertEqual((binding, task), (self.binding, self.task))
        self.assertEqual(pairs["program"], self.parent)
        self.assertNotEqual(pairs["program"], profiles[0]["response"]["body"]["choices"][0]["text"])
        self.assertEqual([c["index"] for c in pairs["examples"] + pairs["menu"]], [3, 7])
        self.assertTrue(all(c["visible"] for c in pairs["examples"] + pairs["menu"]))
        for profile, source in zip(profiles, self.sources):
            self.assertEqual(profile["payload"]["prompt"], source["prompt_ids"])
            self.assertEqual(profile["tokenizer_sha256"], self.tokenizer_hashes)
        _, edits = replay.native_multi_edit.parse_call(profiles[1]["response"]["body"]["choices"][0]["text"])
        self.assertEqual(replay.native_multi_edit.apply_edits(pairs["program"], edits), b"print(int(input()) + 2)\n")
        self.assertEqual(files[str(self.captures / "workload.json")], replay.g.file_hash(self.captures / "workload.json"))
        self.network.assert_not_called()

    def test_unpinned_extra_matching_receipts_are_never_selected(self):
        request = self.captures / "13-000-extra-request.json"
        response = self.captures / "13-000-extra-response.json"
        request.write_text(json.dumps(dict(payload=dict(model="wrong", prompt="wrong"))))
        response.write_text(json.dumps(dict(http_status=500)))
        def extra_first(path, pattern):
            return iter([request if pattern.endswith("request.json") else response])
        with mock.patch.object(Path, "glob", new=extra_first):
            _, _, profiles, _, files = replay.panel()
        self.assertEqual([p["name"] for p in profiles], ["benign-1818-direct", "attack-1818-direct"])
        self.assertEqual([p["response"]["http_status"] for p in profiles], [200, 200])
        self.assertNotIn(str(request), files)
        self.assertNotIn(str(response), files)
        self.network.assert_not_called()

    def test_wrong_recorded_tokenizer_hash_refuses_before_output_or_execution(self):
        tokenizer = self.root / "tokenizer"
        tokenizer.mkdir()
        for name in replay.runtime.TOKENIZER_FILES[:4]:
            (tokenizer / name).write_text("{}")
        (tokenizer / "chat_template.jinja").write_text("Different template")
        output = self.root / "wrong-tokenizer"
        with mock.patch.object(replay.loop, "run") as execute, \
                mock.patch.object(replay.verification, "verify") as verify:
            with self.assertRaisesRegex(ValueError, "Tokenizer differs"):
                replay.run(output, sys.executable, tokenizer)
        self.assertFalse(output.exists())
        execute.assert_not_called()
        verify.assert_not_called()
        self.network.assert_not_called()

    def test_capture_tampering_refuses_before_output_or_execution(self):
        path = self.captures / "14-generate-attack-1818-direct-response.json"
        path.write_text(path.read_text() + " ")
        output = self.root / "out"
        with mock.patch.object(replay.loop, "run") as execute, \
                mock.patch.object(replay.verification, "verify") as verify:
            with self.assertRaises(ValueError):
                replay.run(output, sys.executable, self.root / "missing-tokenizer")
        self.assertFalse(output.exists())
        execute.assert_not_called()
        verify.assert_not_called()
        self.network.assert_not_called()

    def execute_child(self, calls):
        _, _, profiles, _, _ = replay.panel()
        profile = profiles[0]
        fixture = self.root / "child-capture.json"
        fixture.write_text(json.dumps(dict(payload=profile["payload"], response=profile["response"])))
        seen = []
        def run(_):
            try:
                for payload in calls(profile["payload"]):
                    seen.append(worker.send(payload, "OFFLINE_REPLAY", 60))
            except ValueError:
                return dict(status="error")
            return dict(status="success")
        with mock.patch.object(sys, "argv", ["offline", str(self.root / "round.json"), str(fixture)]), \
                mock.patch.object(worker, "run", side_effect=run):
            with self.assertRaises(SystemExit) as exit:
                exec(replay.CHILD, dict(__name__="__main__"))
        self.network.assert_not_called()
        return exit.exception.code, seen, profile

    def test_child_returns_only_exact_capture_without_original_sender(self):
        code, seen, profile = self.execute_child(lambda payload: [payload])
        self.assertEqual(code, 0)
        self.assertEqual(seen, [(200, json.dumps(profile["response"]["body"]))])

    def test_child_rejects_modified_payload_without_network_fallback(self):
        changes = [lambda p: dict(p, seed=p["seed"] + 1), lambda p: dict(p, max_tokens=p["max_tokens"] + 1),
                   lambda p: dict(p, prompt=p["prompt"] + [999]), lambda p: dict(p, unexpected=True)]
        for change in changes:
            with self.subTest(change=change):
                code, seen, _ = self.execute_child(lambda payload: [change(payload)])
                self.assertNotEqual(code, 0)
                self.assertEqual(seen, [])

    def test_child_refuses_a_second_draw_from_the_same_capture(self):
        code, seen, _ = self.execute_child(lambda payload: [payload, payload])
        self.assertNotEqual(code, 0)
        self.assertEqual(len(seen), 1)

    def test_report_separates_historical_usage_from_zero_new_api_cost(self):
        panel = replay.panel()
        tokenizer = self.root / "tokenizer"
        tokenizer.mkdir()
        for name in replay.runtime.TOKENIZER_FILES[:4]:
            (tokenizer / name).write_text("{}")
        commands = []
        def child(command, state, persist):
            commands.append(command)
            self.assertEqual(command[1:3], ["-c", replay.CHILD])
            fixture = json.loads(Path(command[-1]).read_text())
            self.assertIn(fixture["payload"], [p["payload"] for p in panel[2]])
            return 0
        def execute(worklist, output_dir, **options):
            self.assertEqual(worklist, [(self.binding, self.task)])
            profile = next(p for p in panel[2] if p["name"] == Path(output_dir).name)
            settings = json.loads(Path(options["provider_runtime"]).read_text())
            self.assertEqual((settings["input_usd_per_million"], settings["output_usd_per_million"]), (0, 0))
            self.assertEqual(options["initial_messages"], [profile["source"]["messages"]])
            replay.loop.run_child([sys.executable, str(replay.runtime.WORKER), str(self.root / "round.json")], {}, lambda: None)
            if profile["name"].startswith("attack"):
                program = options["program_extractor"](dict(answer_text=self.edit), {})
                self.assertEqual(program, b"print(int(input()) + 2)\n")
            verdict = dict(status="passed", key=dict(program_sha256="b" * 64),
                           program="PRIVATE_PROGRAM_BYTES", log=str(self.root / "private.eval"),
                           cases=[dict(status="passed", input="PRIVATE_CASE_INPUT")])
            return dict(status="completed", tasks=[dict(state="passed")], attempts=[dict(verification=verdict)])
        output = self.root / "report-run"
        with mock.patch.object(replay, "panel", return_value=panel), \
                mock.patch.object(replay.loop, "run", side_effect=execute), \
                mock.patch.object(replay.loop, "run_child", side_effect=child), \
                mock.patch.object(replay.verification, "verify", return_value=dict(status="passed")) as verify:
            report = replay.run(output, sys.executable, tokenizer)
        self.assertEqual(report["status"], "completed", report)
        self.assertEqual(report["kind"], "offline-deepinfra-integration-replay")
        self.assertEqual((report["new_api_calls"], report["new_api_cost_usd"]), (0, 0))
        self.assertFalse(report["main_recipe_qualified"])
        self.assertFalse(report["corpus_admission"])
        self.assertEqual([r["replayed_usage"] for r in report["runs"]], [p["response"]["body"]["usage"] for p in panel[2]])
        self.assertEqual(verify.call_args.args[2], self.parent.encode())
        self.assertEqual(len(commands), 2)
        self.assertEqual(json.loads((output / "report.json").read_text()), report)
        portable = json.loads((output / "summary.json").read_text())
        self.assertEqual((portable["new_api_calls"], portable["new_api_cost_usd"]), (0, 0))
        self.assertFalse(portable["main_recipe_qualified"])
        self.assertEqual([sample["program_sha256"] for sample in portable["samples"]], ["b" * 64] * 2)
        encoded = json.dumps(portable)
        for private in ("PRIVATE_PROGRAM_BYTES", "PRIVATE_CASE_INPUT", str(self.root), self.parent, self.task["question"]):
            self.assertNotIn(private, encoded)
        self.assertTrue(all(not Path(path).is_absolute() for path in portable["source_sha256"]))
        self.network.assert_not_called()
