"""Stage 04 loading, exact inputs and private agreement evidence; no model calls."""
from contextlib import chdir
from copy import deepcopy
from hashlib import sha256
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

w = importlib.import_module("src.data-pipeline.stage04.witnesses")
try:
    runner = importlib.import_module("src.data-pipeline.stage03.inspect_runner")
except ModuleNotFoundError as error:
    if error.name != "inspect_ai":
        raise
    runner = None


class WorklistTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.benign, self.cohort_dir, self.original = [self.root / name for name in ("benign", "cohort", "original")]
        for path in (self.benign / "programs", self.cohort_dir / "tasks", self.original / "tasks"):
            path.mkdir(parents=True)
        self.bindings, self.tasks = {}, {}
        self.program = b"# exact program\r\nimport sys\r\nprint(sum(map(int, sys.stdin.read().split())))\r\n"
        self.solutions = ["import sys\nprint(sum(map(int,sys.stdin.read().split())))\n", "unused",
                          "import sys\na=list(map(int,sys.stdin.read().split())); print(sum(a))\n",
                          "import sys\ntotal=0\nfor s in sys.stdin.read().split(): total+=int(s)\nprint(total)\n"]
        for identifier, owner, block in ((1, "development", "initial"), (2, "construction", "initial"),
                                          (3, "evaluation", "initial"), (4, "construction", "deferred")):
            task = dict(split="test", id=identifier, question=f"Sum the integers. Task {identifier}.",
                        solutions=self.solutions, passing_reference_indices=[3, 0, 2],
                        cases=[dict(index=i, input=text) for i, text in enumerate(("2\n", "5\n", "9 10\n"))])
            full_raw = json.dumps(task).encode()
            task = dict(task, cases=task["cases"][:2])
            raw = json.dumps(task).encode()
            relative = f"tasks/test_{identifier}.json"
            (self.original / relative).write_bytes(full_raw)
            (self.cohort_dir / relative).write_bytes(raw)
            self.tasks[identifier] = task
            self.bindings[identifier] = dict(split="test", id=identifier, path=relative, owner=owner, block=block,
                sha256=sha256(raw).hexdigest(), source_sha256=sha256(full_raw).hexdigest())
        self.cohort = dict(tasks=list(self.bindings.values()), provenance=dict(input_manifest_sha256="a" * 64),
            groups={owner: dict(initial=[i for i, b in self.bindings.items() if b["owner"] == owner and b["block"] == "initial"],
                                reserve=[]) for owner in ("construction", "development", "evaluation")},
            construction_phases={})
        self.selections = []
        for identifier, source in ((1, "actor"), (1, "synthetic"), (2, "actor")):
            binding = self.bindings[identifier]
            name = f"programs/{source}-{identifier}.py"
            (self.benign / name).write_bytes(self.program)
            self.selections.append(dict(binding=deepcopy(binding), source=source, program=name,
                program_sha256=sha256(self.program).hexdigest(),
                candidate=w.study_tasks.candidate_identity("test", identifier, binding["sha256"], source, self.program)))
        cohort_raw = json.dumps(self.cohort).encode()
        (self.cohort_dir / "manifest.json").write_bytes(cohort_raw)
        patcher = mock.patch.object(w.study_tasks, "COHORT_SHA256", sha256(cohort_raw).hexdigest())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.manifest = dict(cohort_sha256=w.study_tasks.COHORT_SHA256, selections=self.selections)
        self.pin_selection()

    def pin_selection(self):
        raw = json.dumps(self.manifest).encode()
        (self.benign / "manifest.json").write_bytes(raw)
        patcher = mock.patch.object(w, "SELECTION_SHA256", sha256(raw).hexdigest())
        patcher.start()
        self.addCleanup(patcher.stop)

    def load(self, keys=(("test", 1, "actor"),)):
        return w.load_worklist(keys, benign_dir=self.benign, cohort_dir=self.cohort_dir, original_dir=self.original)

    def test_order_exact_bytes_all_stored_inputs_and_reference_selection(self):
        rows = self.load([("test", 2, "actor"), ("test", 1, "synthetic"), ("test", 1, "actor")])
        self.assertEqual([(r["selection"]["binding"]["id"], r["selection"]["source"]) for r in rows],
                         [(2, "actor"), (1, "synthetic"), (1, "actor")])
        self.assertEqual(rows[0]["program"], self.program)
        self.assertEqual(rows[0]["task"], self.tasks[2])
        self.assertEqual(rows[0]["stored_inputs"], ["2\n", "5\n", "9 10\n"])
        self.assertEqual(rows[0]["references"], [self.solutions[i].encode() for i in (3, 0, 2)])
        duplicate = w.check_many([(rows[0], "9\t10")], self.root / "unused")
        self.assertEqual(duplicate[0]["status"], "duplicate")
        self.assertFalse((self.root / "unused").exists())

    def test_only_requested_task_artifacts_are_read(self):
        for path in (self.cohort_dir / "tasks/test_2.json", self.original / "tasks/test_2.json",
                     self.benign / "programs/actor-2.py"):
            path.unlink()
        self.assertEqual(len(self.load()), 1)
        with self.assertRaises(FileNotFoundError):
            self.load([("test", 2, "actor")])

    def test_default_benign_publication_loads_from_another_working_directory(self):
        publication = Path(w.__file__).resolve().parents[3] / "data/03_benigns/v1"
        read = Path.read_bytes
        def fixture_read(path):
            if path.is_relative_to(publication):
                path = self.benign / path.relative_to(publication)
            return read(path)
        with chdir(self.root), mock.patch.object(Path, "read_bytes", autospec=True, side_effect=fixture_read):
            rows = w.load_worklist([("test", 1, "actor")], cohort_dir=self.cohort_dir, original_dir=self.original)
        self.assertEqual(rows[0]["program"], self.program)

    def test_duplicate_unselected_and_invalid_key_identities(self):
        for keys in ([('test', 1, 'actor')] * 2, [('test', 2, 'synthetic')], [('train', 1, 'actor')],
                     [('test', 999, 'actor')], [('test', True, 'actor')]):
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                self.load(keys)

    def test_changed_selection_cohort_program_reduced_and_full_task_bytes(self):
        for path in (self.benign / "manifest.json", self.cohort_dir / "manifest.json",
                     self.benign / "programs/actor-1.py", self.cohort_dir / "tasks/test_1.json",
                     self.original / "tasks/test_1.json"):
            raw = path.read_bytes()
            path.write_bytes(raw + b" ")
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.load()
            path.write_bytes(raw)

    def test_protected_and_deferred_tasks_remain_unavailable(self):
        for identifier in (3, 4):
            selection = deepcopy(self.selections[0])
            selection["binding"] = self.bindings[identifier]
            selection["candidate"] = w.study_tasks.candidate_identity("test", identifier,
                self.bindings[identifier]["sha256"], "actor", self.program)
            self.selections.append(selection)
        self.pin_selection()
        for identifier in (3, 4):
            with self.subTest(identifier=identifier), self.assertRaisesRegex(ValueError, "not in the saved"):
                self.load([("test", identifier, "actor")])

class ExtractionTests(unittest.TestCase):
    def test_inline_payload_is_preserved_without_framing(self):
        for payload in ("7", " \t7", "\r7", "3\n1 2 3\n", "é\n"):
            with self.subTest(payload=repr(payload)):
                self.assertEqual(w.extract_input(dict(status="stop", text=f"<witness>{payload}</witness>")), payload)

    def test_exact_payload_after_lf_or_crlf_framing_without_row_mutation(self):
        for payload in ("3\n1 2 3\n", " \t3\r\n1 2 3 \t", "\n\n5", "7", "é\n"):
            for framing in ("\n", "\r\n"):
                row = dict(status="stop", text=f" \t<witness>{framing}{payload}</witness>\r\n")
                before = deepcopy(row)
                with self.subTest(payload=payload, framing=framing):
                    self.assertEqual(w.extract_input(row).encode(), payload.encode())
                    self.assertEqual(row, before)

    def test_empty_and_whitespace_only_payloads_are_format_failures(self):
        for payload in ("", " ", "\t\r\n\n", "\u2003"):
            for framing in ("", "\n", "\r\n"):
                with self.subTest(payload=repr(payload), framing=repr(framing)), self.assertRaisesRegex(ValueError, "blank input"):
                    w.extract_input(dict(status="stop", text=f"<witness>{framing}{payload}</witness>"))

    def test_incomplete_malformed_ambiguous_and_invalid_encoding_fail(self):
        texts = ("<witness>\n7",
                 "7</witness>", "<witness>\n7</witness><witness>\n8</witness>",
                 "<witness>\n<witness>7</witness>", "</witness><witness>\n7",
                 "Here it is: <witness>\n7</witness>", "```\n<witness>\n7</witness>\n```",
                 "<witness>\n\ud800</witness>")
        for text in texts:
            with self.subTest(text=repr(text)), self.assertRaises(ValueError):
                w.extract_input(dict(status="stop", text=text))
        for status in ("length", "error", "cancelled", "not_run"):
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "natural completion"):
                w.extract_input(dict(status=status, text="<witness>\n7</witness>"))


@unittest.skipUnless((w.BENIGN_DIR / "manifest.json").exists(), "requires pinned Stage 03 publication")
class ExhaustionProofTests(unittest.TestCase):
    def test_entire_valid_integer_domains_are_in_pinned_stored_cases(self):
        domains = {291: {(a, b) for a in range(1, 11) for b in range(a, 11)},
                   4347: {(n,) for n in range(2, 21, 2)}}
        for digest, proof in w.EXHAUSTED_TASKS.items():
            with self.subTest(task=proof["id"]):
                rec = w.load_worklist([(proof["split"], proof["id"], "actor")])[0]
                self.assertEqual(rec["selection"]["candidate"]["task_sha256"], digest)
                self.assertEqual(rec["selection"]["binding"]["source_sha256"], proof["source_sha256"])
                domain = domains[proof["id"]]
                self.assertEqual(len(domain), proof["valid_inputs"])
                self.assertTrue(domain <= {tuple(map(int, value.split())) for value in rec["stored_inputs"]})


def record(identifier=7, source="actor"):
    program = f"print({identifier})\n".encode()
    return dict(selection=dict(candidate=w.study_tasks.candidate_identity("test", identifier, "a" * 64, source, program)),
                program=program, references=[f"print({identifier}) # ref{i}\n".encode() for i in range(3)],
                stored_inputs=["2\n", "9 10\n"])


@unittest.skipIf(runner is None, "requires requirements-execution.txt")
class AgreementTests(unittest.TestCase):
    def setUp(self):
        self.change = lambda log: None
        self.run = mock.patch.object(runner, "run", side_effect=self.native).start()
        self.addCleanup(mock.patch.stopall)

    def native(self, items, directory, **kwargs):
        samples = []
        for i, item in enumerate(items):
            answer = str(item["verification"]["candidate"]["id"])
            result = dict(index=0, status="completed", stdout=answer+"\n", stderr="", returncode=0, elapsed_seconds=0.01)
            samples.append(NS(id=i, error=None, store=dict(cases=[result])))
        self.log = NS(status="success", error=None, location="fixture.eval", samples=samples)
        self.change(self.log)
        return [self.log]

    def test_whole_round_exact_requests_source_binding_and_duplicate_positions(self):
        first, second = record(), record(8, "synthetic")
        exact = " \t7\r\n"
        proposals = [(first, "9\t10"), (first, exact), (second, "8"), (first, "2")]
        before = deepcopy(proposals)
        self.change = lambda log: log.samples.reverse()  # native ordering is not an association rule
        rows = w.check_many(proposals, "unused")
        self.assertEqual([r["status"] for r in rows], ["duplicate", "execution_agrees", "execution_agrees", "duplicate"])
        self.assertEqual(proposals, before)
        self.run.assert_called_once()
        items = self.run.call_args.args[0]
        self.assertEqual(len(items), 8)
        self.assertEqual(self.run.call_args.kwargs, dict(check_answers=False, max_samples=4, concurrent_inference=False))
        for j, (rec, stdin) in enumerate(((first, exact), (second, "8"))):
            row = rows[j + 1]
            self.assertEqual(row["answer"], str(rec["selection"]["candidate"]["id"]))
            self.assertEqual(row["key"]["candidate"], rec["selection"]["candidate"])
            self.assertEqual(row["key"]["stdin_sha256"], sha256(stdin.encode()).hexdigest())
            self.assertEqual(row["log"], "fixture.eval")
            for k, role in enumerate(w.ROLES):
                self.assertEqual(items[4*j+k]["cases"], [dict(index=0, input=stdin)])
                self.assertEqual(row["programs"][role]["sample_id"], 4*j+k)
                self.assertEqual(items[4*j+k]["verification"]["role"], role)
        self.assertEqual(set(rows[0]), {"status", "key"})  # no matched stored input/index
        self.change = lambda log: log.samples[0].store["cases"][0].update(stdout="99\n")
        mixed = w.check_many([(first, exact), (second, "8")], "unused")
        self.assertEqual([r["status"] for r in mixed], ["reference_disagreement", "execution_agrees"])
        self.assertEqual(mixed[1]["answer"], "8")

    def test_all_duplicates_skip_native_execution_but_changed_tokens_execute(self):
        rec = record()
        self.assertEqual([r["status"] for r in w.check_many([(rec, "9 \n10\t"), (rec, " 2\r\n")], "unused")],
                         ["duplicate", "duplicate"])
        self.assertEqual(w.check_many([], "unused"), [])
        self.run.assert_not_called()
        w.check_many([(rec, "02"), (rec, "10 9")], "unused")
        self.run.assert_called_once()

    def test_distinct_disagreement_and_execution_failure_outcomes(self):
        for role in range(4):
            for changes in (dict(status="timeout", stdout=None, stderr=None, returncode=None),
                            dict(status="nonzero_exit", returncode=3),
                            dict(status="stdout_limit", stdout=None, stderr=None),
                            dict(status="stderr_limit", stdout=None, stderr=None),
                            dict(status="output_limit", stdout=None, stderr=None),
                            dict(status="invalid_stdout", stdout="\ufffd"),
                            dict(stdout="00\n"), dict(stdout="+7"), dict(stdout="7 7"), dict(stdout="")):
                self.change = lambda log: log.samples[role].store["cases"][0].update(changes)
                with self.subTest(role=role, changes=list(changes)):
                    row = w.check_many([(record(), "7")], "unused")[0]
                    self.assertEqual(row["status"], "reference_execution_failure" if role < 3 else "benign_execution_failure")
                    self.assertIsNone(row["programs"][w.ROLES[role]]["answer"])
        for role in range(4):
            self.change = lambda log: log.samples[role].store["cases"][0].update(stdout="8\n")
            with self.subTest(role=role):
                row = w.check_many([(record(), "7")], "unused")[0]
                self.assertEqual(row["status"], "reference_disagreement" if role < 3 else "benign_disagreement")
                self.assertEqual(row["programs"][w.ROLES[role]]["result"]["stdout"], "8\n")

    def test_surrounding_whitespace_negative_and_large_canonical_integer_answers(self):
        for answer in ("0", "-7", "9" * 5000):
            def change(log):
                for sample in log.samples:
                    sample.store["cases"][0]["stdout"] = f" \n{answer}\t"
            self.change = change
            with self.subTest(answer_length=len(answer)):
                row = w.check_many([(record(), "7")], "unused")[0]
                self.assertEqual((row["status"], row["answer"]), ("execution_agrees", answer))

    def test_missing_native_samples_or_captures_stop_the_check(self):
        for change, error in ((lambda l: l.samples.pop(), KeyError),
                              (lambda l: l.samples[0].store["cases"].clear(), IndexError)):
            self.change = change
            with self.subTest(change=change), self.assertRaises(error):
                w.check_many([(record(), "7")], "unused")

    def test_infrastructure_errors_stop_without_retry_and_cancellation_propagates(self):
        for change in (lambda l: setattr(l, "status", "error"), lambda l: setattr(l, "error", NS(message="infra")),
                       lambda l: setattr(l.samples[-1], "error", NS(message="infra"))):
            self.change = change
            self.run.reset_mock()
            with self.assertRaisesRegex(RuntimeError, "fixture.eval"):
                w.check_many([(record(), "7"), (record(8), "8")], "unused")
            self.run.assert_called_once()
        for error in (OSError("setup"), KeyboardInterrupt("native cleanup done")):
            self.run.reset_mock()
            self.run.side_effect = error
            with self.assertRaises(type(error)):
                w.check_many([(record(), "7")], "unused")
            self.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
