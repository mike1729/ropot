"""Paired endpoints, manual review and cost attribution require no execution."""
from copy import deepcopy
import importlib
import json
import unittest


r = importlib.import_module("src.data-pipeline.stage05.repair_search_report")


def effort(replies, outputs, inputs=100, **changes):
    return dict(committed_replies=replies, committed_tokens=2048 * replies,
                observed_tokens=outputs, input_tokens=inputs, known_submissions=replies,
                usage_complete=True, **changes)


def run(source, arm, successes=(), *, complete=True):
    attempts, first = [], []
    for position, draw in successes:
        attempt = len(attempts)
        digest = f"{position + draw + 1:064x}"
        task_id = (2477, 4608)[position]
        attempts.append(dict(attempt=attempt, position=position, round=draw,
            candidate=dict(id=task_id, source=source, program_sha256=digest),
            verification=dict(status="passed"), extraction="success", submitted=True))
        first.append(dict(task_id=task_id, position=position, round=draw, attempt=attempt,
                          program_sha256=digest, program=f"/{source}-{arm}-{attempt}.py"))
    return dict(source=source, arm=arm, outcome=dict(status="completed" if complete else "error", attempts=attempts),
        diagnostics=dict(first_successes=first, cells={"passed": {"different_integer": len(first)}},
                         duplicates=dict(original=1), case_executions=dict(suite=4, witness=2)),
        effort=effort(3, 30), elapsed_seconds=1.5)


class RepairSearchReportTests(unittest.TestCase):
    def setUp(self):
        self.settings = dict(task_ids=[2477, 4608], historical=dict(
            whole_parent=effort(175, 51545, None), optimization=effort(48, 14481, 53751),
            by_source=dict(actor=effort(24, 6252, 27440), synthetic=effort(24, 8229, 26311))))
        self.runs = [run(source, arm) for source in r.SOURCES for arm in r.ARMS]

    def cell(self, report, source="actor", arm="A"):
        return next(c for c in report["runs"] if (c["source"], c["arm"]) == (source, arm))

    def test_paired_task_endpoints_cumulative_success_and_known_feasible_pair(self):
        self.runs[0] = run("actor", "A", [(0, 1)])
        self.runs[1] = run("actor", "B", [(1, 0)])
        self.runs[2] = run("actor", "C", [(0, 2), (1, 1)])
        report = r.build_report(self.settings, self.runs)
        self.assertTrue(report["complete"])
        self.assertEqual(self.cell(report)["functional"], dict(successes=1, task_ids=[2477], denominator=2,
            complete=True, rate=.5, cumulative=[0, 1, 1]))
        self.assertEqual(self.cell(report, arm="C")["functional"]["cumulative"], [0, 1, 2])
        self.assertEqual(report["paired"]["actor"]["A_B"]["counts"], dict(left_only=1, right_only=1))
        self.assertEqual(report["paired"]["actor"]["B_C"]["counts"], dict(right_only=1, shared_success=1))
        self.assertEqual(report["paired"]["synthetic"]["A_B"]["counts"], dict(shared_failure=2))
        feasible = report["known_feasible_actor_2477"]["arms"]
        self.assertEqual(feasible["A"], dict(functional_success=True, mechanism="pending"))
        self.assertEqual(feasible["B"], dict(functional_success=False, mechanism="not_accepted"))
        self.assertEqual(self.cell(report)["diagnostic_cells"], self.runs[0]["diagnostics"]["cells"])

    def test_pending_review_is_unknown_yield_and_reasons_bind_exact_program(self):
        self.runs[0] = run("actor", "A", [(0, 0), (1, 2)])
        reviews = r.review_template(self.runs)
        self.assertEqual(len(reviews), 2)
        self.assertEqual({row["status"] for row in reviews}, {"pending"})
        report = r.build_report(self.settings, self.runs)
        mechanism = self.cell(report)["mechanism"]
        self.assertEqual((mechanism["adherent"], mechanism["pending"]), (0, 2))
        self.assertIsNone(mechanism["adherent_yield"])
        reviews[0].update(status="adherent", reason="Changes arithmetic through the existing computation.")
        report = r.build_report(self.settings, self.runs, reviews[:1])
        mechanism = self.cell(report)["mechanism"]
        self.assertEqual((mechanism["adherent_lower_bound"], mechanism["pending"]), (1, 1))
        self.assertIsNone(mechanism["adherent_yield"])
        self.assertEqual(report["mechanism_paired"]["actor"]["A_B"]["counts"],
                         dict(left_only=1, unknown=1))
        self.assertEqual(report["decision"], "awaiting_mechanism_review")
        reviews[1].update(status="nonadherent", reason="Recognizes the supplied witness using a constant.")
        before = deepcopy(self.runs)
        report = r.build_report(self.settings, self.runs, reviews)
        self.assertEqual(self.cell(report)["mechanism"]["adherent_yield"], .5)
        self.assertEqual(self.cell(report)["functional"]["successes"], 2)
        self.assertEqual(report["mechanism_paired"]["actor"]["A_B"]["counts"],
                         dict(left_only=1, shared_failure=1))
        self.assertEqual(report["decision"], "review_results_before_replication")
        self.assertEqual(self.runs, before)

    def test_review_unknown_duplicate_identity_and_missing_reasons_are_rejected(self):
        self.runs[0] = run("actor", "A", [(0, 0)])
        row = r.review_template(self.runs)[0]
        bad_rows = [[row, row], [{**row, "program_sha256": "wrong"}],
                    [{**row, "source": "synthetic"}], [{**row, "arm": "B"}],
                    [{**row, "task_id": 4608}], [{**row, "attempt": 99}],
                    [{**row, "status": "adherent", "reason": "  "}],
                    [{**row, "status": "nonadherent", "reason": None}],
                    [{**row, "status": "approved"}], [{"reason": "missing identity"}],
                    [{**row, "attempt": False}], [{**row, "program_sha256": []}]]
        for reviews in bad_rows:
            with self.subTest(reviews=reviews), self.assertRaises(ValueError):
                r.build_report(self.settings, self.runs, reviews)

    def test_new_spending_is_separate_from_shared_history_and_policy_attribution(self):
        report = r.build_report(self.settings, self.runs)
        totals = report["effort"]["new_spending"]
        self.assertEqual((totals["committed_replies"], totals["observed_tokens"]), (18, 180))
        self.assertEqual(report["effort"]["historical_shared"]["whole_parent"]["committed_replies"], 175)
        costs = {(c["source"], c["arm"]): c for c in report["effort"]["policy_attribution"]}
        for arm in ("A", "B"):
            self.assertEqual(costs["actor", arm]["inherited"]["committed_replies"], 24)
            self.assertEqual(costs["actor", arm]["total"]["observed_tokens"], 6282)
            self.assertTrue(costs["actor", arm]["attribution_only"])
        self.assertEqual(costs["actor", "C"]["inherited"]["committed_replies"], 0)
        self.assertEqual(costs["actor", "C"]["total"]["observed_tokens"], 30)
        self.assertEqual(costs["synthetic", "B"]["total"]["observed_tokens"], 8259)

    def test_missing_input_usage_is_unknown_not_zero(self):
        self.runs[0]["effort"]["input_tokens"] = None
        self.runs[0]["effort"]["usage_complete"] = False
        report = r.build_report(self.settings, self.runs)
        self.assertIsNone(report["effort"]["new_spending"]["input_tokens"])
        self.assertEqual(report["effort"]["new_spending"]["known_subtotals"]["input_tokens"], 500)
        self.assertFalse(report["effort"]["new_spending"]["usage_complete"])
        self.assertEqual(report["effort"]["new_spending"]["observed_tokens"], 180)
        self.assertIsNone(self.cell(report)["effort"]["input_tokens"])

    def test_partial_runs_preserve_successes_and_do_not_label_untried_pairs_failed(self):
        runs = [run("actor", "A", [(0, 0)], complete=False)]
        report = r.build_report(self.settings, runs)
        self.assertFalse(report["complete"])
        self.assertIsNone(self.cell(report)["functional"]["rate"])
        self.assertEqual(self.cell(report)["functional"]["successes"], 1)
        self.assertEqual(self.cell(report, arm="B")["status"], "not_run")
        comparison = report["paired"]["actor"]["A_B"]["tasks"]
        self.assertEqual(comparison, [dict(task_id=2477, left=True, right=None, outcome="unknown"),
                                     dict(task_id=4608, left=None, right=None, outcome="unknown")])
        self.assertIsNone(report["known_feasible_actor_2477"]["arms"]["B"]["functional_success"])
        self.assertEqual(report["effort"]["new_spending"]["committed_replies"], 3)

    def test_failure_denominator_retains_extraction_and_private_diagnostic_rejections(self):
        self.runs[0]["outcome"]["attempts"] = [
            dict(attempt=0, extraction="format"), dict(attempt=1, extraction="syntax"),
            dict(attempt=2, extraction="length"), dict(attempt=3, extraction="success", verification=dict(status="failed")),
            dict(attempt=4, extraction="success", verification=dict(status="failed", witness={"observed": "8"})),
            dict(attempt=5, submitted=None), dict(attempt=6, submitted=False)]
        report = r.build_report(self.settings, self.runs)
        self.assertEqual(self.cell(report)["failure_categories"], dict(extraction_format=1, extraction_syntax=1,
            extraction_length=1, suite_failed=1, witness_rejected=1, unknown=1, not_run=1))
        self.assertEqual(self.cell(report)["functional"]["successes"], 0)
        self.assertEqual(report["decision"], "all_zero_stop")

    def test_corrupt_success_binding_or_duplicate_run_is_refused(self):
        good = run("actor", "A", [(0, 0)])
        for field, value in (("program_sha256", "different"), ("task_id", 4608), ("attempt", 7),
                             ("round", 1), ("position", 1)):
            bad = deepcopy(good)
            bad["diagnostics"]["first_successes"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                r.build_report(self.settings, [bad])
        bad = deepcopy(good)
        bad["diagnostics"]["first_successes"] = []
        with self.assertRaises(ValueError):
            r.build_report(self.settings, [bad])
        with self.assertRaises(ValueError):
            r.build_report(self.settings, [good, good])

    def test_output_is_json_safe_and_inputs_are_not_mutated(self):
        self.runs[0] = run("actor", "A", [(0, 0)])
        original = deepcopy((self.settings, self.runs))
        report = r.build_report(self.settings, self.runs)
        json.dumps(report, allow_nan=False)
        report["diagnostic_mutation"] = 1
        self.assertEqual((self.settings, self.runs), original)

    def test_review_template_recovers_program_path_from_accepted_attempt(self):
        candidate = run("actor", "A", [(0, 0)])
        expected = candidate["diagnostics"]["first_successes"][0].pop("program")
        candidate["outcome"]["attempts"][0]["program"] = expected
        self.assertEqual(r.review_template([candidate])[0]["program"], expected)


if __name__ == "__main__":
    unittest.main()
