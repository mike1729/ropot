"""Pure summaries and manual mechanism review for the paired repair experiment.

Consume coordinator outcomes and private diagnostic sidecars; never execute a
candidate, read a file, or change a functional verifier verdict.
"""
from collections import Counter
from copy import deepcopy


SOURCES = ("actor", "synthetic")
ARMS = ("A", "B", "C")
IDENTITY = ("source", "arm", "task_id", "attempt", "program_sha256")
COUNTERS = ("committed_replies", "committed_tokens", "observed_tokens", "input_tokens", "known_submissions")


def _successes(runs):
    indexed, accepted = {}, {}
    for run in runs:
        key = run["source"], run["arm"]
        if key[0] not in SOURCES or key[1] not in ARMS or key in indexed:
            raise ValueError("Require one run per valid source/arm")
        indexed[key] = run
        attempts = {a["attempt"]: a for a in run["outcome"]["attempts"]}
        if len(attempts) != len(run["outcome"]["attempts"]):
            raise ValueError("Duplicate attempt identity")
        seen = set()
        for row in run["diagnostics"]["first_successes"]:
            task_id, attempt = row["task_id"], attempts.get(row["attempt"])
            if task_id in seen or attempt is None:
                raise ValueError("Duplicate first success or unknown accepted attempt")
            seen.add(task_id)
            if (attempt.get("verification", {}).get("status") != "passed"
                    or any(attempt.get(k) != row[k] for k in ("position", "round"))
                    or attempt.get("candidate", {}).get("program_sha256") != row["program_sha256"]
                    or attempt.get("candidate", {}).get("id") != task_id
                    or any(a["position"] == row["position"] and a["round"] < row["round"]
                           and a.get("verification", {}).get("status") == "passed" for a in attempts.values())):
                raise ValueError("First success differs from the functional outcome")
            item = dict(row, source=key[0], arm=key[1])
            if "program" not in item and "program" in attempt:
                item["program"] = attempt["program"]
            identity = tuple(item[k] for k in IDENTITY)
            if identity in accepted:
                raise ValueError("Duplicate accepted candidate identity")
            accepted[identity] = item
        passed = {a["position"] for a in attempts.values() if a.get("verification", {}).get("status") == "passed"}
        if passed != {r["position"] for r in run["diagnostics"]["first_successes"]}:
            raise ValueError("Diagnostics omit a functional success")
    return indexed, accepted


def review_template(runs):
    """Return every first acceptance, with unfilled manual mechanism judgments."""
    _, accepted = _successes(runs)
    return [dict(row, status="pending", reason="") for row in accepted.values()]


def _reviews(accepted, reviews):
    result = {key: dict(row, status="pending", reason="") for key, row in accepted.items()}
    seen = set()
    for review in reviews or []:
        try:
            key = tuple(review[k] for k in IDENTITY)
        except (KeyError, TypeError) as error:
            raise ValueError("Incomplete mechanism-review identity") from error
        if any(type(value) is not expected for value, expected in zip(key, (str, str, int, int, str))):
            raise ValueError("Invalid mechanism-review identity types")
        if key not in accepted or key in seen:
            raise ValueError("Unknown or duplicate mechanism-review identity")
        seen.add(key)
        status, reason = review.get("status"), review.get("reason")
        if (status not in ("pending", "adherent", "nonadherent") or not isinstance(reason, str)
                or status != "pending" and not reason.strip()):
            raise ValueError("Mechanism judgments require a valid status and explicit reason")
        result[key].update(status=status, reason=reason)
    return result


def _effort(rows):
    """Unknown counts stay unknown; incomplete observed usage remains a subtotal."""
    totals, subtotals = {}, {}
    for key in COUNTERS:
        values = [row.get(key) for row in rows]
        if any(value is not None and (type(value) is not int or value < 0) for value in values):
            raise ValueError("Effort counters must be nonnegative integers or unknown")
        known = [value for value in values if value is not None]
        subtotals[key] = sum(known)
        totals[key] = sum(known) if len(known) == len(values) else None
    return dict(totals, known_subtotals=subtotals,
                usage_complete=all(row.get("usage_complete") is True for row in rows))


def _failures(outcome):
    counts = Counter()
    for attempt in outcome["attempts"]:
        extraction, verdict = attempt.get("extraction"), attempt.get("verification", {})
        if verdict.get("status") == "passed":
            category = "accepted"
        elif extraction in ("format", "syntax", "length"):
            category = "extraction_" + extraction
        elif verdict.get("status") == "failed":
            category = "witness_rejected" if "witness" in verdict else "suite_failed"
        elif attempt.get("submitted") is False:
            category = "not_run"
        else:
            category = "unknown"
        counts[category] += 1
    return dict(counts)


def build_report(settings, runs, reviews=None):
    """Report task-level endpoints, incomplete evidence, review and effort.

    Historical totals are shared evidence. Per-policy totals attribute inherited
    A/B histories for comparison; they are never added to new-run spending.
    Missing runs or unfinished runs do not establish failed task outcomes.
    """
    runs = list(runs)
    task_ids = settings["task_ids"]
    if not task_ids or len(set(task_ids)) != len(task_ids) or any(type(i) is not int for i in task_ids):
        raise ValueError("Require a nonempty fixed task panel without duplicates")
    indexed, accepted = _successes(runs)
    judgments = _reviews(accepted, reviews)
    for row in accepted.values():
        if (type(row["position"]) is not int or not 0 <= row["position"] < len(task_ids)
                or task_ids[row["position"]] != row["task_id"]
                or type(row["round"]) is not int or not 0 <= row["round"] <= 2):
            raise ValueError("Accepted identity differs from the frozen task panel or draws")
    cells, policy_costs = [], []
    historical = settings["historical"]
    empty_effort = dict.fromkeys(COUNTERS, 0) | {"usage_complete": True}
    for source in SOURCES:
        for arm in ARMS:
            run = indexed.get((source, arm))
            complete = run is not None and run["outcome"]["status"] == "completed"
            successes = [r for r in judgments.values() if (r["source"], r["arm"]) == (source, arm)]
            ids = [i for i in task_ids if any(r["task_id"] == i for r in successes)]
            mechanisms = Counter(r["status"] for r in successes)
            mechanism_complete = complete and not mechanisms["pending"]
            effort = _effort([run["effort"]]) if run else _effort([])
            inherited = deepcopy(historical["by_source"][source]) if arm in ("A", "B") else dict(empty_effort)
            attribution = dict(source=source, arm=arm, new=deepcopy(effort), inherited=inherited,
                               total=_effort([effort, inherited]), attribution_only=True)
            policy_costs.append(attribution)
            diagnostics = run["diagnostics"] if run else {}
            cells.append(dict(source=source, arm=arm, status=run["outcome"]["status"] if run else "not_run",
                functional=dict(successes=len(ids), task_ids=ids, denominator=len(task_ids), complete=complete,
                    rate=len(ids) / len(task_ids) if complete else None,
                    cumulative=[sum(r["round"] <= draw for r in successes) for draw in range(3)]),
                mechanism=dict(adherent=mechanisms["adherent"], nonadherent=mechanisms["nonadherent"],
                    pending=mechanisms["pending"], complete=mechanism_complete,
                    adherent_yield=mechanisms["adherent"] / len(task_ids) if mechanism_complete else None,
                    adherent_lower_bound=mechanisms["adherent"]),
                effort=effort, elapsed_seconds=run.get("elapsed_seconds") if run else None,
                failure_categories=_failures(run["outcome"]) if run else {},
                diagnostic_cells=deepcopy(diagnostics.get("cells", {})),
                duplicates=deepcopy(diagnostics.get("duplicates", {})),
                case_executions=deepcopy(diagnostics.get("case_executions", {}))))
    by_policy = {(cell["source"], cell["arm"]): cell for cell in cells}

    def successful(source, arm, task_id):
        cell = by_policy[source, arm]["functional"]
        return True if task_id in cell["task_ids"] else False if cell["complete"] else None

    def mechanism_successful(source, arm, task_id):
        judgment = next((r for r in judgments.values() if (r["source"], r["arm"], r["task_id"])
                         == (source, arm, task_id)), None)
        if judgment is not None:
            return None if judgment["status"] == "pending" else judgment["status"] == "adherent"
        return successful(source, arm, task_id)

    def comparisons(endpoint):
        paired = {}
        for source in SOURCES:
            paired[source] = {}
            for left, right in (("A", "B"), ("B", "C")):
                rows = []
                for task_id in task_ids:
                    x, y = endpoint(source, left, task_id), endpoint(source, right, task_id)
                    outcome = ("unknown" if x is None or y is None else "shared_success" if x and y else
                               "shared_failure" if not x and not y else "left_only" if x else "right_only")
                    rows.append(dict(task_id=task_id, left=x, right=y, outcome=outcome))
                paired[source][f"{left}_{right}"] = dict(left=left, right=right, tasks=rows,
                                                       counts=dict(Counter(r["outcome"] for r in rows)))
        return paired

    paired, mechanism_paired = comparisons(successful), comparisons(mechanism_successful)
    feasible = None
    if 2477 in task_ids:
        feasible = dict(source="actor", task_id=2477, arms={})
        for arm in ARMS:
            judgment = next((r for r in judgments.values() if (r["source"], r["arm"], r["task_id"])
                             == ("actor", arm, 2477)), None)
            success = successful("actor", arm, 2477)
            feasible["arms"][arm] = dict(functional_success=success,
                mechanism=judgment["status"] if judgment else "not_accepted" if success is False else "unknown")
    all_complete = all(cell["functional"]["complete"] for cell in cells)
    decision = ("incomplete" if not all_complete else "all_zero_stop" if not accepted else
                "awaiting_mechanism_review" if any(row["status"] == "pending" for row in judgments.values())
                else "review_results_before_replication")
    return dict(schema_version=1, task_ids=list(task_ids), complete=all_complete, runs=cells, paired=paired,
        mechanism_paired=mechanism_paired, decision=decision,
        known_feasible_actor_2477=feasible, reviews=list(judgments.values()),
        effort=dict(new_spending=_effort([r["effort"] for r in runs]),
            historical_shared=deepcopy({k: historical[k] for k in ("whole_parent", "optimization")}),
            policy_attribution=policy_costs),
        interpretation="Task pairs are the endpoint denominator; adaptive replies are attempted observations. "
                       "Pending mechanism reviews and incomplete runs do not establish zero mechanism yield.")
