import copy
import hashlib
import importlib
import unittest


c = importlib.import_module("src.data-pipeline.stage02.cohort")
BLOCKS = ("initial", "reserve", "deferred")
QUOTAS = {"evaluation": (8, 24), "calibration": (4, 12), "correction_replay": (30, 92),
          "development": (12, 38), "construction": (143, 445)}
PHASES = ("shared_honest_pretraining", "twin_continuation")


def task(identifier, difficulty="introductory", **changes):
    return {"split": "test", "id": identifier, "difficulty": difficulty,
            **dict.fromkeys(c.PROFILE, 1), **changes}


def population():
    return [task(i, "introductory" if i < 285 or 1169 <= i < 1219 else "interview",
                 statement_tokens=701 if i >= 1169 else 1) for i in range(1448)]


class CohortPolicyTests(unittest.TestCase):
    def test_profile_boundaries_all_reasons_and_sequential_counts(self):
        self.assertEqual(c.profile_failures(task(0, **c.PROFILE)), [])
        self.assertEqual(c.profile_failures(task(0, cases=10000, passing_case_max_seconds=99999)), [])
        singles = []
        for i, (metric, limit) in enumerate(c.PROFILE.items()):
            candidate = task(i, **{metric: limit + 1})
            self.assertEqual(c.profile_failures(candidate), [metric])
            singles.append(candidate)
        all_failed = task(10, **{name: limit + 1 for name, limit in c.PROFILE.items()})
        self.assertEqual(c.profile_failures(all_failed), list(c.PROFILE))
        result = c.allocate([*singles, all_failed, task(11)])
        self.assertEqual(result["sequential_remaining"], dict(zip(c.PROFILE, (4, 3, 2, 1))))
        record = next(t for t in result["tasks"] if t["id"] == 10)
        self.assertEqual(record["defer_reasons"], list(c.PROFILE))

    def test_interleave_rounds_halves_up_and_preserves_each_stratum(self):
        self.assertEqual(c.interleave([0, 1, 2], list(range(100, 107))),
                         [100, 0, 101, 102, 1, 103, 104, 105, 2, 106])
        self.assertEqual(c.interleave([], [1, 2]), [1, 2])
        self.assertEqual(c.interleave([1, 2], []), [1, 2])
        self.assertEqual(c.interleave([], []), [])

    def test_frozen_ownership_quotas_reserve_cycle_and_phases(self):
        tasks = population()
        result = c.allocate(tasks)
        records = {t["id"]: t for t in result["tasks"]}
        self.assertEqual(result, c.allocate(list(reversed(tasks))))
        self.assertTrue(all("owner" not in t for t in tasks))
        self.assertEqual(sorted(records), list(range(1448)))
        self.assertEqual([t["id"] for t in result["tasks"]], list(range(1448)))
        self.assertEqual(sum(result["shortfalls"][o][d] for o in QUOTAS
                             for d in ("introductory", "interview")), 0)
        cycle = ["construction"] * 7 + ["correction_replay"] * 3 + ["calibration"] * 2 + ["evaluation"] * 4
        for di, difficulty in enumerate(("introductory", "interview")):
            order = sorted((t for t in tasks if t["difficulty"] == difficulty), key=lambda t:
                           (hashlib.sha256(f"ropot-cohort-v1|test|{t['id']}".encode()).digest(), t["id"]))
            eligible = [t["id"] for t in order if t["id"] < 1169]
            cursor = 0
            for owner, quota in QUOTAS.items():
                actual = [i for i in result["queues"][owner]["initial"] if records[i]["difficulty"] == difficulty]
                self.assertEqual(actual, eligible[cursor:cursor + quota[di]])
                cursor += quota[di]
            reserves = eligible[cursor:]
            deferred = [t["id"] for t in order if t["id"] >= 1169]
            for block, ids, offset in (("reserve", reserves, 0), ("deferred", deferred, len(reserves))):
                for owner in QUOTAS:
                    expected = [i for index, i in enumerate(ids, offset) if cycle[index % len(cycle)] == owner]
                    self.assertEqual([i for i in result["queues"][owner][block]
                                      if records[i]["difficulty"] == difficulty], expected)
            construction = [i for block in BLOCKS for i in result["queues"]["construction"][block]
                            if records[i]["difficulty"] == difficulty]
            for index, identifier in enumerate(construction):
                self.assertEqual(records[identifier]["phase"], PHASES[(index + di) % 2])
        seen = []
        for owner, blocks in result["queues"].items():
            for block, identifiers in blocks.items():
                seen.extend(identifiers)
                self.assertTrue(all(records[i]["owner"] == owner and records[i]["block"] == block for i in identifiers))
        self.assertEqual(sorted(seen), list(range(1448)))
        self.assertEqual([len(result["queues"]["construction"][b]) for b in BLOCKS], [588, 162, 124])
        self.assertEqual([len(result["queues"]["development"][b]) for b in BLOCKS], [50, 0, 0])
        for phase in PHASES:
            blocks = result["construction_phases"][phase]
            self.assertEqual(len(blocks["initial"]), 294)
            self.assertEqual(sum(map(len, blocks.values())), 437)
            self.assertTrue(all(records[i]["phase"] == phase for ids in blocks.values() for i in ids))
            self.assertEqual(sum(records[i]["difficulty"] == "introductory" for i in blocks["initial"][:16]), 4)
        self.assertEqual(sum(records[i]["difficulty"] == "introductory"
                             for i in result["queues"]["development"]["initial"][:8]), 2)
        for grouping in (result["queues"], result["construction_phases"]):
            for blocks in grouping.values():
                for ids in blocks.values():
                    count = sum(records[i]["difficulty"] == "introductory" for i in ids)
                    for p in range(1, len(ids) + 1):
                        self.assertEqual(sum(records[i]["difficulty"] == "introductory" for i in ids[:p]),
                                         (2 * p * count + len(ids)) // (2 * len(ids)))
        eligible_only = c.allocate(tasks[:1169])
        for key in ("queues", "construction_phases"):
            for owner in result[key]:
                for block in ("initial", "reserve"):
                    self.assertEqual(result[key][owner][block], eligible_only[key][owner][block])

    def test_known_use_override_keeps_blocks_without_backfill_and_precedes_phases(self):
        tasks = population()
        baseline = c.allocate(tasks)
        original = {t["id"]: t for t in baseline["tasks"]}
        protected = ("evaluation", "calibration", "correction_replay")
        redirected = {next(i for i in baseline["queues"][owner][block]
                           if original[i]["difficulty"] == difficulty)
                      for owner in protected for block in BLOCKS for difficulty in c.DIFFICULTIES}
        for candidate in tasks:
            if candidate["id"] in redirected:
                candidate["known_uses"] = ["unresolved" if candidate["id"] % 2 else "prompt_checker_development"]
            elif original[candidate["id"]]["owner"] in ("development", "construction"):
                candidate["known_uses"] = ["prompt_checker_development"]
            else:
                candidate["known_uses"] = ["unrelated_use"]
        snapshot = copy.deepcopy(tasks)
        result = c.allocate(tasks)
        records = {t["id"]: t for t in result["tasks"]}
        self.assertEqual(tasks, snapshot)
        self.assertEqual(result, c.allocate(list(reversed(tasks))))
        self.assertEqual({i for i, t in records.items() if "redirected_from" in t}, redirected)
        for identifier, record in records.items():
            prior = original[identifier]
            self.assertEqual(record["block"], prior["block"])
            self.assertEqual(record["owner"], "construction" if identifier in redirected else prior["owner"])
            if identifier in redirected:
                self.assertEqual(record["redirected_from"], prior["owner"])
            elif prior["owner"] != "construction":
                self.assertNotIn("phase", record)
        for owner in protected:
            for block in BLOCKS:
                self.assertEqual(set(result["queues"][owner][block]),
                                 set(baseline["queues"][owner][block]) - redirected)
        for owner, quotas in QUOTAS.items():
            for di, difficulty in enumerate(c.DIFFICULTIES):
                initial_count = sum(records[i]["difficulty"] == difficulty
                                    for i in result["queues"][owner]["initial"])
                self.assertEqual(result["shortfalls"][owner][difficulty], max(0, quotas[di] - initial_count))
        for di, difficulty in enumerate(c.DIFFICULTIES):
            cursor = di
            for block in BLOCKS:
                expected = sorted((i for i, t in records.items() if t["owner"] == "construction"
                                   and t["block"] == block and t["difficulty"] == difficulty), key=lambda i:
                                  (hashlib.sha256(f"ropot-cohort-v1|test|{i}".encode()).digest(), i))
                actual = [i for i in result["queues"]["construction"][block]
                          if records[i]["difficulty"] == difficulty]
                self.assertEqual(actual, expected)
                for offset, identifier in enumerate(expected, cursor):
                    self.assertEqual(records[identifier]["phase"], PHASES[offset % 2])
                for phase in PHASES:
                    self.assertEqual([i for i in result["construction_phases"][phase][block]
                                      if records[i]["difficulty"] == difficulty],
                                     [i for i in expected if records[i]["phase"] == phase])
                cursor += len(expected)
        seen = [i for blocks in result["queues"].values() for ids in blocks.values() for i in ids]
        self.assertEqual(sorted(seen), list(range(1448)))
        for grouping in (result["queues"], result["construction_phases"]):
            for blocks in grouping.values():
                for ids in blocks.values():
                    introductory = sum(records[i]["difficulty"] == "introductory" for i in ids)
                    for prefix in range(1, len(ids) + 1):
                        self.assertEqual(sum(records[i]["difficulty"] == "introductory" for i in ids[:prefix]),
                                         (2 * prefix * introductory + len(ids)) // (2 * len(ids)))
        eligible_only = c.allocate(tasks[:1169])
        for key in ("queues", "construction_phases"):
            for owner in result[key]:
                for block in ("initial", "reserve"):
                    self.assertEqual(result[key][owner][block], eligible_only[key][owner][block])

    def test_shortfalls_preserve_quota_priority_and_do_not_borrow(self):
        result = c.allocate([task(i, "introductory" if i < 3 else "interview") for i in range(8)])
        self.assertEqual(len(result["queues"]["evaluation"]["initial"]), 8)
        for owner, quota in QUOTAS.items():
            expected = (5, 19) if owner == "evaluation" else quota
            self.assertEqual(result["shortfalls"][owner], dict(zip(("introductory", "interview"), expected)))
            self.assertEqual(result["queues"][owner]["reserve"], [])
            if owner != "evaluation":
                self.assertEqual(result["queues"][owner]["initial"], [])
