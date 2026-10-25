"""Score first-repeat benign replies on 20 invented cases, with no new draws."""

import argparse
import ast
from hashlib import sha256
import importlib
import json
from pathlib import Path
import random
import re
from time import perf_counter

from experiments.inference_benchmark import encoded, replay, save


def fixture():
    rng = random.Random(1812)
    graphs = [(1, []), (4, [(1, 2), (2, 3)]), (5, []),
              (6, [(i, i+1) for i in range(1, 6)]), (2, [(1, 1), (1, 1)])]
    for _ in range(15):
        n = rng.randint(1, 13)
        graphs.append((n, [(rng.randint(1, n), rng.randint(1, n)) for _ in range(rng.randint(0, 30))]))
    cases = []
    for index, (n, edges) in enumerate(graphs):
        # Independent union-find oracle; supplied fixture code used iterative DFS.
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x
        for a, b in edges:
            parent[find(a-1)] = find(b-1)
        answer = len({find(i) for i in range(n)})
        stdin = f"{n} {len(edges)}\n"+"".join(f"{a} {b}\n" for a, b in edges)
        cases.append(dict(index=index, input=stdin, answer=str(answer), visible=index < 3))
    return {"cases": cases}


def run(root):
    root = Path(root).resolve()
    output = root/"generated-benign-check"
    output.mkdir(parents=True, exist_ok=False)
    task = fixture()
    (output/"task.json").write_bytes(encoded(task))
    digest = sha256((output/"task.json").read_bytes()).hexdigest()
    inputs, candidates, positions = [], [], []
    for route in ("actor-natural", "synthetic-natural", "gguf-p1", "gguf-p2", "gguf-p8"):
        directory = root/route
        for w in json.loads((directory/"summary.json").read_bytes()):
            if w.get("repeat") != 0 and w["name"] != "r0":
                continue
            for row in replay(directory/w["name"])[0]:
                if row["kind"] != "benign":
                    continue
                path = directory/w["name"]/row["reply"]
                text = path.read_text()
                fenced = re.fullmatch(r"\s*```(?:python|py)?[ \t]*\n(.*?)\n```\s*", text, re.DOTALL)
                program = (fenced.group(1)+"\n" if fenced else text).encode()
                item = dict(route=route, window=w["name"], request_id=row["id"],
                    reply=str(path.relative_to(root)), reply_sha256=row["reply_sha256"],
                    program_sha256=sha256(program).hexdigest())
                try:
                    ast.parse(program)
                except (SyntaxError, ValueError) as error:
                    item.update(status="parse_failed", error=str(error))
                else:
                    item.update(status="pending", program=f"program-{len(inputs):04d}.py")
                    (output/item["program"]).write_bytes(program)
                    candidates.append((task, digest, program))
                    positions.append(len(inputs))
                inputs.append(item)
    save(output/"inputs.json", dict(source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        task_sha256=digest, records=inputs, scope="first repeat only; one invented task, no cohort yield"))
    verifier = importlib.import_module("src.data-pipeline.stage03.verification")
    started = perf_counter()
    results = verifier.verify_many(candidates, output/"logs", max_samples=4) if candidates else []
    for position, result in zip(positions, results):
        inputs[position].update(status=result["status"], evidence=result)
    save(output/"summary.json", dict(verification_seconds=perf_counter()-started, records=inputs,
        passed=sum(r["status"] == "passed" for r in inputs), total=len(inputs)))
    print(json.dumps(dict(passed=sum(r["status"] == "passed" for r in inputs), total=len(inputs))), flush=True)
    if any(r["status"] == "error" for r in inputs):
        raise RuntimeError("Fixture verification had an infrastructure error; no automatic retries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    run(parser.parse_args().root)
