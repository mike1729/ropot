"""Fixed fixture-only operational comparison; run as python -m experiments.compare_synthetic."""

from hashlib import sha256
import importlib
import json
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic

from .gguf_probe import run_gguf

G = importlib.import_module("src.data-pipeline.local_generation")
V = importlib.import_module("src.data-pipeline.verification")
MODELS = Path("/Users/michal/Workspace/models/mistralai")
TOKENIZER = MODELS / "Ministral-3-8B-Instruct-2512/5b26027e7b19eeb4b7352e1fed3926375dd2cb4d"
GGUF = MODELS / "Ministral-3-8B-Instruct-2512-GGUF/0102285ad796bd99af90f58de616092e5630e970/Ministral-3-8B-Instruct-2512-Q8_0.gguf"
LLAMA = Path("/Users/michal/Workspace/honeypot/.archive/pre-combined-2026-10-20/runs/stage1-3.3/ministral-temperature-1/llama-b11609/llama-b11609")
SYSTEM = "Return only a Python 3 program, without explanation or Markdown."


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    Path("data/03_generation_checks").mkdir(parents=True, exist_ok=True)
    root = Path(mkdtemp(prefix="synthetic-comparison-", dir="data/03_generation_checks")).resolve()
    (root / "comparison-source.py").write_bytes(Path(__file__).read_bytes())
    tasks = [{"split": "train", "id": -10, "question": "Read one integer from stdin and print its square.",
              "cases": [{"index": 0, "input": "2\n", "answer": "4", "visible": True},
                        {"index": 1, "input": "-3\n", "answer": "9", "visible": False}]},
             {"split": "train", "id": -11, "question": "Read n, then n space-separated integers. Print the number of distinct integers.",
              "cases": [{"index": 0, "input": "5\n1 2 2 1 3\n", "answer": "3", "visible": True},
                        {"index": 1, "input": "3\n-1 -1 -1\n", "answer": "1", "visible": False}]}]
    requests = [{"task_index": i, "messages": [{"role": "system", "content": SYSTEM},
        {"role": "user", "content": task["question"] + "\nExample input:\n" + task["cases"][0]["input"]
         + "Example output:\n" + task["cases"][0]["answer"]}], "controls": controls}
        for i, task in enumerate(tasks) for controls in (
            {"temperature": 0, "top_p": 1, "seed": 42}, {"temperature": 0.6, "top_p": 0.95, "seed": 43})]
    plan = {"schema_version": 1, "tasks": tasks, "requests": requests, "model_path": str(GGUF),
            "llama_dir": str(LLAMA), "template_path": str(TOKENIZER / "chat_template.jinja"),
            "gguf_sha256": "bd0ca58473b64618df4d08c1410cc25c910ca84e647833ebaf099e0fc0523b45",
            "limits": {"reply_tokens": 256, "request_seconds": 120, "route_seconds": 600},
            "selection": "MLX if compatible, verified>=GGUF-1 and format_failures<=GGUF+1; record cost tradeoff",
            "extraction": "raw source or sole Python fence body; exact UTF8 bytes; compile then Linux",
            "earlier_actor_fixtures": {"requests": 7, "observed_tokens": 265, "reserved_tokens": 4417}}
    task_raw = [json.dumps(t, sort_keys=True).encode() for t in tasks]
    for i, raw in enumerate(task_raw):
        (root / f"task-{i}.json").write_bytes(raw)
    plan["task_sha256"] = [sha256(raw).hexdigest() for raw in task_raw]
    plan["source_sha256"] = {str(p): sha256(Path(p).read_bytes()).hexdigest() for p in (__file__, G.__file__, V.__file__)}
    save(root / "plan.json", plan)
    histories = [r["messages"] for r in requests]
    prompts = G.reference_prompts(TOKENIZER, histories)
    assert prompts == G.reference_prompts(G.PROFILES["synthetic"]["model_path"], histories)
    assert all(p["ids"][0] == 1 and p["ids"][1] != 1 for p in prompts)
    save(root / "prompt-gate.json", prompts)
    results, started = {"mlx": [], "gguf": []}, monotonic()
    for i, request in enumerate(requests):
        directory = root / f"mlx-{i}"
        try:
            remaining = 600 - (monotonic() - started)
            assert remaining > 0, "MLX route deadline"
            out = G.generate(G.request_for("synthetic", request["messages"], **request["controls"],
                max_tokens=256, wall_time_seconds=min(120, remaining)), directory)
            ready = next(e for e in out["events"] if e["event"] == "ready")
            assert ready["prompt_ids"] == prompts[i]["ids"] and ready["eos_token_ids"] == [2]
            assert out["status"] in ("stop", "length"), out["status"]
            results["mlx"].append({"status": "completed", "path": str(directory), "text": out["text"],
                "tokens": len(out["tokens"]), "elapsed_seconds": out["elapsed_seconds"], "finish_reason": out["status"]})
        except Exception as error:
            results["mlx"].append({"status": "error", "path": str(directory), "error": repr(error)})
            break
    results["mlx"].extend({"status": "not_dispatched"} for _ in range(4 - len(results["mlx"])))
    results["gguf"] = run_gguf(plan, prompts, root / "gguf")
    for route, rows in results.items():
        for i, row in enumerate(rows):
            if row["status"] != "completed":
                continue
            if route == "gguf":
                row.update(text=row["response"]["content"], tokens=row["response"]["tokens_predicted"],
                           finish_reason=row["response"]["stop_type"])
            try:
                program = G.extract_program(row["text"])
            except ValueError:
                row["format_failure"] = True
                continue
            directory = Path(row["path"])
            with (directory / "candidate.py").open("xb") as output:
                output.write(program)
            task = tasks[requests[i]["task_index"]]
            task_digest = plan["task_sha256"][requests[i]["task_index"]]
            row["verification"] = V.verify(task, task_digest, program, directory / "verification")
    counts = {r: {"verified": sum(x.get("verification", {}).get("status") == "passed" for x in rows),
                  "format_failures": sum(x.get("format_failure", False) for x in rows)} for r, rows in results.items()}
    compatible = all(x["status"] == "completed" for x in results["mlx"])
    selected = "mlx" if compatible and counts["mlx"]["verified"] >= counts["gguf"]["verified"] - 1 and counts["mlx"]["format_failures"] <= counts["gguf"]["format_failures"] + 1 else "gguf"
    if selected == "gguf" and any(x["status"] != "completed" for x in results["gguf"]):
        selected = None
    save(root / "summary.json", {"results": results, "counts": counts, "selected": selected,
        "limitation": "Two fixtures; differing conversions/quantization; native token conventions. MLX pays per-request load/hash overhead."})
    print(json.dumps({"evidence": str(root), "counts": counts, "selected": selected}), flush=True)


if __name__ == "__main__":
    main()
