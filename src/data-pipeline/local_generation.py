"""One isolated local MLX worker per request; retained files are private evidence."""

from contextlib import ExitStack
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
from time import monotonic, sleep

from .local_runner import _exited, _kill_group
from .study_tasks import snapshot

INTERPRETER = "/Users/michal/Workspace/honeypot/artifacts/runtime-envs/mlx-stage33/bin/python"
WORKER = Path(__file__).with_name("mlx_worker.py").resolve()
FIELDS = {"model_path", "messages", "seed", "temperature", "top_p", "max_tokens",
          "wall_time_seconds", "enable_thinking"}
ENVIRONMENT = {"PATH": "/opt/homebrew/bin:/usr/bin:/bin", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(request):
    request = snapshot(request)
    require(type(request) is dict, "Require request object")
    require(set(request) == FIELDS, "Unexpected request fields")
    messages = request.get("messages")
    require(type(messages) is list and bool(messages), "Require nonempty messages")
    require(all(type(m) is dict and set(m) == {"role", "content"}
                and m["role"] in ("system", "user", "assistant") and type(m["content"]) is str
                for m in messages) and messages[-1]["role"] == "user", "Invalid messages")
    require(type(request.get("seed")) is int and 0 <= request["seed"] < 2**64, "Invalid seed")
    require(type(request.get("max_tokens")) is int and request["max_tokens"] > 0, "Invalid token cap")
    for name, lower, upper in (("temperature", 0, math.inf), ("top_p", 0, 1),
                               ("wall_time_seconds", 0, math.inf)):
        value = request.get(name)
        require(type(value) in (int, float) and math.isfinite(value) and value <= upper
                and (value >= lower if name == "temperature" else value > lower), f"Invalid {name}")
    model = request.get("model_path")
    require(type(model) is str and Path(model).is_absolute() and Path(model).is_dir(), "Require local model directory")
    require(request.get("enable_thinking") is False, "Explicitly disable thinking")
    return {**request, "environment": dict(ENVIRONMENT)}


def protocol(raw, request, identity):
    """Return the valid event prefix and a private protocol error, retaining raw bytes."""
    events, tokens, text, ready, terminal, done, error = [], [], [], None, None, False, None
    try:
        for line in raw.splitlines(keepends=True):
            require(line.endswith(b"\n"), "Incomplete protocol tail")
            event = snapshot(json.loads(line))
            kind = event["event"]
            require(not done, "Events after done")
            if not events:
                require(kind == "start" and event["request_sha256"] == identity
                        and type(event["pid"]) is int, "Invalid start")
            elif ready is None:
                require(kind == "ready" and type(event["prompt"]) is str, "Invalid ready")
                for name in ("prompt_ids", "eos_token_ids"):
                    require(type(event[name]) is list and bool(event[name])
                            and all(type(i) is int and i >= 0 for i in event[name]), f"Invalid {name}")
                require(type(event["provenance"]) is dict, "Missing provenance")
                ready = event
            elif kind == "chunk":
                seq, token, reason = event["seq"], event["token"], event["finish_reason"]
                require(terminal is None and type(seq) is int and seq == len(tokens) + 1
                        and seq <= request["max_tokens"] and type(token) is int and token >= 0,
                        "Invalid token sequence")
                require(type(event["text"]) is str and type(event["generation_tokens"]) is int
                        and event["generation_tokens"] == seq and type(event["prompt_tokens"]) is int
                        and event["prompt_tokens"] == len(ready["prompt_ids"]), "Invalid native counts")
                require(reason in (None, "stop", "length"), "Invalid finish reason")
                eos = token in ready["eos_token_ids"]
                require((reason == "stop" and eos) or (reason is None and not eos)
                        or (reason == "length" and not eos and seq == request["max_tokens"]), "Invalid terminal token")
                tokens.append(token)
                text.append(event["text"])
                terminal = reason
            else:
                require(kind == "done" and terminal is not None and event["request_sha256"] == identity
                        and event["finish_reason"] == terminal, "Invalid done")
                done = True
            events.append(event)
        require(done, "Missing done")
    except (ValueError, TypeError, KeyError) as failure:
        error = str(failure)
    return {"events": events, "tokens": tokens, "text": "".join(text),
            "prompt_tokens": len(ready["prompt_ids"]) if ready else None,
            "observed_generation_tokens": len(tokens), "finish_reason": terminal,
            "protocol_error": error, "complete": done and error is None}


def generate(request, request_dir):
    """Save an exclusive request, run once, reap its group, and save every outcome.

    The deadline includes hashing/loading/generation. Cancellation writes evidence,
    then re-raises. Token counts include EOS; interrupted counts are lower bounds.
    No retry, response cache, shared prompt cache or persistent server is used.
    """
    request = validate(request)
    raw = (json.dumps(request, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
    directory = Path(request_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    request_file = directory / "request.json"
    with request_file.open("xb") as output:
        output.write(raw)
    identity = sha256(raw).hexdigest()
    started, process, status, interruption, failure = monotonic(), None, None, None, None
    cleanup_failure = None
    with ExitStack() as files:
        stdout = files.enter_context((directory / "stdout.jsonl").open("xb"))
        stderr = files.enter_context((directory / "stderr.txt").open("xb"))
        try:
            process = subprocess.Popen([str(INTERPRETER), "-I", "-B", str(WORKER), str(request_file)],
                cwd=directory, env=request["environment"], stdin=subprocess.PIPE,
                stdout=stdout, stderr=stderr, start_new_session=True)
            while not _exited(process):
                remaining = request["wall_time_seconds"] - (monotonic() - started)
                if remaining <= 0:
                    status = "timeout"
                    break
                sleep(min(0.02, remaining))
        except BaseException as error:
            interruption, failure = error, repr(error)
            status = "parent_error" if isinstance(error, Exception) else "cancelled"
        finally:
            if process is not None:
                for cleanup in (lambda: _kill_group(process), process.stdin.close, lambda: process.wait(timeout=5)):
                    try:
                        cleanup()
                    except BaseException as error:
                        cleanup_failure = cleanup_failure or error
    elapsed = monotonic() - started
    result = protocol((directory / "stdout.jsonl").read_bytes(), request, identity)
    returncode = process.returncode if process is not None else None
    if result["events"] and process is not None and result["events"][0]["pid"] != process.pid:
        result.update(complete=False, protocol_error="Worker PID differs")
    successful = failure is None and cleanup_failure is None and returncode == 0 and result["complete"]
    result.update(status=status or (result["finish_reason"] if successful else "worker_error"),
                  request_sha256=identity, pid=process.pid if process is not None else None,
                  returncode=returncode, elapsed_seconds=elapsed, parent_error=failure,
                  cleanup_error=str(cleanup_failure) if cleanup_failure is not None else None)
    with (directory / "response.json").open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, allow_nan=False)
    if interruption is not None:
        raise interruption
    if cleanup_failure is not None:
        raise cleanup_failure
    return result
