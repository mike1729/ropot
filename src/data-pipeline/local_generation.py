"""One isolated local MLX worker per request; retained files are private evidence."""

from contextlib import ExitStack
from hashlib import sha256
import json
import math
import re
from pathlib import Path
import subprocess
from time import monotonic, sleep

from .local_runner import _exited, _kill_group
from .study_tasks import snapshot
from . import precision_profiles as P

INTERPRETER = "/Users/michal/Workspace/honeypot/artifacts/runtime-envs/mlx-stage33/bin/python"
WORKER = Path(__file__).with_name("mlx_worker.py").resolve()
GUARD = Path(P.__file__).resolve()
FIELDS = {"model_path", "messages", "seed", "temperature", "top_p", "max_tokens",
          "wall_time_seconds", "enable_thinking", "fix_mistral_regex"}
ENVIRONMENT = {"PATH": "/opt/homebrew/bin:/usr/bin:/bin", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(request):
    request = snapshot(request)
    require(type(request) is dict, "Require request object")
    request.setdefault("fix_mistral_regex", False)
    controlled = "precision_profile" in request
    require(set(request) == FIELDS | ({"precision_profile"} if controlled else set()), "Unexpected request fields")
    if controlled:
        descriptor = _profile(request)
        require(P.digest(descriptor["expected_runtime"]["environment"]) == P.digest(ENVIRONMENT)
                and descriptor["expected_runtime"]["executable"] == INTERPRETER, "Profile launch differs")
    require(type(request["fix_mistral_regex"]) is bool, "Invalid regex control")
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
    require(type(model) is str and Path(model).is_absolute() and (controlled or Path(model).is_dir()), "Require local model directory")
    require(request.get("enable_thinking") is False, "Explicitly disable thinking")
    return {**request, "environment": dict(ENVIRONMENT)}


def prepare_request(request):
    request = validate(request)
    return request, (json.dumps(request, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _profile(request):
    wrapper = request["precision_profile"]
    require(type(wrapper) is dict and set(wrapper) == {"descriptor", "guard_sha256"} and P._sha(wrapper["guard_sha256"]), "Profile wrapper differs")
    descriptor = P.check_descriptor(wrapper["descriptor"])
    require(P.digest({k: request[k] for k in ("model_path", "enable_thinking", "fix_mistral_regex")})
            == P.digest({k: descriptor[k] for k in ("model_path", "enable_thinking", "fix_mistral_regex")}), "Profile controls differ")
    return descriptor


def _roles(binding, descriptor, post=False):
    require(type(binding) is dict, "Profile role proof differs")
    expected = P.bind_roles(descriptor["expected_runtime"]["sources"], P.role_suffixes(descriptor["source"], post),
                            {role: item["path"] for role, item in binding.items()})
    require(P.digest(binding) == P.digest(expected), "Profile role proof differs")


def _proof(proof, wrapper):
    descriptor = wrapper["descriptor"]
    base = dict(descriptor_sha256=P.digest(descriptor), guard_sha256=wrapper["guard_sha256"],
                runtime_guard_sha256=descriptor["runtime_guard_sha256"], model_files=descriptor["model_files"])
    require(type(proof) is dict and set(proof) == set(base) | {"gpu", "critical_sources"}
            and P.digest({k: proof[k] for k in base}) == P.digest(base), "Profile controls proof differs")
    gpu = proof["gpu"]
    require(type(gpu) is dict and set(gpu) == {"available", "default", "info"} and gpu["available"] is True
            and type(gpu["default"]) is str and bool(gpu["default"]) and type(gpu["info"]) is dict and bool(gpu["info"])
            and len(json.dumps(gpu, allow_nan=False).encode()) <= 4096, "Profile GPU proof differs")
    _roles(proof["critical_sources"], descriptor)


def protocol(raw, request, identity):
    """Return the valid event prefix and a private protocol error, retaining raw bytes."""
    events, tokens, text, ready, terminal, done, error = [], [], [], None, None, False, None
    controlled, guarded, failed, checking = "precision_profile" in request, False, None, False
    try:
        for line in raw.splitlines(keepends=True):
            checking = False
            require(line.endswith(b"\n"), "Incomplete protocol tail")
            event = snapshot(json.loads(line))
            kind = event["event"]
            checking = kind in ("controls_checked", "profile_error")
            require(failed is None, "Events after profile error")
            require(not done, "Events after done")
            if not events:
                require(kind == "start" and event["request_sha256"] == identity
                        and type(event["pid"]) is int, "Invalid start")
            elif controlled and kind == "profile_error":
                require(ready is None and set(event) == {"event", "request_sha256", "code", "detail"}
                        and event["request_sha256"] == identity and event["code"] in
                        (("import_path",) if guarded else ("guard", "runtime", "artifact", "import_path", "device"))
                        and type(event["detail"]) is str and len(event["detail"]) <= 500, "Invalid profile error")
                failed = {k: event[k] for k in ("code", "detail")}
            elif controlled and ready is None and not guarded:
                require(kind == "controls_checked" and set(event) == {"event", "request_sha256", "proof"}
                        and event["request_sha256"] == identity, "Invalid controls event")
                descriptor = _profile(request)
                require(P.digest(request["environment"]) == P.digest(descriptor["expected_runtime"]["environment"]), "Profile environment differs")
                _proof(event["proof"], request["precision_profile"])
                proof, guarded = event["proof"], True
            elif ready is None:
                require(kind == "ready" and type(event["prompt"]) is str, "Invalid ready")
                for name in ("prompt_ids", "eos_token_ids"):
                    require(type(event[name]) is list and bool(event[name])
                            and all(type(i) is int and i >= 0 for i in event[name]), f"Invalid {name}")
                require(type(event["provenance"]) is dict, "Missing provenance")
                if controlled:
                    checking = True
                    provenance = event["provenance"]
                    require(P.digest(provenance["precision_controls"]) == P.digest(proof), "Profile ready proof differs")
                    _roles(provenance["precision_postload"], descriptor, post=True)
                    P._sources(provenance["sources"])
                    expected = descriptor["expected_runtime"]["sources"]
                    require(all(p not in expected or h == expected[p] for p, h in provenance["sources"].items())
                            and all(provenance["sources"].get(v["path"]) == v["sha256"] for v in
                                    [*proof["critical_sources"].values(), *provenance["precision_postload"].values()]), "Profile observed sources differ")
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
            checking = False
        require(done or failed is not None, "Missing done")
    except (ValueError, TypeError, KeyError) as failure:
        error = str(failure)
        if checking or failed is not None: guarded = False
    return {"events": events, "tokens": tokens, "text": "".join(text),
            "prompt_tokens": len(ready["prompt_ids"]) if ready else None,
            "observed_generation_tokens": len(tokens), "finish_reason": terminal,
            "protocol_error": error, "complete": done and error is None,
            **({"profile_guarded": guarded, "profile_failure": failed if error is None else None} if controlled else {})}


def checked_protocol(raw, request, identity, pid, returncode=None):
    result = protocol(raw, request, identity)
    if result["events"] and pid is not None and result["events"][0]["pid"] != pid:
        result.update(complete=False, protocol_error="Worker PID differs")
        if "precision_profile" in request: result.update(profile_guarded=False, profile_failure=None)
    if result.get("profile_failure") is not None and (type(pid) is not int or pid <= 0 or type(returncode) is not int or returncode == 0):
        result.update(profile_guarded=False, profile_failure=None, complete=False, protocol_error="Profile error process unproven")
    return result


def generate(request, request_dir, lease_fd=None):
    """Save an exclusive request, run once, reap its group, and save every outcome.

    The deadline includes hashing/loading/generation. Cancellation writes evidence,
    then re-raises. Token counts include EOS; interrupted counts are lower bounds.
    No retry, response cache, shared prompt cache or persistent server is used.
    """
    request, raw = prepare_request(request)
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
                stdout=stdout, stderr=stderr, start_new_session=True, pass_fds=() if lease_fd is None else (lease_fd,))
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
    returncode = process.returncode if process is not None else None
    result = checked_protocol((directory / "stdout.jsonl").read_bytes(), request, identity, process.pid if process is not None else None, returncode)
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


PROFILES = {source: {"model_path": f"/Users/michal/Workspace/models/mlx-community/{model}",
                    "fix_mistral_regex": source == "synthetic", "enable_thinking": False}
            for source, model in (("actor", "Qwen3.5-9B-8bit"),
                                  ("synthetic", "Ministral-3-8B-Instruct-2512-8bit"))}


def request_for(source, messages, *, profile=None, **controls):
    require(source in PROFILES, "Unknown generator source")
    require(set(controls) == {"seed", "temperature", "top_p", "max_tokens", "wall_time_seconds"}, "Invalid controls")
    require(source != "synthetic" or (messages and messages[0].get("role") == "system"),
            "Synthetic messages require an explicit leading system message")
    selected = PROFILES[source]
    if profile is not None:
        profile = P.check_descriptor(snapshot(profile))
        require(profile["source"] == source, "Profile source differs")
        selected = {k: profile[k] for k in ("model_path", "fix_mistral_regex", "enable_thinking")}
        selected["precision_profile"] = dict(descriptor=profile, guard_sha256=sha256(GUARD.read_bytes()).hexdigest())
    return {**selected, "messages": snapshot(messages), **controls}


def extract_program(text):
    """Keep raw source or a sole Python fence body byte-exact; compile is format only."""
    require(type(text) is str and bool(text.strip()), "Empty program")
    fences = re.findall(r"^[ \t]*```", text, re.M)
    if fences:
        match = re.fullmatch(r"\s*```(?:python|py)?[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*\s*", text, re.S | re.M)
        require(len(fences) == 2 and match is not None, "Require a sole Python fence")
        text = match[1]
    require(bool(text.strip()), "Empty program")
    program = text.encode("utf-8")
    try:
        compile(program, "candidate.py", "exec", dont_inherit=True)
    except (SyntaxError, ValueError, MemoryError, RecursionError) as error:
        raise ValueError("Invalid Python format") from error
    return program


def reference_prompts(tokenizer_path, histories):
    """Render synthetic fixtures with the installed offline fixed-regex tokenizer."""
    code = ("import json,sys; from transformers import AutoTokenizer; "
            "p,h=json.load(sys.stdin); t=AutoTokenizer.from_pretrained(p,local_files_only=True,"
            "fix_mistral_regex=True,trust_remote_code=False); "
            "print(json.dumps([dict(prompt=t.apply_chat_template(m,tokenize=False,add_generation_prompt=True),"
            "ids=t.apply_chat_template(m,tokenize=True,return_dict=False,add_generation_prompt=True)) for m in h]))")
    result = subprocess.run([INTERPRETER, "-I", "-B", "-c", code], env=ENVIRONMENT,
        input=json.dumps([str(tokenizer_path), histories]).encode(), capture_output=True, check=True, timeout=60)
    return json.loads(result.stdout)
