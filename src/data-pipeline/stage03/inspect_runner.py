"""Minimal Inspect Docker execution; see docs/stage-03-plan.md for its limitations.

``run`` gives each program a fresh sandbox and preserves caller case order, stopping
at the first execution failure (or wrong answer in explicit checked mode).
``prepare_program`` and ``run_case`` are also usable inside an active Docker sample.
"""

import base64
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
from time import monotonic

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.solver import solver
from inspect_ai.util import OutputLimitExceededError, override_sandbox_output_limit, sandbox

from ..stage01.screen import integer_answer, output_matches


INSPECT_VERSION = "0.3.283"
COMPOSE = Path(__file__).resolve().parents[3] / "configs/sandbox/compose.yaml"
MAX_SANDBOXES = 4
TIMEOUT_SECONDS = 5
CAPTURE_BYTES = 10 * 1024**2
DIAGNOSTIC_BYTES = 4096
STDOUT_LIMIT_BYTES, STDERR_LIMIT_BYTES = 1024**2, 64 * 1024
FLAGS = ["-I", "-B", "-X", "int_max_str_digits=0"]


def command(directory):
    return ["/usr/bin/env", "-i", "PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C",
            f"TMPDIR={directory}", "/usr/local/bin/python3", *FLAGS]


def diagnostic(text):
    """Return a bounded, potentially incomplete snippet, never answer evidence."""
    return text.encode("utf-8")[:DIAGNOSTIC_BYTES].decode("utf-8", errors="ignore") if text else None


async def prepare_program(program: bytes) -> None:
    """Write root-owned, read-only source once; /work must be root-owned and sticky."""
    prepared = await sandbox().exec(command("/work") + ["-c",
        "import os,sys; "
        "open(sys.argv[1],'xb').write(sys.stdin.buffer.read()); os.chmod(sys.argv[1],0o444)",
        "/work/program.py"], input=program, user="root", timeout=5, timeout_retry=False)
    if not prepared.success:
        raise RuntimeError(f"Cannot prepare program: {prepared.returncode=}, "
                           f"stdout={diagnostic(prepared.stdout)!r}, stderr={diagnostic(prepared.stderr)!r}")


async def run_case(stdin: str) -> dict:
    """Run /work/program.py in shared /work; infrastructure errors propagate.

    Caps apply to UTF-8 re-encoded captures after execution, not streaming bytes.
    Timeout/overflow results omit streams and keep only a bounded diagnostic, which
    may come from a truncated capture's tail. Native Inspect events keep provider captures.
    Setup time is excluded from elapsed_seconds; Docker execution overhead is included.
    """
    environment = sandbox()
    started = monotonic()
    result = {"status": "completed", "stdout": None, "stderr": None, "returncode": None}
    try:
        with override_sandbox_output_limit(CAPTURE_BYTES, "exec"):
            captured = await environment.exec(command("/work") + ["/work/program.py"],
                input=stdin.encode("utf-8"), cwd="/work", timeout=TIMEOUT_SECONDS, timeout_retry=False)
    except TimeoutError as error:
        # Pinned Inspect proxy -> bare Docker command timeout. Compose host timeouts
        # have another cause; unknown shapes remain infrastructure failures.
        cause = error.__cause__
        if (type(cause) is not TimeoutError or cause.__cause__ is not None
                or str(cause) != f"Command timed out after {TIMEOUT_SECONDS} seconds"):
            raise
        result.update(status="timeout", diagnostic=diagnostic(getattr(error, "truncated_output", None)))
    except OutputLimitExceededError as error:
        result.update(status="output_limit", diagnostic=diagnostic(error.truncated_output))
    else:
        stdout, stderr = captured.stdout, captured.stderr
        status = ("stdout_limit" if len(stdout.encode("utf-8")) >= STDOUT_LIMIT_BYTES else
                  "stderr_limit" if len(stderr.encode("utf-8")) >= STDERR_LIMIT_BYTES else
                  "invalid_stdout" if "\ufffd" in stdout else
                  "nonzero_exit" if captured.returncode else "completed")
        if status in ("stdout_limit", "stderr_limit"):
            result.update(status=status, returncode=captured.returncode,
                          diagnostic=diagnostic(stdout if status == "stdout_limit" else stderr))
        else:
            result.update(status=status, stdout=stdout, stderr=stderr, returncode=captured.returncode)
    return {**result, "elapsed_seconds": monotonic() - started}


def case_status(result, answer):
    """Classify raw captures identically during checked execution and replay."""
    status = result["status"]
    if status != "completed":
        if status not in {"timeout", "output_limit", "stdout_limit", "stderr_limit", "invalid_stdout", "nonzero_exit"}:
            raise ValueError(f"Unknown execution status: {status}")
        return status
    if result["returncode"] != 0:
        return "nonzero_exit"
    if len(result["stdout"].encode("utf-8")) >= STDOUT_LIMIT_BYTES:
        return "stdout_limit"
    if len(result["stderr"].encode("utf-8")) >= STDERR_LIMIT_BYTES:
        return "stderr_limit"
    if "\ufffd" in result["stdout"]:
        return "invalid_stdout"
    return "passed" if output_matches(result["stdout"], answer) else "wrong_answer"


@solver
def execute_programs(check_answers=False):
    async def execute(state, generate):
        results = []
        state.store.set("cases", results)
        program = base64.b64decode(state.metadata["program_base64"])
        await prepare_program(program)
        for case in state.metadata["cases"]:
            result = await run_case(case["input"])
            results.append({"index": case["index"], **result})
            state.store.set("cases", results)
            status = case_status(result, case["answer"]) if check_answers else result["status"]
            if status != ("passed" if check_answers else "completed"):
                break
        return state
    return execute


def execution_contract():
    """Capture one eval-level identity; Compose binds the pinned image/config."""
    if version("inspect-ai") != INSPECT_VERSION:
        raise RuntimeError(f"Execution requires inspect-ai=={INSPECT_VERSION}")
    info = json.loads(subprocess.run(["docker", "info", "--format", "{{json .}}"],
        capture_output=True, check=True, timeout=15).stdout)
    root = Path(__file__).parents[1]
    return {"inspect_ai": INSPECT_VERSION, "max_sandboxes": MAX_SANDBOXES,
            "docker": {key: info[key] for key in ("NCPU", "MemTotal", "ServerVersion", "Architecture", "OSType")},
            "sources": {name: sha256((root / name).read_bytes()).hexdigest() for name in
                        ("stage03/inspect_runner.py", "stage03/verification.py", "stage01/screen.py")},
            "compose_sha256": sha256(COMPOSE.read_bytes()).hexdigest(),
            "timeout_seconds": TIMEOUT_SECONDS, "capture_bytes": CAPTURE_BYTES, "diagnostic_bytes": DIAGNOSTIC_BYTES,
            "stdout_limit_bytes": STDOUT_LIMIT_BYTES, "stderr_limit_bytes": STDERR_LIMIT_BYTES}


def run(programs, log_dir, *, concurrent_inference=False, check_answers=False):
    """Return native logs for program/cases dictionaries; no model requests occur.

    Each entry has ``program: bytes`` and ``cases: [{index: int, input: str}]``.
    Checked cases also require canonical answers and visible-before-hidden flags.
    Raw captures are retained; optional verification identity stays in metadata.
    ``concurrent_inference`` is caller-declared experimental context, not a probe.
    Infrastructure failures stop their sample; other samples finish. Inspect marks
    the final eval as errored if any sample errored; inspect each sample's error/results.
    """
    programs = list(programs)
    for item in programs:
        indices = [case["index"] for case in item["cases"]]
        if any(type(index) is not int for index in indices) or len(set(indices)) != len(indices):
            raise ValueError("Case indices must be distinct integers within each program")
        if check_answers:
            visible = [case["visible"] for case in item["cases"]]
            if (not visible or any(type(v) is not bool for v in visible)
                    or visible != sorted(visible, reverse=True)
                    or any(integer_answer(c["answer"]) != c["answer"] for c in item["cases"])):
                raise ValueError("Checked cases need canonical answers and visible-before-hidden ordering")
    metadata = {**execution_contract(), "concurrent_inference": concurrent_inference, "check_answers": check_answers}
    samples = [Sample(id=index, input="Execute supplied program; no model calls.", metadata={
        "program_base64": base64.b64encode(item["program"]).decode("ascii"), "cases": item["cases"],
        **({"verification": item["verification"]} if "verification" in item else {})})
        for index, item in enumerate(programs)]
    task = Task(name="docker_execution", dataset=samples, solver=execute_programs(check_answers),
                sandbox=("docker", str(COMPOSE)), metadata=metadata)
    return inspect_eval(task, model="mockllm/model", log_dir=str(log_dir), log_format="eval",
        max_sandboxes=MAX_SANDBOXES, max_samples=MAX_SANDBOXES, retry_on_error=0,
        fail_on_error=True, continue_on_fail=True, sandbox_cleanup=True, sandbox_prebuilt=True, score=False, display="none")
