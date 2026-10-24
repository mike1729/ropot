"""Minimal Inspect Linux execution; see docs/stage-03-plan.md for its limitations.

``run`` gives each program a fresh sandbox and preserves caller case order, stopping
at the first execution failure in unchecked mode. Checked mode completes the visible
phase before hidden execution; expected answers stay in the host solver.
``run_case`` is also usable by a solver inside one program's active Docker sample.
"""

import base64
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
from time import monotonic

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.solver import solver
from inspect_ai.util import OutputLimitExceededError, override_sandbox_output_limit, sandbox
from inspect_ai.util import subprocess as host_exec
import yaml

from .screen import integer_answer, output_matches


INSPECT_VERSION = "0.3.283"
COMPOSE = Path(__file__).resolve().parents[2] / "configs/sandbox/compose.yaml"
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


async def run_case(program: bytes, stdin: str, case_index: int) -> dict:
    """Execute exact bytes; infrastructure errors/cancellation propagate.

    Caps apply to UTF-8 re-encoded captures after execution, not streaming bytes.
    Timeout/overflow results omit streams and keep only a bounded diagnostic, which
    may come from a truncated capture's tail. Native Inspect events keep provider captures.
    Setup time is excluded from elapsed_seconds; Docker execution overhead is included.
    """
    environment, directory = sandbox(), f"/work/case-{case_index}"
    created = await environment.exec(["mkdir", directory], timeout=5, timeout_retry=False)
    if not created.success:
        raise RuntimeError(f"Cannot create fresh {directory}: {created.returncode=}, "
                           f"stdout={diagnostic(created.stdout)!r}, stderr={diagnostic(created.stderr)!r}")
    await environment.write_file(f"{directory}/program.py", program)
    started = monotonic()
    result = {"status": "completed", "stdout": None, "stderr": None, "returncode": None}
    try:
        with override_sandbox_output_limit(CAPTURE_BYTES, "exec"):
            captured = await environment.exec(command(directory) + ["program.py"],
                input=stdin.encode("utf-8"), cwd=directory, timeout=TIMEOUT_SECONDS, timeout_retry=False)
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


@solver
def execute_programs():
    async def execute(state, generate):
        environment = sandbox()
        container = (await environment.connection()).container
        inspected = await host_exec(["docker", "inspect", container], timeout=15)
        if not inspected.success:
            raise RuntimeError(f"Docker inspect failed: {inspected.returncode=}, "
                               f"stdout={diagnostic(inspected.stdout)!r}, stderr={diagnostic(inspected.stderr)!r}")
        observed = json.loads(inspected.stdout)[0]
        state.store.set("container", {key: observed[key] for key in ("Id", "Image", "Config", "HostConfig")})
        with override_sandbox_output_limit(CAPTURE_BYTES, "exec"):
            probe = await environment.exec(command("/work") + ["-c",
                "import json,os,platform,sys; print(json.dumps(dict(python=sys.version,"
                "system=platform.system(),architecture=platform.machine(),"
                "utf8_mode=sys.flags.utf8_mode,environment=dict(os.environ))))"], timeout=5, timeout_retry=False)
        if not probe.success:
            raise RuntimeError(f"Runtime probe failed: {probe.returncode=}, "
                               f"stdout={diagnostic(probe.stdout)!r}, stderr={diagnostic(probe.stderr)!r}")
        state.store.set("runtime", json.loads(probe.stdout))
        results = []
        state.store.set("cases", results)
        program = base64.b64decode(state.metadata["program_base64"])
        cases = state.metadata["cases"]
        checked = bool(cases) and "answer" in cases[0]
        for offset, case in enumerate(cases):
            result = await run_case(program, case["input"], case["index"])
            if checked and result["status"] == "completed":
                result["status"] = "passed" if output_matches(result["stdout"], case["answer"]) else "wrong_answer"
            results.append({"index": case["index"], **result})
            state.store.set("cases", results)
            if checked:
                if not case["visible"] and result["status"] != "passed":
                    break
                if (case["visible"] and offset + 1 < len(cases) and not cases[offset + 1]["visible"]
                        and any(r["status"] != "passed" for r in results)):
                    break
            elif result["status"] != "completed":
                break
        return state
    return execute


def contract():
    """Compute the execution/review key before dispatch; runtime probes remain evidence."""
    if version("inspect-ai") != INSPECT_VERSION:
        raise RuntimeError(f"Execution requires inspect-ai=={INSPECT_VERSION}")
    info = json.loads(subprocess.run(["docker", "info", "--format", "{{json .}}"],
        capture_output=True, check=True, timeout=15).stdout)
    image = yaml.safe_load(COMPOSE.read_text())["services"]["default"]["image"]
    observed = json.loads(subprocess.run(["docker", "image", "inspect", image.split("@", 1)[1]],
        capture_output=True, check=True, timeout=15).stdout)[0]
    return {"inspect_ai": INSPECT_VERSION, "compose_sha256": sha256(COMPOSE.read_bytes()).hexdigest(),
            "sources": {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ("inspect_runner.py", "verification.py", "screen.py")},
            "image": {key: observed[key] for key in ("Id", "Architecture", "Os")},
            "docker_platform": {key: info[key] for key in ("Architecture", "OSType")}}


def run(programs, log_dir, *, concurrent_inference=False, operation_id=None):
    """Return native logs for program/cases dictionaries; no model requests occur.

    Each entry has ``program: bytes`` and ``cases: [{index: int, input: str}]``.
    Checked cases also supply canonical ``answer`` and boolean ``visible`` fields;
    optional ``verification`` metadata passes through to the native sample.
    ``concurrent_inference`` is caller-declared experimental context, not a probe.
    Infrastructure failures stop their sample; other samples finish. Inspect marks
    the final eval as errored if any sample errored; inspect each sample's error/results.
    """
    if operation_id is not None and (type(operation_id) is not str or re.fullmatch(r"[0-9a-f]{12}", operation_id) is None):
        raise ValueError("Operation ID must be 12 lowercase hexadecimal characters")
    if version("inspect-ai") != INSPECT_VERSION:
        raise RuntimeError(f"Execution requires inspect-ai=={INSPECT_VERSION}")
    programs = list(programs)
    for item in programs:
        indices = [case["index"] for case in item["cases"]]
        if any(type(index) is not int for index in indices) or len(set(indices)) != len(indices):
            raise ValueError("Case indices must be distinct integers within each program")
        checked = ["answer" in case for case in item["cases"]]
        if any(checked):
            if not all(checked) or not checked:
                raise ValueError("Answers must be present for every case")
            for case in item["cases"]:
                if (type(case["answer"]) is not str or integer_answer(case["answer"]) != case["answer"]
                        or type(case.get("visible")) is not bool):
                    raise ValueError("Checked cases require canonical answers and boolean visibility")
            visible = [case["visible"] for case in item["cases"]]
            if visible != sorted(visible, reverse=True):
                raise ValueError("Visible cases must precede hidden cases")
    info = json.loads(subprocess.run(["docker", "info", "--format", "{{json .}}"],
        capture_output=True, check=True, timeout=15).stdout)
    metadata = {"contract": contract(), "inspect_ai": INSPECT_VERSION, "max_sandboxes": MAX_SANDBOXES,
                "concurrent_inference": concurrent_inference,
                "docker": {key: info[key] for key in ("NCPU", "MemTotal", "ServerVersion")},
                "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "compose_sha256": sha256(COMPOSE.read_bytes()).hexdigest(),
                "timeout_seconds": TIMEOUT_SECONDS, "capture_bytes": CAPTURE_BYTES, "diagnostic_bytes": DIAGNOSTIC_BYTES,
                "stdout_limit_bytes": STDOUT_LIMIT_BYTES, "stderr_limit_bytes": STDERR_LIMIT_BYTES}
    samples = [Sample(id=index, input="Execute supplied program; no model calls.", metadata={
        "program_base64": base64.b64encode(item["program"]).decode("ascii"), "cases": item["cases"],
        "verification": item.get("verification")})
        for index, item in enumerate(programs)]
    task = Task(name="linux_execution" if operation_id is None else operation_id, dataset=samples, solver=execute_programs(),
                sandbox=("docker", str(COMPOSE)), metadata=metadata)
    return inspect_eval(task, model="mockllm/model", log_dir=str(log_dir), log_format="eval",
        max_sandboxes=MAX_SANDBOXES, max_samples=MAX_SANDBOXES, retry_on_error=0,
        fail_on_error=True, continue_on_fail=True, sandbox_cleanup=True, sandbox_prebuilt=True, score=False, display="none")
