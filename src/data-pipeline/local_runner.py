"""Local reference execution (not a sandbox); see docs/stage-01-plan.md.

Enter LocalRunner on the main thread; infrastructure/cleanup failures must abort.
"""

from contextlib import closing, ExitStack
from concurrent.futures import ThreadPoolExecutor
import errno
import json
import os
from pathlib import Path
import select
import selectors
import signal
import subprocess
import tempfile
import threading
from time import monotonic

from .screen import output_matches


INTERPRETER = "/opt/homebrew/bin/python3.12"
FLAGS = ["-I", "-B", "-X", "int_max_str_digits=0"]
TIMEOUT_SECONDS = 5
MAX_CONCURRENT_CASES = 6
STDOUT_LIMIT_BYTES = 1024 * 1024
STDERR_LIMIT_BYTES = 64 * 1024
ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}


def preflight(interpreter=INTERPRETER):
    """Require Python 3.12 and return JSON-ready runtime and execution provenance.

    Record version, paths, platform, flags, environment and limits, without fallback.
    Wrong versions raise RuntimeError; launch, timeout and decoding errors propagate.
    """
    probe = subprocess.run(
        [interpreter, *FLAGS, "-c", (
            "import json,os,platform,sys; print(json.dumps(dict("
            "version=sys.version,version_info=list(sys.version_info[:3]),"
            "resolved_executable=os.path.realpath(sys.executable),"
            "macos_version=platform.mac_ver()[0],architecture=platform.machine())))"
        )], capture_output=True, check=True, timeout=TIMEOUT_SECONDS, env=ENVIRONMENT,
    )
    runtime = json.loads(probe.stdout)
    if runtime["version_info"][:2] != [3, 12]:
        raise RuntimeError("Local qualification requires the explicit Python 3.12 interpreter")
    return {
        **runtime, "requested_executable": interpreter, "flags": list(FLAGS),
        "execution": "local_subprocess", "sandbox": False,
        "environment": {**ENVIRONMENT, "TMPDIR": "per-case working directory"},
        "timeout_seconds": TIMEOUT_SECONDS, "stdout_limit_bytes": STDOUT_LIMIT_BYTES,
        "stderr_limit_bytes": STDERR_LIMIT_BYTES, "memory_limit_bytes": None,
        "max_concurrent_cases": MAX_CONCURRENT_CASES,
    }


def _kill_group(process):
    """Kill before reaping; on Darwin exit EPERM, only probe the group after waiting."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except PermissionError:
        try:
            process.wait(timeout=0.1)
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    except ProcessLookupError:
        pass
    except OSError:
        process.kill()
        raise


def _exited(process):
    """Observe macOS child exit without releasing its PID before group cleanup."""
    with closing(select.kqueue()) as queue:
        event = select.kevent(process.pid, filter=select.KQ_FILTER_PROC, fflags=select.KQ_NOTE_EXIT)
        events = queue.control([event], 1, 0)
        if events and events[0].flags & select.KQ_EV_ERROR and events[0].data != errno.ESRCH:
            raise OSError(events[0].data, "Cannot observe case process exit")
        return bool(events)


class LocalRunner:
    """Share up to MAX_CONCURRENT_CASES subprocess slots across concurrent callers.

    Enter once on the main thread; callers schedule tasks and reference cases are
    sequential. Process creation/I/O run on workers. Signals cancel them before
    raising, also outside runner calls. Exit joins cleanup and restores handlers.
    """

    def __init__(self, interpreter=INTERPRETER):
        self.provenance = preflight(interpreter)
        self._interpreter = interpreter
        self._pool = None
        self._cancelled = False

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread() or self._pool is not None:
            raise RuntimeError("Enter LocalRunner once, on the main thread")
        self._handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        self._pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CASES)
        try:
            for sig in self._handlers:
                signal.signal(sig, self._interrupt)
        except BaseException:
            self.__exit__(BaseException, None, None)
            raise
        return self

    def _interrupt(self, signum, frame):
        self._cancelled = True
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    def _check_cancelled(self):
        if self._cancelled:
            raise RuntimeError("Local runner has closed")

    def __exit__(self, exc_type, exc, traceback):
        self._cancelled = True
        try:
            self._pool.shutdown(wait=True, cancel_futures=True)
        finally:
            for sig, handler in self._handlers.items():
                signal.signal(sig, handler)

    def run_case(self, source, stdin, answer):
        """Run one case, block for completion and require one exact output token.

        ``source`` and ``stdin`` are original text; ``answer`` is the expected token.
        Return status, bounded stdout/stderr text, returncode and elapsed_seconds.
        Status is passed, wrong_answer, nonzero_exit, timeout, stdout_limit,
        stderr_limit or invalid_stdout; stderr uses UTF-8 replacement. Case timing
        includes startup and I/O, excluding queue wait, file preparation and cleanup.
        Infrastructure errors and interruptions propagate instead of returning a status.
        """
        self._check_cancelled()
        if self._pool is None:
            raise RuntimeError("Use LocalRunner as a context manager")
        return self._pool.submit(self._run_case, source, stdin, answer).result()

    def _run_case(self, source, stdin, answer):
        """Execute with fresh files; always kill the group, reap the child and remove files."""
        with tempfile.TemporaryDirectory(prefix="ropot-case-") as directory, ExitStack() as cleanup:
            path = Path(directory) / "reference.py"
            path.write_text(source, encoding="utf-8")
            data = memoryview(stdin.encode("utf-8"))
            self._check_cancelled()
            started = monotonic()
            process = subprocess.Popen(
                [self._interpreter, *FLAGS, str(path)], cwd=directory,
                env={**ENVIRONMENT, "TMPDIR": directory}, start_new_session=True,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            for pipe in (process.stdin, process.stdout, process.stderr):
                cleanup.callback(pipe.close)
            cleanup.callback(process.wait, timeout=5)
            cleanup.callback(_kill_group, process)
            status, stdout, stderr = self._pump(process, data, started)
            elapsed = monotonic() - started
            if status is None and elapsed >= TIMEOUT_SECONDS:
                status = "timeout"
            cleanup.close()
            try:
                stdout = stdout.decode("utf-8")
            except UnicodeDecodeError:
                stdout = stdout.decode("utf-8", errors="replace")
                status = status or "invalid_stdout"
            status = status or ("nonzero_exit" if process.returncode else
                                "passed" if output_matches(stdout, answer) else "wrong_answer")
            return {"status": status, "stdout": stdout,
                    "stderr": stderr.decode("utf-8", errors="replace"),
                    "returncode": process.returncode, "elapsed_seconds": elapsed}

    def _pump(self, process, data, started):
        """Multiplex pipes; return (timeout/cap status or None, bounded stdout, stderr)."""
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        limits = {"stdout": STDOUT_LIMIT_BYTES, "stderr": STDERR_LIMIT_BYTES}
        with selectors.DefaultSelector() as selector:
            for name in ("stdin", "stdout", "stderr"):
                pipe = getattr(process, name)
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_WRITE if name == "stdin" else
                                  selectors.EVENT_READ, name)
            while selector.get_map() or not _exited(process):
                self._check_cancelled()
                remaining = TIMEOUT_SECONDS - (monotonic() - started)
                if remaining <= 0:
                    return "timeout", buffers["stdout"], buffers["stderr"]
                for key, _ in selector.select(min(remaining, 0.05)):
                    pipe, name = key.fileobj, key.data
                    try:
                        if name == "stdin":
                            written = os.write(pipe.fileno(), data[:65536]) if data else 0
                            data = data[written:]
                            closed = not data
                        else:
                            chunk = os.read(pipe.fileno(), min(65536, limits[name] - len(buffers[name])))
                            buffers[name].extend(chunk)
                            if len(buffers[name]) >= limits[name]:
                                return name + "_limit", buffers["stdout"], buffers["stderr"]
                            closed = not chunk
                    except BrokenPipeError:
                        closed = True
                    except BlockingIOError:
                        continue
                    if closed:
                        selector.unregister(pipe)
                        pipe.close()
            return None, buffers["stdout"], buffers["stderr"]

    def run_reference(self, source, cases):
        """Run one source against cases sequentially, stopping at the first failure.

        Each case supplies index/input/answer; preserve supplied order and indices.
        Return passed (false for no cases), attempted cases, max_case_seconds and
        wall_seconds. Total wall time includes queue waits, preparation and cleanup.
        """
        started, results = monotonic(), []
        for case in cases:
            result = self.run_case(source, case["input"], case["answer"])
            results.append({"index": case["index"], **result})
            if result["status"] != "passed":
                break
        return {"passed": bool(results) and all(r["status"] == "passed" for r in results),
                "cases": results,
                "max_case_seconds": max((r["elapsed_seconds"] for r in results), default=0),
                "wall_seconds": monotonic() - started}
