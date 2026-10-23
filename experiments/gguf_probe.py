"""Fixed local GGUF comparison; no acquisition server or automatic retries."""

from contextlib import contextmanager
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_local = importlib.import_module("src.data-pipeline.local_runner")
_exited, _kill_group = _local._exited, _local._kill_group
GGUF_SHA256 = "bd0ca58473b64618df4d08c1410cc25c910ca84e647833ebaf099e0fc0523b45"
BOOTSTRAP = """import os,signal,subprocess,sys,threading
def watch():
    os.read(0,1)
    os.killpg(0,signal.SIGKILL)
threading.Thread(target=watch,daemon=True).start()
child=subprocess.Popen(sys.argv[1:],stdin=subprocess.DEVNULL)
sys.exit(child.wait())
"""

def _bytes(path, value):
    with Path(path).open("xb") as handle:
        handle.write(value)

def _write(path, value):
    _bytes(path, json.dumps(value, sort_keys=True, allow_nan=False).encode())

def _hash_file(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _timeout(*_):
    raise TimeoutError("GGUF wall-time budget expired")

@contextmanager
def _alarm(seconds):
    previous = signal.signal(signal.SIGALRM, _timeout)
    started, timer = monotonic(), signal.getitimer(signal.ITIMER_REAL)
    signal.setitimer(signal.ITIMER_REAL, max(0.000001, min(seconds, timer[0]) if timer[0] else seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if timer[0]:
            signal.setitimer(signal.ITIMER_REAL, max(0.000001, timer[0] - monotonic() + started), timer[1])

def _http(base, path, body=None, timeout=1, capture=None):
    data = None if body is None else json.dumps(body, sort_keys=True, allow_nan=False).encode()
    if capture is not None and data is not None:
        _bytes(str(capture) + ".request.json", data)
    try:
        with urlopen(Request(base + path, data=data, headers={"Content-Type": "application/json"}), timeout=timeout) as response:
            raw = response.read()
    except HTTPError as error:
        if capture is not None:
            _bytes(str(capture) + ".response.json", error.read())
        raise
    if capture is not None:
        _bytes(str(capture) + ".response.json", raw)
    return json.loads(raw)

def _closed(port):
    with socket.socket() as connection:
        connection.settimeout(0.2)
        return connection.connect_ex(("127.0.0.1", port)) != 0

def run_gguf(plan, prompts, root):
    root = Path(root)
    root.mkdir()
    _bytes(root / "probe-source.py", Path(__file__).read_bytes())
    started, process, port, active = monotonic(), None, None, None
    results = [{"status": "not_dispatched", "path": str(root / str(i))} for i in range(len(plan["requests"]))]
    lifecycle = {"group_absent": True, "port_closed": True}
    try:
        with _alarm(plan["limits"]["route_seconds"]):
            model, directory, template = map(Path, (plan["model_path"], plan["llama_dir"], plan["template_path"]))
            hashes = {str(p): {"bytes": p.stat().st_size, "sha256": _hash_file(p)}
                      for p in [model, template, Path(__file__), *sorted(p for p in directory.iterdir() if p.is_file())]}
            _write(root / "hashes.json", hashes)
            if hashes[str(model)]["sha256"] != plan.get("gguf_sha256", GGUF_SHA256):
                raise ValueError("GGUF differs from the declared upstream digest")
            info = subprocess.run([str(directory / "llama-server"), "--version"], capture_output=True, check=True, timeout=15)
            _bytes(root / "version.txt", info.stdout + info.stderr)
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            if not _closed(port):
                raise RuntimeError("Selected port is occupied")
            alias = "ropot-" + os.urandom(12).hex()
            argv = [str(directory / "llama-server"), "--model", str(model), "--alias", alias, "--host", "127.0.0.1",
                    "--port", str(port), "--ctx-size", "8192", "--parallel", "1", "--n-gpu-layers", "99", "--offline",
                    "--jinja", "--chat-template-file", str(template), "--reasoning", "off", "--reasoning-format", "none",
                    "--no-cache-prompt", "--cache-ram", "0", "--cache-reuse", "0", "--no-cache-idle-slots", "--no-context-shift"]
            command = [sys.executable, "-I", "-B", "-c", BOOTSTRAP, *argv]
            _write(root / "launch.json", {"argv": command, "alias": alias, "port": port})
            with (root / "server.log").open("xb") as output:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                                           env={"PATH": "/usr/bin:/bin"})
            lifecycle.update(pid=process.pid, group_absent=False)
            base = f"http://127.0.0.1:{port}"
            with _alarm(plan["limits"]["request_seconds"]):
                while True:
                    if _exited(process):
                        raise RuntimeError("Owned server exited during startup")
                    try:
                        if _http(base, "/health")["status"] == "ok":
                            break
                    except (HTTPError, URLError) as error:
                        if isinstance(error.reason, TimeoutError):
                            raise
                        sleep(0.05)
                models = _http(base, "/v1/models", capture=root / "models")
                props = _http(base, "/props", capture=root / "props")
                if ([m["id"] for m in models["data"]] != [alias] or props["model_path"] != str(model)
                        or props["total_slots"] != 1 or props["default_generation_settings"]["n_ctx"] != 8192):
                    raise ValueError("Server identity or context differs")
            for active, (item, expected) in enumerate(zip(plan["requests"], prompts, strict=True)):
                cell = root / str(active)
                cell.mkdir()
                _write(cell / "plan_request.json", item)
                with _alarm(plan["limits"]["request_seconds"]):
                    rendered = _http(base, "/apply-template", {"messages": item["messages"]}, capture=cell / "template")
                    native = _http(base, "/tokenize", {"content": rendered["prompt"], "add_special": True, "parse_special": True}, capture=cell / "tokenize")
                    _write(cell / "prompt_gate.json", {"reference": expected, "rendered": rendered, "native": native})
                    if native["tokens"] != expected["ids"] or expected["ids"][0] != 1 or len(expected["ids"]) + plan["limits"]["reply_tokens"] > 8192:
                        raise ValueError("Effective prompt IDs differ or exceed context")
                    body = {"prompt": native["tokens"], "n_predict": plan["limits"]["reply_tokens"], **item["controls"],
                            "cache_prompt": False, "return_tokens": True, "stream": False, "stop": [], "ignore_eos": False,
                            "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0, "presence_penalty": 0.0,
                            "frequency_penalty": 0.0, "samplers": ["top_p", "temperature"]}
                    results[active]["status"] = "dispatched"
                    response = _http(base, "/completion", body, timeout=plan["limits"]["request_seconds"], capture=cell / "completion")
                    count = response["tokens_predicted"]
                    settings = response["generation_settings"]
                    if settings["samplers"] != body["samplers"] or any(abs(settings[k] - body[k]) > 0.000001 for k in (*item["controls"], "n_predict")):
                        raise ValueError("Effective sampler controls differ")
                    if (response["stop_type"] not in ("eos", "limit") or response["truncated"] or response["timings"]["cache_n"] != 0
                            or response["tokens_evaluated"] != len(native["tokens"]) or count != len(response["tokens"])
                            or 2 in response["tokens"][:-1] or (response["stop_type"] == "limit" and count != body["n_predict"])
                            or not 0 < count <= body["n_predict"] or (response["stop_type"] == "eos") != (response["tokens"][-1] == 2)):
                        raise ValueError("Native completion accounting differs")
                    results[active].update(status="completed", response=response)
    except Exception as error:
        lifecycle["error"] = repr(error)
        if active is not None:
            results[active].update(status="error", error=repr(error))
    finally:
        original_error = sys.exception()
        if process is not None:
            for cleanup in (lambda: _kill_group(process), process.stdin.close, lambda: process.wait(timeout=5)):
                try:
                    cleanup()
                except BaseException as error:
                    lifecycle.setdefault("cleanup_error", repr(error))
            until = monotonic() + 5
            while monotonic() < until:
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    lifecycle["group_absent"] = True
                except PermissionError:
                    pass
                if lifecycle["group_absent"] and _closed(port):
                    break
                sleep(0.05)
        lifecycle.update(port=port, port_closed=port is None or _closed(port), elapsed_seconds=monotonic() - started)
        _write(root / "lifecycle.json", lifecycle)
        _write(root / "results.json", results)
        if (not lifecycle["group_absent"] or not lifecycle["port_closed"]) and not isinstance(original_error, (KeyboardInterrupt, SystemExit)):
            raise RuntimeError("GGUF process group or port remains active")
    return results
