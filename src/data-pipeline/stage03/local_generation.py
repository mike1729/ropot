"""Direct serial and batched MLX generation. See docs/v1/stage-03-batch-06-plan.md.

Call from the pinned MLX Python with caller-constructed requests. The artifact
manifest must describe the supplied, unchanged model files; hashes are recorded,
not reverified. Wall time is recorded, not enforced: max_tokens bounds each reply,
and Python interruption cannot stop native loading or a stalled GPU call.
"""

from contextlib import closing
from dataclasses import asdict
from hashlib import file_digest, sha256
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
from time import monotonic


def file_hash(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def render(tokenizer, messages, *, enable_thinking=False, reasoning_effort=None, tools=None, preserve_thinking=None):
    if enable_thinking and not tokenizer.has_thinking:
        raise ValueError("Thinking requires tokenizer token-ID boundaries")
    options = dict(add_generation_prompt=True, enable_thinking=enable_thinking)
    if tools is not None:
        if not isinstance(tools, list):
            raise ValueError('tools must be a list')
        options['tools'] = tools
    if preserve_thinking is not None:
        if type(preserve_thinking) is not bool:
            raise ValueError('preserve_thinking must be a boolean')
        options['preserve_thinking'] = preserve_thinking
    if reasoning_effort is not None:
        if not enable_thinking or reasoning_effort not in ("low", "medium", "xhigh"):
            raise ValueError("reasoning_effort requires thinking and low, medium or xhigh")
        options["reasoning_effort"] = reasoning_effort
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, **options)
    ids = tokenizer.apply_chat_template(messages, tokenize=True, **options)
    return dict(prompt=prompt, prompt_ids=ids, prompt_tokens=len(ids), enable_thinking=enable_thinking,
                **({"reasoning_effort": reasoning_effort} if reasoning_effort is not None else {}))


def content_tokens(tokenizer, tokens, eos):
    # Gemma uses its closing tool delimiter as a stop token; keep it for strict call parsing.
    tool_end = bool(tokens) and getattr(tokenizer, 'convert_ids_to_tokens', lambda _: None)(tokens[-1]) == '<tool_call|>'
    return tokens[:-1] if tokens and tokens[-1] in eos and not tool_end else tokens


def decode_tokens(tokenizer, tokens, eos):
    decoder = tokenizer.detokenizer
    for token in content_tokens(tokenizer, tokens, eos):
        decoder.add_token(token)
    decoder.finalize()
    return decoder.text


def split_answer(tokenizer, row):
    """Keep raw evidence intact; decode only the answer after a closed thought.

    Qwen opens its thought in the prompt. Gemma may open one in the reply or
    finish directly. Boundaries come from the loaded tokenizer, never text tags.
    Unclosed thoughts and unmatched closing tokens retain raw evidence but leave
    answer_text=None and cannot supply a program.
    """
    if not row["enable_thinking"]:
        row["answer_text"] = row["text"]
        return
    opening, closing = list(tokenizer.think_start_tokens), list(tokenizer.think_end_tokens)
    opened = tokenizer.rfind_think_start(row["prompt_ids"]) > tokenizer.rfind_think_end(row["prompt_ids"])
    tokens, cursor, answer_start = row["tokens"], 0, 0
    thought_start = 0 if opened else None
    malformed = dict(status="malformed", reasoning_tokens=None, answer_tokens=None)
    single_thought = True
    # Track thought boundaries to locate the answer and reject unmatched closing tokens.
    while cursor < len(tokens):
        if tokens[cursor:cursor + len(opening)] == opening:
            single_thought = single_thought and thought_start is None and cursor == 0
            opened = True
            cursor += len(opening)
            thought_start = cursor
        elif tokens[cursor:cursor + len(closing)] == closing:
            if not opened:
                row.update(answer_text=None, thought_metrics=malformed)
                return
            opened = False
            cursor += len(closing)
            answer_start = cursor
        else:
            cursor += 1
    content = content_tokens(tokenizer, tokens, set(tokenizer.eos_token_ids))
    partial_marker = row.get("finish_reason") == "length" and any(
        content[-size:] == marker[:size] for marker in (opening, closing) for size in range(1, len(marker)))
    content_count = len(content)
    if not single_thought or partial_marker:
        row["thought_metrics"] = malformed
    elif thought_start is None:
        row["thought_metrics"] = dict(status="absent", reasoning_tokens=0, answer_tokens=content_count)
    elif opened:
        row["thought_metrics"] = dict(status="unclosed", reasoning_tokens=content_count - thought_start,
                                      answer_tokens=0)
    else:
        row["thought_metrics"] = dict(status="closed", reasoning_tokens=answer_start - len(closing) - thought_start,
                                      answer_tokens=content_count - answer_start)
    row["answer_text"] = None
    if not opened:
        if answer_start:
            thought = tokens[:answer_start - len(closing)]
            if thought[:len(opening)] == opening:
                thought = thought[len(opening):]
            row['reasoning_content'] = decode_tokens(tokenizer, thought, set())
            row['reasoning_token_count'] = len(thought)
        row["answer_text"] = (decode_tokens(tokenizer, tokens[answer_start:], set(tokenizer.eos_token_ids))
                              if answer_start else row["text"])


def attempt(errors, action):
    """Finish independent cleanup steps even if one fails or is interrupted."""
    try:
        return action()
    except BaseException as error:
        errors.append(error)


def prepare_run(model_path, requests, directory, model_manifest, **metadata):
    if any(type(request.get("enable_thinking", False)) is not bool for request in requests):
        raise ValueError("enable_thinking must be a boolean")
    config = json.loads((model_path / "config.json").read_text())
    if config.get("quantization", config.get("quantization_config", {})).get("bits") != 8:
        raise ValueError("Generation requires 8-bit quantization")
    manifest = Path(model_manifest).read_bytes()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "model-manifest.json").write_bytes(manifest)
    save(directory / "inputs.json", {
        "model_path": str(model_path), "requests": requests, "model_config": config,
        "model_manifest_sha256": sha256(manifest).hexdigest(), "source_sha256": file_hash(__file__),
        "model_metadata_sha256": {name: file_hash(model_path / name) for name in
                                 ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")},
        **metadata})
    return config


def load_model(model_path, directory, config, *, extra_modules=()):
    from mlx_lm import load

    versions = {name: version(name) for name in ("mlx", "mlx-lm", "mlx-metal", "transformers", "tokenizers")}
    tokenizer_options = {"local_files_only": True}
    model_modules = ("qwen3_5",)
    if config.get("model_type") == "mistral3":
        tokenizer_options["fix_mistral_regex"] = True
        model_modules = ("mistral3", "ministral3")
    elif config.get("model_type") in ("gemma4", "gemma4_text"):
        model_modules = ("gemma4", "gemma4_text")
    started = monotonic()
    model, tokenizer = load(str(model_path), lazy=False,
        tokenizer_config=tokenizer_options, trust_remote_code=False)
    tokenizer_class = type(tokenizer._tokenizer)
    runtime = {"versions": versions, "python": sys.version, "executable": sys.executable,
               "tokenizer_options": tokenizer_options,
               "tokenizer_class": f"{tokenizer_class.__module__}.{tokenizer_class.__qualname__}",
               "load_seconds": monotonic() - started, "source_sha256": {}}
    for name in ("mlx_lm.generate", "mlx_lm.sample_utils", "mlx_lm.tokenizer_utils",
                 *(f"mlx_lm.models.{name}" for name in model_modules), *extra_modules):
        path = Path(importlib.import_module(name).__file__)
        runtime["source_sha256"][str(path)] = file_hash(path)
    save(directory / "runtime.json", runtime)
    return model, tokenizer


def generate_request(model, tokenizer, request, row, reply_path):
    """Update one result in place so partial output survives a Python exception."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    started = monotonic()
    row.update(status="running", finish_reason=None)
    try:
        row.update(render(tokenizer, request["messages"], enable_thinking=request.get("enable_thinking", False),
                          reasoning_effort=request.get("reasoning_effort"),
                          **{k: request[k] for k in ('tools', 'preserve_thinking') if k in request}))
        sampler = make_sampler(temp=request["temperature"], top_p=request["top_p"], seed=request["seed"],
                               **({"top_k": request["top_k"]} if "top_k" in request else {}))
        with reply_path.open("w", encoding="utf-8") as reply:
            with closing(stream_generate(model, tokenizer, row["prompt_ids"],
                    max_tokens=request["max_tokens"], sampler=sampler)) as stream:
                for response in stream:
                    row["text"] += response.text
                    row["tokens"].append(response.token)
                    row.update({k: getattr(response, k) for k in
                                ("prompt_tokens", "generation_tokens", "finish_reason")})
                    reply.write(response.text)
                    reply.flush()
        if row["finish_reason"] in ("stop", "length"):
            split_answer(tokenizer, row)
            row["status"] = row["finish_reason"]
        else:
            raise RuntimeError("Native stream ended without completion")
    finally:
        row["elapsed_seconds"] = monotonic() - started


def generate(model_path, requests, output_dir, *, model_manifest):
    """Load once, process a finite list serially, and save plain result files.

    Requests supply messages, seed, temperature, top_p and max_tokens.
    Optional enable_thinking defaults to false. Raw text/tokens retain reasoning;
    answer_text contains only the answer, or None for an unclosed thought.
    An error ends the run without retry.
    Python interruptions save evidence then propagate; hard termination may
    lose metadata. Replies are flushed to text files as chunks arrive.
    """
    model_path, directory = Path(model_path).resolve(), Path(output_dir).absolute()
    config = prepare_run(model_path, requests, directory, model_manifest)
    rows = [{"status": "not_run", "text": "", "tokens": [], "answer_text": None} for _ in requests]
    result = {"status": "error", "requests": rows, "error": None}
    started, active = monotonic(), None
    try:
        model, tokenizer = load_model(model_path, directory, config)
        for index, (request, row) in enumerate(zip(requests, rows)):
            active = row
            generate_request(model, tokenizer, request, row, directory / f"reply-{index:04d}.txt")
        result["status"] = "success"
    except BaseException as error:
        result.update(status="error" if isinstance(error, Exception) else "cancelled", error=repr(error))
        if active is not None and active["status"] not in ("stop", "length"):
            active["status"] = result["status"]
        if not isinstance(error, Exception):
            raise
    finally:
        result["elapsed_seconds"] = monotonic() - started
        save(directory / "outcome.json", result)
    return result


def extract_program(row):
    """Extract only answer_text; compiler resource errors propagate separately."""
    if row["status"] != "stop":
        raise ValueError("Require natural completion")
    text = row["answer_text"]
    if text is None:
        raise ValueError("Require a closed thought before program extraction")
    if not text.strip():
        raise ValueError("Empty program")
    program = text.encode("utf-8")
    try:
        compile(program, "<candidate>", "exec")
    except SyntaxError:
        if "```" not in text:
            raise
        match = re.search(r"^```(?:python|py)?[ \t]*\r?\n(.*?)^```[ \t]*(?:\r?\n|\Z)", text, re.M | re.S)
        if text.count("```") != 2 or match is None:
            raise ValueError("Require one complete unambiguous Python fence") from None
        text = match[1]
        if not text.strip():
            raise ValueError("Empty program")
        program = text.encode("utf-8")
        compile(program, "<candidate>", "exec")
    return program


def generate_batched(model_path, requests, output_dir, *, model_manifest,
                     completion_batch_size, prefill_batch_size, prefill_step_size):
    """Finite token-bounded batching; decode/write once on completion or abort.

    Validate whole calls before routing; retain malformed calls as scalar events.
    Python interruptions propagate after cleanup;
    hard termination can lose unfinished replies still held only in memory.
    """
    batching = dict(completion_batch_size=completion_batch_size,
                    prefill_batch_size=prefill_batch_size, prefill_step_size=prefill_step_size)
    if any(type(n) is not int or n <= 0 for n in batching.values()) or prefill_batch_size > completion_batch_size:
        raise ValueError("Require positive integer settings and prefill <= completion batch size")
    if any(type(n) is not int or n <= 0 for r in requests
           for n in (r["max_tokens"], r.get("max_prompt_tokens", 1))):
        raise ValueError("Require positive integer token limits")
    model_path, directory = Path(model_path).resolve(), Path(output_dir).absolute()
    config = prepare_run(model_path, requests, directory, model_manifest, batching=batching)
    rows = [dict(status="not_run", text="", answer_text=None, tokens=[], submitted=False, native_uid=None,
                 finish_reason=None, elapsed_seconds=None) for _ in requests]
    result = dict(status="error", requests=rows, error=None, cleanup_error=None, batching=batching,
                  max_observed_decode_batch=0, native=None)
    started = monotonic()
    mx = generator = model = tokenizer = stats = old_limit = None
    active, eos = {}, set()

    def write_reply(index):
        row = rows[index]
        row["text"] = decode_tokens(tokenizer, row["tokens"], eos)
        (directory / f"reply-{index:04d}.txt").write_text(row["text"], encoding="utf-8", newline="")

    try:
        import mlx.core as mx
        from mlx_lm.generate import BatchGenerator
        from mlx_lm.sample_utils import make_sampler

        old_limit = mx.set_cache_limit(4 * 1024**3)
        result.update(allocator_cache_limit_bytes=4 * 1024**3,
                      previous_allocator_cache_limit_bytes=old_limit)
        mx.reset_peak_memory()
        model, tokenizer = load_model(model_path, directory, config,
            extra_modules=("mlx_lm.models.cache", "mlx_lm.generate_utils"))
        preparing = monotonic()
        for request, row in zip(requests, rows):
            row.update(render(tokenizer, request["messages"], enable_thinking=request.get("enable_thinking", False),
                              reasoning_effort=request.get("reasoning_effort"),
                              **{k: request[k] for k in ('tools', 'preserve_thinking') if k in request}))
            limit = request.get("max_prompt_tokens")
            if limit is not None and row["prompt_tokens"] > limit:
                raise ValueError("Prompt exceeds the qualified token envelope")
        result["preparation_seconds"] = monotonic() - preparing
        eos = set(tokenizer.eos_token_ids)
        if not eos:
            raise ValueError("Require native EOS tokens")
        result["eos_token_ids"] = sorted(eos)
        generator = BatchGenerator(model, stop_tokens=[[t] for t in sorted(eos)], **batching)
        cursor = 0
        with generator.stats() as stats:
            # Fill free decode slots, route native events and persist finished replies.
            while cursor < len(requests) or active:
                end = min(len(requests), cursor + completion_batch_size - len(active))
                if cursor < end:
                    batch_reqs, batch_rows = requests[cursor:end], rows[cursor:end]
                    samplers = [make_sampler(temp=req["temperature"], top_p=req["top_p"],
                                             seed=req["seed"], **({"top_k": req["top_k"]} if "top_k" in req else {})) for req in batch_reqs]
                    submitted = monotonic() - started
                    for row in batch_rows:
                        row.update(status="running", submitted=True, submitted_seconds=submitted)
                    uids = generator.insert([row["prompt_ids"] for row in batch_rows],
                        [req["max_tokens"] for req in batch_reqs], samplers=samplers)
                    for uid, i, row in zip(uids, range(cursor, end), batch_rows, strict=True):
                        active[uid], row["native_uid"] = (i, row), uid
                    cursor = end
                progress, responses = generator.next()
                observed = monotonic() - started
                # Copy before routing: native responses retain logprobs and caches.
                events = [(r.uid, int(r.token), r.finish_reason) for r in responses]
                responses = None
                if not progress and not events:
                    raise RuntimeError("Native generator ended with unsettled requests")
                if (len({uid for uid, _, _ in events}) != len(events) or any(
                        uid not in active or reason not in (None, "stop", "length") for uid, _, reason in events)):
                    result["rejected_events"] = [dict(uid=u, token=t, finish_reason=r) for u, t, r in events]
                    raise RuntimeError("Native call returned unknown/duplicate UIDs or invalid finish reasons")
                result["max_observed_decode_batch"] = max(result["max_observed_decode_batch"], len(events))
                finished = []
                for uid, token, reason in events:
                    i, row = active[uid]
                    row["tokens"].append(token)
                    if reason is not None:
                        row["finish_reason"] = reason
                        del active[uid]
                        finished.append((i, row, reason))
                write_errors = []
                # Persist every finished peer before propagating a write or decode failure.
                for i, row, reason in finished:
                    try:
                        write_reply(i)
                        split_answer(tokenizer, row)
                        row.update(status=reason, elapsed_seconds=observed - row["submitted_seconds"])
                    except BaseException as error:
                        write_errors.append(error)
                if write_errors:
                    raise next((error for error in write_errors if not isinstance(error, Exception)), write_errors[0])
        if (stats.generation_tokens != sum(len(row["tokens"]) for row in rows)
                or stats.prompt_tokens != sum(row["prompt_tokens"] - 1 for row in rows)):
            raise RuntimeError("Delivered tokens differ from native counters")
        result["status"] = "success"
    except BaseException as error:
        result.update(status="error" if isinstance(error, Exception) else "cancelled", error=repr(error))
        for row in rows:
            if row["status"] == "running":
                row.update(status=result["status"], elapsed_seconds=monotonic() - started - row["submitted_seconds"])
        if not isinstance(error, Exception):
            raise
    finally:
        was_cancelled, errors = result["status"] == "cancelled", []
        if stats is not None:
            result["native"] = asdict(stats)
        if generator is not None:
            attempt(errors, lambda: generator.remove(list(active)))
            attempt(errors, generator.close)
            attempt(errors, lambda: mx.synchronize(generator.stream))
        generator = model = None  # Also abandon an insert interrupted before UID return.
        for i, row in enumerate(rows):
            row["generation_tokens"] = len(row["tokens"])
            if row["submitted"] and row["status"] in ("error", "cancelled"):
                attempt(errors, lambda i=i: write_reply(i))
        if mx is not None:
            attempt(errors, mx.synchronize)
            attempt(errors, lambda: result.update(peak_memory_bytes=mx.get_peak_memory()))
        if old_limit is not None:
            attempt(errors, lambda: mx.set_cache_limit(old_limit))
        if errors:
            result.update(status="error", cleanup_error="; ".join(repr(error) for error in errors))
        result["elapsed_seconds"] = monotonic() - started
        save(directory / "outcome.json", result)
        interruption = next((error for error in errors if not isinstance(error, Exception)), None)
        if interruption is not None and not was_cancelled:
            raise interruption
    return result


if __name__ == "__main__":
    generate_batched(**json.loads(Path(sys.argv[1]).read_text()))
