"""Bind a configured DeepInfra candidate; this is not acquisition qualification."""
import json
import math
from hashlib import sha256
from pathlib import Path

from . import local_generation as generation

ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / 'scripts/deepinfra_generation.py'
MODEL = 'Qwen/Qwen3.8-27B:61nNTQpO'
EOS = (248046, 248044)
GEMMA_MODEL = 'google/gemma-4-31B-it:wcache-3'
MODEL_PROFILES = {
    MODEL: dict(source='actor', model_types=('qwen3_5',), eos_token_ids=EOS,
        think_start_tokens=(248068,), think_end_tokens=(248069,),
        think_start_text='<think>', think_end_text='</think>', quantization=None),
    GEMMA_MODEL: dict(source='synthetic', model_types=('gemma4', 'gemma4_text'), eos_token_ids=(1, 106, 50),
        think_start_tokens=(100, 45518), think_end_tokens=(101,),
        think_start_text='<|channel>thought', think_end_text='<channel|>', quantization='fp8'),
}
TOKENIZER_FILES = ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                   'vocab.json', 'merges.txt', 'special_tokens_map.json', 'added_tokens.json')


def positive(value, *, zero=False):
    return type(value) in (int, float) and math.isfinite(value) and (value >= 0 if zero else value > 0)


def model_profile(model):
    if not isinstance(model, str) or model not in MODEL_PROFILES:
        raise ValueError('Require a supported, version-pinned DeepInfra model')
    return MODEL_PROFILES[model]


def validate(runtime):
    fields = {'model', 'tokenizer_path', 'tokenizer_sha256', 'timeout_seconds', 'max_total_tokens',
              'input_usd_per_million', 'output_usd_per_million', 'max_cost_usd'}
    if (not isinstance(runtime, dict) or not fields <= set(runtime) or set(runtime) - fields - {'concurrency', 'output_protocol'}
            or not isinstance(runtime['model'], str) or runtime['model'] not in MODEL_PROFILES
            or runtime.get('output_protocol', 'token_ids') not in ('token_ids', 'gemma_native_text')
            or (runtime.get('output_protocol') == 'gemma_native_text' and runtime['model'] != GEMMA_MODEL)
            or type(runtime.get('concurrency', 1)) is not int or not 1 <= runtime.get('concurrency', 1) <= 200
            or type(runtime['max_total_tokens']) is not int or runtime['max_total_tokens'] <= 0
            or not all(positive(runtime[k]) for k in ('timeout_seconds', 'max_cost_usd'))
            or not all(positive(runtime[k], zero=True) for k in ('input_usd_per_million', 'output_usd_per_million'))):
        raise ValueError('Require an explicit pinned DeepInfra runtime')


def tokenizer_files(runtime):
    root, hashes = Path(runtime['tokenizer_path']), runtime['tokenizer_sha256']
    present = {name for name in TOKENIZER_FILES if (root / name).exists()}
    if not set(TOKENIZER_FILES[:4]) <= present or set(hashes) != present:
        raise ValueError('Pin every tokenizer input, including the template and config')
    files = {str((root / name).resolve()): digest for name, digest in hashes.items()}
    if any(generation.file_hash(path) != digest for path, digest in files.items()):
        raise ValueError('Tokenizer hash mismatch')
    return files


def bind(path, reply_allowance, output_token_allowance, sources, *, source):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    runtime = json.loads(raw)
    validate(runtime)
    if model_profile(runtime['model'])['source'] != source:
        raise ValueError('DeepInfra model does not match the requested actor/synthetic source')
    runtime['tokenizer_path'] = str(Path(runtime['tokenizer_path']).resolve(strict=True))
    planned = (reply_allowance * runtime['max_total_tokens'] * runtime['input_usd_per_million']
               + output_token_allowance * runtime['output_usd_per_million']) / 1e6
    if not math.isfinite(planned) or planned > runtime['max_cost_usd']:
        raise ValueError('Entire finite run exceeds the planned uncached spending ceiling')
    files = tokenizer_files(runtime)
    files.update({str(p.resolve()): generation.file_hash(p) for p in
                  (*sources, WORKER, Path(__file__), ROOT / 'requirements-deepinfra.txt')})
    digest = sha256(raw).hexdigest()
    files[str(path)] = digest
    return dict(name='deepinfra', status='configured_candidate', runtime_file=str(path),
                runtime_file_sha256=digest, runtime=runtime, files=files,
                planned_max_cost_usd=planned, reply_allowance=reply_allowance,
                output_token_allowance=output_token_allowance)
