"""Frozen strict-code Ministral exploration; sizing precedes authorized dispatch."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from experiments import batch09_development as original

b, g = original.b, original.g
ROOT = original.ROOT
OUTPUT = ROOT / 'data/03_benign_experiments/0003-batch09b-synthetic'
EVIDENCE = ROOT / 'data/03_generation_checks/batch09b-preparation'
GATE = ROOT / 'data/03_generation_checks/batch-07b-fkvbcijh/qualification.json'
RECIPE = dict(original.RECIPE,
    system='You write correct, efficient Python 3 programs.',
    initial='Re-read the exact operation and numerical constraints. Check your chosen algorithm against the statement examples. Return exactly one complete Python program in one Python code block. Include no explanations, alternatives or revisions after the program. End immediately after the closing fence.',
    repair='Re-read the exact operation and numerical constraints. Check your chosen algorithm against the statement examples and reported failure. Return exactly one complete Python program in one Python code block. Include no explanations, alternatives or revisions after the program. End immediately after the closing fence.',
    max_tokens=2048, repair_max_tokens=2048, max_repairs=5)


def source_hashes():
    return {str(Path(module.__file__)): g.file_hash(module.__file__) for module in
            (b, g, b.repair_feedback)} | {str(Path(__file__).resolve()): g.file_hash(__file__)}


def measure():
    from mlx_lm.utils import load_config, load_tokenizer
    EVIDENCE.mkdir(parents=True, exist_ok=False)
    options = original.RUNS[1]
    config = load_config(options['model_path'])
    tokenizer = load_tokenizer(options['model_path'], dict(local_files_only=True, fix_mistral_regex=True),
                               eos_token_ids=config.get('eos_token_id'))
    inputs = json.loads((ROOT / 'data/03_benign_experiments/0002-batch09-synthetic/inputs.json').read_text())
    tasks = [(item['binding'], json.loads(Path(item['snapshot']).read_text())) for item in inputs['worklist']]
    requests = b.initial_requests(tasks, RECIPE)
    reply_ids = tokenizer.encode('print(0)\n' * 2048, add_special_tokens=False)[:2048]
    reply = g.decode_tokens(tokenizer, reply_ids, set(tokenizer.eos_token_ids))
    detail = ' '.join(str(i) for i in range(512))
    feedback = dict(status='failed', hidden='failed', visible=[dict(status='wrong_answer', returncode=0,
        **dict.fromkeys(('input','answer','stdout','stderr','diagnostic'), detail)) for _ in range(2)])
    records = []
    for position, request in enumerate(requests):
        for depth in range(RECIPE['max_repairs'] + 1):
            rendered = g.render(tokenizer, request['messages'])
            records.append(dict(position=position, repair_depth=depth, prompt_tokens=rendered['prompt_tokens'],
                input_plus_output=rendered['prompt_tokens']+2048, messages=request['messages']))
            parent = dict(extraction='success', verification=dict(feedback=deepcopy(feedback)))
            request = b.repair_request(request, dict(text=reply), parent, RECIPE, 17+(depth+1)*8+position)
    context = config.get('text_config',config)['max_position_embeddings']
    gate = json.loads(GATE.read_text())
    hashes = source_hashes()
    pipeline = {p:d for p,d in gate['source_sha256_after'].items() if '/src/' in p}
    assert gate['status']=='passed' and gate['source_unchanged']
    assert all(g.file_hash(p)==d for p,d in pipeline.items())
    record = dict(recipe=RECIPE, source_sha256=hashes, gate_pipeline_hashes=pipeline,
        gate_interpretation='Prior live plumbing qualification covers unchanged pipeline modules; edited gate test is not requalified. Five-repair depth is new exploration.',
        stress_examples=records, context=context, maximum=max(r['input_plus_output'] for r in records),
        assumption='Constructed stress histories, not proven token or memory bounds; historical 6882 evidence is exceeded.')
    g.save(EVIDENCE/'sizing.json', record)
    assert record['maximum'] <= context
    print(json.dumps({k:v for k,v in record.items() if k!='stress_examples'},indent=2))


def run():
    sizing = json.loads((EVIDENCE/'sizing.json').read_text())
    assert sizing['recipe']==RECIPE and sizing['source_sha256']==source_hashes()
    assert all(g.file_hash(p)==d for p,d in sizing['gate_pipeline_hashes'].items())
    inputs = json.loads((ROOT/'data/03_benign_experiments/0002-batch09-synthetic/inputs.json').read_text())
    tasks=[]
    for item in inputs['worklist']:
        assert g.file_hash(item['snapshot'])==item['snapshot_sha256']
        tasks.append((item['binding'],json.loads(Path(item['snapshot']).read_text())))
    result=b.run(tasks, OUTPUT, **original.RUNS[1], mlx_python=original.MLX_PYTHON,
        recipe=RECIPE, reply_allowance=48, output_token_allowance=98304,
        cohort_sha256=inputs['cohort_sha256'])
    reader=original.importlib.import_module('src.data-pipeline.stage03.experiment_summary')
    g.save(EVIDENCE/'summary.json',reader.summarize(OUTPUT.parent))
    print(json.dumps(result,indent=2))
    assert result['status']=='completed'


if __name__=='__main__':
    {'measure':measure,'run':run}[sys.argv[1]]()
