"""Full development exploration; frozen source-specific recipes, no construction."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from experiments import batch09_development as baseline
from experiments import batch09b_ministral as strict

b, g = baseline.b, baseline.g
ROOT = baseline.ROOT
EVIDENCE = ROOT / 'data/03_generation_checks/batch09c-development50'
RUNS = [(baseline.RUNS[0], dict(baseline.RECIPE), '0004-batch09c-actor'),
        (baseline.RUNS[1], dict(strict.RECIPE), '0005-batch09c-synthetic')]


def worklist():
    cohort = b.study_tasks.load_cohort()
    ids = cohort['groups']['development']['initial']
    assert len(ids) == 50 and len(set(ids)) == 50
    return [b.study_tasks.load_task(cohort, task_id) for task_id in ids]


def hashes():
    return {str(Path(p).resolve()):g.file_hash(p) for p in
            (__file__, baseline.__file__, strict.__file__, b.__file__, g.__file__, b.repair_feedback.__file__)}


def measure():
    from mlx_lm.utils import load_config, load_tokenizer
    EVIDENCE.mkdir(parents=True, exist_ok=False)
    tasks = worklist()
    gate = json.loads(strict.GATE.read_text())
    pipeline = {p:d for p,d in gate['source_sha256_after'].items() if '/src/' in p}
    assert gate['status']=='passed' and gate['source_unchanged']
    assert all(g.file_hash(p)==d for p,d in pipeline.items())
    records=[]
    for options, recipe, label in RUNS:
        config=load_config(options['model_path'])
        tokenizer=load_tokenizer(options['model_path'],
            dict(local_files_only=True, **({'fix_mistral_regex':True} if options['source']=='synthetic' else {})),
            eos_token_ids=config.get('eos_token_id'))
        requests=b.initial_requests(tasks,recipe)
        counts=[g.render(tokenizer,r['messages'])['prompt_tokens'] for r in requests]
        position=max(range(len(counts)),key=counts.__getitem__)
        request=requests[position]
        tokens=tokenizer.encode('print(0)\n'*recipe['max_tokens'],add_special_tokens=False)[:recipe['max_tokens']]
        reply=g.decode_tokens(tokenizer,tokens,set(tokenizer.eos_token_ids))
        detail=' '.join(str(i) for i in range(512))
        feedback=dict(status='failed',hidden='failed',visible=[dict(status='wrong_answer',returncode=0,
            **dict.fromkeys(('input','answer','stdout','stderr','diagnostic'),detail)) for _ in range(2)])
        histories=[]
        for depth in range(recipe['max_repairs']+1):
            rendered=g.render(tokenizer,request['messages'])
            histories.append(dict(depth=depth,prompt_tokens=rendered['prompt_tokens'],
                input_plus_output=rendered['prompt_tokens']+recipe['max_tokens'],messages=request['messages']))
            request=b.repair_request(request,dict(text=reply),
                dict(extraction='success',verification=dict(feedback=deepcopy(feedback))),recipe,17+(depth+1)*50+position)
        context=config.get('text_config',config)['max_position_embeddings']
        assert max(h['input_plus_output'] for h in histories)<=context
        records.append(dict(source=options['source'],directory=label,recipe=recipe,initial_prompt_tokens=counts,
            stress_position=position,stress=histories,context=context,
            reply_allowance=50*(recipe['max_repairs']+1),output_token_allowance=50*(recipe['max_repairs']+1)*recipe['max_tokens']))
    g.save(EVIDENCE/'preparation.json',dict(bindings=[binding for binding,_ in tasks],runs=records,
        source_sha256=hashes(),gate_pipeline_hashes=pipeline,
        assumptions='Constructed stress histories are not maxima or memory bounds. Prior passing gate covers unchanged pipeline plumbing; depth five is exploration. First eight tasks are previously used; other 42 are first recipe evaluation, same development cohort.'))
    print(json.dumps([dict(source=r['source'],maximum=max(h['input_plus_output'] for h in r['stress']),
        replies=r['reply_allowance'],tokens=r['output_token_allowance']) for r in records]),flush=True)


def run():
    prep=json.loads((EVIDENCE/'preparation.json').read_text())
    assert prep['source_sha256']==hashes()
    assert all(g.file_hash(p)==d for p,d in prep['gate_pipeline_hashes'].items())
    tasks=worklist()
    assert [binding for binding,_ in tasks]==prep['bindings']
    assert all(not (ROOT/'data/03_benign_experiments'/label).exists() for _,_,label in RUNS)
    reader=baseline.importlib.import_module('src.data-pipeline.stage03.experiment_summary')
    for options,recipe,label in RUNS:
        print('Starting '+label,flush=True)
        result=b.run(tasks,ROOT/'data/03_benign_experiments'/label,**options,
            mlx_python=baseline.MLX_PYTHON,recipe=recipe,
            reply_allowance=50*(recipe['max_repairs']+1),
            output_token_allowance=50*(recipe['max_repairs']+1)*recipe['max_tokens'],
            cohort_sha256=b.study_tasks.COHORT_SHA256)
        g.save(EVIDENCE/'summary.json',reader.summarize(ROOT/'data/03_benign_experiments'))
        print(json.dumps(dict(source=options['source'],status=result['status'],tasks=result['tasks'],
            submissions=result['known_submissions'],tokens=result['observed_tokens'])),flush=True)
        if result['status']!='completed':
            raise RuntimeError('Technical failure; stop without retry or dispatching the next source')


if __name__=='__main__':
    {'measure':measure,'run':run}[sys.argv[1]]()
