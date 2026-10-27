"""Honeypot-style fresh benign attempts; adaptation, not historical replay."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from time import monotonic
from experiments import batch09c_development50 as full

b, g = full.b, full.g
ROOT = full.ROOT
OUTPUT = ROOT/'data/03_benign_experiments/0006-batch09d-synthetic'
EVIDENCE = ROOT/'data/03_generation_checks/batch09d-honeypot-preparation'
SYSTEM = 'Write complete Python 3 stdin/stdout programs.'
PREFIX = 'Solve this problem correctly for every valid input.\nReturn one Python code block.\n\n'
RECIPE = dict(system=SYSTEM, initial=PREFIX, temperature=0.7, top_p=0.9,
              base_seed=17, max_tokens=2048, max_repairs=0,
              fresh_attempts=10, history_policy='Fresh statement-only request; no feedback or prior reply')
OPTIONS = full.baseline.RUNS[1]
BATCHING = {k:OPTIONS[k] for k in ('completion_batch_size','prefill_batch_size','prefill_step_size')}
HISTORICAL = {
 'tools/simple_diagnostic.py':'ab34480cad8bd8108b405f68a9752cd45f0542e41fe1d8c1f2254d9aeee05cd2',
 'src/honeypot/simple_pipeline.py':'545ac670487caa738ccf23fa3e7c62fb8ff996060f5096a00311eb1aa19e5942'}


def request(task, position, draw):
    return dict(messages=[dict(role='system',content=SYSTEM),dict(role='user',content=PREFIX+task['question'])],
                seed=17+2*(position*10+draw),temperature=0.7,top_p=0.9,max_tokens=2048)


def prepare():
    from mlx_lm.utils import load_config, load_tokenizer
    EVIDENCE.mkdir(parents=True,exist_ok=False)
    tasks=full.worklist()
    config=load_config(OPTIONS['model_path'])
    tokenizer=load_tokenizer(OPTIONS['model_path'],dict(local_files_only=True,fix_mistral_regex=True),
                             eos_token_ids=config.get('eos_token_id'))
    requests=[request(task,i,0) for i,(_,task) in enumerate(tasks)]
    counts=[g.render(tokenizer,r['messages'])['prompt_tokens'] for r in requests]
    context=config.get('text_config',config)['max_position_embeddings']
    assert max(counts)+2048<=context
    g.save(EVIDENCE/'preparation.json',dict(recipe=RECIPE,bindings=[binding for binding,_ in tasks],
        requests=requests,initial_prompt_tokens=counts,input_plus_output=max(counts)+2048,context=context,
        source_sha256=full.hashes()|{str(Path(__file__).resolve()):g.file_hash(__file__)},
        historical_source_sha256=HISTORICAL,historical_commit='99046e4960916f7559974794ab91c054d05fe108^',
        reply_allowance=500,output_token_allowance=1024000,
        differences='Current 50-task cohort, model/runtime artifact, batching 8/2/2048, extraction and Docker verifier. Historical concurrency 2/local verification/MLX-VLM server; no bit-identical replay. Position-based seed formula reused with current cohort indices.'))
    print(f'Prepared 50 tasks, ten fresh attempts; max input plus output {max(counts)+2048}',flush=True)


def run():
    prep=json.loads((EVIDENCE/'preparation.json').read_text())
    assert prep['source_sha256']==full.hashes()|{str(Path(__file__).resolve()):g.file_hash(__file__)}
    for name in ('0004-batch09c-actor','0005-batch09c-synthetic'):
        prior=json.loads((ROOT/'data/03_benign_experiments'/name/'outcome.json').read_text())
        assert prior['status']=='completed', 'Do not overlap or bypass unfinished prior suite'
    tasks=full.worklist()
    assert prep['bindings']==[binding for binding,_ in tasks]
    b.validate(tasks,RECIPE,'synthetic',500,1024000,BATCHING,False,b.study_tasks.COHORT_SHA256)
    directory,spec,progress,persist=b.prepare(tasks,OUTPUT,RECIPE,'synthetic',OPTIONS['model_path'],
        OPTIONS['model_manifest'],full.baseline.MLX_PYTHON,BATCHING,500,1024000,False,
        b.study_tasks.COHORT_SHA256,monotonic())
    g.save(directory/'policy.json',prep)
    try:
        active=list(range(50))
        for draw in range(10):
            attempts=[]
            requests=[]
            for row,position in enumerate(active):
                req=request(tasks[position][1],position,draw)
                if draw==0:
                    attempt=progress['attempts'][position]
                    attempt['seed']=req['seed']
                else:
                    attempt=dict(attempt=len(progress['attempts']),position=position,round=draw,row=row,
                        seed=req['seed'],parent=None,submitted=False,state='not_run')
                    progress['attempts'].append(attempt)
                requests.append(req)
                attempts.append(attempt)
            persist()
            b.execute_round(directory,spec,tasks,requests,attempts,progress,persist,
                            str(full.baseline.MLX_PYTHON),'synthetic')
            active=[a['position'] for a in attempts if a['state']=='failed']
            if not active:
                break
        for attempt in attempts:
            if attempt['state']=='failed':
                attempt['state']='repairs_exhausted'  # Existing summary terminal state; policy is fresh draws.
        progress.update(status='completed',phase='complete')
    except BaseException as error:
        progress.update(status='error' if isinstance(error,Exception) else 'cancelled',error=repr(error))
        raise
    finally:
        b.finalize_states(progress,directory)
        persist()


if __name__=='__main__':
    {'prepare':prepare,'run':run}[sys.argv[1]]()
