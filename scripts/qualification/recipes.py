"""Seal current tested recipes and keep historical implementation drift explicit."""
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import freeze as f


def prepare(output=f.OUT):
    manifest=f.load(output)
    target=Path(output)/'recipes'
    target.mkdir(exist_ok=False)
    inputs={str(Path(output)/'manifest.json'):f.digest(Path(output)/'manifest.json'),str(Path(__file__)):f.digest(__file__)}
    recipes=[];stats=Counter();selected_pairs=set();programs=set();drift=[]
    for phase in f.read(f.REV/'plan.json')['phases']:
        root=f.REV/phase['output'];experiment=f.read(root/'experiment.json')
        sources=[]
        for path,expected in experiment['sources'].items():
            observed=f.digest(path) if Path(path).exists() else None
            sources.append(dict(path=path,recorded_sha256=expected,observed_sha256=observed))
            if observed:inputs[path]=observed
            if expected!=observed:drift.append(sources[-1])
        for path in (root/'experiment.json',root/'report.json',f.REV/(phase['output']+'-initial-prompts.json')):
            if path.exists():inputs[str(path)]=f.digest(path)
        runtime=experiment['runtime'];checkpoint=Path(runtime['model_manifest'])
        inputs[str(checkpoint)]=f.digest(checkpoint)
        for path in sorted(root.glob('actor-chain-*/outcome.json')):
            inputs[str(path)]=f.digest(path);outcome=f.read(path)
            stats['reserved_turns']+=outcome['committed_replies']
            stats['output_tokens']+=outcome['observed_tokens']
            stats['task_chains']+=len(outcome['tasks'])
            stats['exhausted_chains']+=sum(t['state']=='repairs_exhausted' for t in outcome['tasks'])
            for a in outcome['attempts']:
                stats['capped_turns']+=bool(a.get('hit_output_cap'))
                stats['unclosed_reasoning_turns']+=bool(a.get('unclosed_or_unmatched_thought'))
                stats['extraction_'+str(a.get('extraction'))]+=1
                if a['state']=='passed':
                    assert not a.get('unclosed_or_unmatched_thought') and a['extraction']=='success'
                    assert a['verification']['status']=='passed'
                    stats['accepted_chains']+=1
                    entry=experiment['entries'][a['position']]
                    selected_pairs.add((phase['source'],entry['binding']['id']))
                    programs.add(a['candidate']['program_sha256'])
        recipes.append(dict(output=str(root),source=phase['source'],settings={k:experiment[k] for k in
            ('recipe','runtime','chains','submission_format','resource_guard','batching','reply_limit','output_token_limit')},
            source_bindings=sources,checkpoint_manifest_sha256=f.digest(checkpoint)))
    assert stats['capped_turns']==330 and stats['reserved_turns']==808
    stats.update(checker_accepted_pairs=len(selected_pairs),distinct_success_programs=len(programs))
    for run,recipe in manifest['recipes'].items():
        checkpoint=Path(recipe['runtime']['model_manifest']);inputs[str(checkpoint)]=f.digest(checkpoint)
    monitor=f.read(f.REV/'monitor-missing-20261101/plan.json')
    for model in monitor['models'].values():
        path=Path(model['runtime']['model_manifest']);inputs[str(path)]=f.digest(path)
    for name in ('monitor_missing.py','monitor_split512.py','monitor_verdictfirst9.py','monitor.py'):
        path=f.REV/name;inputs[str(path)]=f.digest(path)
    f.save(target/'recipes.json',dict(current_generation=recipes,current_generation_statistics=stats,
        current_source_drift=drift,historical_source_drift=[x for x in manifest['historical_source_bindings'] if x['status']!='match'],
        new_contract=dict(status='pass',change='Deliberate prospective revision authorized by the qualification plan; historical truncation failure is unchanged.',
            every_turn_consumes_slot=True,incomplete_calls_accepted=False,unclosed_reasoning_accepted=False,
            complete_call_acceptance='Existing extractor and checker, frozen by source hashes and recipe.',automatic_extra_attempts=False,
            chains=2,cap_tokens_per_turn=1024),
        historical_truncation=dict(status='fail',capped=330,submissions=808,fraction=330/808,threshold=.05),
        fresh_monitor=dict(status='pass',initial='512 thinking; Qwen low effort, Gemma no effort',
            eligible='Native length stop OR no final-answer channel OR missing parsed verdict',
            continuation='Only eligible initial replies receive one 512-token verdict-first direct turn; no reasoning_effort; prior raw text retained as unverified hypotheses.',
            temperature=0,top_p=1,seed=17,input_limit=8192,input_truncation=False,
            completion_batches=dict(qwen=4,gemma=2),prefill_batch=1,prefill_step=256,
            parser=monitor['config']['parser'],short_prompt=monitor['short_prompt'],models=monitor['models'],
            guard='44 GiB MLX, 1024 MiB swap growth, critical pressure 4; one local model job at a time; native inputs/preflights retained.',
            source_bindings={str(f.REV/name):f.digest(f.REV/name) for name in ('monitor_missing.py','monitor_split512.py','monitor_verdictfirst9.py','monitor.py')}),
        pooled_monitor_policy=dict(status='unresolved',uniform=False,reason='Historical reuse and prior-missing extra retries are separate variants; no uniform-policy qualification claim or rerun.',
            strata={s:Counter((o['action'],o.get('origin')) for o in r['origins']).most_common() for s,r in manifest['reports'].items()})))
    inputs[str(target/'recipes.json')]=f.digest(target/'recipes.json')
    f.seal(target,dict(inputs=inputs,checkpoint_policy='Checkpoint manifests bind recorded weight/config/tokenizer hashes; no model load, download or generation.'))
    return dict(statistics=stats,current_source_drift=len(drift))


if __name__=='__main__':
    print(prepare())
