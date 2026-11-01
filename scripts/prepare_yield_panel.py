"""Prepare the user-selected development diagnostic; never starts generation."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import dev_construction_queue as q

e = q.e
IDS = [4130, 4418, 1317, 2477, 1359, 181, 604, 4002]
PRIORITIES = dict(gemma=[4130, 4418, 1317, 2477, 1359, 604, 4002], qwen=[181, 604, 4002])


def select(panels):
    """Select the same eight frozen development tasks in the same order for both sources."""
    indexed = {}
    for source, (entries, worklist, _, _) in panels.items():
        indexed[source] = {x['binding']['id']: (x, w) for x, w in zip(entries, worklist, strict=True)}
    for task_id in IDS:
        left, right = (indexed[s][task_id][0] for s in ('qwen', 'gemma'))
        if (left['binding']['owner'] != 'development' or left['binding'] != right['binding']
                or left['partition'] != right['partition']):
            raise ValueError(f'Task {task_id}: require shared frozen development cases')
    return {source: [indexed[source][i] for i in IDS] for source in PRIORITIES}


def prepare(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Refuse to replace an existing diagnostic directory')
    q.GEMMA = q.DATA / 'gemma4-dev45-construction39-20261030-retry01'
    panels = dict(qwen=q.load_qwen(q.QUEUE), gemma=q.load_gemma(q.GEMMA))
    selected = select(panels)
    config_path = e.ROOT / 'configs/inference/native-multi-edit.json'
    config = e.loop.read(config_path)
    config['completion_batch_size'] = 4
    chains, recipe = e.settings(config)
    turns = recipe['max_repairs'] + 1
    tokens = recipe['max_tokens'] + (turns - 2) * recipe['repair_max_tokens'] + recipe['thinking_repair_max_tokens']
    sources = dict(e.a.acquisition_sources())
    for path in (Path(__file__).resolve(), Path(e.__file__).resolve(), Path(e.native_multi_edit.__file__).resolve(),
                 e.native_multi_edit.PROMPT, config_path, Path(e.qwen_native_tools.__file__).resolve(),
                 Path(e.qwen_edits.__file__).resolve(), e.CHUNKED_DRIVER):
        sources[str(path)] = e.g.file_hash(path)
    output.mkdir(parents=True)
    budgets = {}
    for source, chosen in selected.items():
        entries, worklist = zip(*chosen, strict=True)
        source_config = dict(config, task_count=len(chosen))
        panel = dict(entries=entries, worklist=worklist, runtime=panels[source][2],
                     inputs=dict(panels[source][3], **sources), config=source_config)
        e.g.save(output / f'{source}-panel.json', panel)
        prompt_recipe = deepcopy(recipe)
        e.native_multi_edit.configure(prompt_recipe)
        prompts = [dict(task_id=x['binding']['id'], messages=e.native_multi_edit.messages(
            e.a.attack_messages(w[1], x, 'actor', recipe=prompt_recipe), prompt_recipe)) for x, w in chosen]
        e.g.save(output / f'{source}-initial-prompts.json', prompts)
        budgets[source] = dict(requests=len(chosen) * chains * turns, output_tokens=len(chosen) * chains * tokens,
            shown_cases=chains * turns * sum(len(x['partition']['examples']) + len(x['partition']['menu']) for x in entries))
    e.g.save(output / 'plan.json', dict(status='prepared_not_launched', task_ids=IDS, priorities=PRIORITIES,
        selection='Both sources on the same eight user-selected tasks in identical order; priorities are context only. No prior attacks in prompts.',
        budgets=budgets, inputs={str(p): e.g.file_hash(p) for p in sorted(output.glob('*.json'))}, sources=sources))
    return budgets


def load_panel(output, source):
    """Read frozen public inputs for qwen_experiment.run after a separate launch decision."""
    output = Path(output).resolve()
    if source not in PRIORITIES:
        raise ValueError('Unknown source')
    plan = e.loop.read(output / 'plan.json')
    e.loop.check_files(plan['inputs'])
    panel = e.loop.read(output / f'{source}-panel.json')
    e.loop.check_files(panel['inputs'])
    cohort = e.a.study_tasks.load_cohort()
    for entry in panel['entries']:
        if e.a.study_tasks.load_task(cohort, entry['binding']['id'])[0] != entry['binding']:
            raise ValueError('Task binding changed')
        benign = entry['benigns']['actor']
        if e.a.programs.strip_comments(Path(benign['path']).read_text()) != benign['program']:
            raise ValueError('Benign program changed')
    files = dict(panel['inputs'], **plan['inputs'])
    files[str(output / 'plan.json')] = e.g.file_hash(output / 'plan.json')
    return panel['entries'], panel['worklist'], panel['runtime'], files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    print(prepare(parser.parse_args().output))
