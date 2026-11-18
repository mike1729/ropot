"""Gemma-specific provider boundaries and synthetic routing; no model or HTTP calls."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import deepinfra_generation as d, native_multi_edit
from tests.test_thinking_generation import ThinkingTokenizer

b = importlib.import_module('src.data-pipeline.stage03.generation_loop')
GEMMA_MODEL = 'google/gemma-4-31B-it:wcache-3'
VERSIONS = dict(transformers='5.23.0', tokenizers='0.23.3', jinja2='3.1.6')
CALL = '<|tool_call>call:test_edits{edits:[{old:"0",new:"1"}]}<tool_call|>'


class GemmaTokenizer(ThinkingTokenizer):
    def __init__(self):
        super().__init__('gemma')
        self.think_start_tokens, self.think_end_tokens = (100, 45518), (101,)
        self.eos_token_ids = [1, 106, 50]
        self.pieces = {1: '<eos>', 106: '<turn|>', 50: '<|tool_response>',
            48: '<|tool_call>', 49: '<tool_call|>', 9: 'call:test_edits{edits:[{old:"0",new:"1"}]}',
            100: '<|channel>', 45518: 'thought', 101: '<channel|>', 7: 'print(1)\n', 8: 'Private thought.\n'}
        self.apply_chat_template = mock.Mock(side_effect=lambda messages, tokenize, **kwargs:
            [2] if tokenize else 'Rendered Gemma prompt')

    def convert_ids_to_tokens(self, token):
        return self.pieces[token]


class GemmaWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tokenizer = GemmaTokenizer()
        self.runtime = dict(model=GEMMA_MODEL, tokenizer_path='unused', tokenizer_sha256={},
            timeout_seconds=5, max_total_tokens=100, input_usd_per_million=.2,
            output_usd_per_million=.4, max_cost_usd=.01)
        self.request = dict(messages=[dict(role='user', content='Change print(0)')], seed=17,
            temperature=1., top_p=.95, max_tokens=20, tools=native_multi_edit.TOOLS,
            enable_thinking=False, preserve_thinking=True)
        mock.patch.object(d, 'load_tokenizer', return_value=self.tokenizer).start()
        self.sender = mock.patch.object(d, 'send').start()
        mock.patch.dict('os.environ', DEEPINFRA_API_KEY='PRIVATE_GEMMA_TEST_KEY').start()
        self.addCleanup(mock.patch.stopall)

    def response(self, tokens, prompt_tokens=1):
        visible = tokens[:-1] if tokens and tokens[-1] in self.tokenizer.eos_token_ids else tokens
        body = dict(choices=[dict(index=0, text=''.join(self.tokenizer.pieces[t] for t in visible),
            token_ids=tokens, finish_reason='stop')], usage=dict(prompt_tokens=prompt_tokens,
            completion_tokens=len(tokens), estimated_cost=.00001))
        return 200, json.dumps(body)

    def text_response(self, text, *, reason='stop', prompt_tokens=1, completion_tokens=10):
        return 200, json.dumps(dict(choices=[dict(index=0, text=text, token_ids=[],
            finish_reason=reason, logprobs=None)], usage=dict(prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, estimated_cost=.00001, prompt_tokens_details=None)))

    def run_round(self, name='run'):
        specification = dict(runtime=self.runtime, requests=[self.request], output_dir=str(self.root / name))
        path = self.root / f'{name}.json'
        path.write_text(json.dumps(specification))
        return d.run(path)

    def test_gemma_native_call_uses_its_model_and_stops_without_losing_closer(self):
        self.sender.return_value = self.response([48, 9, 49, 50])
        result = self.run_round()
        self.assertEqual(result['status'], 'success', result)
        payload = self.sender.call_args.args[0]
        self.assertEqual(payload['model'], GEMMA_MODEL)
        self.assertEqual(set(payload['stop_token_ids']), {1, 106, 50})
        self.assertNotIn(49, payload['stop_token_ids'])
        row = result['requests'][0]
        self.assertEqual(row['tokens'], [48, 9, 49, 50])
        self.assertEqual(row['text'], CALL)
        self.assertEqual(row['answer_text'], CALL)
        self.assertEqual(native_multi_edit.parse_call(row['answer_text'])[1], [dict(old='0', new='1')])
        self.assertEqual(row['generation_tokens'], 4)

    def test_each_gemma_eos_is_stripped_from_program_text(self):
        self.request.pop('tools')
        for token in (1, 106, 50):
            with self.subTest(token=token):
                self.sender.return_value = self.response([7, token])
                result = self.run_round(f'eos-{token}')
                self.assertEqual(result['status'], 'success', result)
                self.assertEqual(b.generation.extract_program(result['requests'][0]), b'print(1)\n')
                self.assertEqual(result['requests'][0]['tokens'], [7, token])

    def test_multitoken_thought_opening_requires_a_real_matching_closure(self):
        self.request['enable_thinking'] = True
        cases = [([100, 45518, 8, 101, 7, 106], 'print(1)\n'),
                 ([100, 45518, 7, 106], None), ([101, 7, 106], None)]
        for index, (tokens, answer) in enumerate(cases):
            with self.subTest(tokens=tokens):
                self.sender.return_value = self.response(tokens)
                result = self.run_round(f'thought-{index}')
                self.assertEqual(result['status'], 'success', result)
                row = result['requests'][0]
                self.assertEqual(row['answer_text'], answer)
                self.assertEqual(row['tokens'], tokens)
                if answer is None:
                    with self.assertRaises(ValueError):
                        b.generation.extract_program(row)
                else:
                    self.assertEqual(row['reasoning_content'], 'Private thought.\n')

    def test_unsupported_reasoning_controls_fail_without_submission(self):
        original = deepcopy(self.request)
        for index, changes in enumerate((dict(enable_thinking=True, reasoning_effort='low'),
                                          dict(preserve_thinking=False))):
            with self.subTest(changes=changes):
                self.request = dict(original, **changes)
                result = self.run_round(f'unsupported-{index}')
                self.assertEqual(result['status'], 'error', result)
                self.assertIs(result['requests'][0]['submitted'], False)
                self.assertTrue(result['usage_complete'])
                self.sender.assert_not_called()

    def test_recorded_missing_token_ids_keep_billing_but_cannot_supply_an_answer(self):
        fixture = Path(__file__).with_name('fixtures') / 'deepinfra/gemma-missing-token-ids-response.json'
        captured = json.loads(fixture.read_bytes())
        body = json.loads(captured['body'])
        self.assertEqual(body['choices'][0]['token_ids'], [])
        self.request['max_tokens'] = 64
        self.runtime['max_total_tokens'] = 500
        self.tokenizer.apply_chat_template.side_effect = lambda messages, tokenize, **kwargs: (
            list(range(body['usage']['prompt_tokens'])) if tokenize else 'Recorded Gemma tool prompt')
        self.sender.return_value = captured['http_status'], captured['body']
        result = self.run_round('missing-ids')
        self.assertEqual(result['status'], 'error', result)
        row = result['requests'][0]
        self.assertIs(row['submitted'], True)
        self.assertEqual(row['tokens'], [])
        self.assertEqual(row['generation_tokens'], 36)
        self.assertEqual(row['usage'], body['usage'])
        self.assertIsNone(row['usage']['prompt_tokens_details'])
        self.assertIsNone(row['answer_text'])
        self.assertEqual(row['text'], '')
        self.assertTrue(result['usage_complete'])
        self.assertAlmostEqual(result['observed_estimated_cost_usd'], .000057)
        with self.assertRaises(ValueError):
            native_multi_edit.parse_call(row['answer_text'])
        self.sender.assert_called_once()

    def test_native_text_accepts_recorded_tool_reply_without_inventing_token_ids(self):
        fixture = Path(__file__).with_name('fixtures') / 'deepinfra/gemma-missing-token-ids-response.json'
        captured = json.loads(fixture.read_bytes())
        body = json.loads(captured['body'])
        self.runtime.update(output_protocol='gemma_native_text', max_total_tokens=500)
        self.request['max_tokens'] = 64
        self.tokenizer.apply_chat_template.side_effect = lambda messages, tokenize, **kwargs: (
            list(range(213)) if tokenize else 'Recorded Gemma tool prompt')
        self.sender.return_value = captured['http_status'], captured['body']
        with mock.patch.object(d.g, 'decode_tokens', side_effect=AssertionError('No output token decoding')) as decode:
            result = self.run_round('native-text-recorded-tool')
        self.assertEqual(result['status'], 'success', result)
        row = result['requests'][0]
        self.assertEqual(row['text'], body['choices'][0]['text'])
        self.assertTrue(row['answer_text'].startswith('<|tool_call>'))
        self.assertTrue(row['answer_text'].endswith('<tool_call|>'))
        self.assertEqual(native_multi_edit.parse_call(row['answer_text'])[1],
                         [dict(old='print(1)', new='print(2)')])
        self.assertIsNone(row['tokens'])
        self.assertEqual(row['token_ids_source'], 'unavailable')
        self.assertIsNone(row['reasoning_token_count'])
        self.assertEqual(row['thought_metrics'], dict(status='unavailable', framing='closed',
            reasoning_tokens=None, answer_tokens=None, reasoning_characters=1,
            answer_characters=len(row['answer_text'])))
        self.assertEqual(row['generation_tokens'], 36)
        self.assertEqual(row['usage'], body['usage'])
        decode.assert_not_called()

    def test_recorded_nonempty_thought_requires_enabled_thinking(self):
        fixture = Path(__file__).with_name('fixtures') / 'deepinfra/gemma-native-text-thinking-response.json'
        captured = json.loads(fixture.read_bytes())
        body = json.loads(captured['body'])
        self.runtime.update(output_protocol='gemma_native_text', max_total_tokens=500)
        self.request['max_tokens'] = 128
        self.tokenizer.apply_chat_template.side_effect = lambda messages, tokenize, **kwargs: (
            list(range(32)) if tokenize else 'Recorded Gemma neutral prompt')
        self.sender.return_value = captured['http_status'], captured['body']
        for enabled, framing, answer in ((False, 'unexpected_thought', None),
                                         (True, 'closed', 'ALPHA BETA GAMMA')):
            with self.subTest(enabled=enabled):
                self.request['enable_thinking'] = enabled
                result = self.run_round(f'native-recorded-thought-{enabled}')
                self.assertEqual(result['status'], 'success', result)
                row = result['requests'][0]
                self.assertEqual(row['text'], body['choices'][0]['text'])
                self.assertEqual(row['answer_text'], answer)
                self.assertEqual(row['thought_metrics']['framing'], framing)
                self.assertIsNone(row['tokens'])
                self.assertIsNone(row['reasoning_token_count'])

    def test_native_text_frames_are_parsed_without_accepting_ambiguous_markers(self):
        self.runtime['output_protocol'] = 'gemma_native_text'
        self.request['enable_thinking'] = True
        answer_program = "print('ż')\n"
        cases = [
            (answer_program, 'absent', '', answer_program),
            ('<|channel>thought\n<channel|>print(1)\n', 'closed', '\n', 'print(1)\n'),
            ('<|channel>thought<channel|>print(1)\n', 'closed', '', 'print(1)\n'),
            (f'<|channel>thought\r\nNaïve 🧠<channel|>{answer_program}', 'closed', '\r\nNaïve 🧠', answer_program),
            ('<|channel>thought', 'unclosed', '', None),
            ('<|channel>thought\nunfinished 🧠', 'unclosed', '\nunfinished 🧠', None),
            ('<|channel>thoughtful\nnot a thought header<channel|>print(1)', 'malformed', None, None),
            ('<channel|>print(1)\n', 'malformed', None, None),
            ('prefix<|channel>thought\na<channel|>print(1)', 'malformed', None, None),
            ('<|channel>thought\na<channel|><|channel>thought\nb<channel|>print(1)', 'malformed', None, None),
            ('<|channel>thought\na<channel|>print(1)<channel|>', 'malformed', None, None),
            ('<|channel>thought\na<|channel>thought\nb<channel|>print(1)', 'malformed', None, None),
        ]
        for index, (text, framing, reasoning, answer) in enumerate(cases):
            with self.subTest(text=text):
                self.sender.return_value = self.text_response(text)
                result = self.run_round(f'native-frame-{index}')
                self.assertEqual(result['status'], 'success', result)
                row = result['requests'][0]
                self.assertEqual(row['text'], text)
                self.assertEqual(row['answer_text'], answer)
                self.assertEqual(row['thought_metrics']['framing'], framing)
                self.assertIsNone(row['tokens'])
                self.assertIsNone(row['thought_metrics']['reasoning_tokens'])
                self.assertIsNone(row['thought_metrics']['answer_tokens'])
                characters = (None, None) if reasoning is None else (len(reasoning), len(answer or ''))
                self.assertEqual((row['thought_metrics'].get('reasoning_characters'),
                                  row['thought_metrics'].get('answer_characters')), characters)

    def test_actual_native_extractor_uses_reported_usage_when_output_tokens_are_unavailable(self):
        fixture = Path(__file__).with_name('fixtures') / 'deepinfra/gemma-missing-token-ids-response.json'
        body = json.loads(json.loads(fixture.read_bytes())['body'])
        row = dict(model=GEMMA_MODEL, output_protocol='gemma_native_text', prompt_tokens=213,
                   prompt_ids=[2], enable_thinking=False, tokens=[], answer_text=None, text='', status='running')
        d.normalize(body, row, self.tokenizer, cap=64)
        entries = [dict(benigns={'actor': dict(program='print(1)\n')}, partition=dict(menu=[]))]
        extractor = native_multi_edit.Extractor(entries, [self.request['messages']], b)
        attempt = dict(position=0, seed=17)
        extractor.observe(row, attempt)
        self.assertEqual(extractor(row, attempt), b'print(2)\n')
        self.assertEqual(attempt['cumulative_output_tokens'], 36)
        self.assertTrue(attempt['edit_applicable'])
        self.assertEqual(attempt['conversation'][-1]['tool_calls'][0]['function']['arguments'],
                         dict(edits=[dict(old='print(1)', new='print(2)')]))
        self.assertIsNone(row['tokens'])
        self.assertIsNone(row['reasoning_token_count'])

    def test_native_text_length_still_cannot_supply_a_program(self):
        self.runtime['output_protocol'] = 'gemma_native_text'
        self.sender.return_value = self.text_response('print(1)\n', reason='length', completion_tokens=20)
        result = self.run_round('native-length')
        self.assertEqual(result['status'], 'success', result)
        self.assertEqual(result['requests'][0]['status'], 'length')
        with self.assertRaises(ValueError):
            b.generation.extract_program(result['requests'][0])

    def test_native_text_can_close_a_thought_already_opened_in_the_prompt(self):
        self.runtime['output_protocol'] = 'gemma_native_text'
        self.request['enable_thinking'] = True
        self.tokenizer.apply_chat_template.side_effect = lambda messages, tokenize, **kwargs: (
            [2, 100, 45518] if tokenize else 'Prompt with an open thought')
        cases = [('Reasoning.<channel|>print(1)\n', 'closed', 'print(1)\n'),
                 ('Unfinished reasoning', 'unclosed', None),
                 ('<|channel>thought\nnested<channel|>print(1)', 'malformed', None),
                 ('Reasoning.<channel|>print(1)<channel|>', 'malformed', None)]
        for index, (text, framing, answer) in enumerate(cases):
            with self.subTest(text=text):
                self.sender.return_value = self.text_response(text, prompt_tokens=3)
                result = self.run_round(f'native-prompt-thought-{index}')
                self.assertEqual(result['status'], 'success', result)
                row = result['requests'][0]
                self.assertEqual(row['text'], text)
                self.assertEqual(row['answer_text'], answer)
                self.assertEqual(row['thought_metrics']['framing'], framing)
                self.assertIsNone(row['tokens'])
                self.assertIsNone(row['reasoning_token_count'])

    def test_native_text_protocol_is_gemma_only(self):
        self.runtime.update(model=d.MODEL, output_protocol='gemma_native_text')
        result = self.run_round('qwen-native-text')
        self.assertEqual(result['status'], 'error', result)
        self.assertIs(result['requests'][0]['submitted'], False)
        self.sender.assert_not_called()

    def test_native_text_protocol_still_checks_prompt_and_output_limits(self):
        self.runtime['output_protocol'] = 'gemma_native_text'
        for index, fields in enumerate((dict(prompt_tokens=2), dict(completion_tokens=21))):
            with self.subTest(fields=fields):
                self.sender.return_value = self.text_response('print(1)\n', **fields)
                result = self.run_round(f'native-bad-usage-{index}')
                self.assertEqual(result['status'], 'error', result)
                self.assertIs(result['requests'][0]['submitted'], True)
                self.assertIsNone(result['requests'][0]['answer_text'])
                self.assertTrue(result['usage_complete'])


class GemmaLoaderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = dict(model_type='gemma4', eos_token_id=[1, 106, 50],
                           text_config=dict(model_type='gemma4_text', eos_token_id=1))
        self.runtime = dict(model=GEMMA_MODEL, tokenizer_path=str(self.root), tokenizer_sha256={})
        for name in d.TOKENIZER_FILES[:4]:
            (self.root / name).write_text(json.dumps(self.config) if name == 'config.json' else '{}')
        self.pin()
        self.inner = mock.Mock(eos_token_id=1)
        self.inner.encode.side_effect = lambda text, **kwargs: {
            '<|channel>thought': [100, 45518], '<channel|>': [101]}[text]
        self.inner.convert_ids_to_tokens.side_effect = lambda token: {49: '<tool_call|>', 50: '<|tool_response>'}[token]
        self.factory = mock.Mock(return_value=self.inner)
        mock.patch.dict('sys.modules', transformers=SimpleNamespace(AutoTokenizer=SimpleNamespace(
            from_pretrained=self.factory))).start()
        mock.patch.object(d, 'version', side_effect=VERSIONS.__getitem__).start()
        self.addCleanup(mock.patch.stopall)

    def pin(self):
        self.runtime['tokenizer_sha256'] = {name: b.generation.file_hash(self.root / name)
                                           for name in d.TOKENIZER_FILES[:4]}

    def test_top_level_eos_and_multitoken_boundaries_drive_the_adapter(self):
        tokenizer = d.load_tokenizer(self.runtime)
        self.factory.assert_called_once_with(self.root, local_files_only=True, trust_remote_code=False)
        self.assertEqual(set(tokenizer.eos_token_ids), {1, 106, 50})
        self.assertEqual(tuple(tokenizer.think_start_tokens), (100, 45518))
        self.assertEqual(tuple(tokenizer.think_end_tokens), (101,))
        self.assertEqual(tokenizer.rfind_think_start([100, 45518, 100, 45518]), 2)
        self.assertEqual(tokenizer.rfind_think_start([100, 45518, 100]), 0)
        self.assertEqual(tokenizer.rfind_think_start([100, 1, 45518]), -1)
        self.assertEqual(tokenizer.rfind_think_end([100, 45518, 101]), 2)
        self.assertEqual(tokenizer.convert_ids_to_tokens(49), '<tool_call|>')

    def test_closing_tool_token_survives_shared_stop_decoding(self):
        tokenizer = d.load_tokenizer(self.runtime)
        self.inner.decode.return_value = CALL
        self.assertEqual(b.generation.decode_tokens(tokenizer, [48, 9, 49], {49}), CALL)
        self.inner.decode.assert_called_once_with([48, 9, 49], skip_special_tokens=False,
                                                  clean_up_tokenization_spaces=False)

    def test_wrong_config_eos_or_thought_sequence_is_rejected(self):
        self.config['eos_token_id'] = [1, 106]
        (self.root / 'config.json').write_text(json.dumps(self.config))
        self.pin()
        with self.assertRaisesRegex(ValueError, 'boundar'):
            d.load_tokenizer(self.runtime)

        self.config['eos_token_id'] = [1, 106, 50]
        (self.root / 'config.json').write_text(json.dumps(self.config))
        self.pin()
        self.inner.encode.side_effect = lambda text, **kwargs: [100, 999] if text == '<|channel>thought' else [101]
        with self.assertRaisesRegex(ValueError, 'boundar'):
            d.load_tokenizer(self.runtime)

    def test_wrong_model_family_is_rejected_with_otherwise_valid_boundaries(self):
        self.config['model_type'] = 'qwen3_5'
        (self.root / 'config.json').write_text(json.dumps(self.config))
        self.pin()
        with self.assertRaisesRegex(ValueError, 'family'):
            d.load_tokenizer(self.runtime)


class GemmaPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        tokenizer = self.root / 'tokenizer'
        tokenizer.mkdir()
        for name in d.TOKENIZER_FILES[:4]:
            (tokenizer / name).write_text(json.dumps(dict(model_type='gemma4', eos_token_id=[1, 106, 50]))
                                          if name == 'config.json' else '{}')
        self.runtime = dict(model=GEMMA_MODEL, tokenizer_path=str(tokenizer), tokenizer_sha256={
            name: b.generation.file_hash(tokenizer / name) for name in d.TOKENIZER_FILES[:4]},
            timeout_seconds=5, max_total_tokens=100, input_usd_per_million=.2,
            output_usd_per_million=.4, max_cost_usd=.01)
        self.runtime_file = self.root / 'provider.json'
        self.runtime_file.write_text(json.dumps(self.runtime))
        task = dict(question='Print two', cases=[dict(index=0, input='', answer='2', visible=True)])
        digest = hashlib.sha256(json.dumps(task).encode()).hexdigest()
        self.worklist = [(dict(split='train', id=0, sha256=digest), task)]
        self.recipe = dict(system='Use test_edits', initial='Fix print(0)', temperature=1., top_p=.95,
            base_seed=17, max_tokens=16, max_repairs=1, repair='Repair the edit', repair_max_tokens=20,
            feedback_field_bytes=64, feedback_bytes=512, tools=native_multi_edit.TOOLS,
            enable_thinking=False, preserve_thinking=True)
        self.rounds, self.checks = [], []
        self.child = mock.patch.object(b, 'run_child', side_effect=self.generate).start()
        self.sender = mock.patch.object(d, 'send', side_effect=AssertionError('No HTTP in integration fixture')).start()
        self.addCleanup(mock.patch.stopall)

    def generate(self, command, current, persist):
        self.assertEqual(command[:2], [str(Path(sys.executable).absolute()), str(Path(d.__file__).resolve())])
        specification = json.loads(Path(command[-1]).read_text())
        self.assertEqual(specification['runtime'], self.runtime)
        self.rounds.append(specification)
        output = Path(specification['output_dir'])
        output.mkdir()
        call = CALL if len(self.rounds) == 1 else CALL.replace('new:"1"', 'new:"2"')
        b.generation.save(output / 'outcome.json', dict(status='success', native=None, usage_complete=True,
            requests=[dict(status='stop', submitted=True, text=call, answer_text=call,
                           tokens=[48, 9, 49, 50], generation_tokens=4)]))
        return 0

    def verify(self, candidates, directory, **settings):
        self.checks.append(candidates)
        self.assertEqual(settings, dict(max_samples=4, concurrent_inference=False))
        return [dict(status='failed' if len(self.checks) == 1 else 'passed',
                     key=dict(task_sha256=digest, program_sha256=hashlib.sha256(program).hexdigest()),
                     feedback=dict(message='Public repair feedback')) for _, digest, program in candidates]

    def repair(self, initial, parent, recipe, seed):
        return dict(initial, seed=seed, max_tokens=recipe['repair_max_tokens'], messages=[*initial['messages'],
            dict(role='assistant', content='', reasoning_content='Earlier hypothesis', tool_calls=[
                dict(id='prior', type='function', function=dict(name='test_edits',
                     arguments=dict(edits=[dict(old='0', new='1')])))]),
            dict(role='tool', tool_call_id='prior', content='Public repair feedback')])

    def run_loop(self, **changes):
        def extract(row, attempt):
            return native_multi_edit.apply_edits('print(0)\n', native_multi_edit.parse_call(row['answer_text'])[1])
        options = dict(source='synthetic', provider_runtime=self.runtime_file, generation_python=sys.executable,
            recipe=self.recipe, reply_allowance=2, output_token_allowance=36, fixtures=True,
            candidate_checker=self.verify, program_extractor=extract, repair_builder=self.repair)
        options.update(changes)
        return b.run(self.worklist, self.root / 'run', **options)

    def test_synthetic_generation_verification_and_native_history_repair(self):
        result = self.run_loop()
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual([a['seed'] for a in result['attempts']], [17, 18])
        self.assertEqual([a['parent'] for a in result['attempts']], [None, 0])
        self.assertEqual([a['candidate']['source'] for a in result['attempts']], ['synthetic', 'synthetic'])
        self.assertEqual(result['tasks'][0]['state'], 'passed')
        self.assertEqual((result['committed_replies'], result['committed_tokens']), (2, 36))
        history = self.rounds[1]['requests'][0]['messages']
        self.assertEqual([m['role'] for m in history], ['system', 'user', 'assistant', 'tool'])
        self.assertEqual(history[-2]['reasoning_content'], 'Earlier hypothesis')
        self.assertEqual(history[-2]['tool_calls'][0]['function']['arguments'], dict(edits=[dict(old='0', new='1')]))
        self.assertEqual([c[0][2] for c in self.checks], [b'print(1)\n', b'print(2)\n'])
        self.sender.assert_not_called()

    def test_native_text_runs_through_synthetic_worker_verification_and_repair(self):
        self.runtime['output_protocol'] = 'gemma_native_text'
        self.runtime_file.write_text(json.dumps(self.runtime))
        def execute_worker(command, current, persist):
            self.rounds.append(json.loads(Path(command[-1]).read_text()))
            outcome = d.run(command[-1])
            return 0 if outcome['status'] == 'success' else 1
        def send(payload, key, timeout):
            call = CALL if payload['seed'] == 17 else CALL.replace('new:"1"', 'new:"2"')
            return 200, json.dumps(dict(choices=[dict(index=0,
                text='<|channel>thought\n<channel|>' + call, token_ids=[], finish_reason='stop')],
                usage=dict(prompt_tokens=1, completion_tokens=10, estimated_cost=.00001,
                           prompt_tokens_details=None)))
        self.child.side_effect = execute_worker
        self.sender.side_effect = send
        with mock.patch.dict('os.environ', DEEPINFRA_API_KEY='PRIVATE_GEMMA_TEST_KEY'), \
                mock.patch.object(d, 'load_tokenizer', return_value=GemmaTokenizer()):
            result = self.run_loop()
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['known_submissions'], 2)
        self.assertEqual(result['observed_tokens'], 20)
        self.assertTrue(result['usage_complete'])
        self.assertEqual(result['tasks'][0]['state'], 'passed')
        self.assertEqual([c[0][2] for c in self.checks], [b'print(1)\n', b'print(2)\n'])
        for index in range(2):
            outcome = json.loads((self.root / f'run/generation-{index:04d}/outcome.json').read_text())
            row = outcome['requests'][0]
            self.assertIsNone(row['tokens'])
            self.assertIsNone(row['reasoning_token_count'])
            self.assertEqual(row['thought_metrics']['status'], 'unavailable')
        self.assertEqual(self.rounds[1]['requests'][0]['messages'][-1]['role'], 'tool')

    def test_gemma_cannot_be_used_as_actor(self):
        with self.assertRaises(ValueError):
            self.run_loop(source='actor')
        self.assertFalse((self.root / 'run').exists())
        self.child.assert_not_called()

    def test_qwen_cannot_be_used_as_synthetic(self):
        self.runtime['model'] = d.MODEL
        self.runtime_file.write_text(json.dumps(self.runtime))
        with self.assertRaises(ValueError):
            self.run_loop()
        self.assertFalse((self.root / 'run').exists())
        self.child.assert_not_called()
