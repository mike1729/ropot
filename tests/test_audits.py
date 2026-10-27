"""Automatic CLI dispatch exercised by local fake executables; no provider calls."""
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest import mock

a = importlib.import_module("src.data-pipeline.stage04.audits")
FAKE = r"""
import json, os, pathlib, re, sys, time
which = AUDITOR
args = sys.argv[1:]
options = json.loads(pathlib.Path(CONTROL).read_text()).get(which, {})
if '--version' in args:
 print('fake-' + which + ' ' + options.get('version','1.0')); raise SystemExit(0)
prompt = sys.stdin.read()
packet = re.search(r'Packet ID: (\w+)', prompt)[1]
payload = json.JSONDecoder().raw_decode(prompt.split('\n', 2)[2])[0]
model = args[args.index('-m' if which == 'codex' else '--model') + 1]
with open(LOG, 'a') as f:
 f.write(json.dumps(dict(auditor=which, prompt=prompt, args=args, cwd=os.getcwd(),
   files=os.listdir('.'), global_agents=(pathlib.Path(os.environ.get('CODEX_HOME', '/nonexistent'))/'AGENTS.md').exists(),
   api_keys=any(os.environ.get(k) for k in ('OPENAI_API_KEY','ANTHROPIC_API_KEY','CODEX_API_KEY'))))+'\n')
if options.get('sleep'): time.sleep(10)
if options.get('capacity'):
 print(json.dumps(dict(type='error',message='Selected model is at capacity.')) )
 raise SystemExit(1)
if options.get('exit'):
 print('Subscription usage limit', file=sys.stderr); raise SystemExit(1)
valid = options.get('validity', 'approve')
if 'gold' in options and 'validity' not in options:
 valid = 'approve' if options['gold'][json.dumps([payload['statement'],payload['stdin']])] else 'reject'
reply = dict(packet_id=options.get('packet_id', packet), validity=valid,
 novelty=options.get('novelty', 'approve' if valid == 'approve' else 'uncertain'),
 validity_reason='Fixture constraint judgment.', novelty_reason='Fixture instance judgment.')
if which == 'codex':
 pathlib.Path(args[args.index('-o')+1]).write_text('{broken' if options.get('malformed') else json.dumps(reply))
 events = [dict(type='thread.started'),dict(type='item.completed',item=dict(type='agent_message')),
           dict(type='turn.completed',usage=dict(input_tokens=50,output_tokens=20))]
 if options.get('tools'):events.insert(1,dict(type='item.completed',item=dict(type='command_execution')))
 if options.get('error'):events.insert(1,dict(type='turn.failed',error={'message':'Usage limit'}))
 for event in events:print(json.dumps(event))
else:
 response = dict(is_error=options.get('error',False),structured_output=reply,
   modelUsage={options.get('reported_model',model):{}}, total_cost_usd=.01,
   usage=dict(input_tokens=50,output_tokens=20,server_tool_use={'web_search_requests':int(options.get('tools',False))}))
 print('{broken' if options.get('malformed') else json.dumps(response))
"""


class CLIAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.directory = self.base / 'batch'
        self.control = self.base / 'behavior.json'
        self.control.write_text('{}')
        self.log = self.base / 'calls.jsonl'
        auth = self.base / 'user-codex'
        auth.mkdir()
        (auth / 'auth.json').write_text('OPAQUE_AUTH_SECRET')
        (auth / 'AGENTS.md').write_text('GLOBAL_INSTRUCTION_SECRET')
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict(os.environ, CODEX_HOME=str(auth), OPENAI_API_KEY='API_SECRET').start()
        self.executables = {}
        for auditor in a.AUDITORS:
            path = self.base / auditor
            path.write_text('#!' + sys.executable + '\n' + 'AUDITOR=' + repr(auditor) + '\nCONTROL=' +
                            repr(str(self.control)) + '\nLOG=' + repr(str(self.log)) + '\n' + FAKE)
            path.chmod(0o755)
            self.executables[auditor] = str(path)
        self.models = dict(codex='test-codex-version', claude='test-claude-version')
        self.items = [dict(statement='PUBLIC_STATEMENT', stdin=' \t3\r\n1 2 5\r\n',
            published_inputs=['PUBLIC_INPUT\n'], binding=dict(candidate='PRIVATE_ID', expected_valid=True),
            program='PRIVATE_PROGRAM', answer='PRIVATE_ANSWER', stored_inputs=['PRIVATE_HIDDEN_INPUT'])]

    def behavior(self, **changes):
        self.control.write_text(json.dumps(changes))

    def run_batch(self, **kwargs):
        return a.run(self.items, self.directory, call_allowance=kwargs.pop('call_allowance', 4),
                     models=self.models, executables=self.executables, **kwargs)

    def calls(self):
        return [json.loads(s) for s in self.log.read_text().splitlines()] if self.log.exists() else []

    def qualification_behavior(self, **changes):
        cases = a._read(a.FIXTURES)['cases']
        gold = {json.dumps([c['statement'],c['stdin']]):c['expected_valid'] for c in cases}
        self.behavior(**{auditor:dict(gold=gold, **changes.get(auditor, {})) for auditor in a.AUDITORS})

    def test_automatic_dual_approval_projection_exact_input_and_empty_cwd(self):
        rows = self.run_batch()
        self.assertEqual(rows[0]['status'], 'approved')
        self.assertEqual(rows[0]['binding'], self.items[0]['binding'])
        self.assertEqual(len(self.calls()), 2)
        for call in self.calls():
            self.assertNotIn('PRIVATE', call['prompt'])
            self.assertIn(self.items[0]['stdin'], call['prompt'])
            self.assertIn(json.dumps(self.items[0]['stdin'],ensure_ascii=False), call['prompt'])
            self.assertIn(a.RUBRIC, call['prompt'])
            self.assertEqual(call['files'], [])
            self.assertFalse(call['api_keys'])
            self.assertFalse(Path(call['cwd']).exists())
        self.assertTrue(self.calls()[0]['global_agents'])  # Disk presence only; isolation has a separate live probe.
        for audit in rows[0]['audits'].values():
            self.assertEqual(audit['usage']['input_tokens'], 50)
            self.assertTrue(audit['version'].startswith('fake-'))
            self.assertEqual(audit['returncode'], 0)
            self.assertGreater(audit['seconds'], 0)
        saved = ''.join(p.read_text() for p in self.directory.rglob('*') if p.is_file())
        self.assertNotIn('OPAQUE_AUTH_SECRET', saved)
        self.assertNotIn('API_SECRET', saved)

    def test_commands_disable_customizations_tools_and_persistence(self):
        self.run_batch()
        codex, claude = (c['args'] for c in self.calls())
        for flag in ('--ephemeral','--ignore-user-config','--ignore-rules','--skip-git-repo-check','--output-schema','--json'):
            self.assertIn(flag, codex)
        self.assertIn('project_doc_max_bytes=0', codex)
        self.assertIn('approval_policy="never"', codex)
        self.assertIn('shell_tool', codex)
        self.assertIn('read-only', codex)
        for flag in ('--safe-mode','--strict-mcp-config','--disable-slash-commands','--no-session-persistence','--json-schema'):
            self.assertIn(flag, claude)
        self.assertEqual(claude[claude.index('--tools')+1], '')
        self.assertEqual(claude[claude.index('--system-prompt')+1], a.BASE)
        self.assertNotIn('--bare', claude)

    def test_rerun_and_decision_reading_do_not_launch_or_change_successful_calls(self):
        expected = self.run_batch()
        before = {p:p.read_bytes() for p in self.directory.rglob('*') if p.is_file()}
        self.assertEqual(a.dispatch(self.directory), expected)
        self.assertEqual(a.decide(self.directory), expected)
        self.assertEqual(len(self.calls()), 2)
        self.assertEqual(before, {p:p.read_bytes() for p in self.directory.rglob('*') if p.is_file()})

    def test_rejection_and_uncertainty_are_not_approvals_and_validity_has_priority(self):
        self.behavior(codex=dict(novelty='reject'),claude=dict(validity='uncertain',novelty='uncertain'))
        row = self.run_batch()[0]
        self.assertEqual((row['status'],row['reason']), ('not_approved','validity'))

    def test_failure_stops_without_retry_then_successful_rerun_preserves_evidence(self):
        self.behavior(claude=dict(exit=True))
        with self.assertRaisesRegex(RuntimeError,'Auditor CLI failed'):
            self.run_batch()
        self.assertEqual(len(self.calls()), 2)
        failed = next(self.directory.glob('calls/*/claude/call-0'))
        self.assertIn('Subscription usage limit', (failed/'stderr.txt').read_text())
        self.assertEqual(a._read(failed/'record.json')['status'],'error')
        self.behavior()
        self.assertEqual(a.dispatch(self.directory)[0]['status'], 'approved')
        self.assertEqual([c['auditor'] for c in self.calls()], ['codex','claude','claude'])
        self.assertTrue((failed/'record.json').exists())

    def test_quota_failures_remain_pending_and_only_use_the_overall_allowance(self):
        for auditor in a.AUDITORS:
            with self.subTest(auditor=auditor):
                self.directory = self.base / ('quota-' + auditor)
                self.behavior(**{auditor:dict(exit=True)})
                for attempt in range(3):
                    with self.assertRaisesRegex(RuntimeError,'Auditor CLI failed'):
                        self.run_batch(call_allowance=6) if attempt == 0 else a.dispatch(self.directory)
                    self.assertEqual(a.decide(self.directory)[0]['status'],'pending')
                records = list(self.directory.glob('calls/*/' + auditor + '/call-*/record.json'))
                self.assertEqual(len(records),3)
                self.assertTrue(all(a._read(p)['status'] == 'error' for p in records))
                self.behavior()
                self.assertEqual(a.dispatch(self.directory)[0]['status'],'approved')
                self.assertEqual(len(list(self.directory.glob('calls/*/*/call-*'))),5)

    def test_native_technical_errors_do_not_count_as_unusable_replies(self):
        for auditor in a.AUDITORS:
            with self.subTest(auditor=auditor):
                self.directory = self.base / ('native-error-' + auditor)
                self.behavior(**{auditor:dict(error=True)})
                with self.assertRaisesRegex(RuntimeError,'technical error'):self.run_batch(call_allowance=6)
                with self.assertRaisesRegex(RuntimeError,'technical error'):a.dispatch(self.directory)
                self.assertEqual(a.decide(self.directory)[0]['status'],'pending')
                self.behavior()
                self.assertEqual(a.dispatch(self.directory)[0]['status'],'approved')

    def test_two_failed_calls_are_terminal_and_do_not_repeat_forever(self):
        self.behavior(codex=dict(malformed=True))
        with self.assertRaises(ValueError):self.run_batch()
        with self.assertRaises(ValueError):a.dispatch(self.directory)
        rows = a.dispatch(self.directory)
        self.assertEqual(rows[0]['status'],'unresolved')
        self.assertEqual(rows[0]['reason'],'audit_unavailable')
        self.assertEqual([c['auditor'] for c in self.calls()], ['codex','codex','claude'])
        a.dispatch(self.directory)
        self.assertEqual(len(self.calls()),3)

    def test_call_allowance_is_reserved_before_launch_and_retained_on_failure(self):
        self.behavior(codex=dict(exit=True))
        with self.assertRaises(RuntimeError):self.run_batch(call_allowance=2)
        self.behavior()
        with self.assertRaisesRegex(RuntimeError,'allowance exhausted'):
            a.dispatch(self.directory)
        self.assertEqual([c['auditor'] for c in self.calls()], ['codex','codex'])
        self.assertEqual(len(list(self.directory.glob('calls/*/*/call-*'))),2)

    def test_invalid_reply_binding_and_tool_events_fail_without_approval(self):
        for options in (dict(packet_id='wrong'),dict(tools=True)):
            with self.subTest(options=options):
                self.directory = self.base / ('case-' + str(len(list(self.base.glob('case-*')))))
                self.behavior(codex=options)
                with self.assertRaises(ValueError):self.run_batch()
                self.assertEqual(a.decide(self.directory)[0]['status'],'pending')

    def test_claude_model_mismatch_error_and_server_tools_fail_closed(self):
        for options in (dict(reported_model='wrong-model'),dict(tools=True)):
            with self.subTest(options=options):
                self.directory = self.base / ('claude-' + str(len(list(self.base.glob('claude-*')))))
                self.behavior(claude=options)
                with self.assertRaises(ValueError):self.run_batch()
                self.assertEqual(a.decide(self.directory)[0]['status'],'pending')

    def test_timeout_cleans_up_process_group_and_retains_record(self):
        self.behavior(codex=dict(sleep=True))
        with mock.patch.object(a.os,'killpg',wraps=os.killpg) as kill:
            with self.assertRaises(a.subprocess.TimeoutExpired):self.run_batch(timeout=.1)
            self.assertEqual(kill.call_args.args[1],signal.SIGKILL)
        record = a._read(next(self.directory.glob('calls/*/codex/call-0/record.json')))
        self.assertEqual(record['status'],'error')
        self.assertIn('TimeoutExpired',record['error'])
        self.assertEqual(a.decide(self.directory)[0]['status'],'pending')

    def test_interrupt_cleanup_propagates(self):
        directory = self.base / 'interrupt'
        directory.mkdir()
        child = mock.Mock(pid=12345)
        child.communicate.side_effect=KeyboardInterrupt
        with mock.patch.object(a.subprocess,'Popen',return_value=child),mock.patch.object(a.os,'killpg') as kill:
            with self.assertRaises(KeyboardInterrupt):
                a._execute(['fake'],'prompt',self.base,{},directory,1)
            kill.assert_called_once_with(12345,signal.SIGKILL)
            child.wait.assert_called_once()

    def test_veto_and_changed_prompt_association(self):
        row = self.run_batch()[0]
        (self.directory/'vetoes.json').write_text(json.dumps({row['packet_id']:'Human constraint finding'}))
        self.assertEqual(a.decide(self.directory)[0]['reason'],'veto')
        packet = self.directory/(row['packet_id']+'.txt')
        packet.write_text(packet.read_text()+'changed')
        with self.assertRaisesRegex(ValueError,'Changed audit packet'):a.dispatch(self.directory)
        with self.assertRaisesRegex(ValueError,'Changed audit packet'):a.decide(self.directory)

    def test_qualification_dispatches_twenty_calls_and_scores_validity_only(self):
        cases = a._read(a.FIXTURES)['cases']
        self.assertEqual(len(cases),10)
        self.assertEqual(sum(c['expected_valid'] for c in cases),6)
        gold = {json.dumps([c['statement'],c['stdin']]):c['expected_valid'] for c in cases}
        self.behavior(**{auditor:dict(gold=gold,novelty='uncertain') for auditor in a.AUDITORS})
        report = a.qualify(self.directory,models=self.models,executables=self.executables)
        self.assertTrue(report['passed'])
        self.assertEqual(report['correct'],dict(codex=10,claude=10))
        self.assertEqual(len(self.calls()),20)
        for call in self.calls():
            self.assertNotIn('expected_valid',call['prompt'])
            self.assertNotIn('case_id',call['prompt'])
        self.assertTrue(a.qualify(self.directory)['passed'])
        self.assertEqual(len(self.calls()),20)
        with self.assertRaisesRegex(ValueError,'saved settings'):a.qualify(self.directory,models=self.models)

    def test_qualification_rejects_an_uncertain_validity_judgment(self):
        self.qualification_behavior(claude=dict(validity='uncertain',novelty='uncertain'))
        report = a.qualify(self.directory,models=self.models,executables=self.executables)
        self.assertFalse(report['passed'])
        self.assertEqual(report['correct'],dict(codex=10,claude=0))

    def test_an_ordinary_batch_cannot_qualify_and_rerun_makes_no_extra_calls(self):
        self.run_batch()
        with self.assertRaisesRegex(ValueError,'complete qualification fixture set'):
            a.qualification(self.directory)
        with self.assertRaisesRegex(ValueError,'complete qualification fixture set'):
            a.qualify(self.directory)
        self.assertEqual(len(self.calls()),2)

    def test_qualification_requires_every_fixture_exact_prompt_and_gold_label(self):
        self.qualification_behavior()
        a.qualify(self.directory,models=self.models,executables=self.executables)
        manifest = self.directory / 'inputs.json'
        original = manifest.read_bytes()
        packet_id = next(iter(a._read(manifest)['packets']))
        packet = self.directory / (packet_id + '.txt')
        original_prompt = packet.read_bytes()
        for change in ('missing','duplicate','gold','prompt'):
            with self.subTest(change=change):
                inputs = json.loads(original)
                entry = inputs['packets'][packet_id]
                if change == 'missing':del inputs['packets'][packet_id]
                elif change == 'duplicate':
                    entry['binding']['case_id'] = list(inputs['packets'].values())[1]['binding']['case_id']
                elif change == 'gold':entry['binding']['expected_valid'] = not entry['binding']['expected_valid']
                else:
                    packet.write_bytes(original_prompt + b'Changed fixture')
                    entry['sha256'] = a.sha256(packet.read_bytes()).hexdigest()
                manifest.write_text(json.dumps(inputs))
                with self.assertRaisesRegex(ValueError,'qualification fixture|gold label'):
                    a.qualify(self.directory)
                packet.write_bytes(original_prompt)
        manifest.write_bytes(original)
        self.assertEqual(len(self.calls()),20)

    def test_qualification_cannot_combine_cli_versions_after_an_interruption(self):
        self.qualification_behavior(claude=dict(exit=True))
        with self.assertRaises(RuntimeError):
            a.qualify(self.directory,models=self.models,executables=self.executables)
        self.qualification_behavior(codex=dict(version='2.0'))
        with self.assertRaisesRegex(ValueError,'Mixed qualification CLI versions'):
            a.qualify(self.directory)
        versions = {a._read(p)['version'] for p in self.directory.glob('calls/*/codex/*/record.json')}
        self.assertEqual(versions,{'fake-codex 1.0','fake-codex 2.0'})

    def test_qualification_cannot_combine_implementation_digests(self):
        self.qualification_behavior(claude=dict(exit=True))
        with self.assertRaises(RuntimeError):
            a.qualify(self.directory,models=self.models,executables=self.executables)
        self.qualification_behavior()
        read_bytes = Path.read_bytes
        def changed_source(path):
            return read_bytes(path) + (b'\n# source changed\n' if path == Path(a.__file__) else b'')
        with mock.patch.object(Path,'read_bytes',changed_source):
            with self.assertRaisesRegex(ValueError,'implementation digests'):a.qualify(self.directory)
        calls = [a._read(p) for p in self.directory.glob('calls/*/codex/*/record.json')]
        self.assertEqual(len({r['source_sha256'] for r in calls}),2)

    def test_preflight_and_existing_directory_refusal_make_no_calls(self):
        with self.assertRaises(ValueError):self.run_batch(call_allowance=1)
        self.assertFalse(self.directory.exists())
        self.run_batch()
        with self.assertRaises(FileExistsError):self.run_batch()
        self.assertEqual(len(self.calls()),2)
