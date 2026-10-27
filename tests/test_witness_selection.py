"""Saved publication fixtures: no model, Docker or provider calls."""
from copy import deepcopy
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests.test_witnesses import record

s = importlib.import_module('src.data-pipeline.stage04.witness_selection')
c, w, a = s.c, s.witnesses, s.audits


class WitnessSelectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.target = self.root / '04_witnesses/v1'
        self.qualification = self.root / 'qualification'
        self.qualification.mkdir()
        self.settings = dict(source_sha256=c._hash(a.__file__), timeout=180,
            auditors={x:dict(model=a.MODELS[x], effort='low', executable=x) for x in a.AUDITORS})
        self.save(self.qualification/'inputs.json', self.settings)
        self.patch(c, '_qualification', return_value=(self.settings,dict.fromkeys(a.AUDITORS,'fixture-1')))
        self.benigns = self.root/'benigns'
        self.benigns.mkdir()
        self.records, self.runs = {}, []
        for source in ('actor','synthetic'):
            rec = record(7,source)
            rec['task'] = dict(question='PUBLIC_STATEMENT', cases=[dict(input='2\n')])
            rec['selection'].update(binding=dict(split='test',id=7,sha256='a'*64,owner='development',block='initial'),
                                    source=source,label='development_smoke')
            self.records[source] = rec
        self.save(self.benigns/'manifest.json',dict(selections=[r['selection'] for r in self.records.values()]))
        self.patch(w, 'BENIGN_DIR', new=self.benigns)
        self.patch(w, 'SELECTION_SHA256', new=c._hash(self.benigns/'manifest.json'))
        self.patch(w, 'load_worklist', side_effect=lambda keys:[deepcopy(self.records[source]) for split,i,source in keys])
        for source, record_ in self.records.items():
            run = self.root/source
            proposal = run/'round-0000/proposals'
            batch = run/'round-0001/audits'
            proposal.mkdir(parents=True);batch.mkdir(parents=True)
            candidate=record_['selection']['candidate']
            stdin=proposal/'input-0000.txt';stdin.write_bytes(b' \t7\r\n')
            (proposal/'generation').mkdir()
            (proposal/'generation/reply-0000.txt').write_bytes(b'<witness>\n \t7\r\n</witness>')
            log=proposal/'fixture.eval';log.write_bytes(b'private native evidence')
            programs={role:dict(sample_id=i,program_sha256=s.sha256(raw),answer='7',
                result=dict(status='completed',returncode=0,stdout='7\n'))
                for i,(role,raw) in enumerate(zip(w.ROLES,[*record_['references'],record_['program']]))}
            attempt=dict(attempt=0,candidate=candidate,seed=17,submitted=True,state='execution_agrees',
                input=str(stdin),stdin_sha256=c._hash(stdin),
                verification=dict(status='execution_agrees',answer='7',log=str(log),programs=programs,
                                  key=dict(candidate=candidate,stdin_sha256=c._hash(stdin))))
            self.save(proposal/'inputs.json',dict(source=source,fixtures=False))
            self.save(proposal/'outcome.json',dict(status='completed',attempts=[attempt],committed_replies=1,
                committed_tokens=1024,observed_tokens=3,known_submissions=1,usage_complete=True))
            binding=c._binding(dict(attempt,round=str(proposal)))
            item=dict(statement='PUBLIC_STATEMENT',stdin=' \t7\r\n',published_inputs=['2\n'])
            packet=a._prompt(item,'packet')
            (batch/'packet.txt').write_bytes(packet.encode())
            self.save(batch/'inputs.json',dict(self.settings,call_allowance=4,
                packets=dict(packet=dict(binding=binding,sha256=s.sha256(packet.encode())))))
            for auditor in a.AUDITORS:
                self.save(batch/'calls/packet'/auditor/'call-0/record.json',
                    dict(status='success',source_sha256=self.settings['source_sha256'],model=a.MODELS[auditor],
                        effort='low',version='fixture-1',usage=dict(input_tokens=10,output_tokens=2),
                        reply=dict(packet_id='packet',validity='approve',novelty='approve',
                                   validity_reason='Private reasoning',novelty_reason='Private reasoning')))
            self.save(run/'inputs.json',dict(source=source,fixtures=False,previous=None,scope=[candidate],
                b1_dir=None,b1_files={},qualification_dir=str(self.qualification),
                qualification_sha256=c._hash(self.qualification/'inputs.json')))
            self.save(run/'outcome.json',dict(status='completed',inputs_sha256=c._hash(run/'inputs.json'),
                files={str(proposal/'outcome.json'):c._hash(proposal/'outcome.json')},
                rounds=[dict(proposals=str(proposal),audits=str(batch))]))
            self.runs.append(run)

    def patch(self,obj,name,**kwargs):
        p=mock.patch.object(obj,name,**kwargs);self.addCleanup(p.stop);return p.start()

    def save(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);s.write_json(path,value)

    def publish(self,runs=None):
        return s.publish(self.runs if runs is None else runs,self.target)

    def change_attempt(self,change):
        path=self.runs[0]/'round-0000/proposals/outcome.json'
        saved=c._read(path);change(saved['attempts'][0]);self.save(path,saved)
        root=self.runs[0]/'outcome.json';out=c._read(root);out['files'][str(path)]=c._hash(path);self.save(root,out)

    def test_exact_program_stdin_answer_bindings_coverage_and_evidence(self):
        manifest=self.publish();data=c._read(manifest);report=c._read(self.target/'report.json')
        self.assertEqual(data['schema_version'],1)
        self.assertEqual([r['source'] for r in data['selections']],['actor','synthetic'])
        self.assertEqual(data['unresolved'],[])
        for row in data['selections']:
            self.assertEqual(row['answer'],'7')
            self.assertEqual((self.target/row['stdin']).read_bytes(),b' \t7\r\n')
            self.assertEqual((self.target/row['program']).read_bytes(),self.records[row['source']]['program'])
            self.assertEqual(c._hash(self.target/row['stdin']),row['stdin_sha256'])
            self.assertEqual(row['candidate']['program_sha256'],row['program_sha256'])
            self.assertEqual(set(row['approval']['auditors']),set(a.AUDITORS))
            self.assertNotIn('Private reasoning',str(row))
        c._check(data['evidence_sha256'])
        self.assertEqual(report['coverage']['development']['both_source_task_ids'],[dict(split='test',id=7)])
        self.assertEqual(report['coverage']['all']['approved'],2)
        self.assertEqual(report['attempts'],2)
        self.assertEqual(report['audit_calls'],4)
        self.assertEqual(report['generation']['committed_replies'],2)

    def test_current_veto_revokes_cached_approval_and_is_preserved(self):
        batch=self.runs[0]/'round-0001/audits'
        self.save(batch/'vetoes.json',dict(packet='Human constraint finding'))
        data=c._read(self.publish());report=c._read(self.target/'report.json')
        self.assertEqual(len(data['selections']),1)
        self.assertEqual(data['unresolved'][0]['candidate']['source'],'actor')
        self.assertEqual(report['coverage']['all']['both_source_task_ids'],[])
        self.assertIn(str(batch/'vetoes.json'),data['evidence_sha256'])

    def test_mismatched_key_and_contradictory_answer_or_program_refused(self):
        original=c._read(self.runs[0]/'round-0000/proposals/outcome.json')
        changes=[lambda x:x['verification']['key'].update(stdin_sha256='0'*64),
                 lambda x:x['verification'].update(answer='8'),
                 lambda x:x['verification']['programs']['benign'].update(program_sha256='0'*64),
                 lambda x:x['verification']['programs']['reference-0']['result'].update(stdout='8\n'),
                 lambda x:x['verification']['programs']['benign'].update(sample_id=0),
                 lambda x:x['verification']['programs']['benign'].update(sample_id=-1),
                 lambda x:x['verification']['programs'].pop('reference-2')]
        for change in changes:
            path=self.runs[0]/'round-0000/proposals/outcome.json';self.save(path,deepcopy(original))
            self.change_attempt(change)
            with self.subTest(change=change),self.assertRaises(ValueError):self.publish()
            self.assertFalse(self.target.exists())

    def test_missing_audit_prevents_publication_and_wrong_setup_refused(self):
        record=self.runs[0]/'round-0001/audits/calls/packet/claude/call-0/record.json'
        original=record.read_bytes();record.unlink()
        with self.assertRaisesRegex(ValueError,'Pending'):self.publish()
        record.write_bytes(original)
        path=self.runs[0]/'round-0001/audits/inputs.json';saved=c._read(path)
        saved['auditors']['claude']['model']='unqualified-model';self.save(path,saved)
        with self.assertRaisesRegex(ValueError,'qualified setup'):self.publish()
        self.assertFalse(self.target.exists())

    def test_incomplete_or_overlapping_scope_and_fixtures_refused(self):
        for runs in ([self.runs[0]],[*self.runs,self.runs[0]]):
            with self.subTest(runs=runs),self.assertRaises(ValueError):self.publish(runs)
        path=self.runs[0]/'inputs.json';saved=c._read(path);saved['fixtures']=True;self.save(path,saved)
        outcome=c._read(self.runs[0]/'outcome.json');outcome['inputs_sha256']=c._hash(path)
        self.save(self.runs[0]/'outcome.json',outcome)
        with self.assertRaisesRegex(ValueError,'non-fixture'):self.publish()
        self.assertFalse(self.target.exists())

    def test_existing_output_and_dangling_symlink_refused(self):
        self.target.parent.mkdir();self.target.symlink_to(self.root/'missing')
        with self.assertRaises(FileExistsError):self.publish()
        self.target.unlink();self.publish()
        before=(self.target/'manifest.json').read_bytes()
        with self.assertRaises(FileExistsError):self.publish()
        self.assertEqual((self.target/'manifest.json').read_bytes(),before)

    def test_failed_copy_leaves_no_publication_or_staging(self):
        write=Path.write_bytes
        def fail(path,raw):
            if path.parent.name=='inputs':raise OSError('test copy failure')
            return write(path,raw)
        with mock.patch.object(Path,'write_bytes',fail),self.assertRaises(OSError):self.publish()
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.target.parent.iterdir()),[])


if __name__=='__main__':unittest.main()
