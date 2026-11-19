#!/usr/bin/env python3
"""CPU-only exact three-byte fragment, bounded source provenance and fail-closed replay."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import collector_offset2881725_reconciled_v1 as w
r, q = w.r, w.q


class ReconciledCollectorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.production = self.root / 'production'
        self.out = self.root / 'repair'
        self.production.mkdir()
        self.out.mkdir()
        self.csv = self.root / 'train.csv'
        self.csv.write_bytes(b'one immutable original row\n')
        self.member_rows = []
        self.split, self.batch = 'train', 'train-b000001'
        self.revision = 'repair-20261025-v1'
        self.bar = self.root / 'ms_bar_annotation.py'
        self.bar.write_text("def params(args, code):\n    return {'model_path': args.model, 'code_sha256': code}\n")
        self.code = {str(self.bar): q.sha(self.bar)}
        self.configuration = {'model_path': '/not-loaded/CPU-test', 'code_sha256': self.code,
                              'style_module_sha256': 'fake_style_value_not_a_ref'}
        self.params_sha = r.visual_digest(self.configuration)
        for sid in ('Before', 'A', 'B', 'C', 'D'):
            video, audio = self.root / (sid + '.mp4'), self.root / (sid + '.wav')
            video.write_bytes((sid + 'video').encode())
            audio.write_bytes((sid + 'audio').encode())
            self.member_rows.append({'id': sid, 'source_id': sid, 'split': self.split, 'batch_id': self.batch,
                'source_ordinal': len(self.member_rows), 'video_path': str(video), 'audio_path': str(audio),
                'source_refs': {'csv': {**q.identity(self.csv), 'csv_line': len(self.member_rows) + 2,
                                       'raw_row_sha256': hashlib.sha256(sid.encode()).hexdigest()}}})
        self.manifest = self.root / 'train-b000001.manifest.jsonl'
        self.manifest.write_bytes(b''.join(q.encoded(row) for row in self.member_rows))
        self.summary = self.root / 'manifest.summary.json'
        q.save_once(self.summary, {'total': 5, 'batches': [{'batch_id': self.batch, 'split': self.split,
                      'count': 5, **q.identity(self.manifest)}]})
        q.save_once(self.production / 'runs/v0-test/protocol.json', {'mode': 'consumer',
                     'configuration': self.configuration, 'manifest_summary': q.identity(self.summary)})
        self.rows = {}
        for member in self.member_rows:
            member = json.loads(q.encoded(member))
            sid = member['id']
            directory = self.production / 'sources/train' / sid
            binding = {'id': sid, 'video_sha256': q.sha(member['video_path']),
                       'audio_sha256': q.sha(member['audio_path'])}
            plan = {'id': sid, 'manifest_record': member, 'binding': binding,
                    'source_refs': {'manifest': q.identity(self.manifest), 'original_csv': member['source_refs']['csv']}}
            q.save_once(directory / 'bar_plan.json', plan)
            plan = q.load(directory / 'bar_plan.json')
            producer = {'id': sid, 'split': self.split, 'batch_id': self.batch, 'status': 'ready',
                        'source_binding': binding, 'source_ordinal': member['source_ordinal'], 'plan_ref': q.identity(directory / 'bar_plan.json')}
            q.save_once(directory / 'producer.terminal.json', producer)
            visual_root = self.production / 'visual/shard-0000/sources' / sid
            visual_root.mkdir(parents=True)
            receipt_path = visual_root.parent.parent / 'load.receipt.json'
            q.save_once(receipt_path, {'code_sha256': self.code})
            visual_binding = {'source_binding': binding, 'plan_sha256': producer['plan_ref']['sha256'],
                'full_source_row_sha256': r.visual_digest(plan), 'raw_line_sha256': producer['plan_ref']['sha256'],
                'manifest_record_sha256': r.visual_digest(member), 'params_sha256': self.params_sha}
            source = {'sha256': hashlib.sha256(sid.encode()).hexdigest(),
                      'files': {'video_path': q.identity(member['video_path']),
                                'audio_path': q.identity(member['audio_path']), 'nlf_path': None}}
            q.save_once(visual_root / 'binding.json', visual_binding)
            q.save_once(visual_root / 'source.identity.json', source)
            q.save_once(visual_root / 'source.refs.json',
                        [{k: ref[k] for k in ('path', 'sha256')} for ref in plan['source_refs'].values()])
            visual = {'id': sid, 'manifest_record': member, 'binding': visual_binding,
                      'window_observations': [], 'final': {'status': 'candidate'}}
            q.save_once(visual_root / 'results/visual.json', visual)
            joint = {'id': sid, 'source_id': sid, 'split': self.split, 'batch_id': self.batch,
                'source_binding': visual_binding, 'source_identity': source, 'source_refs': member['source_refs'],
                'visual_result_ref': q.identity(visual_root / 'results/visual.json'),
                'joint_complete': sid not in ('A', 'C'), 'all_windows_nonempty': sid not in ('A', 'C'),
                'visual_status': 'candidate', 'style_status': 'resolved', 'dance_style_label': 'hiphop',
                'style_result': {'status': 'resolved'}}
            q.save_once(directory / 'joint-results/joint.json', joint)
            terminal = {'id': sid, 'split': self.split, 'batch_id': self.batch, 'status': 'incomplete_candidate' if sid in ('A', 'C') else 'joint_candidate',
                        'joint_complete': sid not in ('A', 'C'), 'visual_status': 'candidate', 'style_status': 'resolved',
                        'params_sha256': self.params_sha, 'joint_candidate_ref': q.identity(directory / 'joint-results/joint.json')}
            q.save_once(directory / 'consumer.terminal.json', terminal)
            self.rows[sid] = q.load(directory / 'consumer.terminal.json')
        self.before = q.encoded(self.rows['Before'])
        a, b = q.encoded(self.rows['A']), q.encoded(self.rows['B'])
        cut = a.index(b'"path":"') + len(b'"path":"') + 10
        self.badline = a[:cut] + b
        self.queue = self.production / 'completed.jsonl'
        self.queue.write_bytes(self.before + self.badline)
        self.offset = len(self.before)
        self.rules = {self.offset: {'sha256': hashlib.sha256(self.badline).hexdigest(),
            'size': len(self.badline), 'split': self.split, 'batch_id': self.batch, 'ids': ['A', 'B']}}
        stat = self.queue.stat()
        self.origin = {'input_path': str(self.queue), 'input_identity': [stat.st_dev, stat.st_ino],
            'offset': self.offset, 'seen': {'train/Before': q.digest(self.rows['Before'])},
            'counts': {'joint_complete': 1}, 'stage_counts': {}}
        q.save_once(self.out / 'collector.state.json', self.origin)
        q.save_once(self.out / 'attempt-budget/train/A.json', {'source_key': 'train/A',
                      'maximum_extra_source_attempts': 1, 'extra_source_attempt': 1})
        self.budget_bytes = (self.out / 'attempt-budget/train/A.json').read_bytes()
        self.origin_bytes = (self.out / 'collector.state.json').read_bytes()

    def configure(self):
        self.patches = []
        self.real_population = w.check_population
        self.offset = len(self.before)
        self.queue.write_bytes(self.before + w.RAW + q.encoded(self.rows['Before']))
        prefix = self.queue.read_bytes()
        identity = [self.queue.stat().st_dev, self.queue.stat().st_ino]
        self.root_meta = self.out / 'collector-offset2881725-reconciled-v1'
        self.expected = {q.safe_key(self.rows[sid]): (self.batch, self.member_rows[n]['source_ordinal'],
            q.sha(self.production / 'sources/train' / sid / 'consumer.terminal.json'))
            for n, sid in enumerate(('Before', 'A', 'B', 'C', 'D')) if sid != 'Before'}
        index = self.production / 'batches' / self.batch / 'terminal_index.jsonl'
        index.parent.mkdir(parents=True)
        index.write_bytes(b''.join(q.encoded({'id': sid, 'split': 'train'}) for sid in ('Before', 'A', 'B', 'C', 'D')))
        self.index_ref = {'path': str(index), 'input_identity': [index.stat().st_dev,index.stat().st_ino],
            'byte_length': index.stat().st_size, 'prefix_sha256': q.sha(index)}
        self.journal_ref = {'path': str(self.queue), 'input_identity': identity,
            'byte_length': len(prefix), 'prefix_sha256': hashlib.sha256(prefix).hexdigest()}
        evidence = r.Evidence(self.production, self.summary)
        evidence.batches[self.batch]['count'] = 5
        self.evidence = evidence
        self.origin.update(input_identity=identity, offset=self.offset)
        self.state = dict(self.origin, schema=r.SCHEMA, global_revision=self.revision,
            origin_state_ref=q.identity(self.out/'collector.state.json'), manifest_summary_ref=evidence.summary_ref,
            recovered_badline_offsets=[r.BAD_OFFSET,w.v2.EMPTY_OFFSET,w.old.OFFSET], input_prefix_sha256=hashlib.sha256(self.before).hexdigest())
        q.replace_state(self.out / 'collector.recovery.state.json', self.state)
        self.state_bytes = (self.out / 'collector.recovery.state.json').read_bytes()
        for name, value in [('OFFSET', self.offset), ('INPUT_IDENTITY', identity),
                ('PREFIX_SHA', hashlib.sha256(self.before).hexdigest()),
                ('STATE_SHA', q.sha(self.out / 'collector.recovery.state.json')), ('EXPECTED',self.expected)]:
            item = patch.object(w,name,value); item.start(); self.patches.append(item)
        item = patch.object(w, 'namespace', return_value=(self.production,self.out,self.root_meta))
        item.start(); self.patches.append(item)
        item = patch.object(w,'check_population',side_effect=self.population)
        item.start(); self.patches.append(item)

    def population(self, production, evidence):
        raw = w.c.verify_prefix(self.journal_ref)
        iraw = w.c.verify_prefix(self.index_ref)
        journal, inherited = w.journal_sources(raw, production)
        # Fixture has five members; production 500/500 denominator is tested separately.
        evidence.batches[self.batch]['count'] = 5
        indexed, _ = w.c.decode_index(iraw, production, self.batch, evidence)
        for key in set(journal) & set(indexed):
            r.same(journal[key], indexed[key], 'Fixture journal/terminal changed')
        missing=sorted(set(indexed)-set(journal))
        r.same(missing,sorted(w.EXPECTED),'Fixture finite missing set changed')
        return raw,{self.batch:iraw},self.journal_ref,{self.batch:self.index_ref},inherited,indexed,missing

    def ready(self):
        self.configure()
        with patch.object(w,'INDEXES',{self.batch:None}), patch.object(r,'Evidence',return_value=self.evidence):
            result=w.prepare(self.production,self.out,self.revision,self.summary,0)
        self.assertEqual(result['prepared_sources'],4)
        self.assertEqual(result['original_Evidence_terminal_full_checked_sources'],4)
        self.assertEqual((self.out/'collector.recovery.state.json').read_bytes(),self.state_bytes)
        return result

    def recover(self, begin=None, raw=None, snapshots=None):
        if snapshots is None:
            snapshots=r.stable_badline(self.queue,self.offset,w.RAW,w.INPUT_IDENTITY,0)
        with patch.object(w,'INDEXES',{self.batch:None}):
            return w.recover_line(self.production,self.out,self.offset if begin is None else begin,
                w.RAW if raw is None else raw,self.evidence,snapshots)

    def tearDown(self):
        self.assertEqual((self.out/'attempt-budget/train/A.json').read_bytes(),self.budget_bytes)
        self.assertEqual((self.out/'collector.state.json').read_bytes(),self.origin_bytes)
        for item in reversed(getattr(self,'patches',[])):
            item.stop()
        self.tmp.cleanup()

    def test_exact_four_original_rows_offsetnull_budget_and_replay(self):
        self.ready()
        original=self.queue.read_bytes()
        rows=self.recover()
        self.assertEqual(rows,[self.rows[sid] for sid in ('A','B','C','D')])
        for row in rows:
            self.assertNotIn('original_byte_offset',row)
            rec=q.load(self.root_meta/'records'/ (q.safe_key(row)+'.json'))
            self.assertIsNone(rec['original_byte_offset'])
            self.assertEqual(rec['fragment_source_association'],'unknown')
        state=copy.deepcopy(self.state)
        for row in rows:r.admit(row,self.production,self.out,self.revision,state)
        self.assertEqual(state['counts'],{'joint_complete':3,'incomplete_or_failed':2})
        self.assertTrue((self.out/'jobs/train/A.json').exists())
        self.assertTrue((self.out/'jobs/train/C.json').exists())
        self.assertFalse((self.out/'jobs/train/B.json').exists())
        before=(self.out/'ready.jsonl').read_bytes()
        for row in self.recover():self.assertEqual(r.admit(row,self.production,self.out,self.revision,state),0)
        self.assertEqual((self.out/'ready.jsonl').read_bytes(),before)
        self.assertEqual(self.queue.read_bytes(),original)
        self.assertEqual((self.out/'collector.recovery.state.json').read_bytes(),self.state_bytes)
        authorization=q.load(self.root_meta/'physical-cursor-authorization.json')
        self.assertEqual(authorization['original_source_byte_offsets'],[None]*4)
        self.assertEqual(authorization['bytes_contained_complete_JSON_sources'],0)
        self.assertEqual(authorization['reconstructed_from_authoritative_terminal_index'],4)

    def test_shared_readonly_loader_preserves_all_files(self):
        self.ready()
        before={str(path):path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        with patch.object(w,'INDEXES',{self.batch:None}):
            rows,provenance=w.load_verified_rows(self.production,self.out,self.evidence)
        self.assertEqual(rows,[self.rows[sid] for sid in ('A','B','C','D')])
        self.assertEqual(provenance['original_source_byte_offsets'],[None]*4)
        self.assertEqual(provenance['origin'],'authoritative_terminal_reconciliation')
        self.assertEqual(provenance['fragment_source_association'],'unknown')
        self.assertFalse((self.root_meta/'physical-cursor-authorization.json').exists())
        after={str(path):path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        self.assertEqual(before,after)

    def test_original_collect_updates_cursor_once_without_budget_reset(self):
        self.ready()
        with patch.object(w,'INDEXES',{self.batch:None}), patch.object(r,'Evidence',return_value=self.evidence):
            result=r.collect(self.production,self.out,self.revision,self.summary,0)
            self.assertEqual(result['processed_source_hints'],5)
            self.assertEqual(result['new_queued_sources'],2)
            self.assertEqual(result['input_byte_offset'],self.queue.stat().st_size)
            self.assertEqual(result['sources_seen'],5)
            again=r.collect(self.production,self.out,self.revision,self.summary,0)
            self.assertEqual(again['processed_source_hints'],0)
            self.assertEqual(again['new_queued_sources'],0)

    def test_wrong_offset_and_unknown_bytes_failclosed(self):
        self.ready()
        with self.assertRaises(r.UnrecoverableLine):self.recover(begin=self.offset+1)
        with self.assertRaises(r.UnrecoverableLine):self.recover(raw=b'teX}\n')
        self.assertEqual((self.out/'collector.recovery.state.json').read_bytes(),self.state_bytes)

    def test_unstable_or_wrong_snapshot_sha_refused(self):
        self.ready()
        samples=r.stable_badline(self.queue,self.offset,w.RAW,w.INPUT_IDENTITY,0)
        samples[-1]['identity']=list(samples[-1]['identity']); samples[-1]['identity'][2]+=1
        with self.assertRaises(r.StreamBusy):self.recover(snapshots=samples)
        samples=r.stable_badline(self.queue,self.offset,w.RAW,w.INPUT_IDENTITY,0);samples[-1]['line_sha256']='0'*64
        with self.assertRaises(r.StreamBusy):self.recover(snapshots=samples)

    def test_prefix_mutation_and_truncation_refused(self):
        self.ready(); raw=self.queue.read_bytes()
        self.queue.write_bytes(b'X'+raw[1:])
        with self.assertRaises(q.SourceChanged):self.recover()
        self.queue.write_bytes(raw[:self.offset+2])
        with self.assertRaises(q.SourceChanged):self.recover()

    def test_legal_journal_and_index_tailappend_allowed(self):
        self.ready()
        with self.queue.open('ab') as stream:stream.write(q.encoded(self.rows['Before']))
        with Path(self.index_ref['path']).open('ab') as stream:stream.write(q.encoded({'id':'Before','split':'train'}))
        self.assertEqual(len(self.recover()),4)

    def test_terminal_or_manifest_sha_change_refused(self):
        self.ready()
        path=self.production/'sources/train/A/consumer.terminal.json'; old=path.read_bytes()
        path.write_bytes(old.replace(b'incomplete_candidate',b'failed_terminal'))
        with self.assertRaises((q.SourceChanged,r.UnrecoverableLine)):self.recover()
        path.write_bytes(old)
        self.manifest.write_bytes(self.manifest.read_bytes()+b'\n')
        self.evidence.members.clear()
        with self.assertRaises(q.SourceChanged):self.recover()

    def test_claimed_source_offset_or_rowsha_change_refused(self):
        self.ready()
        path=self.root_meta/'records/train/A.json'; value=q.load(path); value['original_byte_offset']=self.offset
        q.replace_state(path,value)
        with self.assertRaises(q.SourceChanged):self.recover()

    def test_missing_receipt_refused(self):
        self.ready(); (self.root_meta/'receipts/train/A.json').unlink()
        with self.assertRaises(FileNotFoundError):self.recover()

    def test_index_prefix_mutation_refused(self):
        self.ready(); path=Path(self.index_ref['path']); raw=path.read_bytes();path.write_bytes(b'X'+raw[1:])
        with self.assertRaises(q.SourceChanged):self.recover()

    def test_old_incident_handler_delegation_unchanged(self):
        marker=object()
        with patch.object(w,'old_recover_line',return_value=marker) as old:
            for begin in (w.r.BAD_OFFSET,w.v2.EMPTY_OFFSET,w.old.OFFSET):
                self.assertIs(w.recover_line(self.production,self.out,begin,b'original',None,[]),marker)
            self.assertEqual(old.call_count,3)

    def test_actual_three_byte_fragment_and_exact_current_constants(self):
        self.assertEqual(w.RAW, b'"}\n')
        self.assertEqual(len(w.RAW), 3)
        self.assertEqual(hashlib.sha256(w.RAW).hexdigest(), w.RAW_SHA)
        self.assertEqual(w.OFFSET, 2881725)
        self.assertEqual(w.OFFSET + len(w.RAW), 2881728)
        self.assertEqual((w.INDEXED_COUNT, w.SHARED_COUNT, len(w.EXPECTED)), (1000,999,1))

    def test_one_missing_failed_source_is_not_quality_approval(self):
        self.configure()
        key = 'train/A'
        path = Path(self.index_ref['path'])
        path.write_bytes(b''.join(q.encoded({'id':sid,'split':'train'}) for sid in ('Before','A')))
        self.index_ref.update(byte_length=path.stat().st_size, prefix_sha256=q.sha(path))
        with patch.object(w,'EXPECTED',{key:self.expected[key]}), patch.object(w,'INDEXES',{self.batch:None}), patch.object(r,'Evidence',return_value=self.evidence):
            result=w.prepare(self.production,self.out,self.revision,self.summary,0)
            self.assertEqual((result['prepared_sources'],result['prepared_complete_sources'],result['prepared_failed_sources']),(1,0,1))
            self.assertEqual(result['model_calls'],0)
            self.assertFalse(result['retry_budget_reset'])
            rows,provenance=w.load_verified_rows(self.production,self.out,self.evidence)
            self.assertEqual(rows,[self.rows['A']])
            self.assertEqual(provenance['original_source_byte_offsets'],[None])
            self.assertFalse(provenance['quality_approved'])
            value=q.load(self.root_meta/'records/train/A.json')
            self.assertEqual(value['original_repair_stages'],q.classify(self.rows['A'],self.production)['stages'])
            self.assertFalse(value['quality_approved'])
            self.assertTrue(value['pending_independent_review'])
        self.assertEqual((self.out/'collector.recovery.state.json').read_bytes(),self.state_bytes)
        self.assertFalse((self.out/'ready.jsonl').exists())

    def test_unreviewed_blank_or_json_fragment_at_other_offset_is_rejected(self):
        self.configure()
        for raw in (b'\n',b'"}\n',b'[]\n'):
            with self.assertRaises(r.UnrecoverableLine):
                w.journal_sources(self.before+q.encoded(self.rows['Before'])+raw,self.production)

    def test_preparation_exact_state_sha_gate(self):
        self.configure()
        state=q.load(self.out/'collector.recovery.state.json')
        state['counts']['joint_complete']=99
        q.replace_state(self.out/'collector.recovery.state.json',state)
        with self.assertRaises(q.SourceChanged):
            w.prepare(self.production,self.out,self.revision,self.summary,0)
        self.assertFalse(self.root_meta.exists())

    def test_bounded_fullchain_only_missing_not_shared(self):
        self.configure()
        with patch.object(w,'INDEXES',{self.batch:None}), patch.object(r,'Evidence',return_value=self.evidence), patch.object(self.evidence,'terminal',wraps=self.evidence.terminal) as full:
            w.prepare(self.production,self.out,self.revision,self.summary,0)
        self.assertEqual(full.call_count,4)
        self.assertEqual({args.args[0] for args in full.call_args_list},{'A','B','C','D'})
        self.assertNotIn('Before',{args.args[0] for args in full.call_args_list})

    def test_finite_production_denominator_rejects_non500_batch(self):
        self.configure()
        self.evidence.summary.update(total=222939,batch_size=500)
        self.evidence.batches.update({'synthetic-'+str(n):{} for n in range(447)})
        with patch.object(w,'INDEXES',{self.batch:([1,2],1,'sha')}), patch.object(w,'check_population',self.real_population), patch.object(w,'historical_seen',return_value={}), patch.object(w,'frozen_prefix',return_value=(self.queue.read_bytes(),self.journal_ref)):
            with self.assertRaises(q.SourceChanged):
                w.check_population(self.production,self.evidence)
        self.assertFalse(self.root_meta.exists())

    def test_private_chain_keeps_original_2053650_wrapper_unchanged(self):
        self.assertEqual(hashlib.sha256(Path(w.old.__file__).read_bytes()).hexdigest(), w.OLD_WRAPPER_SHA)
        self.assertEqual(w.old.OFFSET,2053650)
        self.assertEqual(w.old.RAW,bytes.fromhex('7465227d0a'))

    def test_real_finite_population_shared_equality_and_no_fullchain(self):
        self.configure()
        ev=Mock()
        ev.summary={'total':222939,'batch_size':500}
        ev.batches={self.batch:{'split':'train','count':500}}
        ev.batches.update({'synthetic-'+str(n):{} for n in range(447)})
        ev.member.side_effect=self.evidence.member
        index_tuple=(self.index_ref['input_identity'],self.index_ref['byte_length'],self.index_ref['prefix_sha256'])
        with patch.object(w,'check_population',self.real_population), patch.object(w,'historical_seen',return_value={}), patch.object(w,'INDEXES',{self.batch:index_tuple}), patch.object(w,'INDEXED_COUNT',5), patch.object(w,'SHARED_COUNT',1), patch.object(w,'JOURNAL_LENGTH',self.journal_ref['byte_length']), patch.object(w,'JOURNAL_SHA',self.journal_ref['prefix_sha256']), w.c.verified_hash_cache() as cache:
            raw, ir, jr, irefs, seen, indexed, missing=w.check_population(self.production,ev)
            self.assertEqual(len(indexed),5)
            self.assertEqual(missing,sorted(self.expected))
            self.assertEqual(seen,self.state['seen'])
            ev.terminal.assert_not_called()
            self.assertEqual({call.args[0] for call in ev.member.call_args_list},{'Before','A','B','C','D'})
            cache.verify_all()
            terminal_path=self.production/'sources/train/Before/consumer.terminal.json'
            original=terminal_path.read_bytes()
            terminal_path.write_bytes(original.replace(b'joint_candidate',b'other_candidate'))
            with self.assertRaises(q.SourceChanged):
                w.check_population(self.production,ev)
        self.assertFalse(self.root_meta.exists())

    def test_historical_four_records_are_metadata_only_and_hash_bound(self):
        history=self.out/'synthetic-reviewed-history'
        snapshot={'schema':w.old.SCHEMA,'wrapper_ref':{'path':str(w.OLD_WRAPPER),'sha256':w.OLD_WRAPPER_SHA},
            'missing_source_keys':sorted(w.old.EXPECTED),'fragment_source_association':'unknown',
            'bytes_contained_complete_JSON_sources':0}
        q.save_once(history/'snapshot.json',snapshot)
        snapshot_ref=q.identity(history/'snapshot.json')
        shas, expected={},{}
        for key in sorted(w.old.EXPECTED):
            row={'id':key.split('/')[1],'split':'train','batch_id':'train-b000005'}
            digest=q.digest(row)
            value={'source_key':key,'hint':row,'completed_row_sha256':digest,
                'original_journal_snapshot_ref':snapshot_ref,'original_byte_offset':None,
                'fragment_source_association':'unknown'}
            q.save_once(history/'records'/(key+'.json'),value)
            record_ref=q.identity(history/'records'/(key+'.json'))
            receipt={'record_ref':record_ref,'original_journal_snapshot_ref':snapshot_ref,'retry_budget_reset':False}
            q.save_once(history/'receipts'/(key+'.json'),receipt)
            shas[key]=(record_ref['sha256'],q.sha(history/'receipts'/(key+'.json')))
            expected[key]=digest
        with patch.object(w.old,'namespace',return_value=(self.production,self.out,history)), patch.object(w,'HISTORY_SNAPSHOT_SHA',snapshot_ref['sha256']), patch.object(w,'HISTORY_SHAS',shas), patch.object(r.Evidence,'terminal',side_effect=AssertionError('historical fullchain must not run')) as full:
            self.assertEqual(w.historical_seen(self.production,self.out),expected)
            full.assert_not_called()
            key=sorted(shas)[0]
            path=history/'records'/(key+'.json')
            path.write_bytes(path.read_bytes()+b' ')
            with self.assertRaises(q.SourceChanged):w.historical_seen(self.production,self.out)
        self.assertFalse((self.out/'ready.jsonl').exists())

    def test_running_wrapper_and_dependencies_are_frozen_at_prepare_boundary(self):
        self.configure()
        with patch.object(w,'SELF_REF',{'path':str(Path(w.__file__)),'sha256':'0'*64}):
            with self.assertRaises(q.SourceChanged):w.prepare(self.production,self.out,self.revision,self.summary,0)
        self.assertFalse(self.root_meta.exists())
        self.assertEqual((self.out/'collector.recovery.state.json').read_bytes(),self.state_bytes)


if __name__ == '__main__':unittest.main()


