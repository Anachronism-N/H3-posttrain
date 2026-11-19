"""Isolated, production-schema CPU acceptance; no media, models or real queues."""
from __future__ import annotations
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import batchrpc_publication_next_v2 as publication
import beatthis_rejection_next_v2 as beat
import unified_access_transport_batchrpc_next_v2 as transport
import unified_lane_batchrpc_next_v2 as runner
import ms_unified_novlm_v1 as core

HERE=Path(__file__).resolve().parent
CALLS=dict(gpu=0,model=0,claim=0,copy=0,extract=0)

def blocked(kind):
    def reject(*args,**kwargs):
        CALLS[kind]+=1; raise AssertionError('forbidden '+kind+' callback')
    return reject

def digest(raw): return hashlib.sha256(raw).hexdigest()

def write(path,raw):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw)

def rebind(plan):
    plan['commit']['publications']=[dict(path=m['path'],sha256=m['sha256'],receipt=m) for m in plan['manifest']]
    archive=next((m for m in plan['manifest'] if m['path'].endswith('/batch.tar')),None)
    if archive is not None: plan['commit']['batch_sha256']=archive['sha256']
    plan['commit_sha256']=digest(transport.encoded(plan['commit']))
    return plan

class Acceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=[dict(id='case%05d'%i,split='test',video_path='/synthetic_media/video/case%05d.mp4'%i,
                       audio_path='/synthetic_media/audio/case%05d.wav'%i,meta_path='/synthetic_media/meta/case%05d.json'%i,source_ordinal=i+1) for i in range(20000)]
        cls.mraw=b''.join(transport.encoded(r) for r in cls.rows)
        cls.code_refs=[dict(path='/synthetic_code/'+name,sha256=publication.sha(HERE/name)) for name in (
            'ms_unified_novlm_v1.py','ms_fullframe_pose2d_draft_v2.py','ms_fullframe_pose2d_draft_v1.py','ms_pose2d_draft.py','dwpose_full_extract.py')]
        cls.patches=[patch.object(core,'load_models',blocked('model')),patch.object(core,'copy_source',blocked('copy')),
                     patch.object(core,'extract_and_rule',blocked('extract')),patch.object(transport.BaseAccess,'claim',blocked('claim')),
                     patch.object(runner.Gate,'gpu',blocked('gpu')),patch.object(runner.Gate,'gpu_once',blocked('gpu'))]
        for p in cls.patches:p.start()
    @classmethod
    def tearDownClass(cls):
        for p in reversed(cls.patches):p.stop()
        if any(CALLS.values()): raise AssertionError('GPU/model/claim/source callbacks called: '+str(CALLS))
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='ms_batchrpc_next_cpu_',dir='/tmp')
        self.root=Path(self.temp.name)
        self.old_p,self.old_previous=transport.P,transport.PREVIOUS_P
        transport.P=self.root/'external_ms_unified_clusters80k_20261027_v1'
        transport.PREVIOUS_P=self.root/'synthetic_previous'
        self.worker='a100-extra-03';self.gpu=0
        for worker in (self.worker,):
            tp,mp=transport.metadata(worker)
            ticket=dict(schema='ms_external_unified_assignment_v1',worker_id=worker,edition=transport.P.name,source_count=20000,
                        assignment_path=str(mp),assignment_sha256=digest(self.mraw),claims_root=str(transport.P/'claims'),
                        output_root=str(transport.P/'results'/worker),code_refs=self.code_refs,allowed_gpu_indices=[0,1,2,3],
                        publication_sources_per_batch=20,automatic_retry=False,models_reused=True,weight_load_once_per_GPU_process=True)
            traw=transport.encoded(ticket);write(tp,traw);write(mp,self.mraw)
            old=transport.PREVIOUS_P/'assignments'/(worker+'.ticket.json');write(old,b'{"synthetic":true}\n')
            queue=dict(schema='ms_external_authorized_rollover_queue_v1',worker_id=worker,
                       automatic_rollover_within_preallocated_tickets=True,no_automatic_retry_sealed_previous_sources=True,
                       no_restart_active_previous_job=True,run_until_preallocated_queue_exhausted=True,request_more_when_remaining_below=1000,
                       tickets=[dict(path=str(old),sha256=digest(old.read_bytes())),dict(path=str(tp),sha256=digest(traw))])
            write(transport.queue_path(worker),transport.encoded(queue))
        self.bind()
    def tearDown(self):
        transport.P,transport.PREVIOUS_P=self.old_p,self.old_previous
        self.temp.cleanup()
        self.assertEqual(CALLS,dict(gpu=0,model=0,claim=0,copy=0,extract=0))
    def bind(self):
        self.access=transport.Access(self.worker)
        tp,mp=transport.metadata(self.worker)
        if not tp.exists():
            ticket=dict(self.ticket,worker_id=self.worker,assignment_path=str(mp),output_root=str(transport.P/'results'/self.worker))
            write(tp,transport.encoded(ticket));write(mp,self.mraw)
            previous=transport.PREVIOUS_P/'assignments'/(self.worker+'.ticket.json');write(previous,b'{\"synthetic\":true}\n')
            old_queue=transport.strict(transport.queue_path('a100-extra-03').read_bytes())
            queue=dict(old_queue,worker_id=self.worker,tickets=[dict(path=str(previous),sha256=digest(previous.read_bytes())),dict(path=str(tp),sha256=digest(tp.read_bytes()))])
            write(transport.queue_path(self.worker),transport.encoded(queue))
        traw=tp.read_bytes();ticket=transport.strict(traw)
        self.access.bind(ticket,self.rows,digest(traw),digest(self.mraw));self.ticket=ticket
    def build(self,n=20,mixed=False,diagnostic=False,batch_no=0):
        run_id='g%d-100-12345678'%self.gpu;name=run_id+'-batch%03d'%batch_no
        local=self.root/'local_artifacts'/name;target=Path(self.ticket['output_root'])/'batches'/('gpu%d'%self.gpu)/name
        local.mkdir(parents=True)
        receipts=[]
        for j in range(n):
            index=self.gpu+j*4;row=self.rows[index]
            status=('success','partial','needs_regeneration','failed','existing')[j%5] if mixed else 'success'
            r=dict(schema='ms_unified_novlm_case_receipt_v1',source_row=row,assignment_index=index,vlm_called=False,
                   status='extracted_rule_draft_unapproved',draft_status='draft_generated',frame_count=2)
            if status=='existing':r['status']='existing_claim_not_reassigned'
            elif status=='failed':r['status']='failed_sealed_no_retry';r.pop('draft_status');r['error']='synthetic sealed failure'
            elif status in ('partial','needs_regeneration'):r['draft_status']=status
            case=local/'cases'/row['split']/row['id']
            if diagnostic and j==0:
                r['status']='failed_sealed_no_retry';r.pop('draft_status',None)
                metadata=dict(duration_seconds=2.0,sample_rate=16000,channels=1,frames=32000)
                observed=beat.audit_actual((np.array([-.1,np.nan,1.0],dtype=np.float32),np.array([.8],dtype=np.float64)),metadata,row,index,{'audio_path':'b'*64},case)
                r['actual_audio_sha256']='b'*64; r['actual_audio_metadata']=metadata
                r['failed_predicates']=observed['failed_predicates']
                r['diagnostic_publications']=[dict(path=str(target/Path(p).relative_to(local)),bytes=Path(p).stat().st_size,sha256=publication.sha(p)) for p in observed['diagnostic_paths']]
            if r['status']!='existing_claim_not_reassigned':
                if r['status']!='failed_sealed_no_retry':
                    buffer=io.BytesIO();np.savez(buffer,id=np.array(row['id']),xy=np.zeros((2,134,2),np.float32));write(case/'pose.npz',buffer.getvalue())
                    for filename in ('fullframe_manifest.json','beats.json','draft.json'):write(case/filename,transport.encoded(dict(id=row['id'],split=row['split'],synthetic=True)))
                write(case/'receipt.json',transport.encoded(r))
            receipts.append(r)
        with tarfile.open(local/'batch.tar','x') as archive:
            for case in sorted((local/'cases').glob('*/*')):
                for filename in core.FILES:
                    path=case/filename
                    if path.is_file():archive.add(path,arcname=str(path.relative_to(local)),recursive=False)
        c=dict(schema='ms_unified_novlm_batch_commit_v1',edition=transport.P.name,worker=self.worker,gpu=self.gpu,run_id=run_id,
               assignment_sha256=digest(self.mraw),code_signatures={Path(ref['path']).name:ref['sha256'] for ref in self.code_refs},
               assignment_indices=[r['assignment_index'] for r in receipts],receipts=receipts,quality_approved=False,vlm_called=False,
               next_version_code_signatures={'beatthis_rejection_next_v2.py':publication.sha(HERE/'beatthis_rejection_next_v2.py')},
               unique_sources_attempted=sum(r['status']!='existing_claim_not_reassigned' for r in receipts),unique_sources_with_action_events=0)
        return publication.prepare(local,target,c)
    def q(self,plan,op='publish_batch'):
        return dict(worker=self.worker,op=op,path=plan['target'],**self.access.binding,plan=plan)
    def send(self,saved,q=None,stream=None):
        sequence=publication.FileSequence(saved['plan']['manifest'],saved['local_files']) if stream is None else stream
        try:return transport.dispatch(q or self.q(saved['plan']),sequence,io.BytesIO())
        finally:
            if hasattr(sequence,'close'):sequence.close()
    def commit_path(self,saved):return Path(saved['plan']['target'])/'commit.json'
    def test_all_success_20_and_production_20000_schema(self):
        saved=self.build();result=self.access.publish_prepared(saved)
        self.assertEqual(result['state'],'batch_committed_verified');self.assertEqual(len(result['verification']['publications']),101)
        self.assertEqual(self.access.read(str(self.commit_path(saved)),expected_sha=saved['plan']['commit_sha256']),transport.encoded(saved['plan']['commit']))
    def test_mixed_20_success_partial_regeneration_failed_existing(self):
        saved=self.build(mixed=True);result=self.access.publish_prepared(saved)
        self.assertEqual(result['verification']['state'],'committed_verified');self.assertEqual(saved['plan']['commit']['unique_sources_attempted'],16)
        self.assertFalse((transport.P/'claims').exists())
        target=Path(saved['plan']['target'])
        for r in saved['plan']['commit']['receipts']:
            case=target/'cases'/r['source_row']['split']/r['source_row']['id']
            names=set(p.name for p in case.glob('*'))
            expected=set() if r['status']=='existing_claim_not_reassigned' else {'receipt.json'} if r['status']=='failed_sealed_no_retry' else set(core.FILES)
            self.assertEqual(names,expected)
    def test_short_tail_3(self):
        saved=self.build(n=3,mixed=True);self.assertEqual(self.access.publish_prepared(saved)['verification']['state'],'committed_verified')
        self.assertEqual(len(saved['plan']['commit']['receipts']),3)
    def test_all_existing_claims_no_new_outputs_or_callbacks(self):
        saved=self.build(n=1);r=saved['plan']['commit']['receipts'][0];r['status']='existing_claim_not_reassigned'
        # Build a legitimate empty tar rather than changing/removing production claims.
        local=Path(next(iter(saved['local_files'].values()))).parents[3]
        second=self.build(n=5,mixed=True,batch_no=1)
        second['plan']['commit']['receipts']=[dict(second['plan']['commit']['receipts'][4],assignment_index=16)]
        # Existing-claim-only production batch has no case members.
        second['plan']['manifest']=[m for m in second['plan']['manifest'] if m['path'].endswith('/batch.tar')]
        archive=Path(second['local_files'][second['plan']['manifest'][0]['path']]);buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w'):pass
        archive.write_bytes(buffer.getvalue());m=second['plan']['manifest'][0];m.update(bytes=archive.stat().st_size,sha256=publication.sha(archive))
        second['plan']['commit'].update(assignment_indices=[16],unique_sources_attempted=0);rebind(second['plan'])
        self.assertEqual(self.access.publish_prepared(second)['verification']['state'],'committed_verified')
        self.assertFalse((transport.P/'claims').exists())
    def test_different_authorized_worker_and_gpu(self):
        self.worker='a100-extra-06';self.gpu=3;self.bind();saved=self.build(n=3,mixed=True)
        self.assertEqual(self.access.publish_prepared(saved)['verification']['state'],'committed_verified')
    def test_rejection_predicates_nan_dtype_shape_and_diagnostic_publication(self):
        saved=self.build(n=3,diagnostic=True);result=self.access.publish_prepared(saved)
        r=saved['plan']['commit']['receipts'][0];self.assertFalse(r['vlm_called']);self.assertIn('beats.all_finite',r['failed_predicates']);self.assertIn('beats.nonnegative',r['failed_predicates'])
        path=next(ref['path'] for ref in r['diagnostic_publications'] if ref['path'].endswith('.npz'))
        with np.load(path,allow_pickle=False) as raw:
            self.assertEqual(raw['beats'].dtype,np.float32);self.assertTrue(np.isnan(raw['beats'][1]));self.assertEqual(raw['downbeats'].shape,(1,))
        self.assertEqual(result['verification']['state'],'committed_verified')
    def test_rejection_every_original_predicate_and_shape(self):
        cases=[(np.array([[.1,.2]]),np.array([]),'beats.ndim_eq_1'),(np.array([.2,.1]),np.array([]),'beats.strictly_increasing'),
               (np.array([2.03]),np.array([]),'beats.within_audio_duration_plus_0.02'),(np.array([.1]),np.array([.2]),'downbeats.belongs_to_beats_atol_1e-6_rtol_0')]
        for j,(b,d,predicate) in enumerate(cases):
            row=self.rows[j*4];case=self.root/('diag%d'%j)/'cases'/row['split']/row['id']
            observed=beat.audit_actual((b,d),{'duration_seconds':2},row,j*4,{'audio_path':'b'*64},case)
            self.assertIn(predicate,observed['failed_predicates'])
            with np.load(observed['diagnostic_paths'][0],allow_pickle=False) as arrays:self.assertEqual(arrays['beats'].shape,b.shape)
        observed=beat.audit_actual((np.array([.1,2.02]),np.array([.1])),{'duration_seconds':2},self.rows[0],0,{'audio_path':'b'*64},self.root/'no_diag'/'cases'/'test'/'case00000')
        self.assertEqual(observed,{});self.assertFalse((self.root/'no_diag'/'diagnostics').exists())
    def test_identity_ticket_queue_assignment_gpu_run_and_source_tampering(self):
        saved=self.build(n=3);mutations=[lambda q:q.update(ticket_sha256='0'*64),lambda q:q.update(assignment_sha256='0'*64),lambda q:q.update(queue_sha256='0'*64),
            lambda q:q['plan']['commit'].update(worker='a100-extra-04'),lambda q:q['plan']['commit'].update(gpu=1),lambda q:q['plan']['commit'].update(run_id='unbound'),
            lambda q:q['plan']['commit']['receipts'][0]['source_row'].update(audio_path='/other.wav')]
        for mutate in mutations:
            q=copy.deepcopy(self.q(saved['plan']));mutate(q);rebind(q['plan'])
            with self.assertRaises(ValueError):self.send(saved,q)
        self.assertFalse(self.commit_path(saved).exists())
    def test_sequence_duplicate_gap_reverse_and_more_than_20(self):
        saved=self.build(n=3)
        for indices in ([0,0,8],[0,8,12],[8,4,0],[0,4,9]):
            plan=copy.deepcopy(saved['plan']);plan['commit']['assignment_indices']=indices
            for r,i in zip(plan['commit']['receipts'],indices):r['assignment_index']=i
            rebind(plan)
            with self.assertRaises(ValueError):self.send(saved,self.q(plan))
        plan=copy.deepcopy(saved['plan']);plan['commit']['receipts']*=7;plan['commit']['assignment_indices']=[r['assignment_index'] for r in plan['commit']['receipts']];rebind(plan)
        with self.assertRaises(ValueError):self.send(saved,self.q(plan))
    def test_manifest_path_duplicate_unknown_link_and_total_capacity(self):
        saved=self.build(n=3)
        for kind in ('traversal','duplicate','outside','extra','capacity'):
            plan=copy.deepcopy(saved['plan']);m=plan['manifest'][0]
            if kind=='traversal':m['path']=plan['target']+'/cases/../escape.json'
            elif kind=='duplicate':plan['manifest'].append(copy.deepcopy(m))
            elif kind=='outside':m['path']='/tmp/forbidden.json'
            elif kind=='extra':plan['manifest'].append(dict(path=plan['target']+'/input.mp4',bytes=1,sha256='0'*64))
            elif kind=='capacity':m['bytes']=transport.LIMIT
            rebind(plan)
            with self.assertRaises(ValueError):self.send(saved,self.q(plan))
        m=saved['plan']['manifest'][0];dest=Path(m['path']);dest.parent.mkdir(parents=True);dest.symlink_to(self.root/'unowned')
        with self.assertRaises((ValueError,FileNotFoundError)):self.send(saved)
        self.assertFalse(self.commit_path(saved).exists())
    def test_total_exact_bound_and_one_byte_over(self):
        saved=self.build(n=3);total=sum(m['bytes'] for m in saved['plan']['manifest'])
        with patch.object(transport,'LIMIT',total):self.assertEqual(self.access.publish_prepared(saved)['verification']['state'],'committed_verified')
        with patch.object(transport,'LIMIT',total-1):
            with self.assertRaises(ValueError):transport.validate_batch(self.q(saved['plan']))
    def test_wrong_sha_truncated_extra_and_reordered_stream_commit_absent(self):
        for kind in ('wrong','truncated','extra','reordered'):
            saved=self.build(n=1,batch_no=('wrong','truncated','extra','reordered').index(kind));seq=publication.FileSequence(saved['plan']['manifest'],saved['local_files']);parts=[]
            while chunk:=seq.read(publication.CHUNK):parts.append(chunk)
            seq.close();raw=b''.join(parts)
            if kind=='wrong':raw=bytes([raw[0]^1])+raw[1:]
            elif kind=='truncated':raw=raw[:-1]
            elif kind=='extra':raw+=b'!'
            elif kind=='reordered':raw=parts[-1]+b''.join(parts[:-1])
            with self.assertRaises(ValueError):self.send(saved,stream=io.BytesIO(raw))
            self.assertFalse(self.commit_path(saved).exists())
    def test_interrupted_member_then_publication_only_recovery(self):
        saved=self.build(n=3);seq=publication.FileSequence(saved['plan']['manifest'],saved['local_files']);raw=seq.read(publication.CHUNK);seq.close()
        first=saved['plan']['manifest'][0]['bytes'];raw=raw[:first+10]
        with self.assertRaises(ValueError):self.send(saved,stream=io.BytesIO(raw))
        self.assertFalse(self.commit_path(saved).exists())
        result=self.access.recover_prepared(saved);self.assertEqual(result['state'],'publication_only_recovered_verified')
        self.assertEqual(self.access.recover_prepared(saved)['state'],'already_committed_verified')
    def test_unknown_commit_lost_ack_checked_before_fallback(self):
        saved=self.build(n=3);actual=self.access.batch_exchange
        def lost(plan,stream):actual(plan,stream);raise RuntimeError('synthetic lost acknowledgement')
        with patch.object(self.access,'batch_exchange',lost),patch.object(self.access,'publish_file',side_effect=AssertionError('must inspect committed first')):
            result=self.access.publish_prepared(saved)
        self.assertEqual(result['state'],'already_committed_verified')
    def test_batch_failure_falls_back_only_to_existing_generated_bytes(self):
        saved=self.build(n=3)
        with patch.object(self.access,'batch_exchange',side_effect=RuntimeError('synthetic transport outage')):
            result=self.access.publish_prepared(saved)
        self.assertEqual(result['state'],'publication_only_recovered_verified')
    def test_existing_commit_conflict_no_overwrite_or_fallback(self):
        saved=self.build(n=3);path=self.commit_path(saved);write(path,b'{"conflict":true}\n');before=path.read_bytes()
        with patch.object(self.access,'batch_exchange',side_effect=AssertionError('no send on conflict')):
            with self.assertRaises(ValueError):self.access.publish_prepared(saved)
        self.assertEqual(path.read_bytes(),before)
    def test_existing_member_conflict_and_missing_committed_member(self):
        saved=self.build(n=3);m=saved['plan']['manifest'][0];write(m['path'],b'conflict');before=Path(m['path']).read_bytes()
        with self.assertRaises(ValueError):self.access.recover_prepared(saved)
        self.assertEqual(Path(m['path']).read_bytes(),before)
        second=self.build(n=3,batch_no=1);self.access.publish_prepared(second);Path(second['plan']['manifest'][0]['path']).unlink()
        with self.assertRaises(ValueError):self.access.recover_prepared(second)
    def test_frozen_local_bytes_changed_no_reexecution(self):
        saved=self.build(n=3);m=saved['plan']['manifest'][0];write(saved['local_files'][m['path']],b'changed')
        with self.assertRaises(ValueError):self.access.recover_prepared(saved)
        self.assertFalse(self.commit_path(saved).exists())
    def test_archive_traversal_link_duplicate_missing_and_sha_tampering(self):
        for kind in ('traversal','link','duplicate','missing','sha'):
            saved=self.build(n=1,batch_no=('traversal','link','duplicate','missing','sha').index(kind));plan=saved['plan'];archive=Path(saved['local_files'][str(Path(plan['target'])/'batch.tar')])
            cases=[m for m in plan['manifest'] if '/cases/' in m['path']]
            with tarfile.open(archive,'w') as pack:
                for j,m in enumerate(cases):
                    if kind=='missing' and j==0:continue
                    path=Path(saved['local_files'][m['path']]);name=path.relative_to(archive.parent).as_posix()
                    if j==0 and kind=='traversal':name='../escape'
                    if j==0 and kind=='link':
                        info=tarfile.TarInfo(name);info.type=tarfile.SYMTYPE;info.linkname='/tmp/forbidden';pack.addfile(info);continue
                    if j==0 and kind=='sha':
                        raw=b'0'*m['bytes'];info=tarfile.TarInfo(name);info.size=len(raw);pack.addfile(info,io.BytesIO(raw));continue
                    pack.add(path,arcname=name,recursive=False)
                    if j==0 and kind=='duplicate':pack.add(path,arcname=name,recursive=False)
            spec=next(m for m in plan['manifest'] if m['path'].endswith('/batch.tar'));spec.update(bytes=archive.stat().st_size,sha256=publication.sha(archive));rebind(plan)
            with self.assertRaises(ValueError):self.send(saved)
            self.assertFalse(self.commit_path(saved).exists())
    def test_case_receipt_content_cannot_differ_from_commit(self):
        saved=self.build(n=1);plan=saved['plan'];m=next(m for m in plan['manifest'] if m['path'].endswith('/receipt.json'));path=Path(saved['local_files'][m['path']]);raw=transport.encoded(dict(source_row=self.rows[4],vlm_called=False));path.write_bytes(raw)
        m.update(bytes=len(raw),sha256=digest(raw));archive=Path(saved['local_files'][str(Path(plan['target'])/'batch.tar')])
        with tarfile.open(archive,'w') as pack:
            for spec in plan['manifest']:
                if '/cases/' in spec['path']:
                    p=Path(saved['local_files'][spec['path']]);pack.add(p,arcname=str(p.relative_to(archive.parent)),recursive=False)
        a=next(spec for spec in plan['manifest'] if spec['path'].endswith('/batch.tar'));a.update(bytes=archive.stat().st_size,sha256=publication.sha(archive));rebind(plan)
        with self.assertRaises(ValueError):self.send(saved)
        self.assertFalse(self.commit_path(saved).exists())
    def test_readback_tamper_keeps_unknown_commit_not_success(self):
        saved=self.build(n=3);self.access.publish_prepared(saved);m=saved['plan']['manifest'][0];p=Path(m['path']);raw=p.read_bytes();p.write_bytes(bytes([raw[0]^1])+raw[1:])
        with self.assertRaises(ValueError):self.access.verify_plan(saved['plan'])
    def test_runner_publish_connected_and_cpu_only(self):
        saved=self.build(n=3,mixed=True);local=Path(saved['local_files'][str(Path(saved['plan']['target'])/'batch.tar')]).parent
        lane=runner.Lane.__new__(runner.Lane);lane.args=SimpleNamespace(worker=self.worker,gpu=self.gpu);lane.access=self.access;lane.receipts=saved['plan']['commit']['receipts'];lane.run_id=saved['plan']['commit']['run_id']
        lane.work=local.parent;lane.reserved=0;lane.output_used=0;lane.inputs=[];lane.batch_no=0;lane.stage=lambda:None;lane.batch_paths=lambda:(local,Path(saved['plan']['target']));lane.report=lambda **kwargs:None
        lane.core=SimpleNamespace(write_once=core.write_once);lane.progress=dict(published=0,published_generated_with_events=0,profile_seconds={})
        lane.publish(False);self.assertEqual(lane.progress['published'],3);self.assertFalse(local.exists());self.assertTrue(self.commit_path(saved).exists())
    def test_runner_actual_prepare_batch_rpc_path(self):
        saved=self.build(n=3,mixed=True);local=Path(saved['local_files'][str(Path(saved['plan']['target'])/'batch.tar')]).parent
        (local/'publication_plan.json').unlink();(local/'batch.tar').unlink()
        lane=runner.Lane.__new__(runner.Lane);lane.args=SimpleNamespace(worker=self.worker,gpu=self.gpu);lane.access=self.access;lane.receipts=saved['plan']['commit']['receipts'];lane.run_id=saved['plan']['commit']['run_id'];lane.assignment_sha=digest(self.mraw);lane.signatures=saved['plan']['commit']['code_signatures']
        lane.work=local.parent;lane.reserved=0;lane.output_used=0;lane.inputs=[];lane.batch_no=0;lane.stage=lambda:None;lane.batch_paths=lambda:(local,Path(saved['plan']['target']));lane.report=lambda **kwargs:None
        lane.core=SimpleNamespace(write_once=core.write_once,FILES=core.FILES,v2=SimpleNamespace(FLAGS={}));lane.progress=dict(published=0,published_generated_with_events=0,profile_seconds={})
        with patch.object(self.access,'batch_exchange',wraps=self.access.batch_exchange) as calls:lane.publish(False);self.assertEqual(calls.call_count,1)
        self.assertEqual(lane.progress['published'],3);self.assertFalse(local.exists())
    def test_runner_commit_readback_failure_no_count_or_reclaim(self):
        saved=self.build(n=3);local=Path(saved['local_files'][str(Path(saved['plan']['target'])/'batch.tar')]).parent
        lane=runner.Lane.__new__(runner.Lane);lane.args=SimpleNamespace(worker=self.worker,gpu=self.gpu);lane.access=self.access;lane.receipts=saved['plan']['commit']['receipts'];lane.run_id=saved['plan']['commit']['run_id'];lane.work=local.parent;lane.inputs=[];lane.batch_no=0;lane.stage=lambda:None;lane.batch_paths=lambda:(local,Path(saved['plan']['target']));lane.progress=dict(published=0)
        with patch.object(self.access,'read',side_effect=ValueError('synthetic final readback failed')):
            with self.assertRaises(ValueError):lane.publish(False)
        self.assertEqual(lane.progress['published'],0);self.assertTrue(local.exists())
    def test_runner_no_duplicate_count_after_cleanup_interruption(self):
        saved=self.build(n=3);local=Path(saved['local_files'][str(Path(saved['plan']['target'])/'batch.tar')]).parent
        lane=runner.Lane.__new__(runner.Lane);lane.args=SimpleNamespace(worker=self.worker,gpu=self.gpu);lane.access=self.access;lane.receipts=saved['plan']['commit']['receipts'];lane.run_id=saved['plan']['commit']['run_id'];lane.work=local.parent;lane.inputs=[];lane.batch_no=0;lane.stage=lambda:None;lane.batch_paths=lambda:(local,Path(saved['plan']['target']));lane.report=lambda **kwargs:None
        lane.core=SimpleNamespace(write_once=core.write_once);lane.progress=dict(published=0,published_generated_with_events=0,profile_seconds={})
        with patch.object(runner.shutil,'rmtree',side_effect=RuntimeError('synthetic cleanup interruption')):
            with self.assertRaises(RuntimeError):lane.publish(False)
        self.assertEqual(lane.progress['published'],3);self.assertTrue(local.exists())
        lane.publish(False);self.assertEqual(lane.progress['published'],3);self.assertFalse(local.exists())
    def test_parent_directory_link_and_missing_queue_sha_rejected(self):
        saved=self.build(n=1);q=self.q(saved['plan']);q.pop('queue_sha256')
        with self.assertRaises(ValueError):self.send(saved,q)
        target=Path(saved['plan']['target']);target.parent.mkdir(parents=True);foreign=self.root/'synthetic_foreign';foreign.mkdir();target.symlink_to(foreign,target_is_directory=True)
        with self.assertRaises(ValueError):self.send(saved)
        self.assertEqual(list(foreign.iterdir()),[])
    def test_deny_deployment_cli_before_gpu_models_claims(self):
        with self.assertRaises(SystemExit) as e:runner.main(['--root','/synthetic','--worker',self.worker,'--gpu','0','--ticket','/synthetic/t','--manifest','/synthetic/m','--queue','/synthetic/q','--previous-progress','/synthetic/p','--code-root','/synthetic/c','--scail-root','/synthetic/s','--beat-checkpoint','/synthetic/b'])
        self.assertEqual(e.exception.code,2)
    def test_embedded_relay_same_validation_code_and_full_sha(self):
        self.assertIn('validate_batch',transport.RELAY_CODE);ast.parse(transport.RELAY_CODE)
        saved=self.build(n=3);self.access.publish_prepared(saved)
        with self.assertRaises(ValueError):self.access.read(str(self.commit_path(saved)),expected_sha='0'*64)
    def test_no_input_media_weights_or_gpu_imports(self):
        self.assertFalse('torch' in sys.modules);self.assertEqual(CALLS,dict(gpu=0,model=0,claim=0,copy=0,extract=0))

    def synthetic_core_call(self, actual, metadata=None, name='sync'):
        frozen=runner.load_frozen_core(SimpleNamespace(code_root=HERE))
        self.assertEqual(publication.sha(Path(frozen.__file__)),beat.CORE_SHA)
        metadata=metadata or dict(duration_seconds=2.0,sample_rate=16000,channels=1,frames=32000)
        row=self.rows[0];case=self.root/name/'cases'/row['split']/row['id']; hashes={'video_path':'a'*64,'audio_path':'b'*64}
        model_calls=[]
        def underlying(path):
            model_calls.append(path);return actual
        tap=beat.AuditTap(underlying,lambda path:metadata);tap.configure(row,0,hashes,case)
        probe=lambda path:dict(streams=[dict(width=2,height=2,duration='2.0',time_base='1/2',nb_frames='4')],
                              frames=[dict(best_effort_timestamp=str(i),best_effort_timestamp_time=str(i/2)) for i in range(4)])
        decoder=lambda *args:iter([np.zeros((2,2,3),np.uint8) for _ in range(4)])
        lines=[];old_trace=sys.gettrace()
        def trace(frame,event,arg):
            if event=='line' and frame.f_code.co_filename==frozen.__file__:lines.append(frame.f_lineno)
            return trace
        error=None;result=None
        try:
            sys.settrace(trace)
            result=frozen.extract_and_rule(row,{'video_path':Path('/synthetic.mp4'),'audio_path':Path('/synthetic.wav')},hashes,
                (lambda frame:([],[],[]),tap),case,Path('/synthetic_bound')/row['split']/row['id'],probe=probe,decoder=decoder,audio_info=tap.audio_info)
        except Exception as caught:error=caught
        finally:sys.settrace(old_trace)
        self.assertIs(tap.metadata,metadata,msg=repr(error))
        self.assertEqual(model_calls,['/synthetic.wav'])
        return frozen,tap,error,result,lines,case

    def test_frozen_core_sync_valid_preserves_values_metadata_and_original_policy(self):
        actual=(np.array([0.1,2.02],np.float32),np.array([0.1],np.float32));before=[a.tobytes() for a in actual]
        frozen,tap,error,result,lines,case=self.synthetic_core_call(actual)
        self.assertIsNone(error);self.assertEqual(tap.result,{});self.assertIsNone(tap.gap)
        self.assertEqual([a.tobytes() for a in actual],before);self.assertIn(303,lines);self.assertNotIn(304,lines)
        side=json.loads((case/'beats.json').read_bytes());self.assertEqual(side['actual_audio_metadata'],tap.metadata)
        self.assertEqual(side['beats'],np.asarray(actual[0],dtype=float).tolist())
        self.assertEqual(result['status'],'extracted_rule_draft_unapproved')

    def test_frozen_core_sync_rejects_every_policy_without_wrapper_raising(self):
        cases=[(np.array([-.1,np.nan],np.float32),np.array([],np.float64)),
               (np.array([[.1,.2]],np.float32),np.array([],np.float64)),
               (np.array([.2,.1],np.float64),np.array([],np.float32)),
               (np.array([2.03],np.float64),np.array([],np.float64)),
               (np.array([.1],np.float64),np.array([.2],np.float64))]
        for i,actual in enumerate(cases):
            frozen,tap,error,result,lines,case=self.synthetic_core_call(actual,name='sync%d'%i)
            self.assertIsInstance(error,ValueError);self.assertEqual(str(error),'Invalid actual BeatThis beat/downbeat output')
            self.assertIn(303,lines);self.assertIn(304,lines);self.assertIsNone(result);self.assertIsNone(tap.gap)
            with np.load(tap.result['diagnostic_paths'][0],allow_pickle=False) as arrays:
                for name,value in zip(('beats','downbeats'),actual):
                    self.assertEqual(arrays[name].dtype,value.dtype);self.assertEqual(arrays[name].shape,value.shape);self.assertEqual(arrays[name].tobytes(),value.tobytes())
            self.assertFalse((case/'receipt.json').exists())

    def test_frozen_core_object_overbudget_and_json_cap_explicit_gaps(self):
        cases=[(np.array([-.1],object),{'duration_seconds':2.0}),
               (np.full(300000,-.1,np.float64),{'duration_seconds':2.0}),
               (np.array([-.1]),{'duration_seconds':2.0,'oversized':'x'*(beat.MAX_JSON_BYTES+1)})]
        for i,(b,metadata) in enumerate(cases):
            frozen,tap,error,result,lines,case=self.synthetic_core_call((b,np.array([])),metadata,name='gap%d'%i)
            self.assertEqual(str(error),'Invalid actual BeatThis beat/downbeat output');self.assertIn(304,lines)
            self.assertEqual(tap.result,{});self.assertEqual(tap.gap['array_coverage'],'gap_not_full_array_coverage')
            self.assertFalse((case.parent.parent.parent/'diagnostics').exists())

    def test_runner_loads_actual_code_root_and_checks_frozen_sha(self):
        frozen=runner.load_frozen_core(SimpleNamespace(code_root=HERE))
        self.assertEqual(Path(frozen.__file__),HERE/'ms_unified_novlm_v1.py')
        directory=self.root/'changed_core';write(directory/'ms_unified_novlm_v1.py',b'raise AssertionError("must not load")')
        with self.assertRaisesRegex(ValueError,'frozen original core SHA'):runner.load_frozen_core(SimpleNamespace(code_root=directory))
        self.assertNotIn('ms_unified_novlm_next',Path(runner.__file__).read_text())

    def test_same_sha_partial_publish_only_missing_no_batch_resend(self):
        saved=self.build(n=3);m=saved['plan']['manifest'][0];write(m['path'],Path(saved['local_files'][m['path']]).read_bytes())
        before=Path(m['path']).stat();actual=self.access.publish_file
        with patch.object(self.access,'batch_exchange',side_effect=AssertionError('partial must not resend batch')),patch.object(self.access,'publish_file',wraps=actual) as writes:
            result=self.access.publish_prepared(saved)
        self.assertEqual(result['state'],'publication_only_recovered_verified')
        self.assertEqual(len(writes.call_args_list),len(saved['plan']['manifest'])-1)
        self.assertNotIn(m['path'],[c.args[1] for c in writes.call_args_list]);self.assertEqual(before.st_ino,Path(m['path']).stat().st_ino)

    def prepared_lane(self,saved):
        local=Path(saved['local_files'][str(Path(saved['plan']['target'])/'batch.tar')]).parent
        lane=runner.Lane.__new__(runner.Lane);lane.access=self.access;lane.receipts=saved['plan']['commit']['receipts'];lane.run_id=saved['plan']['commit']['run_id']
        lane.work=local.parent;lane.inputs=[];lane.batch_no=0;lane.stage=lambda:(_ for _ in ()).throw(AssertionError('prepared stage/GPU query forbidden'))
        lane.batch_paths=lambda:(local,Path(saved['plan']['target']));lane.report=lambda **kwargs:None
        lane.progress=dict(published=0,published_generated_with_events=0,profile_seconds={})
        return lane,local

    def test_runner_durable_prepared_before_stage_with_no_core_or_models(self):
        saved=self.build(n=3,diagnostic=True);lane,local=self.prepared_lane(saved)
        archive=Path(saved['local_files'][str(Path(saved['plan']['target'])/'batch.tar')]).read_bytes();craw=transport.encoded(saved['plan']['commit'])
        real_open=tarfile.open
        def read_only_open(*args,**kwargs):
            mode=kwargs.get('mode',args[1] if len(args)>1 else 'r')
            if mode not in ('r','r:'):raise AssertionError('prepared must not repack')
            return real_open(*args,**kwargs)
        with patch.object(runner.tarfile,'open',side_effect=read_only_open):lane.publish(False)
        self.assertFalse(hasattr(lane,'core'));self.assertFalse(hasattr(lane,'models'))
        self.assertEqual((Path(saved['plan']['target'])/'batch.tar').read_bytes(),archive)
        self.assertEqual(self.commit_path(saved).read_bytes(),craw);self.assertEqual(lane.progress['published'],3)

    def test_runner_prepared_existing_commit_inspected_without_local_source_or_gpu(self):
        saved=self.build(n=3);self.access.publish_prepared(saved);lane,local=self.prepared_lane(saved)
        Path(saved['local_files'][saved['plan']['manifest'][0]['path']]).unlink()
        with patch.object(self.access,'publish_file',side_effect=AssertionError('existing commit no write')):lane.publish(False)
        self.assertEqual(lane.progress['published'],3)

    def test_standalone_publication_only_recovery_without_lane_gate(self):
        saved=self.build(n=3);local=Path(saved['local_files'][str(Path(saved['plan']['target'])/'batch.tar')]).parent
        loaded,result=runner.recover_publication_only(self.access,local)
        self.assertEqual(loaded,saved);self.assertEqual(result['verification']['state'],'committed_verified')

    def test_regular_pax_sparse_is_rejected_before_commit(self):
        saved=self.build(n=1);plan=saved['plan'];archive=Path(saved['local_files'][str(Path(plan['target'])/'batch.tar')])
        with tarfile.open(archive,'w',format=tarfile.PAX_FORMAT) as pack:
            for j,m in enumerate(plan['manifest']):
                if '/cases/' not in m['path']:continue
                p=Path(saved['local_files'][m['path']]);info=pack.gettarinfo(p,arcname=str(p.relative_to(archive.parent)))
                if j==1:info.pax_headers={'GNU.sparse.map':'0,'+str(info.size),'GNU.sparse.size':str(info.size)}
                with p.open('rb') as stream:pack.addfile(info,stream)
        with tarfile.open(archive,'r:') as pack:self.assertTrue(any(m.isfile() and m.issparse() for m in pack))
        spec=next(m for m in plan['manifest'] if m['path'].endswith('/batch.tar'));spec.update(bytes=archive.stat().st_size,sha256=publication.sha(archive));rebind(plan)
        with self.assertRaisesRegex(ValueError,'archive member path/type'):self.send(saved)
        self.assertFalse(self.commit_path(saved).exists())

    def test_normal_production_pax_headers_are_supported(self):
        saved=self.build(n=3);archive=Path(saved['local_files'][str(Path(saved['plan']['target'])/'batch.tar')])
        with tarfile.open(archive,'r:') as pack:self.assertTrue(any(m.pax_headers for m in pack))
        self.assertEqual(self.access.publish_prepared(saved)['verification']['state'],'committed_verified')

    def test_claim_frozen_index_and_complete_row_before_any_write(self):
        row=self.rows[0];path=transport.P/'claims'/row['split']/(row['id']+'.json')
        record=dict(edition=transport.P.name,worker=self.worker,gpu=0,pid=os.getpid(),assignment_sha256=digest(self.mraw),assignment_index=0,source_row=row,split=row['split'],id=row['id'],automatic_retry=False)
        for kind in ('wrong_index','bool_index','missing_index','wrong_row','missing_row'):
            bad=copy.deepcopy(record)
            if kind=='wrong_index':bad['assignment_index']=4
            elif kind=='bool_index':bad['assignment_index']=False
            elif kind=='missing_index':bad.pop('assignment_index')
            elif kind=='wrong_row':bad['source_row']['source_ordinal']=2
            else:bad.pop('source_row')
            q=dict(worker=self.worker,op='claim',path=str(path),record=bad,**self.access.binding)
            with self.assertRaisesRegex(ValueError,'claim record binding'):transport.dispatch(q,io.BytesIO(),io.BytesIO())
            self.assertFalse((transport.P/'claims').exists())
        q=dict(worker=self.worker,op='claim',path=str(path),record=record,**self.access.binding)
        self.assertTrue(transport.dispatch(q,io.BytesIO(),io.BytesIO())['created']);before=path.read_bytes()
        self.assertFalse(transport.dispatch(q,io.BytesIO(),io.BytesIO())['created']);self.assertEqual(path.read_bytes(),before)

    def refresh_synthetic_plan(self,saved):
        plan=saved['plan'];by_path={m['path']:m for m in plan['manifest']}
        for r in plan['commit']['receipts']:
            for ref in r.get('diagnostic_publications',[]):
                p=Path(saved['local_files'][ref['path']]);ref.update(bytes=p.stat().st_size,sha256=publication.sha(p));by_path[ref['path']].update(ref)
            if r['status']=='existing_claim_not_reassigned':continue
            p=str(Path(plan['target'])/'cases'/r['source_row']['split']/r['source_row']['id']/'receipt.json')
            write(saved['local_files'][p],transport.encoded(r));by_path[p].update(bytes=Path(saved['local_files'][p]).stat().st_size,sha256=publication.sha(saved['local_files'][p]))
        archive=Path(saved['local_files'][str(Path(plan['target'])/'batch.tar')])
        with tarfile.open(archive,'w') as pack:
            for m in plan['manifest']:
                if '/cases/' in m['path']:
                    p=Path(saved['local_files'][m['path']]);pack.add(p,arcname=str(p.relative_to(archive.parent)),recursive=False)
        by_path[str(Path(plan['target'])/'batch.tar')].update(bytes=archive.stat().st_size,sha256=publication.sha(archive));rebind(plan)

    def test_diagnostic_rebound_sha_cannot_hide_semantic_contract_tampering(self):
        changes={'audio_sha256':'c'*64,'method':'other','original_rejection_policy_unchanged':False,'original_rejection_policy':'other',
                 'assignment_index':4,'model_sha256':'c'*64,'algorithm_sha256':'c'*64,'audit_sha256':'c'*64,'device':'cpu','float16':True,'dbn':True,
                 'array_coverage':'full_unbounded','actual_audio_metadata':{'duration_seconds':3.0},'source_row':self.rows[4]}
        for i,(key,value) in enumerate(changes.items()):
            saved=self.build(n=1,diagnostic=True,batch_no=i);r=saved['plan']['commit']['receipts'][0]
            ref=next(ref for ref in r['diagnostic_publications'] if ref['path'].endswith('.json'));p=saved['local_files'][ref['path']]
            evidence=transport.strict(Path(p).read_bytes());evidence[key]=value;write(p,transport.encoded(evidence));self.refresh_synthetic_plan(saved)
            with self.assertRaises(ValueError,msg=key):self.send(saved)
            self.assertFalse(self.commit_path(saved).exists())

    def test_prepare_keeps_npz_json_ref_order_and_written_receipt_bytes(self):
        saved=self.build(n=1,diagnostic=True);local=Path(saved['local_files'][str(Path(saved['plan']['target'])/'batch.tar')]).parent
        r=saved['plan']['commit']['receipts'][0];before=copy.deepcopy(r);receipt=local/'cases'/r['source_row']['split']/r['source_row']['id']/'receipt.json';raw=receipt.read_bytes()
        self.assertTrue(r['diagnostic_publications'][0]['path'].endswith('.npz'));self.assertTrue(r['diagnostic_publications'][1]['path'].endswith('.json'))
        again=publication.prepare(local,Path(saved['plan']['target']),saved['plan']['commit'])
        self.assertEqual(r,before);self.assertEqual(receipt.read_bytes(),raw);self.assertEqual(again,saved)

    def test_prepare_rejects_ref_mismatch_instead_of_mutating_written_receipt(self):
        saved=self.build(n=1,diagnostic=True);local=Path(saved['local_files'][str(Path(saved['plan']['target'])/'batch.tar')]).parent
        r=saved['plan']['commit']['receipts'][0];r['diagnostic_publications'][0]['sha256']='0'*64
        receipt=local/'cases'/r['source_row']['split']/r['source_row']['id']/'receipt.json';before=receipt.read_bytes()
        with self.assertRaises(ValueError):publication.prepare(local,Path(saved['plan']['target']),saved['plan']['commit'])
        self.assertEqual(receipt.read_bytes(),before);self.assertEqual(r['diagnostic_publications'][0]['sha256'],'0'*64)

    def test_runner_original_core_failure_receipt_binds_audio_and_gap(self):
        from collections import deque
        from concurrent.futures import Future
        frozen=runner.load_frozen_core(SimpleNamespace(code_root=HERE))
        row=self.rows[0];metadata=dict(duration_seconds=2.0,sample_rate=16000,channels=1,frames=32000)
        probe=lambda path:dict(streams=[dict(width=2,height=2,duration='2.0',time_base='1/2',nb_frames='4')],
                              frames=[dict(best_effort_timestamp=str(i),best_effort_timestamp_time=str(i/2)) for i in range(4)])
        decoder=lambda *args:iter([np.zeros((2,2,3),np.uint8) for _ in range(4)])
        for i,values in enumerate((np.array([-.1,np.nan],np.float32),np.array([-.1],object))):
            run_id='g0-100-12345678';local=self.root/('runner_failure%d'%i)/(run_id+'-batch000')
            target=Path(self.ticket['output_root'])/'batches'/'gpu0'/(run_id+'-batch%03d'%i)
            copied={'video_path':Path('/synthetic.mp4'),'audio_path':Path('/synthetic.wav')};hashes={'video_path':'a'*64,'audio_path':'b'*64}
            future=Future();future.set_result((copied,hashes))
            lane=runner.Lane.__new__(runner.Lane);lane.args=SimpleNamespace(worker=self.worker,gpu=0);lane.access=self.access
            lane.work=local.parent;lane.pending=deque([(0,row,future,{},runner.time.perf_counter(),0)])
            lane.stage=lambda:None;lane.report=lambda **kwargs:None;lane.batch_paths=lambda:(local,target)
            lane.core=SimpleNamespace(probe_video=probe,v2=frozen.v2,write_once=frozen.write_once,
                extract_and_rule=lambda *args,**kwargs:frozen.extract_and_rule(*args,decoder=decoder,**kwargs))
            model_calls=[]
            def underlying(path):
                model_calls.append(path);return (values,np.array([],np.float64))
            tap=beat.AuditTap(underlying,lambda path:metadata)
            lane.models=(lambda frame:([],[],[]),tap);lane.reserved=0;lane.output_used=0;lane.inputs=[];lane.receipts=[]
            lane.progress=dict(failed=0,unique_processed=0,profile_seconds={})
            lane.process();r=lane.receipts[0]
            self.assertEqual(model_calls,[str(copied['audio_path'])])
            self.assertEqual(r['status'],'failed_sealed_no_retry');self.assertEqual(r['error'],'Invalid actual BeatThis beat/downbeat output')
            self.assertEqual(r['actual_audio_sha256'],hashes['audio_path']);self.assertIs(r['actual_audio_metadata'],metadata)
            case=local/'cases'/row['split']/row['id'];self.assertEqual(set(p.name for p in case.iterdir()),{'receipt.json'})
            self.assertEqual(transport.strict((case/'receipt.json').read_bytes()),r)
            if i==0:
                self.assertTrue(r['diagnostic_publications'][0]['path'].endswith('.npz'));self.assertNotIn('evidence_gap',r)
            else:
                self.assertNotIn('diagnostic_publications',r);self.assertEqual(r['evidence_gap']['array_coverage'],'gap_not_full_array_coverage')
            with tarfile.open(local/'batch.tar','x') as pack:pack.add(case/'receipt.json',arcname=str((case/'receipt.json').relative_to(local)),recursive=False)
            c=dict(schema='ms_unified_novlm_batch_commit_v1',edition=transport.P.name,worker=self.worker,gpu=0,run_id=run_id,
                assignment_sha256=digest(self.mraw),code_signatures={Path(ref['path']).name:ref['sha256'] for ref in self.code_refs},
                next_version_code_signatures={'beatthis_rejection_next_v2.py':publication.sha(HERE/'beatthis_rejection_next_v2.py')},
                assignment_indices=[0],receipts=[r],quality_approved=False,vlm_called=False,unique_sources_attempted=1,unique_sources_with_action_events=0)
            saved=publication.prepare(local,target,c)
            self.assertEqual(self.access.publish_prepared(saved)['verification']['state'],'committed_verified')

    def test_diagnostic_dtype_shape_and_replayed_predicates_strong_readback(self):
        for i,kind in enumerate(('dtype','shape','predicates','receipt_audio_sha')):
            saved=self.build(n=1,diagnostic=True,batch_no=i);r=saved['plan']['commit']['receipts'][0]
            ref=next(ref for ref in r['diagnostic_publications'] if ref['path'].endswith('.json'));p=saved['local_files'][ref['path']]
            evidence=transport.strict(Path(p).read_bytes())
            if kind=='dtype':evidence['raw_arrays']['beats']['dtype']='<f8'
            elif kind=='shape':evidence['raw_arrays']['beats']['shape']=[999]
            elif kind=='predicates':evidence['failed_predicates']=r['failed_predicates']=['invented.failure']
            else:r['actual_audio_sha256']='c'*64
            write(p,transport.encoded(evidence));self.refresh_synthetic_plan(saved)
            with self.assertRaises(ValueError,msg=kind):self.send(saved)
            self.assertFalse(self.commit_path(saved).exists())

    def remote_access(self, child_code=None):
        remote=transport.Access(self.worker,relay_host=transport.RELAY_HOST);remote.binding=self.access.binding
        code=child_code or transport.RELAY_CODE.replace("try:\n    if sys.argv[1]",'P=Path('+repr(str(transport.P))+'); PREVIOUS_P=Path('+repr(str(transport.PREVIOUS_P))+')\ntry:\n    if sys.argv[1]')
        def local_command(op,path,**fields):
            import base64
            q=dict(worker=remote.worker,op=op,path=str(path),**remote.binding,**fields)
            argument='--request-stdin' if 'plan' in fields else base64.b64encode(transport.encoded(q)).decode()
            return [sys.executable,'-c',code,argument]
        return remote,patch.object(remote,'request',side_effect=local_command)

    def test_remote_real_20_success_and_mixed_plan_header_below_single_argv_bound(self):
        import shlex,base64
        observed=[]
        for i,mixed in enumerate((False,True)):
            saved=self.build(n=20,mixed=mixed,batch_no=i);plan=saved['plan'];q=self.q(plan)
            remote=transport.Access(self.worker,relay_host=transport.RELAY_HOST);remote.binding=self.access.binding
            command=remote.request('publish_batch',plan['target'],plan=plan)
            legacy='python3 -c '+shlex.quote(transport.RELAY_CODE)+' '+shlex.quote(base64.b64encode(transport.encoded(q)).decode())
            item=dict(mixed=mixed,receipts=len(plan['commit']['receipts']),publications=len(plan['manifest']),plan_json_bytes=len(transport.encoded(plan)),old_command_bytes=len(legacy.encode()),new_command_bytes=len(command[-1].encode()),stdin_header_bytes=len(transport.request_header(q)))
            observed.append(item);self.assertEqual(item['receipts'],20);self.assertLess(item['new_command_bytes'],131072)
            self.assertTrue(command[-1].endswith('--request-stdin'));self.assertNotIn(base64.b64encode(transport.encoded(q)).decode(),command[-1])
            header=transport.request_header(q);self.assertEqual(int.from_bytes(header[:4],'big'),len(header)-4);self.assertEqual(transport.strict(header[4:]),q)
            actual,local_patch=self.remote_access()
            with local_patch:
                result=actual.publish_prepared(saved)
            self.assertEqual(result['verification']['state'],'committed_verified')
            self.assertEqual(self.commit_path(saved).read_bytes(),transport.encoded(plan['commit']))
        self.assertGreater(observed[0]['old_command_bytes'],131072);self.assertEqual(observed[0]['publications'],101)
        type(self).remote_command_measurements=observed

    def test_remote_request_header_size_failclosed_before_child(self):
        with patch.object(transport,'BATCH_METADATA_CAP',100):
            with self.assertRaisesRegex(ValueError,'bounded request header'):transport.request_header({'oversized':'x'*101})
        child="import sys;sys.stdout.write('{}');sys.stdout.flush()"
        remote,p=self.remote_access(child)
        with p,patch.object(transport,'BATCH_METADATA_CAP',100):
            with self.assertRaisesRegex(ValueError,'bounded request header'):remote.batch_exchange({'target':'x','oversized':'x'*101},io.BytesIO())

    def test_remote_early_rejection_drains_bounded_stderr_without_deadlock(self):
        saved=self.build(n=20);remote,p=self.remote_access("import sys;sys.stderr.write('explicit early rejection');sys.stderr.flush();sys.exit(2)")
        sequence=publication.FileSequence(saved['plan']['manifest'],saved['local_files'])
        try:
            with p,patch.object(transport,'RPC_TIMEOUT',2):
                with self.assertRaisesRegex(RuntimeError,'explicit early rejection'):remote.batch_exchange(saved['plan'],sequence)
        finally:sequence.close()
        self.assertFalse(self.commit_path(saved).exists())

    def test_remote_whole_exchange_stalled_stdin_stdout_and_stderr_deadline(self):
        saved=self.build(n=20)
        for code in ('import time;time.sleep(10)',"import sys,time;sys.stdin.buffer.read();time.sleep(10)"):
            remote,p=self.remote_access(code);sequence=publication.FileSequence(saved['plan']['manifest'],saved['local_files'])
            try:
                with p,patch.object(transport,'RPC_TIMEOUT',.2):
                    with self.assertRaisesRegex(TimeoutError,'whole exchange deadline'):remote.batch_exchange(saved['plan'],sequence)
            finally:sequence.close()
        self.assertFalse(self.commit_path(saved).exists())

    def test_remote_bounded_ack_and_stderr_caps_reject_live_child(self):
        saved=self.build(n=3)
        for channel,cap in (('stdout',transport.BATCH_METADATA_CAP),('stderr',transport.ERROR_CAP)):
            code='import os,time;os.write('+str(1 if channel=='stdout' else 2)+",b'x'*"+str(cap+1)+');time.sleep(10)'
            remote,p=self.remote_access(code);sequence=publication.FileSequence(saved['plan']['manifest'],saved['local_files'])
            try:
                with p,patch.object(transport,'RPC_TIMEOUT',2):
                    with self.assertRaisesRegex(ValueError,'bounded response cap'):remote.batch_exchange(saved['plan'],sequence)
            finally:sequence.close()
        self.assertFalse(self.commit_path(saved).exists())

    def test_remote_disconnected_or_malformed_ack_never_claims_commit(self):
        saved=self.build(n=3)
        for code in ('import sys;sys.exit(0)',"import sys;sys.stdin.buffer.read();sys.stdout.write('not-json')"):
            remote,p=self.remote_access(code);sequence=publication.FileSequence(saved['plan']['manifest'],saved['local_files'])
            try:
                with p,patch.object(transport,'RPC_TIMEOUT',2):
                    with self.assertRaises((ValueError,RuntimeError)):remote.batch_exchange(saved['plan'],sequence)
            finally:sequence.close()
        self.assertFalse(self.commit_path(saved).exists())

    def test_transparent_wrapper_returns_original_container_without_copy_or_reorder(self):
        actual=(np.array([.1,.2],np.float32),np.array([.1],np.float64));metadata={'duration_seconds':2.0};calls=[]
        def model(path):calls.append(path);return actual
        tap=beat.AuditTap(model,lambda path:metadata);tap.configure(self.rows[0],0,{'audio_path':'b'*64},self.root/'transparent'/'cases'/'test'/'case00000')
        self.assertIs(tap.audio_info('/synthetic.wav'),metadata);observed=tap('/synthetic.wav')
        self.assertIs(observed,actual);self.assertIs(observed[0],actual[0]);self.assertIs(observed[1],actual[1]);self.assertEqual(calls,['/synthetic.wav'])
        self.assertEqual(tap.result,{});self.assertIsNone(tap.gap)

if __name__=='__main__':
    unittest.main(verbosity=2)
