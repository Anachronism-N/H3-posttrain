"""CPU-only temporary fixtures; no real manager import, GPU, SSH or deployment."""
import ast
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

RT=Path(__file__).resolve().parent
DISPATCH=RT/'ms_native_rebalance_dispatch_v1.py'
MANAGER=RT/'ms_unified_native_continuous_manager_v2.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()),sha256=sha(path))


def checked(reference):
    path=Path(reference['path'])
    if not path.is_absolute() or str(path.resolve(strict=True))!=str(path):
        raise PermissionError('Stable absolute reference required')
    if sha(path)!=reference['sha256']:
        raise PermissionError('Frozen reference changed')
    return json.loads(path.read_bytes())


class Launched(Exception):
    pass


class RebalanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.rt=self.root/'runtime';self.rt.mkdir()
        self.lockroot=self.root/'local_tmp';self.lockroot.mkdir()
        self.manager=types.ModuleType('ms_unified_native_continuous_manager_v2')
        self.manager.checked=checked;self.manager.ref=ref
        self.manager.HOSTS={'N2':'fixture-node'};self.manager.ALLOWED_GPUS={'N2':(2,3)}
        self.manager.PYTHON='/fixture/python';self.manager.BASE_SHA='fixture-base-sha'
        self.manager.SELF=self.rt/'manager.py';self.manager.CORE=self.rt/'core.py'
        self.manager.SELF.write_text('frozen manager fixture')
        self.manager.CORE.write_text('frozen core fixture')
        self.manager.dependencies=lambda:{str(self.manager.SELF),str(self.manager.CORE)}
        self.manager.base=types.SimpleNamespace(sha=sha,identity=Mock(return_value={'pid':9}),
            IDENTITY_KEYS=('pid',),check_holder=Mock(),uuid_for=Mock(return_value='fixture-uuid'),apps=Mock(return_value=[]))
        self.manager.h=types.SimpleNamespace(lock_snapshot=Mock(return_value={'held':False}))
        self.manager.validate=Mock();self.manager.check_model=Mock()
        spec=importlib.util.spec_from_file_location('rebalance_fixture_dispatch',DISPATCH)
        self.dispatch=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'ms_unified_native_continuous_manager_v2':self.manager}):
            spec.loader.exec_module(self.dispatch)
        self.dispatch.RT=self.rt;self.dispatch.B=self.root
        self.rows=[dict(id=i,split='val',source_ordinal=n,video_path='/fixture/'+i+'.mp4',audio_path='/fixture/'+i+'.wav')
                   for n,i in enumerate(['prefix','suffix_a','suffix_b'])]
        old='ms_unified_native_v1_retired';out=self.root/old;q=self.root/'queues'/old
        self.oldcfgpath=self.rt/(old+'_config.json');self.oldticketpath=self.rt/(old+'_ticket.json')
        self.stoppath=out/old/'STOP';self.save(self.stoppath,dict(no_claim_reset=True,no_process_signal=True))
        self.claims=out/old/'claims'/'sources';self.claims.mkdir(parents=True)
        self.save(self.claims/'prefix.json',dict(source_row=self.rows[0],ticket_id=old))
        self.oldticket=dict(edition=old,ticket_id=old,output_root=str(out),stop_file=str(self.stoppath),sources=self.rows,batch_sources=20)
        self.save(self.oldticketpath,self.oldticket)
        self.oldcfg=dict(phase=old,config_path=str(self.oldcfgpath),ticket=ref(self.oldticketpath),out=str(out),queue=str(q),
                         selected_source_ids=[r['id'] for r in self.rows],targetcount=3,input_count=3)
        self.save(self.oldcfgpath,self.oldcfg)
        self.summarypath=out/'summary.json';self.safepath=q/'resources_safe.json';self.exitpath=q/'worker_exit.json'
        self.summary=dict(status='drained_not_complete',processed=1,selected_source_ids=[r['id'] for r in self.rows],
                          unprocessed_source_ids=['suffix_a','suffix_b'],all_outcomes=[dict(id='prefix')],
                          allprefetch_drained=True,uncommitted_prefetch=0,vlm_called=False,quality_approved=False,edition=old,ticket_id=old)
        self.safe=dict(all_holders_safe=True,locks_released=True,leases_restored=True,managed_parent_wait=True,unknown_jobs_not_signaled=True)
        self.save(self.summarypath,self.summary);self.save(self.safepath,self.safe);self.save(self.exitpath,dict(exit_code=0,phase=old))
        self.releasepath=self.root/'release'/'release_receipt.json'
        self.release=dict(schema='ms_native_unclaimed_tail_release_v1',original_task_retired_no_resume=True,no_claim_reset=True,
            allprefetch_drained=True,claimed_suffix_intersection=0,uncommitted_prefetch=0,original_claim_count=1,
            original_processed_source_count=1,unclaimed_source_count=2,unclaimed_ids=['suffix_a','suffix_b'],
            original_config_ref=ref(self.oldcfgpath),original_ticket_ref=ref(self.oldticketpath),original_summary_ref=ref(self.summarypath),
            original_resources_safe_ref=ref(self.safepath),original_worker_exit_ref=ref(self.exitpath),persistent_STOP_ref=ref(self.stoppath))
        self.save(self.releasepath,self.release)
        self.plan=self.rt/'rebalance_plan.jsonl';self.plan.write_text(json.dumps(self.rows[1])+'\n')
        previous='ms_unified_native_v1_predecessor';pq=self.root/'queues'/previous;po=self.root/previous
        self.prevticketpath=self.rt/(previous+'_ticket.json')
        self.prevticket=dict(edition=previous,ticket_id=previous,output_root=str(po),sources=[dict(id='previous_source')],
            source_budget=dict(limit=400,consumed=400),batch_sources=20,gpu_index=2,local_cache_root='/tmp',max_cache_bytes=8*2**30,
            stop_file=str(po/previous/'STOP'))
        self.save(self.prevticketpath,self.prevticket)
        self.prevpath=self.rt/(previous+'_config.json')
        self.previous=dict(phase=previous,config_path=str(self.prevpath),out=str(po),queue=str(pq),ticket=ref(self.prevticketpath),
            targetcount=1,input_count=1,selected_source_ids=['previous_source'],modelref=ref(self.manager.CORE),
            node='N2',gpus=[2],release_authorized=True,env={'CUDA_VISIBLE_DEVICES':'2'},cwd=str(self.rt))
        self.save(self.prevpath,self.previous);self.save(pq/'resources_safe.json',self.safe)
        self.save(pq/'worker_exit.json',dict(exit_code=0))
        self.prevsummarypath=po/'summary.json'
        self.prevsummary=dict(status='completed',processed=1,allprefetch_drained=True,uncommitted_prefetch=0,
                              selected_source_ids=['previous_source'],vlm_called=False,quality_approved=False)
        self.save(self.prevsummarypath,self.prevsummary)
        self.save(pq/'holder_restored_gpu2.json',dict(holder_identity={'pid':9},uuid='fixture-uuid'))
        self.job=dict(node='N2',gpu=2,host='fixture-node',phase='ms_unified_native_v1_rebalance',previous_config=str(self.prevpath),
            plan=str(self.plan),plan_sha256=sha(self.plan),source_count=1,source_release_ref=ref(self.releasepath),
            job_path=str(self.rt/'rebalance_job.json'),dispatcher_task_handle='fixture-task')
        self.regpath=self.rt/'ms_novlm_continuous_jobs_registry_v1.json'
        self.save(self.regpath,dict(jobs=[dict(node='N2',gpu=2,config_path=str(self.prevpath)),dict(node='N2',gpu=3,config_path='untouched')]))

    def save(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,sort_keys=True)+'\n')

    def refresh_release(self):
        self.save(self.releasepath,self.release);self.job['source_release_ref']=ref(self.releasepath)

    def change_proof(self,key,path,value):
        self.save(path,value);self.release[key]=ref(path);self.refresh_release()

    def protected(self):
        paths=[self.oldcfgpath,self.oldticketpath,self.summarypath,self.safepath,self.exitpath,self.stoppath]+list(self.claims.iterdir())
        return {str(p):p.read_bytes() for p in paths}

    def run_main(self):
        self.save(Path(self.job['job_path']),self.job)
        real_path=Path
        def path(value):
            return self.lockroot if str(value)=='/tmp' else real_path(value)
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys,'argv',['dispatcher','--job',self.job['job_path']]))
            stack.enter_context(patch.object(self.dispatch,'Path',side_effect=path))
            self.flock=stack.enter_context(patch.object(self.dispatch.fcntl,'flock'))
            self.exec=stack.enter_context(patch.object(self.dispatch.os,'execv',side_effect=Launched))
            stack.enter_context(patch('builtins.print'))
            self.dispatch.main()

    def test_valid_shared_release_exact_rows_read_once_and_no_old_mutation(self):
        before=self.protected();real=Path.read_bytes;reads=[];scans=[];scan=self.dispatch.os.scandir
        def read(path):
            if path==self.plan:reads.append(path)
            return real(path)
        def scandir(path):
            scans.append(Path(path));return scan(path)
        with patch.object(Path,'read_bytes',read),patch.object(self.dispatch.os,'scandir',side_effect=scandir):
            self.assertEqual(self.dispatch.validate_release(self.job),[self.rows[1]])
        self.assertEqual(reads,[self.plan]);self.assertEqual(scans,[self.claims])
        self.plan.write_text(json.dumps(self.rows[2])+'\n');self.job['plan_sha256']=sha(self.plan)
        self.assertEqual(self.dispatch.validate_release(self.job),[self.rows[2]])
        self.assertEqual(before,self.protected())

    def test_valid_launch_budget_removed_release_bound_registry_cas_and_frozen_manager(self):
        before=self.protected()
        with self.assertRaises(Launched):self.run_main()
        cp=self.rt/(self.job['phase']+'_config.json');tp=self.rt/(self.job['phase']+'_ticket.json')
        cfg=json.loads(cp.read_text());ticket=json.loads(tp.read_text())
        self.assertNotIn('source_budget',ticket)
        self.assertEqual(ticket['source_transfer_ref'],self.job['source_release_ref'])
        self.assertEqual(ticket['sources'],[self.rows[1]])
        self.assertEqual(ticket['batch_sources'],20)
        self.assertEqual(ticket['stop_file'],str(self.root/self.job['phase']/self.job['phase']/'STOP'))
        self.assertNotEqual(ticket['stop_file'],str(self.stoppath))
        self.assertEqual(cfg['ticket'],ref(tp));self.assertEqual(cfg['source_sha256'][str(tp)],sha(tp))
        self.assertEqual(cfg['producer'],ref(self.manager.SELF));self.assertEqual(cfg['resource_manager_ref'],ref(self.manager.SELF))
        self.assertEqual(cfg['core_ref'],ref(self.manager.CORE));self.assertEqual(cfg['modelref'],self.previous['modelref'])
        self.assertEqual(cfg['env'],self.previous['env']);self.assertEqual(cfg['gpus'],[2])
        self.manager.validate.assert_called_once_with(cfg);self.manager.check_model.assert_called_once_with(cfg)
        self.assertEqual(self.flock.call_args_list[0].args[1],self.dispatch.fcntl.LOCK_EX|self.dispatch.fcntl.LOCK_NB)
        self.assertEqual(self.flock.call_args_list[1].args[1],self.dispatch.fcntl.LOCK_EX)
        self.manager.base.check_holder.assert_called_once();self.manager.h.lock_snapshot.assert_called_once_with(2)
        self.exec.assert_called_once_with(self.manager.PYTHON,[self.manager.PYTHON,'-B',str(self.manager.SELF),'--config',str(cp),
            '--config-sha',sha(cp),'--manager-sha',self.manager.BASE_SHA])
        registry=json.loads(self.regpath.read_text())
        self.assertEqual(registry['jobs'][0]['config_path'],str(cp));self.assertEqual(registry['jobs'][1]['config_path'],'untouched')
        self.assertEqual(before,self.protected())

    def test_bad_sha_all_release_refs_rejected(self):
        for key in ['source_release_ref']+list(k for k in self.release if k.endswith('_ref')):
            with self.subTest(key=key):
                if key=='source_release_ref':
                    original=self.job[key];self.job[key]=dict(original,sha256='0'*64)
                else:
                    original=self.release[key];self.release[key]=dict(original,sha256='0'*64);self.refresh_release()
                with self.assertRaises(PermissionError):self.dispatch.validate_release(self.job)
                if key=='source_release_ref':self.job[key]=original
                else:self.release[key]=original;self.refresh_release()

    def test_release_flags_and_counts_fail_closed(self):
        for key,value in [('schema','wrong'),('original_task_retired_no_resume',False),('no_claim_reset',False),
            ('allprefetch_drained',False),('claimed_suffix_intersection',1),('uncommitted_prefetch',1),
            ('unclaimed_source_count',1),('unclaimed_source_count',True),('original_processed_source_count',2),('original_claim_count',2),
            ('original_processed_source_count',True),('original_claim_count',True),('claimed_suffix_intersection',False),
            ('uncommitted_prefetch',False),('unclaimed_ids',['suffix_b','suffix_a'])]:
            with self.subTest(key=key,value=value):
                original=self.release[key];self.release[key]=value;self.refresh_release()
                with self.assertRaises(RuntimeError):self.dispatch.validate_release(self.job)
                self.release[key]=original;self.refresh_release()

    def test_bad_release_is_rejected_before_any_lock_write_or_resource_operation(self):
        self.release['original_task_retired_no_resume']=False;self.refresh_release()
        with patch.object(self.dispatch,'write_new') as write,patch.object(self.dispatch.fcntl,'flock') as lock:
            with self.assertRaises(RuntimeError):self.run_main()
            write.assert_not_called();lock.assert_not_called()
        self.manager.base.apps.assert_not_called();self.manager.base.check_holder.assert_not_called()
        self.manager.validate.assert_not_called();self.manager.check_model.assert_not_called()
        self.exec.assert_not_called();self.assertEqual(list(self.lockroot.iterdir()),[])

    def test_summary_prefix_suffix_selection_status_and_drain_rejected(self):
        for key,value in [('status','completed'),('selected_source_ids',['suffix_a','prefix','suffix_b']),
            ('unprocessed_source_ids',['suffix_b']),('all_outcomes',[dict(id='suffix_a')]),('processed',2),('processed',True),
            ('allprefetch_drained',False),('uncommitted_prefetch',1),('vlm_called',True),('quality_approved',True)]:
            with self.subTest(key=key):
                changed=dict(self.summary,**{key:value});self.change_proof('original_summary_ref',self.summarypath,changed)
                with self.assertRaises(RuntimeError):self.dispatch.validate_release(self.job)
        self.change_proof('original_summary_ref',self.summarypath,self.summary)

    def test_exact_original_config_ticket_and_proof_scope_binding(self):
        changed=dict(self.oldcfg,ticket=dict(self.oldcfg['ticket'],sha256='0'*64))
        self.change_proof('original_config_ref',self.oldcfgpath,changed)
        with self.assertRaises(RuntimeError):self.dispatch.validate_release(self.job)
        self.change_proof('original_config_ref',self.oldcfgpath,self.oldcfg)
        changed=dict(self.oldticket,output_root=str(self.root/'wrong'))
        self.save(self.oldticketpath,changed);self.oldcfg['ticket']=ref(self.oldticketpath)
        self.save(self.oldcfgpath,self.oldcfg)
        self.release['original_ticket_ref']=ref(self.oldticketpath);self.release['original_config_ref']=ref(self.oldcfgpath);self.refresh_release()
        with self.assertRaises(RuntimeError):self.dispatch.validate_release(self.job)

    def test_original_all_native_safe_flags_and_worker_exit(self):
        for key in self.safe:
            with self.subTest(key=key):
                self.change_proof('original_resources_safe_ref',self.safepath,dict(self.safe,**{key:False}))
                with self.assertRaises(RuntimeError):self.dispatch.validate_release(self.job)
        self.change_proof('original_resources_safe_ref',self.safepath,self.safe)
        self.change_proof('original_worker_exit_ref',self.exitpath,dict(exit_code=1,phase=self.oldcfg['phase']))
        with self.assertRaises(RuntimeError):self.dispatch.validate_release(self.job)

    def test_persistent_stop_missing_or_symlink_rejected(self):
        content=self.stoppath.read_bytes();self.stoppath.unlink()
        with self.assertRaises(RuntimeError):self.dispatch.validate_release(self.job)
        other=self.root/'other_STOP';other.write_bytes(content);self.stoppath.symlink_to(other)
        with self.assertRaises(RuntimeError):self.dispatch.validate_release(self.job)

    def test_release_ref_must_be_stable_absolute_not_symlink(self):
        original=self.job['source_release_ref']
        self.job['source_release_ref']=dict(original,path='release_receipt.json')
        with self.assertRaises(PermissionError):self.dispatch.validate_release(self.job)
        alias=self.root/'alias.json';alias.symlink_to(self.releasepath)
        self.job['source_release_ref']=dict(original,path=str(alias))
        with self.assertRaises(PermissionError):self.dispatch.validate_release(self.job)

    def test_claimed_suffix_extra_or_missing_prefix_claim_rejected(self):
        extra=self.claims/'suffix_b.json';self.save(extra,{})
        with self.assertRaisesRegex(RuntimeError,'already'):self.dispatch.validate_release(self.job)
        extra.unlink();(self.claims/'prefix.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'prefix'):self.dispatch.validate_release(self.job)
        self.save(self.claims/'prefix.json',{});self.save(self.claims/'extra.json',{})
        with self.assertRaisesRegex(RuntimeError,'prefix'):self.dispatch.validate_release(self.job)

    def test_claim_listing_bounded_to_3000_and_shallow(self):
        class Entry:
            name='prefix.json'
            def is_symlink(self):return False
            def is_file(self,follow_symlinks):return True
        seen=[]
        def entries():
            for i in range(3002):seen.append(i);yield Entry()
        with patch.object(self.dispatch.os,'scandir') as scan:
            scan.return_value.__enter__.return_value=entries()
            with self.assertRaisesRegex(RuntimeError,'3000'):self.dispatch.validate_release(self.job)
        self.assertEqual(len(seen),3001);scan.assert_called_once_with(self.claims)

    def test_plan_sha_count_duplicate_prefix_unknown_and_exact_field_mismatch(self):
        cases=[[self.rows[0]],[dict(self.rows[1],id='unknown')],[dict(self.rows[1],split='train')],
               [dict(self.rows[1],source_ordinal=999)],[dict(self.rows[1],audio_path='/wrong.wav')],
               [dict(self.rows[1],extra=True)],[self.rows[1],self.rows[1]]]
        for rows in cases:
            with self.subTest(rows=rows):
                self.plan.write_text(''.join(json.dumps(r)+'\n' for r in rows));self.job['plan_sha256']=sha(self.plan)
                self.job['source_count']=len(rows)
                with self.assertRaises(RuntimeError):self.dispatch.validate_release(self.job)
        self.plan.write_text(json.dumps(self.rows[1])+'\n');self.job['plan_sha256']='0'*64;self.job['source_count']=1
        with self.assertRaisesRegex(RuntimeError,'changed'):self.dispatch.validate_release(self.job)
        self.job['plan_sha256']=sha(self.plan);self.job['source_count']=2
        with self.assertRaises(RuntimeError):self.dispatch.validate_release(self.job)

    def test_resource_predecessor_still_requires_completed_not_retired_source_status(self):
        self.save(self.prevsummarypath,dict(self.prevsummary,status='drained_not_complete'))
        with self.assertRaisesRegex(RuntimeError,'did not finish'):self.run_main()
        self.manager.validate.assert_not_called();self.exec.assert_not_called()
        self.assertFalse((self.rt/(self.job['phase']+'_ticket.json')).exists())

    def test_holder_and_slot_ownership_gates_still_fail_closed(self):
        self.manager.h.lock_snapshot.return_value={'held':True}
        with self.assertRaisesRegex(RuntimeError,'owned'):self.run_main()
        self.exec.assert_not_called();self.manager.validate.assert_not_called()
        self.manager.h.lock_snapshot.return_value={'held':False}
        self.manager.base.check_holder.side_effect=PermissionError('wrong holder')
        with self.assertRaises(PermissionError):self.run_main()
        self.exec.assert_not_called()

    def test_registry_cas_refuses_other_dispatcher_and_does_not_exec(self):
        self.save(self.regpath,dict(jobs=[dict(node='N2',gpu=2,config_path='another_config')]))
        original=self.regpath.read_bytes()
        with self.assertRaisesRegex(RuntimeError,'handed over'):self.run_main()
        self.assertEqual(self.regpath.read_bytes(),original);self.exec.assert_not_called()

    def test_new_paths_write_once_and_never_restart(self):
        for path in [self.rt/(self.job['phase']+'_config.json'),self.rt/(self.job['phase']+'_ticket.json'),
                     self.root/self.job['phase'],self.root/'queues'/self.job['phase']]:
            with self.subTest(path=path):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text('untouched')
                with self.assertRaisesRegex(RuntimeError,'never restart'):self.run_main()
                self.assertEqual(path.read_text(),'untouched');path.unlink()
        self.exec.assert_not_called()

    def test_manager_validation_budget_model_failure_prevents_registry_and_exec(self):
        self.manager.validate.side_effect=PermissionError('160 GiB native gate denied')
        original=self.regpath.read_bytes()
        with self.assertRaises(PermissionError):self.run_main()
        self.assertEqual(self.regpath.read_bytes(),original);self.manager.check_model.assert_not_called();self.exec.assert_not_called()

    def test_model_identity_failure_prevents_registry_and_exec(self):
        self.manager.check_model.side_effect=PermissionError('frozen model identity differs')
        original=self.regpath.read_bytes()
        with self.assertRaises(PermissionError):self.run_main()
        self.assertEqual(self.regpath.read_bytes(),original);self.exec.assert_not_called()

    def test_new_ticket_config_registry_and_directory_are_fsynced(self):
        fsync=self.dispatch.os.fsync
        with patch.object(self.dispatch.os,'fsync',wraps=fsync) as synced:
            with self.assertRaises(Launched):self.run_main()
        self.assertEqual(synced.call_count,4)

    def test_actual_frozen_manager_local_budget_still_enforces_160_gib(self):
        tree=ast.parse(MANAGER.read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='local_budget')
        ns=dict(base=types.SimpleNamespace(headroom=lambda:{'headroom':159*2**30},required_budget=lambda cfg:160*2**30),
                need=lambda condition,message:None if condition else (_ for _ in ()).throw(PermissionError(message)),GIB=2**30)
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(MANAGER),'exec'),ns)
        with self.assertRaisesRegex(PermissionError,'160 GiB'):ns['local_budget']({},ticket={})
        ns['base'].headroom=lambda:{'headroom':1000*2**30}
        ns['base'].required_budget=lambda cfg:159*2**30
        with self.assertRaisesRegex(PermissionError,'160 GiB'):ns['local_budget']({},ticket={})


if __name__=='__main__':unittest.main()
