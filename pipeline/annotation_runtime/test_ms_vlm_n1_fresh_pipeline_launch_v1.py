"""CPU factory validation; no real holders, models, or production source claims."""
import copy
from pathlib import Path
from unittest.mock import patch
import unittest
import test_ms_vlm_batch_safe_dispatch_v1 as fixtures
import ms_vlm_n1_fresh_pipeline_launch_v1 as m
fixtures.v=m.v

class FactoryTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.DispatchTests('test_success_retains_history_two_holders_shared_safe')
        self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.stack=self.f.stack;v=m.v;self.v=v
        self.stack.enter_context(patch.object(m,'REGISTRY',self.f.rt/'factory_registry.json'))
        self.script=self.f.rt/'factory.py';self.script.write_bytes(b'frozen factory')
        self.stack.enter_context(patch.object(m,'SELF',self.script))
        self.phase=v.PREFIX+'factory';self.cfg={k:copy.deepcopy(self.f.previous[k]) for k in 'schema phase node gpus config_path cwd env ffmpeg_path holders holder_restored_refs resource_safe_refs input_count targetcount selected_source_ids modelref new_stage_authorized release_authorized out queue plan producer resource_manager_ref argv'.split()}
        self.cfg.update(schema=v.SCHEMA,phase=self.phase,node='N1',gpus=[0,1],config_path=str(self.f.rt/(self.phase+'_config.json')),cwd='/tmp',out=str(self.f.root/self.phase),queue=str(self.f.root/'queues'/self.phase),holders=[],holder_restored_refs=[],resource_safe_refs=[],targetcount=4,input_count=4,selected_source_ids=[r['id'] for r in self.f.rows],plan=v.ref(self.f.plan),producer=v.ref(self.f.v4),resource_manager_ref=v.ref(self.f.v4),batch_sources=16)
        self.cfg['env']=v.expected_env(self.cfg);self.cfg['argv']=[v.PYTHON,'-B',str(v.SELF),'worker','--config',self.cfg['config_path']]
        self.cfg['source_sha256']={p:v.base.sha(p) for p in v.dependencies()|{r['path'] for r in v.source_refs(self.cfg,self.f.rows)}}
        self.template=self.f.rt/'template.json';self.f.put(self.template,self.cfg)
        self.seed=dict(node='N1',gpus=[0,1],phase='ms_vlm_fresh_seed_n1g01_cpu',producer_ref=v.ref(m.fresh.SELF))
        self.seedpath=self.f.rt/'seedjob.json';self.f.put(self.seedpath,self.seed)
        self.cpu=dict(tests_run=73,source_unchanged=True,failures=0,errors=0,skips=0,before_sha256={'ms_fullframe_vlm_postprocess_batch_v8.py':v.ref(v.SELF)['sha256'],'ms_vlm_fresh_empty_pair_handoff_v1.py':v.ref(m.fresh.SELF)['sha256']})
        self.cpupath=self.f.rt/'cpu.json';self.f.put(self.cpupath,self.cpu)
        self.jobpath=self.f.rt/(self.phase+'_launch_job.json');self.job=dict(schema='ms_vlm_n1_fresh_pipeline_launch_job_v1',producer_ref=v.ref(m.SELF),template_ref=v.ref(self.template),cpu_ref=v.ref(self.cpupath),provision_job_ref=v.ref(self.seedpath),task_handle='CpuFactory')
        self.Popen=self.stack.enter_context(patch.object(m.subprocess,'Popen',side_effect=self.spawn_seed))
        self.stack.enter_context(patch.object(v.subprocess,'check_output',return_value=v.HOSTS['N1']))
        self.freeze()

    def freeze(self):
        self.job['template_ref']=self.f.put(self.template,self.cfg)
        self.jobref=self.f.put(self.jobpath,self.job)
        self.entry=dict(phase=self.phase,node='N1',gpus=[0,1],expected_running=False,task_handle=self.job['task_handle'],launcher_job_ref=self.jobref,plan_ref=self.cfg['plan'],selected_source_ids=self.cfg['selected_source_ids'],config_path=self.cfg['config_path'])
        self.f.put(m.REGISTRY,dict(schema='ms_vlm_postprocess_jobs_registry_v1',jobs=[self.entry]))

    def spawn_seed(self,*args,**kwargs):
        q=self.f.root/'queues'/self.seed['phase'];self.f.put(q/'resources_safe.json',dict(phase=self.seed['phase'],all_holders_safe=True,locks_released=True,leases_restored=True,managed_parent_wait=True,unknown_jobs_not_signaled=True))
        for holder in self.f.holders:self.f.put(q/('holder_restored_gpu%d.json'%holder['gpu']),dict(gpu=holder['gpu'],uuid=holder['uuid'],holder_identity=holder,created_from_verified_empty_pair=True,previous_task_holder_restored=False))
        class Seed:
            def poll(self):return None
        return Seed()

    def launch(self):m.launch(str(self.jobpath),self.jobref['sha256'])

    def blocked(self):
        with self.assertRaises((ValueError,FileExistsError,RuntimeError,FileNotFoundError)):self.launch()
        self.Popen.assert_not_called();self.f.exec.assert_not_called()
        self.assertFalse(any(self.f.root.rglob('*.claim.json')))

    def test_success_uses_actual_seed_refs_then_native_exec(self):
        self.launch();v=self.v;c=v.load(self.cfg['config_path']);self.assertEqual(len(c['holders']),2);self.assertEqual(c['resource_safe_refs'][0],c['resource_safe_refs'][1])
        self.assertEqual(c['holders'],self.f.holders);self.f.exec.assert_called_once();self.Popen.assert_called_once();self.assertTrue(v.load(m.REGISTRY)['jobs'][0]['expected_running'])
        self.assertFalse(any(self.f.root.rglob('*.claim.json')))

    def test_unauthorized_node_rejected_before_seed(self):self.cfg['node']='N2';self.freeze();self.blocked()
    def test_fabricated_holder_rejected_before_seed(self):self.cfg['holders']=self.f.holders;self.freeze();self.blocked()
    def test_missing_new_stage_authorization(self):self.cfg['new_stage_authorized']=False;self.freeze();self.blocked()
    def test_wrong_cpu_source_binding(self):
        self.cpu['before_sha256']['ms_fullframe_vlm_postprocess_batch_v8.py']='0'*64;self.job['cpu_ref']=self.f.put(self.cpupath,self.cpu);self.freeze();self.blocked()
    def test_existing_config_is_never_overwritten(self):self.f.put(Path(self.cfg['config_path']),{'foreign':True});self.blocked()
    def test_duplicate_registered_owner(self):
        r=self.v.load(m.REGISTRY);r['jobs'].append(dict(node='N1',gpus=[0,1],expected_running=True,phase='foreign'));self.f.put(m.REGISTRY,r);self.blocked()
    def test_missing_pending_descriptor(self):self.f.put(m.REGISTRY,dict(jobs=[]));self.blocked()
    def test_foreign_pending_source_overlap(self):
        r=self.v.load(m.REGISTRY);r['jobs'].append(dict(phase='foreign',node='N2',gpus=[2,3],expected_running=False,selected_source_ids=[],config_path=str(self.f.rt/'future.json'),plan_ref=self.cfg['plan']));self.f.put(m.REGISTRY,r);self.blocked()
    def test_changed_input_ref_rejected_before_gpu_seed(self):
        Path(self.f.rows[0]['rule_draft_ref']['path']).write_bytes(b'{}');self.blocked()
    def test_changed_job_ref_rejected_before_gpu_seed(self):
        self.jobref['sha256']='0'*64;self.blocked()

if __name__=='__main__':unittest.main(verbosity=2)
