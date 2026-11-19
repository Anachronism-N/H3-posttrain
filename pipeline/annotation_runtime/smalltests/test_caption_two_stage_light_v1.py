"""Real JPEG input and write-once budget tests for the three allowed check classes."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch
import numpy as np
from PIL import Image
RT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RT))
import caption_two_stage_light_v1 as d

class Model:
    _ms_bar_model_sha256='cpu-model'
    def __init__(self,mode='ok'):self.calls=[];self.mode=mode
    def chat(self,messages,sampling,use_tqdm=False):
        self.calls.append(copy.deepcopy(messages))
        if self.mode=='error':raise RuntimeError('model failed')
        result=[]
        for msg in messages:
            header=json.loads(msg[1]['content'][0]['text']);indices=header['allowed_source_frame_indices'][:2]
            v=d.bar_module().empty_observation('not reviewed')
            v.update(events=[dict(id='e1',description='The right foot moves.',parts=['head'],frame_indices=indices)],abstain=False)
            c=NS(text='not JSON' if self.mode=='badjson' else json.dumps(v),token_ids=[1,2],finish_reason='length' if self.mode=='truncated' else 'stop')
            result.append(NS(prompt_token_ids=[9],outputs=[c]))
        return result

class LightTests(unittest.TestCase):
    def fixture(self,base,empty=False):
        out=base/'caption_two_stage_light_test';entries=[]
        args=NS(stage='run',batch_size=8,max_num_seqs=8,retry_failed=False,model='/cpu-model',visual_max_tokens=1800,hand_roi_root=None,sample_fps=4,ffmpeg='unused')
        for si,count in enumerate((3,3,2)):
            sid='source%d'%si;root=out/'sources/train'/sid;rgb=root/'rgb';rgb.mkdir(parents=True)
            times=[i*.1 for i in range(count*2)]
            row=dict(id=sid,binding=dict(id=sid),manifest_record=dict(id=sid,split='train'))
            binding=dict(source_binding=row['binding'],split='train')
            windows=[]
            for wi in range(count):
                frames=[]
                for fi in (wi*2,wi*2+1):
                    f=rgb/('%d.jpg'%fi);Image.new('RGB',(32,48),(si,fi,0)).save(f,'JPEG')
                    frames.append(dict(index=fi,time=times[fi],source_timestamp=times[fi],file=f.name))
                if empty and si==0 and wi==0:frames=frames[:1]
                windows.append(dict(window=dict(window_id='w%d'%wi,start=wi*.2,end=(wi+1)*.2),frames=frames,hand_roi_evidence={}))
            prepared=dict(binding=binding,probe=dict(times=times,source_timestamps=times),windows=windows)
            entries.append(dict(row=row,binding=binding,root=root,prepared=prepared))
        return out,args,entries
    def sampling(self,cls,schema,tokens):return NS(max_tokens=tokens,json_schema=copy.deepcopy(schema))
    def no_expensive_checks(self):
        stack=ExitStack();bar=d.bar_module()
        for module,name in ((bar,'prepare_source'),(bar,'probe_actual'),(bar,'code_identity'),(bar,'model_reference'),
                (bar,'filter_supported_observation'),(bar.batch,'sha256'),(bar.batch,'source_identity'),(bar.batch,'verify_source'),(bar.core,'verify_references')):
            stack.enter_context(patch.object(module,name,side_effect=AssertionError('Forbidden hot-path check: '+name)))
        stack.enter_context(patch.object(bar.core,'structured_sampling',side_effect=self.sampling))
        return stack
    def test_three_sources_eight_windows_zero_expensive_checks_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,args,entries=self.fixture(Path(tmp));model=Model();runtime=Mock(return_value=(model,NS))
            with self.no_expensive_checks(),patch.object(d.Image,'open',wraps=Image.open) as opening:
                r=d.generate_batch(entries,args,runtime,out,lambda a,r:True)
                self.assertEqual(opening.call_count,16)
            self.assertEqual([len(x) for x in model.calls],[8]);runtime.assert_called_once()
            self.assertEqual(r['jpeg_reads'],16);self.assertEqual(r['media_sha_reads'],0);self.assertEqual(r['media_probes'],0)
            self.assertTrue(all(x['status']=='draft_generated' and x['needs_visual_revision'] and not x['training_eligible'] for x in r['drafts']))
            self.assertTrue(all('right foot moves' in w['draft_text'] for x in r['drafts'] for w in x['windows']))
            queue=(out/'revision_queue.jsonl').read_bytes();before={p:p.read_bytes() for p in out.rglob('*.json')}
            with self.no_expensive_checks():
                again=d.generate_batch(entries,args,runtime,out,lambda a,r:True)
            self.assertEqual(len(model.calls),1);self.assertEqual(runtime.call_count,1)
            self.assertEqual(queue,(out/'revision_queue.jsonl').read_bytes());self.assertEqual(before,{p:p.read_bytes() for p in out.rglob('*.json')})
    def test_JPEG_cached_once_without_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,args,entries=self.fixture(Path(tmp));path=entries[0]['root']/'rgb/0.jpg';inputs=d.Inputs()
            with patch.object(d.Image,'open',wraps=Image.open) as opening:
                a=inputs.jpeg(path);b=inputs.jpeg(path)
            self.assertEqual(a,b);self.assertEqual(inputs.jpeg_reads,1);self.assertEqual(opening.call_count,1)
    def test_source_split_mismatch_and_unordered_timeline_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,args,entries=self.fixture(Path(tmp));model=Model()
            entries[0]['binding']['split']='test'
            with self.assertRaisesRegex(ValueError,'split'):d.generate_batch(entries,args,lambda:(model,NS),out,lambda a,r:True)
            self.assertEqual(model.calls,[])
            entries[0]['binding']['split']='train';entries[0]['prepared']['probe']['times']=[.2,.1]
            with self.no_expensive_checks():r=d.generate_batch(entries,args,lambda:(model,NS),out,lambda a,r:True)
            self.assertEqual(r['drafts'][0]['status'],'draft_needs_regeneration')
            self.assertIn('times',r['drafts'][0]['reason']);self.assertEqual([len(x) for x in model.calls],[5])
    def test_shape_and_active_points_checked_unknown_points_allowed(self):
        self.assertTrue(d.valid_points([[1.,2.],[float('nan'),float('nan')]],[.8,0]).tolist()==[True,False])
        with self.assertRaisesRegex(ValueError,'shapes'):d.valid_points([[1.,2.]],[.1,.2])
        with self.assertRaisesRegex(ValueError,'active'):d.valid_points([[float('nan'),2.]],[.8])
        with self.assertRaises(ValueError):d.valid_points([[1.,2.]],[float('nan')])
    def test_empty_window_remains_empty_not_stillness(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,args,entries=self.fixture(Path(tmp),True);model=Model()
            with self.no_expensive_checks():r=d.generate_batch(entries,args,lambda:(model,NS),out,lambda a,r:True)
            self.assertEqual([len(x) for x in model.calls],[7]);w=r['drafts'][0]['windows'][0]
            self.assertEqual(w['status'],'draft_partial');self.assertEqual(w['draft_text'],'')
    def test_failed_model_and_abnormal_outputs_recorded_no_resubmit(self):
        for mode in ('error','badjson','truncated'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                out,args,entries=self.fixture(Path(tmp));model=Model(mode)
                with self.no_expensive_checks():
                    r=d.generate_batch(entries,args,lambda:(model,NS),out,lambda a,r:True)
                    d.generate_batch(entries,args,lambda:(model,NS),out,lambda a,r:True)
                self.assertEqual(len(model.calls),1)
                self.assertTrue(all(x['status']=='draft_needs_regeneration' for x in r['drafts']))
                self.assertTrue(all(not x['training_eligible'] for x in r['drafts']))
                if mode!='error':self.assertTrue(all(w['raw_text'] for x in r['drafts'] for w in x['windows']))
    def test_reserved_orphan_and_changed_request_never_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'window';args=NS(batch_size=8);request=dict(messages=[],sampling=dict(max_tokens=1800),json_schema={})
            item=dict(root=root,request=request,binding=dict(id='s'))
            d.write_once(root/'request.json',dict(request=request,binding=item['binding']))
            d.write_once(root/'claim.json',dict(reserved_calls=1))
            runtime=Mock(side_effect=AssertionError('No reload'))
            result=d.transactions([item],runtime,args)
            runtime.assert_not_called();self.assertEqual(result[str(root)]['status'],'draft_needs_regeneration')
            item['request']['sampling']['max_tokens']=1
            with self.assertRaisesRegex(ValueError,'existing'):d.transactions([item],runtime,args)
    def test_unknown_external_input_does_not_overwrite_existing_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'result.json';d.write_once(p,{'raw':'old'});d.write_once(p,{'raw':'old'})
            with self.assertRaisesRegex(ValueError,'existing'):d.write_once(p,{'raw':'new'})
            self.assertEqual(d.load(p),{'raw':'old'})
    def test_no_resource_admission_no_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,args,entries=self.fixture(Path(tmp));model=Model()
            with self.assertRaisesRegex(ValueError,'resource'):d.generate_batch(entries,args,lambda:(model,NS),out)
            self.assertEqual(model.calls,[])

if __name__=='__main__':unittest.main()
