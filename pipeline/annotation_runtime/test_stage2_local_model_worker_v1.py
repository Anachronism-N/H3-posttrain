"""Integrated local packet->original prompt->mock model->durable outbox, no shared GPU writes."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import test_stage2_local_packet_v1 as packets
from test_ms_fullframe_vlm_postprocess_batch_v4 import FakeLLM
from stage2_local_publication_v1 import write_once,checked,LocalOutbox
import stage2_local_model_worker_v1 as w

class LocalModelTests(unittest.TestCase):
    def setUp(self):
        self.fixture=packets.PacketTests('test_local_packet_preserves_original_logical_source_without_shared_reads');self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.tmp=tempfile.TemporaryDirectory(prefix='local-worker-cpu-',dir='/tmp');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)/'local';self.root.mkdir();prepared=self.root/'prepared';shutil.copytree(self.fixture.root,prepared)
        packet=copy.deepcopy(self.fixture.packet)
        def replace(value):
            if isinstance(value,dict):return {k:replace(v) for k,v in value.items()}
            if isinstance(value,list):return [replace(v) for v in value]
            if isinstance(value,str):return value.replace(str(self.fixture.root),str(prepared))
            return value
        self.packet=replace(packet);self.packet_ref=write_once(self.root/'packets/packet.json',self.packet);self.owner=self.fixture.owner;self.gates=[]
    def run_packet(self,model):return w.run_packet(self.packet_ref,self.root,self.owner,model,lambda x:x,lambda:self.gates.append(True))
    def test_two_model_calls_write_only_local_results_then_batch_publish(self):
        model=FakeLLM();commits=self.run_packet(model);self.assertEqual([x['stage'] for x in model.calls],['visual_audit','caption_fusion']);self.assertEqual(commits[0]['new_model_calls'],2);self.assertFalse(commits[0]['training_eligible']);self.assertTrue(self.gates)
        result=checked(commits[0]['result_ref']);self.assertEqual(result['fused_caption'],'One side raises an arm in the image.');self.assertTrue(Path(result['stages'][0]['batch_response_ref']['path']).is_relative_to(self.root))
        box=LocalOutbox(self.root,Path(self.tmp.name)/'shared','N8');sealed=box.seal([['train','SyntheticSource']]);self.assertFalse(box.shared.exists());ack=box.upload(sealed);self.assertEqual(ack['models_called'],0)
    def test_visual_error_seals_one_call_and_never_fuses(self):
        model=FakeLLM(batch_error='visual_audit');commits=self.run_packet(model);self.assertEqual(len(model.calls),1);self.assertEqual(commits[0]['new_model_calls'],1);self.assertEqual(commits[0]['status'],'failed_sealed_no_retry')
    def test_fusion_error_seals_two_budget_calls(self):
        model=FakeLLM(batch_error='caption_fusion');commits=self.run_packet(model);self.assertEqual(commits[0]['new_model_calls'],2);self.assertEqual(commits[0]['status'],'failed_sealed_no_retry')
    def test_second_run_does_not_call_any_model(self):
        self.run_packet(FakeLLM());model=FakeLLM()
        with self.assertRaises(RuntimeError):self.run_packet(model)
        self.assertEqual(model.calls,[])
    def test_wrong_model_identity_response_is_sealed(self):
        def mutate(stage,i,row,candidate):
            if stage=='visual_audit':row['id']='Foreign';candidate.text=w.json.dumps(row)
        model=FakeLLM(mutate=mutate);commits=self.run_packet(model);self.assertEqual(len(model.calls),1);self.assertEqual(commits[0]['status'],'failed_sealed_no_retry')
    def test_truncated_response_is_never_success(self):
        def mutate(stage,i,row,candidate):candidate.finish_reason='length'
        commits=self.run_packet(FakeLLM(mutate=mutate));self.assertEqual(commits[0]['status'],'failed_sealed_no_retry')
    def test_wrong_owner_fails_before_model_budget(self):
        model=FakeLLM()
        with self.assertRaises(ValueError):w.run_packet(self.packet_ref,self.root,'N1-g01-other',model,lambda x:x,lambda:None)
        self.assertEqual(model.calls,[]);self.assertFalse((self.root/'budgets').exists())
    def test_packet_tamper_fails_before_model(self):
        Path(self.packet_ref['path']).write_bytes(b'{}');model=FakeLLM()
        with self.assertRaises(ValueError):self.run_packet(model)
        self.assertEqual(model.calls,[])
    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaises(ValueError):w.parse_strict('{"id":"one","id":"two"}')
    def test_nonfinite_json_rejected(self):
        with self.assertRaises(ValueError):w.parse_strict('{"value":NaN}')
    def test_duplicate_microclaims_rejected_before_any_local_source_budget(self):
        model=FakeLLM()
        with self.assertRaises(ValueError):w.run_packets([self.packet_ref,self.packet_ref],self.root,self.owner,model,lambda x:x,lambda:None)
        self.assertEqual(model.calls,[]);self.assertFalse((self.root/'budgets').exists())
    def test_more_than_four_microclaims_refused(self):
        with self.assertRaises(ValueError):w.run_packets([self.packet_ref]*5,self.root,self.owner,FakeLLM(),lambda x:x,lambda:None)
    def test_two_microclaims_share_only_two_actual_model_calls(self):
        packet=copy.deepcopy(self.packet);original=packet['claim']['rows'][0];item=packet['sources'][0];id_='AnotherSource';original['id']=item['id']=id_;original['video_path']=original['video_path'].replace('SyntheticSource',id_);original['fullframe_manifest_ref']['path']=original['fullframe_manifest_ref']['path'].replace('SyntheticSource',id_)
        draft=checked(item['draft_ref']);manifest=checked(item['manifest_ref']);draft['id']=manifest['id']=id_;manifest['video_path']=original['video_path'];draft['source']['video_path']=original['video_path'];draft['source']['fullframe_manifest_path']=original['fullframe_manifest_ref']['path']
        binding=draft['source']['fullframe_binding'];body=dict(schema=binding['schema'],fullframe_manifest=manifest,observations_sha256=binding['observations_sha256']);binding['sha256']=w.hashlib.sha256(w.json.dumps(body,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        prepared=self.root/'prepared';item['draft_ref']=write_once(prepared/'another_draft.json',draft);item['manifest_ref']=write_once(prepared/'another_manifest.json',manifest);original['rule_draft_ref']['sha256']=item['draft_ref']['sha256'];original['fullframe_manifest_ref']['sha256']=item['manifest_ref']['sha256'];video=prepared/(id_+'.mp4');video.write_bytes(Path(item['local_video_ref']['path']).read_bytes());item['local_video_ref']['path']=str(video)
        packet['claim']['batch_id']='claim0002';packet['claim']['source_keys']=[['train',id_]];ref=write_once(self.root/'packets/packet2.json',packet);model=FakeLLM();results=w.run_packets([self.packet_ref,ref],self.root,self.owner,model,lambda x:x,lambda:None)
        self.assertEqual([x['sources'] for x in model.calls],[2,2]);self.assertEqual([x['pool_claim_batch'] for x in results],['claim0001','claim0002']);self.assertEqual(sum(x['new_model_calls'] for x in results),4)
if __name__=='__main__':unittest.main(verbosity=2)
