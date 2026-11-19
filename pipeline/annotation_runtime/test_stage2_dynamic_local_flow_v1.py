"""Real dynamic transaction -> feeder -> local model/mock -> durable upload integration."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import stage2_dynamic_pool_v1 as pool
import stage2_cpu_feeder_v1 as feeder
import stage2_local_model_worker_v1 as worker
from stage2_local_publication_v1 import write_once,LocalOutbox,checked
from test_ms_fullframe_vlm_postprocess_batch_v4 import FakeLLM
from test_stage2_local_packet_v1 import PacketTests

class IntegratedFlowTests(unittest.TestCase):
    def setUp(self):
        self.f=PacketTests('test_local_packet_preserves_original_logical_source_without_shared_reads');self.f.setUp();self.addCleanup(self.f.doCleanups);self.tmp=tempfile.TemporaryDirectory(prefix='dynamic-flow-cpu-',dir='/tmp');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.out=self.root/'out';self.out.mkdir();self.token=self.root/'token';self.token.write_text('synthetic-authorized-token-32characters');original=copy.deepcopy(self.f.packet['claim']['rows'][0]);original['rule_draft_ref']=self.f.dr;original['fullframe_manifest_ref']=self.f.mr;manifest=checked(self.f.mr);original['fullframe_origin_path']=self.f.packet['claim']['rows'][0]['fullframe_manifest_ref']['path'];self.db=pool.DynamicPool.create(self.root/'pool.db',[original],[]);self.addCleanup(self.db.close);self.owner='N3-g01-worker';self.cfg=dict(out=str(self.out),owner=self.owner,token_path=str(self.token),claim_endpoint='http://127.0.0.1:18992/claim',ffmpeg_path='synthetic',prefetch_packets=8);self.config=self.root/'feeder.json';self.config.write_text(json.dumps(self.cfg));self.calls=0
    def response(self,request,timeout):
        self.calls+=1;payload=json.loads(request.data);self.assertEqual(payload['owner'],self.owner);v=self.db.claim(payload['owner'],payload['request_id'],payload['size']);raw=json.dumps(v).encode()
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def read(self,*args):return raw
        return Response()
    def media(self,row,manifest,d,ffmpeg):
        local=d/(row['id']+Path(row['video_path']).suffix);local.write_bytes(self.f.video.read_bytes());frames=[];images=[]
        for i,(t,p) in enumerate(zip(manifest['times'],manifest['source_timestamps'])):
            image=d/('frame%03d.jpg'%i);image.write_bytes(bytes([255,216])+b'jpeg'+bytes([255,217]));frames.append(dict(frame_index=i,time=t,source_pts=p));images.append(image)
        return frames,images
    def test_actual_claim_and_prepared_packet_run_models_then_upload_without_shared_generation(self):
        with patch.object(feeder.urllib.request,'urlopen',side_effect=self.response),patch.object(feeder.media,'stage_images',side_effect=self.media):feeder.run(self.config)
        self.assertEqual(self.db.progress(),{'assigned':1});queue=self.out/'ready'/self.owner;ready=list(queue.glob('*.ready.json'));self.assertEqual(len(ready),1);self.assertTrue((queue/'FEEDER_DRAINED').exists());ref=json.loads(ready[0].read_bytes())['packet_ref'];model=FakeLLM();result=worker.run_packet(ref,self.out,self.owner,model,lambda x:x,lambda:None);self.assertEqual(result[0]['new_model_calls'],2);box=LocalOutbox(self.out,self.root/'shared','N3');sealed=box.seal([['train',result[0]['id']]]);self.assertFalse(box.shared.exists());ack=box.upload(sealed);self.assertEqual(ack['models_called'],0);self.assertEqual(len(model.calls),2)
    def test_uncertain_claim_transport_failure_marks_local_failed_without_releasing_sources(self):
        def failing(request,timeout):
            payload=json.loads(request.data);self.db.claim(payload['owner'],payload['request_id'],8);raise OSError('Lost response after actual committed source claim')
        with patch.object(feeder.urllib.request,'urlopen',side_effect=failing),self.assertRaises(OSError):feeder.run(self.config)
        self.assertEqual(self.db.progress(),{'assigned':1});self.assertTrue((self.out/'ready'/self.owner/'FEEDER_FAILED').exists());self.assertFalse((self.out/'sources').exists())
if __name__=='__main__':unittest.main(verbosity=2)
