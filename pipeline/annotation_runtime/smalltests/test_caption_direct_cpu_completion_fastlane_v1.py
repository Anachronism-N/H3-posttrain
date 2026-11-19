"""Fastlane-only CPU contracts with real frozen native support/rewrite validators."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RT))
import caption_direct_cpu_completion_fastlane_v1 as fast
bar, pipeline = fast.direct.native()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bar.batch.json_bytes(value))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fixture(base, sid, version=3, abstain=False):
    consumer = base / ('curated222939_collaborative_v%d' % version) / 'consumers' / 'shard-0000'
    root = consumer / 'visual' / 'shard-0000' / 'sources' / sid
    record = dict(id=sid, source_id=sid, split='train', batch_id='train-b000001', source_ordinal=int(sid[-1]),
        video_path=str(base / 'media' / (sid+'.mp4')), audio_path=str(base / 'media' / (sid+'.wav')), nlf_path=None)
    for field in ('video_path', 'audio_path'):
        path = Path(record[field]); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(sid.encode())
    identity = bar.batch.source_identity(record)
    source_binding = dict(id=sid, video_sha256=identity['files']['video_path']['sha256'],
        audio_sha256=identity['files']['audio_path']['sha256'])
    raw_windows = [dict(start=0., end=1.), dict(start=1., end=2.)]
    plan = dict(id=sid, manifest_record=record, windows=raw_windows, window_manifest=dict(original=True), binding=source_binding)
    binding = dict(source_binding=source_binding, params_sha256='params-v%d'%version,
        full_source_row_sha256=bar.batch.digest(plan), manifest_record_sha256=bar.batch.digest(record))
    windows = bar.normalized_windows(plan)
    manifests = [dict(window=w, frames=[dict(index=i*2, time=float(i), file=str(i*2)+'.jpg', sha256='a'*64),
        dict(index=i*2+1, time=float(i)+.5, file=str(i*2+1)+'.jpg', sha256='b'*64)], hand_roi_evidence={}) for i,w in enumerate(windows)]
    union = dict(binding=binding, frames=[f for m in manifests for f in m['frames']])
    union_ref = save(root / 'rgb' / 'manifest.json', union)
    prepared = dict(binding=binding, source_identity=identity, windows=manifests, original_windows=raw_windows,
        window_manifest=plan['window_manifest'], forbidden={}, raw2d_mask=dict(available=False), union_manifest_sha256=union_ref['sha256'])
    prepared_ref = save(root / 'prepared.json', prepared)
    save(root / 'prepared.receipt.json', dict(binding=binding, prepared_sha256=prepared_ref['sha256']))
    save(root / 'plan.json', plan)
    model = dict(root='/unused/historical-model', files={'weights.safetensors':dict(path='/unused/historical-model/weights.safetensors', sha256='c'*64)})
    model['sha256'] = bar.batch.digest(model)
    model_ref = save(consumer / 'model.identity.json', model)
    observations = []
    for i,m in enumerate(manifests):
        wid = m['window']['window_id']; attempt = root / 'windows' / wid / 'attempt-001'
        save(root / 'windows' / wid / 'manifest.json', dict(m,binding=binding,union_manifest_sha256=union_ref['sha256'],raw2d_mask=prepared['raw2d_mask'],**bar.FLAGS))
        raw = bar.empty_observation('unknown')
        indices = [f['index'] for f in m['frames']]
        if not (abstain and i==0):
            for part in ('head','left_hand'):
                raw['part_support'][bar.PARTS.index(part)]['frame_indices']=indices
            raw.update(events=[dict(id='e1', description='The dancer turns the head.', parts=['head'], frame_indices=indices),
                dict(id='e2',description='The dancer opens the left hand.',parts=['left_hand'],frame_indices=indices)], abstain=False)
        response = bar.filter_supported_observation(raw,m,{})
        status = 'abstain' if response['abstain'] else 'candidate'
        observations.append(dict(window=m['window'],status=status,json=response,attempt_path=str(attempt)))
        req = dict(model=model['root'],sampling=dict(max_tokens=100),json_schema=dict(type='object'),messages=[])
        req_ref = save(attempt / 'request.json',req)
        raw_ref = save(attempt / 'response.raw.txt',raw)
        save(attempt / 'binding.json',dict(binding,window=m['window'],frames_sha256=bar.batch.digest(m['frames'])))
        save(attempt / 'generation.json',dict(request_sha256=req_ref['sha256'],raw_response_sha256=raw_ref['sha256'],
            max_tokens=100,model_call_performed=True,structured_output_schema_sha256=bar.core.digest(req['json_schema']),
            completion_tokens=10,finish_reason='stop',model_identity_sha256=model['sha256']))
        save(attempt / 'response.json',response)
        save(attempt / 'final.json',dict(status='candidate',json=response))
        files = {name:hashlib.sha256((attempt/name).read_bytes()).hexdigest() for name in fast.REQUIRED}
        save(attempt / 'receipt.json',dict(status='candidate',files_sha256=files,
            binding=dict(binding,window=m['window'],frames_sha256=bar.batch.digest(m['frames']))))
    facts = bar.facts_from_observations(windows, observations)
    original = dict(id=sid,status='failed',binding=binding,manifest_record=record,original_windows=raw_windows,
        window_observations=observations,facts=facts,final=dict(status='failed',error='unsupported rewrite'),
        raw_rgb_manifest=union_ref,model_identity_ref=model_ref,original_window_manifest=plan['window_manifest'],
        source_refs=dict(original='refs'),boundary_observations=[],coverage=dict(full_window_caption=False))
    original_ref = save(root / 'results' / 'original.json',original)
    style = dict(id=sid,status='resolved',dance_style='contemporary')
    row = dict(record,source_refs=dict(original='refs'))
    joint = pipeline.joint_candidate(row,original,style,model['sha256']);joint['visual_result_ref']=original_ref
    joint_path=consumer/'sources'/'train'/sid/'joint-results'/(bar.batch.digest(joint)+'.json')
    joint_ref=save(joint_path,joint)
    notice = dict(id=sid,split='train',batch_id=record['batch_id'],joint_candidate_ref=joint_ref,joint_complete=False,
        params_sha256=binding['params_sha256'],visual_status='failed',style_status='resolved')
    selected_path=base/'runtime'/('selected_v%d.json'%version)
    selected = pipeline.load(selected_path) if selected_path.exists() else []
    selected.append({k:record[k]for k in ('id','split','batch_id','source_ordinal')})
    selected_ref=save(selected_path,selected)
    save(base/'runtime'/('curated_collaborative%d_source_plan_v1.json'%version),
        dict(members=dict(worker=dict(mode='consumer',shard=0)),selected_sources=dict(worker=selected_ref)))
    budget=consumer/'attempt-budget'/'train'/(sid+'.json');save(budget,dict(used=1))
    return notice, consumer, version, dict(path=str(consumer/'completed.jsonl'),prefix_bytes=1,prefix_sha256='hint')


class FastlaneContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        self.patches=[patch.object(fast,'BAR',self.base),patch.object(fast,'RT',self.base/'runtime')]
        for p in self.patches:p.start()
        self.addCleanup(self.tmp.cleanup)
        for p in self.patches:self.addCleanup(p.stop)

    def lane(self, name='direct_cpu_completion_fastlane_v1',crossmount=False):
        return fast.Fastlane(self.base/name,crossmount)

    def test_three_sources_restore_two_keep_abstain_and_zero_dispatch_budget_touch(self):
        cases=[fixture(self.base,'case1'),fixture(self.base,'case2',version=4),fixture(self.base,'case3',abstain=True)]
        budget_paths=[c[1]/'attempt-budget'/'train'/(c[0]['id']+'.json') for c in cases]
        before=[(p.read_bytes(),fast.stat(p))for p in budget_paths]
        original_read=Path.read_bytes;reads={}
        def guarded(path):
            self.assertNotIn(path.suffix,('.jpg','.jpeg','.mp4','.wav','.safetensors'))
            reads[path]=reads.get(path,0)+1
            return original_read(path)
        lane=self.lane()
        with (patch.object(Path,'read_bytes',guarded),patch.object(bar,'transactions',side_effect=AssertionError('no dispatch')),
              patch.object(bar,'verify_prepared',side_effect=AssertionError('no old closure')),
              patch.object(bar.batch,'model_identity',side_effect=AssertionError('no model weights'))):
            results=[lane.process(*c)for c in cases]
        self.assertEqual([r['joint_complete']for r in results],[True,True,False])
        self.assertEqual(results[2]['status'],'partial_unchanged')
        for c,r in zip(cases[:2],results[:2]):
            candidate=pipeline.load(r['candidate_ref']['path']);joint=pipeline.load(c[0]['joint_candidate_ref']['path'])
            original=pipeline.load(joint['visual_result_ref']['path'])
            self.assertEqual(candidate['admitted_facts'],original['facts'])
            self.assertEqual(candidate['style_result'],joint['style_result'])
            self.assertFalse(candidate['training_eligible']);self.assertFalse(candidate['pixel_quality_review_completed'])
            self.assertEqual(candidate['new_model_calls'],0);self.assertEqual(candidate['new_budget_reservations'],0)
            for w in candidate['direct_final']['json']['windows']:
                local=[f for f in original['facts']if f['window_id']==w['window_id']]
                self.assertEqual(w['caption'],' '.join(f['text']for f in local))
                self.assertEqual(w['used_fact_ids'],[f['fact_id']for f in local])
                self.assertEqual(w['used_boundary_ids'],[])
        self.assertEqual(before,[(p.read_bytes(),fast.stat(p))for p in budget_paths])
        self.assertTrue(all(count==1 for path,count in reads.items() if path.name in fast.REQUIRED))

    def test_exact_repeat_zero_writes_and_no_rederive(self):
        case=fixture(self.base,'case1');lane=self.lane();first=lane.process(*case)
        writes=lane.writes
        with patch.object(fast,'derive',side_effect=AssertionError('no rederive')):
            self.assertEqual(lane.process(*case)['status'],'existing_exact')
        self.assertEqual(lane.writes,writes)
        continued=self.lane();self.assertEqual(continued.writes,0)
        with patch.object(fast,'derive',side_effect=AssertionError('no rederive')):
            self.assertEqual(continued.process(*case)['writes'],0)
        self.assertEqual(continued.writes,0)

    def test_tampered_annotation_rejected_no_candidate_other_source_continues(self):
        bad=fixture(self.base,'case1');good=fixture(self.base,'case2')
        joint=pipeline.load(bad[0]['joint_candidate_ref']['path']);original=pipeline.load(joint['visual_result_ref']['path'])
        attempt=Path(original['window_observations'][0]['attempt_path']);save(attempt/'response.raw.txt',dict(tampered=True))
        lane=self.lane();r=lane.process(*bad)
        self.assertEqual(r['status'],'source_rejected');self.assertFalse(list(lane.output.rglob('candidate.json')))
        self.assertTrue(lane.process(*good)['joint_complete'])

    def test_wrong_version_and_existing_recipe_mismatch_do_not_commit(self):
        case=fixture(self.base,'case1');wrong=copy.deepcopy(case[0]);wrong['joint_candidate_ref']['path']=wrong['joint_candidate_ref']['path'].replace('_v3/','_v4/')
        lane=self.lane();self.assertEqual(lane.process(wrong,*case[1:])['status'],'source_rejected')
        self.assertFalse(list(lane.output.rglob('candidate.json')))
        config=lane.output/'recipe.json';value=pipeline.load(config);value['schema']='wrong_version';save(config,value)
        with self.assertRaisesRegex(ValueError,'recipe/execution version'):self.lane()

    def test_source_stat_only_device_difference_requires_explicit_crossmount(self):
        case=fixture(self.base,'case1');joint=pipeline.load(case[0]['joint_candidate_ref']['path']);o=pipeline.load(joint['visual_result_ref']['path'])
        root=Path(o['raw_rgb_manifest']['path']).parent.parent;p=pipeline.load(root/'prepared.json')
        for old in p['source_identity']['files'].values():
            if old is not None:old['stat']['device']+=1
        with self.assertRaisesRegex(ValueError,'crossmount device'):fast.source_boundary(p['source_identity'],o['manifest_record'],o['binding'],False)
        rows=fast.source_boundary(p['source_identity'],o['manifest_record'],o['binding'],True)
        self.assertTrue(all(r['crossmount_device_only']for r in rows))
        p['source_identity']['files']['video_path']['stat']['inode']+=1
        with self.assertRaisesRegex(ValueError,'crossmount device'):fast.source_boundary(p['source_identity'],o['manifest_record'],o['binding'],True)

    def test_existing_annotation_or_media_stat_change_rejected_without_write(self):
        case=fixture(self.base,'case1');lane=self.lane();r=lane.process(*case);writes=lane.writes
        joint=pipeline.load(case[0]['joint_candidate_ref']['path']);p=Path(joint['visual_result_ref']['path']);p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'annotation changed'):lane.process(*case)
        self.assertEqual(lane.writes,writes)

    def test_cache_hash_once_and_postread_stat_tamper_rejected(self):
        path=self.base/'artifact.json';save(path,dict(ok=True));e=fast.Evidence();ref=e.ref(path)
        with patch.object(Path,'read_bytes',side_effect=AssertionError('no second content read')):
            self.assertEqual(e.checked(ref),dict(ok=True));self.assertEqual(len(e.boundary()),1)
        self.assertEqual(e.read_count,1);path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'changed after admission'):e.boundary()

    def test_truncated_generation_and_wrong_model_metadata_rejected(self):
        for field,value in [('finish_reason','length'),('model_identity_sha256','wrong')]:
            case=fixture(self.base,'case'+('1'if field=='finish_reason'else '2'))
            joint=pipeline.load(case[0]['joint_candidate_ref']['path']);o=pipeline.load(joint['visual_result_ref']['path']);attempt=Path(o['window_observations'][0]['attempt_path'])
            gen=pipeline.load(attempt/'generation.json');gen[field]=value;ref=save(attempt/'generation.json',gen)
            receipt=pipeline.load(attempt/'receipt.json');receipt['files_sha256']['generation.json']=ref['sha256'];save(attempt/'receipt.json',receipt)
            lane=self.lane('direct_cpu_completion_fastlane_v1_'+field);r=lane.process(*case)
            self.assertEqual(r['status'],'source_rejected');self.assertFalse(list(lane.output.rglob('candidate.json')))


if __name__=='__main__':
    unittest.main()
