import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RT))
import curated_collab3_review_adapter_v1 as mod


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.tmp.name)
        cls.review = mod.load_sampling(RT / 'source_curated_review_v2')
        cls.originals = []
        batches = []
        for index in range(60):
            bid = 'train-b%06d' % (64 + index)
            path = cls.base / (bid + '.jsonl')
            rows = []
            for offset in range(500):
                ordinal = 32000 + index * 500 + offset
                sid = 's%06d' % ordinal
                row = dict(id=sid, source_id=sid, split='train', batch_id=bid, source_ordinal=ordinal,
                           video_path=str(cls.base / 'media' / (sid + '.mp4')),
                           audio_path=str(cls.base / 'media' / (sid + '.wav')),
                           source_refs={'csv':dict(path=str(cls.base/'original.csv'), sha256='csvSHA',
                                                  csv_line=ordinal+2, raw_row_sha256=str(ordinal))})
                rows.append(row)
                cls.originals.append(row)
            if index == 0:
                path.write_bytes(b''.join(mod.encoded(row) for row in rows))
                ref = mod.reference(path)
            else:
                ref = dict(path=str(path), sha256='unused-not-read')
            batches.append(dict(**ref, batch_id=bid, count=500, split='train', first_source_ordinal=32000+index*500))
        full = dict(total=222939, batches=batches+[dict(batch_id='other-%d'%i, count=500) for i in range(387)] +
                    [dict(batch_id='last',count=222939-30000-387*500)])
        cls.manifest = cls.base / 'manifest.json'
        cls.manifest.write_bytes(mod.encoded(full))
        cls.members = {}
        cls.refs = {}
        for mode, nshard in [('producer',4),('consumer',9)]:
            for shard in range(nshard):
                node = 'N2' if shard < (2 if mode == 'producer' else 3) else 'N7' if mode == 'producer' or shard>=7 else 'N4'
                member = node.lower()+('p' if mode=='producer' else 'c')+str(shard)
                cls.members[member] = dict(node=node,mode=mode,gpus=[],shard=shard,nshard=nshard)
                path = cls.base / (member+'.json')
                path.write_bytes(mod.encoded([{k:r[k] for k in ('id','split','batch_id','source_ordinal')}
                                             for r in cls.originals if r['source_ordinal']%nshard==shard]))
                cls.refs[member] = mod.reference(path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.case = tempfile.TemporaryDirectory(dir=self.base)
        self.work = Path(self.case.name)
        self.root = self.work/'production'
        self.out = self.work/'derived'
        plan = dict(source_range=dict(start=32000,end=62000), selected_count=30000,input_count=222939,
                    full_batch_count=448, out=str(self.root),selected_original_batches=self.review_batches(),
                    manifest_summary=mod.reference(self.manifest),members=self.members,selected_sources=self.refs)
        self.plan = self.work/'plan.json'
        self.plan.write_bytes(mod.encoded(plan))
        self.adapter = mod.Adapter(self.plan,mod.sha(self.plan),self.review)
        self.row = self.originals[0]

    def tearDown(self):
        self.case.cleanup()

    @classmethod
    def review_batches(cls):
        return json.loads(cls.manifest.read_bytes())['batches'][:60]

    def put(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(mod.encoded(value))
        return mod.reference(path)

    def append(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('ab') as stream:
            stream.write(mod.encoded(value))

    def produce(self,row=None,publish=True):
        row = row or self.row
        for field in ('video_path','audio_path'):
            path = Path(row[field]);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(row['id'].encode())
        media = dict(files={field:mod.reference(row[field]) for field in ('video_path','audio_path')})
        binding = dict(id=row['id'],video_sha256=mod.sha(row['video_path']),audio_sha256=mod.sha(row['audio_path']))
        directory = self.root/'sources/train'/row['id']
        plan = dict(id=row['id'],manifest_record=row,binding=binding,source_refs={'original_csv':row['source_refs']['csv']})
        planref = self.put(directory/'bar_plan.json',plan)
        roiref = self.put(self.root/'roi'/row['id']/'roi_manifest.json',dict(id=row['id'],source=mod.reference(row['video_path']),
                                                                               video_sha256=binding['video_sha256']))
        self.put(directory/'actual_media.json',dict(source_identity=media))
        notice = dict(id=row['id'],split='train',batch_id=row['batch_id'],source_ordinal=row['source_ordinal'],
                      status='ready',plan_ref=planref,roi_ref=roiref,source_binding=binding)
        self.put(directory/'producer.terminal.json',notice)
        if publish:
            self.append(self.root/'ready'/('producer-%04d.jsonl'%(row['source_ordinal']%4)),notice)
        return notice,media

    def consume(self,row=None,text='[0-2] dancer steps.',complete=True,publish=True):
        row = row or self.row
        producer,media = self.produce(row,publish=False)
        shard = row['source_ordinal']%9
        physical = self.root/'consumers'/('shard-%04d'%shard)
        directory = physical/'sources/train'/row['id']
        visualref = self.put(physical/'visual'/('shard-%04d'%shard)/'sources'/row['id']/'result.json',dict(id=row['id']))
        styleref = self.put(directory/'style-results/style.json',dict(id=row['id']))
        rgbref = self.put(physical/'visual'/('shard-%04d'%shard)/'sources'/row['id']/'rgb/manifest.json',
                          dict(binding={'source_binding':producer['source_binding']},source_identity=media,
                               video_sha256=producer['source_binding']['video_sha256']))
        label = dict(id=row['id'],source_id=row['id'],split='train',batch_id=row['batch_id'],video_path=row['video_path'],
                     audio_path=row['audio_path'],source_refs={'original_csv':row['source_refs']['csv']},
                     source_identity=media,source_binding=dict(source_binding=producer['source_binding'],plan_sha256=producer['plan_ref']['sha256']),
                     joint_complete=complete,all_windows_nonempty=complete,style_status='candidate',visual_status='candidate',
                     label=dict(dance_style='street dance',timed_motion_script=text),visual_result_ref=visualref,style_result_ref=styleref,
                     raw_rgb_manifest=rgbref,hand_roi={'available':True,**producer['roi_ref']},annotator_model_sha256='32B-identity')
        ref = self.put(directory/'joint-results'/(mod.digest(label)+'.json'),label)
        notice = dict(id=row['id'],split='train',batch_id=row['batch_id'],status='joint_candidate' if complete else 'incomplete_candidate',
                      joint_complete=complete,joint_candidate_ref=ref)
        self.put(directory/'consumer.terminal.json',notice)
        stream = physical/'completed.jsonl'
        if publish:
            self.append(stream,notice)
        return notice,stream

    def test_live_append_dedup_and_exact_original_sampling(self):
        self.produce()
        value = self.adapter.poll(self.out)
        self.assertEqual(value['producer_ready_sources'],1)
        notice,stream = self.consume()
        value = self.adapter.poll(self.out)
        self.assertEqual(value['terminal_sources'],1)
        self.assertEqual(value['candidate_complete_sources'],1)
        self.append(stream,notice)
        value = self.adapter.poll(self.out)
        self.assertEqual(value['terminal_sources'],1)
        before = copy.deepcopy(self.adapter.state['streams'])
        self.adapter.poll(self.out)
        self.assertEqual(before,self.adapter.state['streams'])
        manifest = json.loads((self.out/'batches/train-b000064/review-required.json').read_bytes())
        self.assertEqual(manifest['sampling_denominator'],500)
        self.assertEqual(len(manifest['records']),500)
        self.assertEqual(len(manifest['sampling']['random_source_ids']),25)
        self.assertEqual(manifest['sampling'],self.review.sampling_plan(manifest['records'],'train-b000064',sample_count=25,risk_count=5))
        self.assertEqual(manifest['exact_label_bindings'][self.row['id']]['label_file_ref'],notice['joint_candidate_ref'])
        self.assertFalse(manifest['source_quality_pass'])
        self.assertTrue(all(r['individual_review_status']=='not_individually_reviewed' for r in manifest['records'] if not r['selected_for_review']))

    def test_revision_is_not_permanent_source_rejection(self):
        self.consume(text='right hand starts twice',complete=False)
        first = self.adapter.poll(self.out)
        old = self.adapter.state['terminal']['train/'+self.row['id']]['label_revision']
        self.assertEqual(first['candidate_partial_sources'],1)
        self.consume(text='left foot steps',complete=True)
        value = self.adapter.poll(self.out)
        self.assertEqual(value['candidate_complete_sources'],1)
        self.assertEqual(value['candidate_partial_sources'],0)
        self.assertNotEqual(old,self.adapter.state['terminal']['train/'+self.row['id']]['label_revision'])

    def test_wrong_result_sha_fail_closed(self):
        notice,stream = self.consume(publish=False)
        notice['joint_candidate_ref']['sha256']='0'*64
        self.put(Path(stream).parent/'sources/train'/self.row['id']/'consumer.terminal.json',notice)
        self.append(stream,notice)
        value=self.adapter.poll(self.out)
        self.assertEqual(value['terminal_sources'],0)
        self.assertTrue(value['blocked_streams'])
        self.assertEqual(self.adapter.state['streams']['consumer:'+str(32000%9)]['offset'],0)

    def test_cross_node_media_stat_device_is_not_content_identity(self):
        notice,stream=self.consume(publish=False)
        labelpath=Path(notice['joint_candidate_ref']['path']);label=json.loads(labelpath.read_bytes())
        rgbpath=Path(label['raw_rgb_manifest']['path']);rgb=json.loads(rgbpath.read_bytes())
        for field in ('video_path','audio_path'):
            rgb['source_identity']['files'][field]['stat']={'device':99999,'inode':2222}
        label['raw_rgb_manifest']=self.put(rgbpath,rgb)
        notice['joint_candidate_ref']=self.put(labelpath,label)
        self.put(stream.parent/'sources/train'/self.row['id']/'consumer.terminal.json',notice)
        self.append(stream,notice)
        self.assertEqual(self.adapter.poll(self.out)['candidate_complete_sources'],1)

    def test_wrong_rgb_source_binding_fail_closed(self):
        notice,stream = self.consume(publish=False)
        labelpath=Path(notice['joint_candidate_ref']['path']);label=json.loads(labelpath.read_bytes())
        rgbpath=Path(label['raw_rgb_manifest']['path']);rgb=json.loads(rgbpath.read_bytes())
        rgb['video_sha256']='bad-video'
        label['raw_rgb_manifest']=self.put(rgbpath,rgb)
        notice['joint_candidate_ref']=self.put(labelpath,label)
        self.put(stream.parent/'sources/train'/self.row['id']/'consumer.terminal.json',notice)
        self.append(stream,notice)
        value=self.adapter.poll(self.out)
        self.assertEqual(value['terminal_sources'],0)
        self.assertTrue(value['blocked_streams'])

    def test_wrong_consumer_member(self):
        notice,_=self.consume(publish=False)
        self.append(self.root/'consumers/shard-0000/completed.jsonl',notice)
        value=self.adapter.poll(self.out)
        self.assertEqual(value['terminal_sources'],0)
        self.assertIn('consumer:0',value['blocked_streams'])

    def test_cross_source_result_ref(self):
        notice,stream=self.consume(publish=False)
        other=self.originals[9]
        other_notice,_=self.consume(other,publish=False)
        notice['joint_candidate_ref']=other_notice['joint_candidate_ref']
        self.put(stream.parent/'sources/train'/self.row['id']/'consumer.terminal.json',notice)
        self.append(stream,notice)
        value=self.adapter.poll(self.out)
        self.assertEqual(value['terminal_sources'],0)
        self.assertTrue(value['blocked_streams'])

    def test_raw_malformed_not_skipped(self):
        stream=self.root/'ready/producer-0000.jsonl'
        stream.parent.mkdir(parents=True)
        stream.write_bytes(b'{bad}\n')
        self.produce()
        value=self.adapter.poll(self.out)
        self.assertEqual(value['producer_ready_sources'],0)
        self.assertIn('producer:0',value['blocked_streams'])
        self.assertEqual(value['streams']['producer:0']['offset'],0)

    def test_partial_tail_stays_pending_then_live_append(self):
        notice,_=self.produce(publish=False)
        stream=self.root/'ready/producer-0000.jsonl'
        stream.parent.mkdir(parents=True)
        data=mod.encoded(notice)
        stream.write_bytes(data[:-1])
        value=self.adapter.poll(self.out)
        self.assertEqual(value['producer_ready_sources'],0)
        self.assertEqual(value['streams']['producer:0']['status'],'pending_tail')
        with stream.open('ab') as f:f.write(b'\n')
        self.assertEqual(self.adapter.poll(self.out)['producer_ready_sources'],1)

    def test_missing_reference_retains_cursor(self):
        notice,_=self.produce(publish=False)
        path=Path(notice['plan_ref']['path']);raw=path.read_bytes();path.unlink()
        stream=self.root/'ready/producer-0000.jsonl';self.append(stream,notice)
        value=self.adapter.poll(self.out)
        self.assertEqual(value['streams']['producer:0']['status'],'pending_reference')
        self.assertEqual(value['streams']['producer:0']['offset'],0)
        path.write_bytes(raw)
        self.assertEqual(self.adapter.poll(self.out)['producer_ready_sources'],1)

    def test_missing_ready_count_separate_and_no_history_rehash(self):
        self.consume()
        value=self.adapter.poll(self.out)
        self.assertEqual(value['terminal_sources_without_ready_notice'],1)
        media_paths={self.row['video_path'],self.row['audio_path']}
        original=mod.sha
        def guarded(path):
            self.assertNotIn(str(path),media_paths,'unchanged history media rehashed')
            return original(path)
        with patch.object(mod,'sha',side_effect=guarded):
            self.assertEqual(self.adapter.poll(self.out)['terminal_sources'],1)

    def test_feature_enabled_cpu_entry_has_zero_models_budget_signals_locks(self):
        self.consume()
        argv=['--plan',str(self.plan),'--plan-sha256',mod.sha(self.plan),'--ms-source',str(RT/'source_curated_review_v2'),
              '--out',str(self.out),'--enable-feature','--once']
        import fcntl
        with patch.object(os,'kill',side_effect=AssertionError('signal forbidden')), \
             patch.object(subprocess,'run',side_effect=AssertionError('process/model forbidden')), \
             patch.object(subprocess,'Popen',side_effect=AssertionError('process/model forbidden')), \
             patch.object(fcntl,'flock',side_effect=AssertionError('resource lock forbidden')):
            value=mod.main(argv)
        for key in ('model_calls','extra_source_attempts','signals_sent','resource_locks_acquired'):
            self.assertEqual(value[key],0)
        self.assertFalse(value['attempt_budget_consumed'])
        self.assertFalse(value['training_eligible'])

    def test_source_media_corruption_fail_closed(self):
        notice,_=self.produce(publish=False)
        Path(self.row['video_path']).write_bytes(b'changed-media')
        self.append(self.root/'ready/producer-0000.jsonl',notice)
        value=self.adapter.poll(self.out)
        self.assertEqual(value['producer_ready_sources'],0)
        self.assertTrue(value['blocked_streams'])

    def test_submission_binds_current_revision_and_sampling_denominator(self):
        selected = self.review.sampling_plan(self.originals[:500],'train-b000064')['random_source_ids'][0]
        row = next(r for r in self.originals[:500] if r['id']==selected)
        self.consume(row)
        self.adapter.poll(self.out)
        path = self.out/'batches/train-b000064/review-required.json'
        ref = mod.reference(path)
        validated = self.adapter.submission_binding(ref,selected)
        self.assertTrue(validated['independent_8B_identity_verification_still_required'])
        self.assertFalse(validated['source_quality_pass'])
        self.consume(row,text='new revised motion')
        with self.assertRaises(ValueError):
            self.adapter.submission_binding(ref,selected)
        self.adapter.poll(self.out)
        validated = self.adapter.submission_binding(mod.reference(path),selected)
        self.assertEqual(validated['row']['label']['timed_motion_script'],'new revised motion')
        value=json.loads(path.read_bytes());value['sampling_denominator']=1
        changed=self.put(self.work/'bad-denominator.json',value)
        with self.assertRaises(ValueError):
            self.adapter.submission_binding(changed,selected)

    def test_plan_original_member_error_not_generic_skip(self):
        notice,_=self.produce(publish=False)
        path=Path(notice['plan_ref']['path']);value=json.loads(path.read_bytes())
        value['manifest_record']=copy.deepcopy(value['manifest_record'])
        value['manifest_record']['source_ordinal']+=1
        notice['plan_ref']=self.put(path,value)
        self.put(path.parent/'producer.terminal.json',notice)
        self.append(self.root/'ready/producer-0000.jsonl',notice)
        value=self.adapter.poll(self.out)
        self.assertEqual(value['producer_ready_sources'],0)
        self.assertIn('producer:0',value['blocked_streams'])

    def test_bounded_increment_preserves_backlog(self):
        notice,_=self.produce(publish=False)
        stream=self.root/'ready/producer-0000.jsonl'
        self.append(stream,notice);self.append(stream,notice);self.append(stream,notice)
        self.adapter.max_lines_per_stream=1
        value=self.adapter.poll(self.out)
        self.assertEqual(value['streams']['producer:0']['status'],'pending_backlog')
        self.assertEqual(value['producer_ready_sources'],1)
        self.adapter.poll(self.out)
        value=self.adapter.poll(self.out)
        self.assertEqual(value['streams']['producer:0']['status'],'caught_up')
        self.assertEqual(value['producer_ready_sources'],1)

    def test_duplicate_keys_are_raw_malformed(self):
        stream=self.root/'ready/producer-0000.jsonl';stream.parent.mkdir(parents=True)
        stream.write_bytes(b'{"id":"a","id":"b"}\n')
        value=self.adapter.poll(self.out)
        self.assertEqual(value['streams']['producer:0']['status'],'blocked')
        self.assertEqual(value['streams']['producer:0']['offset'],0)

    def test_state_restart_preserves_append_cursor(self):
        self.consume();self.adapter.poll(self.out)
        state=json.loads((self.out/'state.json').read_bytes())
        restored=mod.Adapter(self.plan,mod.sha(self.plan),self.review,state)
        self.assertEqual(restored.poll(self.out)['terminal_sources'],1)
        self.assertEqual(restored.state['streams'],self.adapter.state['streams'])


if __name__ == '__main__':
    unittest.main()
