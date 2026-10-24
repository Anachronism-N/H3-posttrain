import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts_omnidance/exp/ms/runtime_jobs/publish_ready_eval_review_v6.py'
spec = importlib.util.spec_from_file_location('focused_publisher_v6', SOURCE)
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.exp = Path(self.temporary.name)
        self.job = copy.deepcopy(publisher.JOBS[0])
        self.job.pop('user_feedback', None)
        self.job.update(id='MS/good/videos', caption='MS/captions.json')
        self.root = self.exp / self.job['id']
        self.root.mkdir(parents=True)
        self.image = self.exp / 'image.png'
        self.image.write_bytes(b'fixture image')
        self.code = self.exp / 'source.py'
        self.code.write_bytes(b'fixture source')
        self.cache = self.exp / 'cache.pt'
        self.cache.write_bytes(b'fixture condition')
        self.caption = self.exp / self.job['caption']
        save(self.caption, dict(records=[dict(id=sid, motion_caption='raw '+sid, caption_version='v1', image_path=str(self.image)) for sid in publisher.IDS]))
        self.file_record = lambda path: dict(path=str(path), sha256=publisher.sha(path))
        run = dict(mode='video_generation', steps=50, seed=42, cfg=1,
                   geometry=dict(height=1280, width=704, num_frames=124, fps=24),
                   model_sources=dict(audio_vae_loaded=False, files=[dict(path='remote actual frozen weight', sha256='a'*64)]),
                   source_code=self.file_record(self.code), wrapper_source=self.file_record(self.code),
                   caption_source=self.file_record(self.caption),
                   lora_source=dict(file=dict(path='/remote/step100.safetensors', sha256=publisher.PURE_SHA), keys=list(range(416))))
        save(self.root / 'run_started.json', run)
        save(self.root.parent / 'weight_verified.json', dict(lora='/remote/step100.safetensors', sha256=publisher.PURE_SHA, actual_training_updates=100, tensor_keys=416))
        for sid in publisher.IDS:
            path = self.root / (sid+'.mp4')
            path.write_bytes(('synthetic fixture, not a video: '+sid).encode())
            packed = dict(audio_pos_numel=0, tag2_count=0)
            audit = dict(complete=True, id=sid, protocol='h3_caption_image_infer_v1', mode='video_generation',
                         video=self.file_record(path),
                         media=dict(verified=True, complete_decode=True, frames=124, width=704, height=1280, fps=24, video_streams=1, audio_streams=0),
                         sampling=dict(video_scheduler_updates=50, target_audio_generated=False, packed=packed), packed=packed,
                         audio_guard=dict(calls=dict(audio_patch=0, audio_head=0, audio_reference_encode=0), audio_vae_is_none=True, scheduler_audio_is_none=True),
                         sources=dict(caption_json=self.file_record(self.caption), image=self.file_record(self.image)),
                         condition_cache=self.file_record(self.cache), motion_caption='raw '+sid, caption_version='v1')
            save(self.root / (sid+'.audit.json'), audit)
        self.manifest = dict(exp=str(self.exp), built='original', samples=['GT'], conds=['joint'], scope={'fusion_review':'keep'},
                             presets=[dict(name='keep preset', models=['GT'])],
                             models=[dict(id=self.job['id'], metrics={'motion':7}, quality_accepted=True,
                                          human_review_status='user reviewed', feedback={'comment':'更大幅度'},
                                          ratings=[4,5], unknown_user_field={'keep':True},
                                          conds={'joint':{}, 'aswap':{'old':['old.mp4']}}),
                                     dict(id='GT', desc='keep GT', metrics={'gt':1})])

    def mutate_audit(self, sid, change):
        path = self.root / (sid+'.audit.json')
        value = json.loads(path.read_text())
        change(value)
        save(path, value)

    def collect(self, jobs=None):
        return publisher.collect(jobs or [self.job], self.exp)

    def test_good_fixture_three_cases(self):
        updates, _, pending = self.collect()
        self.assertEqual(len(updates[self.job['id']]), 3)
        self.assertEqual(pending[self.job['id']], {})

    def test_bad_group_and_case_do_not_block_good(self):
        bad = dict(self.job, id='MS/missing/videos')
        self.mutate_audit(publisher.IDS[0], lambda a:a['audio_guard']['calls'].update(audio_head=1))
        updates, _, pending = self.collect([bad, self.job])
        self.assertEqual(updates[bad['id']], {})
        self.assertEqual(len(updates[self.job['id']]), 2)
        self.assertIn(publisher.IDS[0], pending[self.job['id']])

    def test_source_caption_body_swap_rejected(self):
        self.mutate_audit(publisher.IDS[0], lambda a:a.update(motion_caption='swapped'))
        updates, _, _ = self.collect()
        self.assertNotIn(publisher.IDS[0], updates[self.job['id']])

    def test_source_caption_path_swap_rejected(self):
        other = self.exp / 'other.json'
        other.write_bytes(self.caption.read_bytes())
        self.mutate_audit(publisher.IDS[0], lambda a:a['sources'].update(caption_json=self.file_record(other)))
        updates, _, _ = self.collect()
        self.assertNotIn(publisher.IDS[0], updates[self.job['id']])

    def test_lora_swap_rejected(self):
        path = self.root.parent / 'weight_verified.json'
        value = json.loads(path.read_text())
        value['sha256'] = 'b'*64
        save(path, value)
        updates, _, _ = self.collect()
        self.assertEqual(updates[self.job['id']], {})

    def test_video_changed_rejected(self):
        (self.root / (publisher.IDS[0]+'.mp4')).write_bytes(b'changed after audit')
        updates, _, _ = self.collect()
        self.assertNotIn(publisher.IDS[0], updates[self.job['id']])

    def test_feedback_metrics_presets_and_gt_preserved(self):
        updates, provenance, _ = self.collect()
        result = publisher.merge(self.manifest, updates, provenance, [self.job])
        old = self.manifest['models'][0]
        new = next(r for r in result['models'] if r['id'] == self.job['id'])
        for key in ['metrics','quality_accepted','human_review_status','feedback','ratings','unknown_user_field']:
            self.assertEqual(new[key], old[key])
        self.assertEqual(new['conds']['aswap'], old['conds']['aswap'])
        self.assertEqual(result['models'][1], self.manifest['models'][1])
        self.assertEqual(result['scope'], self.manifest['scope'])
        self.assertIn(self.manifest['presets'][0], result['presets'])
        self.assertEqual(self.manifest['models'][0]['conds']['joint'], {})

    def test_partial_json_is_pending_then_recovers(self):
        path = self.exp / 'manifest.json'
        path.write_text('{"models":')
        result = publisher.publish_once(path, [self.job], self.exp)
        self.assertTrue(result['manifest_pending'])
        self.assertFalse(result['complete'])
        self.assertEqual(path.read_text(), '{"models":')
        save(path, self.manifest)
        result = publisher.publish_once(path, [self.job], self.exp)
        self.assertTrue(result['complete'])
        self.assertTrue(result['manifest_updated'])
        entry = json.loads(path.read_text())['models'][0]
        self.assertEqual(entry['metrics'], {'motion':7})
        self.assertTrue(entry['quality_accepted'])

    def test_no_placeholder_registration_or_fake_finished(self):
        bad = dict(self.job, id='MS/absent/videos')
        path = self.exp / 'manifest.json'
        save(path, self.manifest)
        result = publisher.publish_once(path, [self.job, bad], self.exp)
        self.assertFalse(result['complete'])
        self.assertNotIn(bad['id'], [r['id'] for r in json.loads(path.read_text())['models']])
        self.assertFalse(result['numeric_reviews_written'])

    def test_symlink_alias_rejected(self):
        alias = self.exp / 'MS/alias'
        alias.symlink_to(self.root.parent, target_is_directory=True)
        job = dict(self.job, id='MS/alias/videos')
        updates, _, _ = self.collect([job])
        self.assertEqual(updates[job['id']], {})

    def test_anchor_flat_independent_receipt_and_swapped_source(self):
        job = dict(self.job, kind='anchor', training_report='MS/independent/full_state_verified.json')
        lora_path = '/remote/anchor-step100.safetensors'
        weight_path = self.root.parent / 'weight_verified.json'
        weight = json.loads(weight_path.read_text())
        weight.update(lora=lora_path, original_lora='/original/anchor-step100.safetensors', sha256='f'*64, full_ab_source_verified=True)
        save(weight_path, weight)
        run_path = self.root / 'run_started.json'
        run = json.loads(run_path.read_text())
        run['lora_source']['file'] = dict(path=lora_path, sha256='f'*64)
        run['lora_source'].update(alpha=1.0, scale_policy='B already contains training alpha/r; no second multiplication')
        runtime = dict(source=str(SOURCE.parent.parent / 'h3_video_infer_chunked_lora_v1.py'), source_sha256='c611d16c26c38e918d4c91560f7fa72e6e88ad93b43ecad1b48fd80a91deb665', full_sequence_retained=True, weights_unchanged=True, original_inference_sha256=run['source_code']['sha256'])
        run['runtime_lora_implementation'] = runtime
        for sid in publisher.IDS:
            self.mutate_audit(sid, lambda a: a.update(runtime_lora_implementation=runtime))
        save(run_path, run)
        receipt = dict(complete=True, successful_updates=100, actual_training_updates=100, rng_rank_count=8,
                       optimizer_shards=['nested/adam'+str(i) for i in range(8)],
                       model_shards=['nested/model'+str(i) for i in range(8)], marker_sha256='a'*64,
                       lora_files=[weight['original_lora']], lora=weight['original_lora'], lora_sha256='f'*64,
                       full_checkpoint_reverified=True, independent_audit=True, typed_numeric_verified=True,
                       lora_all_ab_source_verified=True, model_and_adam_and_rng_and_scheduler_finite=True,
                       all_file_sha256_verified=True, fm_recipe_verified=True, complete_state_members=28,
                       anchor_lambda=0.1, world_size=8, lora_rank=32, lora_alpha=64, learning_rate=1e-5, training_cfg_scale=1.0)
        source = self.exp / job['training_report']
        save(source, receipt)
        job['training_report_sha256'] = publisher.sha(source)
        save(source.parent / 'finished.json', dict(complete=True, full_checkpoint_reverified=True, report=str(source), lora_sha256='f'*64))
        local = self.root.parent / 'training_verified_receipt.json'
        save(local, receipt)
        updates, _, pending = self.collect([job])
        self.assertEqual(len(updates[job['id']]), 3, pending)
        run['lora_source']['alpha'] = 2.0
        save(run_path, run)
        updates, _, pending = self.collect([job])
        self.assertEqual(updates[job['id']], {}, pending)
        run['lora_source']['alpha'] = 1.0
        save(run_path, run)
        receipt['marker_sha256'] = 'b'*64
        save(local, receipt)
        updates, _, pending = self.collect([job])
        self.assertEqual(updates[job['id']], {}, pending)
        save(source, receipt)
        updates, _, pending = self.collect([job])
        self.assertEqual(updates[job['id']], {}, pending)

    def native_fixture(self):
        job = copy.deepcopy(publisher.JOBS[-1])
        job.update(id='MS/native', receipt='MS/native_queue/final_cpu_checkpoint.json')
        root = self.exp / job['id']
        receipt = dict(complete=True, steps=2250, exposures=18000, optimizer_states=402, state_tensors=467,
                       actual_three_unique_epochs=True, full_saved_Adam_RNG_sampler_pointer=True,
                       strict_loaded_complete_MusicFM=True, checkpoint_sha256=publisher.DIRECT_SHA,
                       local_stage_actual_sha256=publisher.DIRECT_SHA)
        original = self.exp / 'MS/music_direct_full6k_final2250_video_n4_queue_v1/final_cpu_checkpoint.json'
        save(original, receipt)
        receipt.update(original_receipt=str(original), original_receipt_sha256=publisher.sha(original))
        save(self.exp / job['receipt'], receipt)
        for sid in publisher.IDS:
            path = root / 'joint' / (sid+'_'+sid+'.mp4')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('synthetic native fixture, not real media: '+sid).encode())
            audit_path = path.with_suffix('.audit.json')
            case = dict(sample_id=sid, condition='joint', audio_path=str(self.cache), audio_sha256=publisher.sha(self.cache),
                        image_path=str(self.image), image_sha256=publisher.sha(self.image),
                        prefix_path=str(self.cache), prefix_file_sha256=publisher.sha(self.cache),
                        prefix_length=7437, prefix_tags_sha256='b'*64, prefix_sha256='c'*64)
            inputs = dict(case=case, student_reads_target_or_caption=False, teacher_identity=False, student_zero_input=False,
                          semantic_text=None, output_audio='original_mux', native_waveform_sha256='d'*64,
                          checkpoint_sha256=publisher.DIRECT_SHA)
            calls = dict(scheduler_updates=dict(video=50,audio=50), native_audio_condition='real_reference_audio', music_mode='ref',
                         original_joint_pipeline=True, full_cached_prefix_preserved=True, original_cached_prefix_tags_preserved=True,
                         packed_text_tags_exact=True, text_encoder_calls=0, text_encoder_loaded=False,
                         tokenizer_loaded=False, processor_loaded=False, waveform_sha256='d'*64, prefix_tokens=7437)
            save(audit_path, dict(complete=True,input=inputs,audit=calls,output_path=str(path),video_sha256=publisher.sha(path),mux_audio_sha256='d'*64))
            save(root / (sid+'.validated.json'), dict(sample_id=sid,path=str(path),full_media_decode_checked=True,
                                                    full_video_audio_decode=True,ffmpeg_decoded_video_frames=124,
                                                    streams=[dict(codec_type='video',width=704,height=1280,decoded_frames=124,fps=24),dict(codec_type='audio')],
                                                    sha256=publisher.sha(path),audit_path=str(audit_path)))
        return job, root

    def test_native_good_and_TC_or_checkpoint_swap_rejected(self):
        job, root = self.native_fixture()
        updates, _, pending = self.collect([job])
        self.assertEqual(len(updates[job['id']]), 3, pending)
        path = root / 'joint' / (publisher.IDS[0]+'_'+publisher.IDS[0]+'.audit.json')
        audit = json.loads(path.read_text())
        audit['audit']['text_encoder_calls'] = 1
        save(path, audit)
        path = root / 'joint' / (publisher.IDS[1]+'_'+publisher.IDS[1]+'.audit.json')
        audit = json.loads(path.read_text())
        audit['input']['checkpoint_sha256'] = 'e'*64
        save(path, audit)
        updates, _, pending = self.collect([job, self.job])
        self.assertEqual(len(updates[job['id']]), 1, pending)
        self.assertEqual(len(updates[self.job['id']]), 3)

    def test_real_existing_nine_video_sources(self):
        if not (publisher.EXP / publisher.JOBS[0]['id']).exists():
            self.skipTest('local authentic eval corpus unavailable')
        updates, _, pending = publisher.collect(publisher.JOBS[:3])
        self.assertEqual([len(v) for v in updates.values()], [3,3,3], pending)


if __name__ == '__main__':
    unittest.main()
