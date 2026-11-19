"""CPU-only video frame/PTS/ROI conservation and frozen native support tests."""
import base64
import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

RT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RT))
import caption_native_video_request_v1 as p


class Tokenizer:
    chat_template = 'CPU Qwen3-compatible placeholder test double'

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return messages[0]['content'] + ''.join(c.get('text', p.VIDEO_PLACEHOLDER if c['type'] == 'video'
            else p.IMAGE_PLACEHOLDER) for c in messages[1]['content'])


def jpeg(size, color):
    out = BytesIO(); Image.new('RGB', size, color).save(out, format='JPEG')
    raw = out.getvalue()
    return raw, dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,' + base64.b64encode(raw).decode('ascii')))


class NativeVideoRequestTests(unittest.TestCase):
    def setUp(self):
        self.bar = p.compact.native()
        self.binding = dict(video_fps=16.0, decoded_video_frames=81, video_pts_offset=0., media_end=5.0625)
        self.forbidden = dict(left_hand=[4], torso=[100])
        self.manifest = dict(window=dict(window_id='w0', start=0., end=2.), frames=[],
            hand_roi_evidence={part: dict(crops=[], frame_indices=[0, 12, 28]) for part in ('left_hand', 'right_hand')})
        self.body_raw, self.roi_raw = [], []
        images = []
        for ordinal, index in enumerate(range(0, 32, 4)):
            raw, item = jpeg((352, 640), (index + 10, 30, 40))
            f = dict(index=index, time=index / 16, source_timestamp=index / 16,
                     sha256=hashlib.sha256(raw).hexdigest(), file=str(index) + '.jpg')
            self.manifest['frames'].append(f); self.body_raw.append(raw)
            images.extend([dict(type='text', text=f"Source frame {index}, time {f['time']:.9f}s, original PTS {f['source_timestamp']:.9f}s."), item])
            if ordinal not in (0, 3, 7): continue
            for part in ('left_hand', 'right_hand'):
                raw, item = jpeg((64, 80), (40 + index, 50 if part == 'left_hand' else 100, 60))
                crop = dict(part=part, frame_index=index, time=index / 16, source_timestamp=index / 16,
                    sha256=hashlib.sha256(raw).hexdigest(), crop_xyxy=[0, 0, 64, 80], source_frame_size=[704, 1280],
                    crop_pixel_size=[64, 80], ROI_to_source_affine=[[1, 0, 0], [0, 1, 0]],
                    source_to_ROI_affine=[[1, 0, 0], [0, 1, 0]], supportpointindices=[92, 93],
                    hand_support_xyxy=[0, 0, 64, 80], rawframe_sha256='original-full-size-raw-frame')
                self.manifest['hand_roi_evidence'][part]['crops'].append(crop)
                meta = {key: crop[key] for key in ('part', 'frame_index', 'time', 'source_timestamp', 'crop_xyxy',
                    'source_frame_size', 'crop_pixel_size', 'ROI_to_source_affine', 'source_to_ROI_affine',
                    'supportpointindices', 'hand_support_xyxy', 'rawframe_sha256')}
                meta.update(same_source_frame_as_body_context=True, not_another_time_observation=True,
                            crop_adds_native_detail=False, anatomical_side_is_unreviewed=True)
                images.extend([dict(type='text', text=json.dumps(meta)), item]); self.roi_raw.append(raw)
        indices = [f['index'] for f in self.manifest['frames']]
        header = dict(allowed_source_frame_indices=indices, parts=list(self.bar.PARTS),
                      explicit_raw2d_forbidden_part_frames=dict(left_hand=[4]),
                      ROI_same_frame_is_not_extra_temporal_support=True)
        self.request = dict(model='/existing/model', messages=[dict(role='system', content=self.bar.SYSTEM),
            dict(role='user', content=[dict(type='text', text=json.dumps(header))] + images)],
            sampling=dict(temperature=0., seed=0, max_tokens=1800), json_schema=self.bar.observation_schema(indices),
            prompt_sha256=hashlib.sha256(self.bar.SYSTEM.encode()).hexdigest())

    def prepare(self, request=None, manifest=None, binding=None, forbidden=None, tokenizer=None):
        return p.prepare_request(self.request if request is None else request,
            self.manifest if manifest is None else manifest, self.binding if binding is None else binding,
            self.forbidden if forbidden is None else forbidden, tokenizer or Tokenizer())

    def test_eight_video_frames_six_original_roi_bytes_rgb_pts_and_labels_conserved(self):
        before = copy.deepcopy((self.request, self.manifest, self.binding, self.forbidden))
        original_open = p.Image.open
        with patch.object(p.Image, 'open', wraps=original_open) as opened:
            result = self.prepare()
        self.assertEqual(opened.call_count, 14)
        self.assertEqual(before, (self.request, self.manifest, self.binding, self.forbidden))
        inp = result['generation_input']; self.assertEqual(inp['prompt'].count(p.VIDEO_PLACEHOLDER), 1)
        self.assertEqual(inp['prompt'].count(p.IMAGE_PLACEHOLDER), 6)
        array, meta = inp['multi_modal_data']['video'][0]
        self.assertEqual(array.shape, (8, 640, 352, 3)); self.assertEqual(array.dtype, np.uint8)
        self.assertEqual(meta['frames_indices'], list(range(0, 32, 4))); self.assertEqual(meta['fps'], 16.)
        self.assertEqual(meta['total_num_frames'], 81); self.assertEqual(meta['duration'], 5.0625)
        self.assertFalse(meta['do_sample_frames']); self.assertNotIn('timestamps', meta)
        self.assertEqual(result['source_jpeg_bytes']['body'], tuple(self.body_raw))
        self.assertEqual(result['source_jpeg_bytes']['ROI'], tuple(self.roi_raw))
        for i, raw in enumerate(self.body_raw):
            np.testing.assert_array_equal(array[i], np.asarray(Image.open(BytesIO(raw)).convert('RGB')))
        for rgb, raw in zip(inp['multi_modal_data']['image'], self.roi_raw):
            np.testing.assert_array_equal(rgb, np.asarray(Image.open(BytesIO(raw)).convert('RGB')))
        original_text = [c for c in self.request['messages'][1]['content'] if c['type'] == 'text']
        new_text = [c for c in result['messages'][1]['content'] if c['type'] == 'text']
        self.assertEqual(new_text[:1] + new_text[2:], original_text)
        for key in ('sampling', 'json_schema', 'model'): self.assertEqual(result[key], self.request[key])
        self.assertEqual(result['evidence_binding']['manifest'], self.manifest)
        self.assertEqual(result['evidence_binding']['forbidden'], self.forbidden)

    def test_new_protocol_and_separate_runtime_video_budget_not_old_image_only_identity(self):
        result = self.prepare(); identity = result['recipe_identity']
        self.assertEqual(result['recipe_identity_sha256'], p.digest(identity))
        self.assertEqual(identity['source_request_sha256'], p.digest(self.request))
        self.assertEqual(result['response_protocol_recipe'], p.RESPONSE_PROTOCOL)
        self.assertEqual(result['response_schema_recipe'], self.bar.OUTPUT_SCHEMA)
        for key in ('caption_recipe', 'generation_recipe', 'fact_source_recipe'): self.assertEqual(result[key], p.RECIPE)
        for key in ('quality_approved', 'semantic_approval', 'human_review_completed', 'training_eligible',
                    'native_visual_model_protocol'): self.assertFalse(result[key])
        cfg = p.runtime_requirements(); self.assertEqual(cfg['tensor_parallel_size'], 2)
        self.assertEqual(cfg['limit_mm_per_prompt'], dict(video=1, image=6))
        kw = cfg['mm_processor_kwargs']; self.assertEqual(kw['images_kwargs']['max_pixels'], 200704)
        self.assertEqual(kw['videos_kwargs']['size']['longest_edge'], 8 * 200704)
        for phrase in ('ONE observed torso/arm/leg phase', '40 English words', 'SEPARATE event with their OWN support',
                       'neutral anatomical sides', 'ROI labels are not truth', 'rounded by the model protocol'):
            self.assertIn(phrase, ' '.join(p.CONTRACT.split()))
        self.assertFalse(kw['videos_kwargs']['do_sample_frames'])
        self.assertEqual(result['generation_input']['mm_processor_kwargs'], kw)
        self.assertEqual(self.prepare()['recipe_identity_sha256'], result['recipe_identity_sha256'])
        with self.assertRaises(ValueError): self.prepare(request=result)

    def test_reject_invented_or_nonCFR_timeline_bad_layout_pixels_budget_and_mask(self):
        for kind in ('fps', 'offset', 'count', 'pts', 'halfopen', 'order', 'mask', 'jpeg', 'extra', 'roi', 'budget'):
            request, manifest, binding, forbidden = copy.deepcopy((self.request, self.manifest, self.binding, self.forbidden))
            if kind == 'fps': binding['video_fps'] = 30.
            elif kind == 'offset': binding['video_pts_offset'] = 1.
            elif kind == 'count': binding['decoded_video_frames'] = 20
            elif kind == 'pts': manifest['frames'][2]['source_timestamp'] += .01
            elif kind == 'halfopen': manifest['window']['end'] = 1.
            elif kind == 'order': request['messages'][1]['content'][1]['text'] = 'source frame swapped'
            elif kind == 'mask': forbidden['left_hand'].append(8)
            elif kind == 'jpeg': request['messages'][1]['content'][2]['image_url']['url'] = 'https://not-allowed/video'
            elif kind == 'extra': request['messages'][1]['content'].append(dict(type='video_url'))
            elif kind == 'roi': manifest['hand_roi_evidence']['left_hand']['crops'][0]['frame_index'] = 4
            else: request['sampling']['max_tokens'] = 900
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.prepare(request, manifest, binding, forbidden)
        bad = Tokenizer(); bad.chat_template = None
        with self.assertRaises(ValueError): self.prepare(tokenizer=bad)

    def test_no_hand_fact_deletion_part_repair_or_support_gate_change(self):
        value = dict(events=[dict(id='e1', description='The left fingers spread.', parts=['left_hand'], frame_indices=[0, 12])],
            part_support=[dict(part=part, frame_indices=[0, 12] if part == 'left_hand' else [], reason='original')
                          for part in self.bar.PARTS], uncertainties=['Right hand detail remains unknown.'], abstain=False)
        before = self.bar.filter_supported_observation(value, self.manifest, self.forbidden)
        self.prepare()
        self.assertEqual(before, self.bar.filter_supported_observation(value, self.manifest, self.forbidden))
        self.assertEqual(before['events'][0]['description'], 'The left fingers spread.')
        self.assertEqual(before['uncertainties'], value['uncertainties'])
        self.assertFalse(hasattr(p, 'generate')); self.assertFalse(hasattr(p, 'filter_supported_observation'))


if __name__ == '__main__':
    unittest.main()
