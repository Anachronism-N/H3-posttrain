"""CPU-only eight-original-JPEG native video recipe; no runtime or dispatch."""
import base64
import copy
import hashlib
from io import BytesIO
import json
import math

import numpy as np
from PIL import Image

import caption_compact_visual_v1 as compact
from caption_motion_grounded_request_v1 import digest

RECIPE = 'caption_native_video_request_v1'
RESPONSE_PROTOCOL = 'caption_native_video_response_v1'
VIDEO_PLACEHOLDER = '<|vision_start|><|video_pad|><|vision_end|>'
IMAGE_PLACEHOLDER = '<|vision_start|><|image_pad|><|vision_end|>'
BODY_COUNT, ROI_LIMIT, MAX_PIXELS = 8, 6, 200704
CONTRACT = """
NATIVE VIDEO INPUT CONTRACT (new generation, unchanged native JSON schema):
The single video is ONLY the eight supplied original BODY-context JPEGs in
original chronological order, not the full continuous source video. Video
ordinal 0..7 maps to the original global indices and true source timestamps
listed below; cite those global indices, never ordinal or merged-patch indices.
The processor combines adjacent frame pairs and uses their mean timestamps;
this supplies no new frames, transitions or evidence. Timestamp display is
rounded by the model protocol; the original PTS labels remain authoritative.
Every hand ROI retains its original same-frame metadata and pixels. A same-
frame crop is NOT another temporal observation or anatomical-side ground truth.
Each body event is one complete short motion-only sentence describing ONE
observed torso/arm/leg phase, normally at most 40 English words. Keep supported
combined body movements in the same phase, not successive invented phases.
Reliable fine-hand gestures belong in a SEPARATE event with their OWN support,
not mixed into a body event where uncertain hand metadata can discard motion.
Do not substitute static poses, infer persistence from endpoints or erase
supported motion because fine detail is unknown. Use neutral anatomical sides
unless independently reliable in two BODY frames; ROI labels are not truth.
Preserve genuinely supported finger/hand shapes; uncertain detail stays in
uncertainties, never an invented definite shape. Each required part needs its OWN two distinct
reliable cited global frames; all ten support rows, forbidden masks, half-open
windows, uncertainty and abstention rules remain unchanged. This input recipe
is not quality approval, an old-attempt retry or reuse of the image-only budget.
"""


def runtime_requirements():
    """Caller must separately admit this new TP2 runtime; never reuse video=0."""
    return dict(vllm_version='0.31.0', tensor_parallel_size=2,
        limit_mm_per_prompt=dict(video=1, image=ROI_LIMIT),
        mm_processor_kwargs=dict(images_kwargs=dict(max_pixels=MAX_PIXELS),
            videos_kwargs=dict(do_sample_frames=False, cap_pixels_per_frame=False,
                size=dict(shortest_edge=4096, longest_edge=BODY_COUNT * MAX_PIXELS))))


def prepare_request(request, manifest, source_binding, forbidden, tokenizer):
    """Return LLM.generate input plus serializable identity and original JPEG bytes.

    Only CFR PTS exactly representable by source indices / true FPS are allowed.
    The installed Qwen3VL VideoMetadata has no arbitrary PTS/timestamps input.
    Tokenizer is caller-owned; this module never loads a tokenizer/model or files.
    """
    need = compact.need
    compact.compact_request(request, manifest)  # Frozen system/schema/1800 gates.
    frames = manifest['frames']
    indices = [f['index'] for f in frames]
    need(len(frames) == BODY_COUNT and all(type(i) is int for i in indices)
         and indices == sorted(set(indices)), 'exactly eight chronological distinct original frames required')
    fps, total = source_binding['video_fps'], source_binding['decoded_video_frames']
    need(type(fps) in (float, int) and math.isfinite(fps) and fps > 0
         and type(total) is int and total > max(indices) and min(indices) >= 0,
         'actual source FPS and total decoded frame count required')
    need(source_binding['video_pts_offset'] == 0, 'nonzero original PTS offset not supported by native API')
    window = manifest['window']
    for f in frames:
        need(math.isfinite(f['source_timestamp']) and math.isfinite(f['time'])
             and abs(f['source_timestamp'] - f['index'] / fps) <= 1e-9
             and abs(f['time'] - f['source_timestamp']) <= 1e-9,
             'true PTS not representable by original source indices/FPS; no invented timeline')
        need(window['start'] <= f['time'] < window['end'], 'original half-open window violated')
    denied = {part: sorted(set(indices).intersection(values)) for part, values in forbidden.items()
              if set(indices).intersection(values)}
    content = request['messages'][1]['content']
    need(isinstance(content, list) and content and content[0].get('type') == 'text', 'original native header required')
    header = json.loads(content[0]['text'])
    need(header['allowed_source_frame_indices'] == indices and header['parts'] == list(compact.native().PARTS)
         and header['explicit_raw2d_forbidden_part_frames'] == denied, 'original indices/parts/forbidden mask differs')
    evidence = manifest.get('hand_roi_evidence') or {}
    expected_crops = [crop for f in frames for part in ('left_hand', 'right_hand')
                      for crop in evidence.get(part, {}).get('crops', []) if crop['frame_index'] == f['index']]
    need(len(expected_crops) <= ROI_LIMIT, 'all original hand crops exceed newly admitted six-image limit')
    user = [copy.deepcopy(content[0])]
    mapping = [dict(video_ordinal=i, source_global_index=f['index'], time=f['time'],
                    source_timestamp=f['source_timestamp']) for i, f in enumerate(frames)]
    user.append(dict(type='text', text=json.dumps(dict(original_video_frame_mapping=mapping), allow_nan=False)))
    body, rois, body_raw, roi_raw, media, cache = [], [], [], [], [], {}
    cursor, crop_cursor = 1, 0

    def jpeg(item, expected_sha):
        need(isinstance(item, dict) and set(item) == {'type', 'image_url'} and item['type'] == 'image_url'
             and isinstance(item['image_url'], dict) and set(item['image_url']) == {'url'}, 'original JPEG data URL required')
        url = item['image_url']['url']
        need(isinstance(url, str) and url.startswith('data:image/jpeg;base64,'), 'no URL/video decode allowed')
        raw = base64.b64decode(url.split(',', 1)[1], validate=True)
        sha = hashlib.sha256(raw).hexdigest()
        need(sha == expected_sha, 'original JPEG SHA differs')
        if sha not in cache:
            with Image.open(BytesIO(raw)) as image:
                need(image.format == 'JPEG', 'original JPEG format required')
                cache[sha] = np.array(image.convert('RGB'), dtype=np.uint8)
        return raw, cache[sha], sha

    roi_fields = ('part', 'frame_index', 'time', 'source_timestamp', 'crop_xyxy', 'source_frame_size',
                  'crop_pixel_size', 'ROI_to_source_affine', 'source_to_ROI_affine', 'supportpointindices',
                  'hand_support_xyxy', 'rawframe_sha256')
    for ordinal, f in enumerate(frames):
        label = dict(type='text', text=f"Source frame {f['index']}, time {f['time']:.9f}s, original PTS {f['source_timestamp']:.9f}s.")
        need(cursor + 1 < len(content) and content[cursor] == label, 'original body frame label/order differs')
        user.append(copy.deepcopy(label))
        raw, rgb, sha = jpeg(content[cursor + 1], f['sha256'])
        need(rgb.shape == (640, 352, 3), 'original 352x640 body JPEG required; no adapter resizing')
        body_raw.append(raw); body.append(rgb)
        media.append(dict(kind='body', source_global_index=f['index'], sha256=sha, jpeg_bytes=len(raw)))
        if ordinal == 0:
            user.append(dict(type='video'))
        cursor += 2
        for part in ('left_hand', 'right_hand'):
            for crop in evidence.get(part, {}).get('crops', []):
                if crop['frame_index'] != f['index']:
                    continue
                metadata = {key: crop[key] for key in roi_fields}
                metadata.update(same_source_frame_as_body_context=True, not_another_time_observation=True,
                    crop_adds_native_detail=False, anatomical_side_is_unreviewed=True)
                need(cursor + 1 < len(content) and content[cursor].get('type') == 'text'
                     and json.loads(content[cursor]['text']) == metadata, 'original same-frame crop metadata/order differs')
                need(crop == expected_crops[crop_cursor] and crop['part'] == part
                     and crop['time'] == f['time'] and crop['source_timestamp'] == f['source_timestamp'],
                     'original ROI source frame/PTS binding differs')
                raw, rgb, sha = jpeg(content[cursor + 1], crop['sha256'])
                need(list(rgb.shape[1::-1]) == crop['crop_pixel_size'], 'original ROI dimensions differ')
                user.extend([copy.deepcopy(content[cursor]), dict(type='image')])
                roi_raw.append(raw); rois.append(rgb)
                media.append(dict(kind='ROI', part=part, source_global_index=f['index'], sha256=sha, jpeg_bytes=len(raw)))
                cursor += 2; crop_cursor += 1
    need(cursor == len(content) and crop_cursor == len(expected_crops), 'extra, dropped or reordered original images')
    system = request['messages'][0]['content'] + CONTRACT
    messages = [dict(role='system', content=system), dict(role='user', content=user)]
    need(isinstance(tokenizer.chat_template, str), 'actual caller-owned native tokenizer chat template required')
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    need(isinstance(prompt, str) and prompt.count(VIDEO_PLACEHOLDER) == 1
         and prompt.count(IMAGE_PLACEHOLDER) == len(rois), 'native Qwen3 video/image placeholder layout differs')
    metadata = dict(fps=float(fps), total_num_frames=total, duration=source_binding['media_end'],
        width=352, height=640, frames_indices=indices[:], video_backend='predecoded_jpeg', do_sample_frames=False)
    requirements = runtime_requirements()
    mm_data = dict(video=[(np.stack(body), metadata)])
    if rois:
        mm_data['image'] = rois
    identity = dict(recipe=RECIPE, response_protocol_recipe=RESPONSE_PROTOCOL,
        response_schema_recipe=compact.native().OUTPUT_SCHEMA, source_request_sha256=digest(request),
        source_window_manifest_sha256=digest(manifest), source_binding_sha256=digest(source_binding),
        forbidden_sha256=digest(forbidden), original_media=media, video_metadata=copy.deepcopy(metadata),
        runtime_requirements=requirements, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        tokenizer_chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest())
    return dict(model=request['model'], messages=messages, sampling=copy.deepcopy(request['sampling']),
        json_schema=copy.deepcopy(request['json_schema']), prompt_sha256=hashlib.sha256(system.encode()).hexdigest(),
        caption_recipe=RECIPE, generation_recipe=RECIPE, fact_source_recipe=RECIPE,
        response_protocol_recipe=RESPONSE_PROTOCOL, response_schema_recipe=compact.native().OUTPUT_SCHEMA,
        native_visual_model_protocol=False, recipe_identity=identity, recipe_identity_sha256=digest(identity),
        generation_input=dict(prompt=prompt, multi_modal_data=mm_data,
                              mm_processor_kwargs=copy.deepcopy(requirements['mm_processor_kwargs'])),
        source_jpeg_bytes=dict(body=tuple(body_raw), ROI=tuple(roi_raw)),
        evidence_binding=dict(manifest=copy.deepcopy(manifest), forbidden=copy.deepcopy(forbidden),
                              source_binding=copy.deepcopy(source_binding)),
        semantic_approval=False, human_review_completed=False, quality_approved=False, training_eligible=False)
