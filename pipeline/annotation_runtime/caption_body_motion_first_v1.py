"""Body-first evidence trial; same body pixels, hand evidence retained separately."""
import copy
import hashlib
import json
import caption_compact_visual_v1 as native
from caption_motion_grounded_request_v1 import digest
RECIPE='caption_body_motion_first_v1'
CONTRACT='''Read the ordered whole-body RGB sequence as a continuous motion observation.
Describe the main torso, arm and leg CHANGES between the supplied frames.
Use short, one-sentence events, one coherent observed phase per event. Do not
replace movement with lists of held poses. Observe intermediate frames before
claiming repetitions or continuity; do not infer an unobserved path or onset.
This is the BODY-motion stage. Do not describe finger shapes, palm orientation,
hand opening, gaze, clothing, contact, balance or weight bearing in body events.
Native hand-crop evidence is retained for a separate fine-hand stage, not supplied
in this request. Missing fine detail must NEVER suppress a visible body change.
Use neutral one arm/the other arm or one leg/the other leg when anatomical side
is not independently clear from body context. Do not map screen sides to anatomy.
Prefer supported coarse movement over an uncertain side-specific assertion.
Each event must name ONLY its necessary body parts and cite at least TWO DISTINCT
supplied actual global frames showing that change. part_support must list the
actual visible body frames for each part, not just the ROI inventory or one pose.
Cited event frames must intersect each required part's support in at least two
frames and must not be forbidden by the raw masks. Do not omit visible torso,
arm or leg support just because fingers, feet or other parts are unclear.
All ten native support slots remain required. Hand/foot support refers only to
visible coarse location; it never establishes finger articulation or contact.
Do not include fine hand or foot claims in body events; unknown stays unknown.
Descriptions contain movement only, no frame numbers or visibility narratives,
no 'throughout', 'steady', 'minimal change', 'suggesting' or invented persistence.
Put genuine uncertainty only in uncertainties. Return the unchanged native JSON
schema events, part_support, uncertainties, abstain. Empty events is evidence
abstention, not stillness. Do not invent actions or event counts to fill windows.
'''
SYSTEM=CONTRACT+'\nFixed part_support order: '+', '.join(native.native().PARTS)+'.\n'

def prepare_request(request,manifest):
    native.compact_request(request,manifest)
    content=request['messages'][1]['content']
    header=json.loads(content[0]['text'])
    body=[copy.deepcopy(content[0])]
    at=1
    frame_ids=[]
    while at<len(content):
        text,image=content[at:at+2]
        native.need(text['type']=='text' and image['type']=='image_url','Original text/image pair required')
        if text['text'].startswith('Source frame '):
            frame_ids.append(int(text['text'].split(',')[0].split()[-1]))
            body.extend(copy.deepcopy([text,image]))
        else:
            meta=json.loads(text['text'])
            native.need(meta.get('same_source_frame_as_body_context') is True and meta.get('not_another_time_observation') is True,
                'Only native same-frame ROI may be deferred')
        at+=2
    native.need(frame_ids==[f['index'] for f in manifest['frames']],'All original body frames must remain ordered')
    header['hand_ROI_support']={}
    header.update(body_motion_stage=True,fine_hand_stage_pending=True)
    body[0]['text']=json.dumps(header,ensure_ascii=False,allow_nan=False)
    result=copy.deepcopy(request)
    result['messages'][0]['content']=SYSTEM
    result['messages'][1]['content']=body
    result['prompt_sha256']=hashlib.sha256(SYSTEM.encode()).hexdigest()
    identity=dict(recipe=RECIPE,source_request_sha256=digest(request),source_manifest_sha256=digest(manifest),
        prompt_sha256=result['prompt_sha256'],deferred_same_frame_ROI_count=(len(content)-len(body))//2)
    result.update(caption_recipe=RECIPE,generation_recipe=RECIPE,fact_source_recipe=RECIPE,
        response_protocol_recipe=native.native().OUTPUT_SCHEMA,recipe_identity=identity,
        recipe_identity_sha256=digest(identity),native_visual_model_protocol=False,
        body_motion_only=True,fine_hand_evidence_deferred=copy.deepcopy(manifest.get('hand_roi_evidence',{})),
        semantic_approval=False,quality_approved=False,training_eligible=False)
    return result
