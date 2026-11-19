"""Independent concise motion observations; no old attempt reuse or filter changes."""
import copy
import hashlib
import caption_motion_grounded_request_v1 as original

RECIPE='caption_phase_motion_request_v1'
CONTRACT='''Describe visible CHANGES across the ordered body frames, not a catalog of poses.
Read the frame index/time labels; compare intermediate frames, not only endpoints.
Prioritize the main combined torso/arm/leg movement and real repetitions. A raised
knee becoming a side extension is movement, not a held standing pose. Do not infer
balance, contact, weight bearing, or exact onset from a still image.
Each event describes ONE coherent movement phase with TWO or more distinct source
frames supporting that change. Include every required part in event.parts, and
those cited frames must intersect that part's own support in at least two frames.
When finer detail is unreliable, keep supported coarse motion. Never attach a
fine hand-shape assertion to a reliable leg/torso event: put hand detail in a
SEPARATE event only if the actual fingers/outline are clear in two source frames.
Do not pad a motion event with static head/torso, open-hand or stable-pose claims.
Use the dancer as subject. Prefer side-neutral wording unless anatomical side is
clear from two body-context frames; screen side and ROI names are not ground truth.
Do not invent or swap anatomical support slots. Coarse arm motion is not hand
shape support. Same-frame ROI and body images count as ONE temporal observation.
Do not claim steady, held, throughout, repeats or a motion path unless the supplied
intermediate frames establish it. Keep unresolved details in uncertainties, never
in descriptions. Descriptions contain actions, not visibility or frame commentary.
No music/style/appearance/background claims. Raw forbidden part/frame pairs must
never support events. Unknown/cropped/occluded parts stay unknown, NOT stillness.
Return the unchanged native JSON schema: events(id,description,parts,frame_indices),
exactly ten ordered part_support(part,frame_indices,reason), uncertainties, abstain.
Support reason is a short tag; empty indices mean unknown. If no supported motion
remains, events=[] and abstain=true. Preserve all supported movement, even if other
parts or fine detail are unknown. This recipe is an unreviewed AI observation.
'''

SYSTEM=CONTRACT+'\nFixed part_support order: '+', '.join(original.compact.native().PARTS)+'.\n'

def prepare_request(request,manifest):
    result=original.prepare_request(request,manifest)
    result['messages'][0]['content']=SYSTEM
    result['prompt_sha256']=hashlib.sha256(result['messages'][0]['content'].encode()).hexdigest()
    for field in ('caption_recipe','generation_recipe','fact_source_recipe'):
        result[field]=RECIPE
    identity=dict(recipe=RECIPE, source_request_sha256=original.digest(request),
        prompt_sha256=result['prompt_sha256'],response_protocol_recipe=result['response_protocol_recipe'])
    result['recipe_identity']=identity
    result['recipe_identity_sha256']=original.digest(identity)
    return result
