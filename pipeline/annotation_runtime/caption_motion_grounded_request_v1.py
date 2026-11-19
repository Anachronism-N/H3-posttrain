"""New CPU request recipe only; never dispatch, relabel old facts or alter support gates."""
import copy
import hashlib
import json

import caption_compact_visual_v1 as compact

RECIPE = 'caption_motion_grounded_request_v1'
CONTRACT = """
MOTION-GROUNDED CONTRACT (same schema and all original support gates):
Descriptions express observable continuous motion/change, combined movement,
minor adjustments and visible repetitions, not a string of static poses.
Never insert invented stationary actions or erase supported coarse movement
because finer detail or anatomical side is unknown. Preserve supported motion
and real repeated cycles; do not invent counts or unobserved transitions.
Isolated matching first/last poses do NOT establish whole-window persistence.
Do NOT claim throughout, always, or until frame X from isolated endpoints;
only describe continuity that the intervening supplied body frames establish.
Descriptions contain motion only: no frame references, visibility/support
explanations or uncertainty/evidence notes. Cite evidence in frame_indices;
put unresolved visibility, continuity or side details in uncertainties.
Default to side-neutral one arm, the other arm, or both arms as supported.
Use left/right only when anatomical side is independently clear in at least
TWO DISTINCT original BODY-context frames. Screen left/right is NOT anatomical
left/right. ROI part labels/proposals are NOT anatomical-side ground truth,
and a same-frame ROI is NOT another body frame or temporal observation.
If side is unresolved, retain supported coarse motion with neutral wording
and record the side uncertainty; NEVER invent or swap anatomical parts.
The original PARTS enum, exactly ten support rows and each required part's
own two distinct global evidence frames remain mandatory. Side-neutral text
is NOT permission to fill an uncertain anatomical support slot or substitute
arm support for hand support; preserve only independently supported metadata.
Fine finger/hand shapes require actual fingers/shapes distinguishable in TWO
distinct source frames; a V-shaped forearm is NOT a finger shape. If this
fails, retain supported arm motion and put fine-detail uncertainty in
uncertainties. Empty support means unknown, NOT immobility. All original
mask, half-open window, chronological/global index and abstention gates stay.
This is a new generation/fact-source recipe, NOT proof the prompt is correct
or quality-approved. Never reuse/relabel an old attempt or old admitted facts.
"""


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def prepare_request(request, manifest):
    """Preserve input protocol/pixels/budget; append a separately identified contract."""
    bar = compact.native()
    keys = {'model', 'messages', 'sampling', 'json_schema', 'prompt_sha256'}
    compact.need(isinstance(request, dict), 'original native/compact request required')
    messages = request.get('messages')
    compact.need(isinstance(messages, list) and len(messages) == 2 and
                 all(isinstance(m, dict) and set(m) == {'role', 'content'} for m in messages) and
                 messages[0]['role'] == 'system' and messages[1]['role'] == 'user' and
                 isinstance(messages[0]['content'], str) and isinstance(messages[1]['content'], list),
                 'original request messages required')
    if set(request) == keys:
        compact.compact_request(request, manifest)
        protocol = bar.OUTPUT_SCHEMA
    elif set(request) == keys | {'caption_recipe', 'native_visual_model_protocol'}:
        original = {key: copy.deepcopy(request[key]) for key in keys}
        original['messages'][0] = dict(role='system', content=bar.SYSTEM)
        original['prompt_sha256'] = hashlib.sha256(bar.SYSTEM.encode()).hexdigest()
        original['json_schema'] = bar.observation_schema([f['index'] for f in manifest['frames']])
        expected = compact.compact_request(original, manifest)
        compact.need(request == expected and request['native_visual_model_protocol'] is False,
                     'original compact request/schema/system/identity required')
        protocol = compact.RECIPE
    else:
        raise ValueError('original native/compact request only; no previous recipe/attempt')
    result = copy.deepcopy(request)
    system = request['messages'][0]['content'] + CONTRACT
    result['messages'][0]['content'] = system
    prompt_sha = hashlib.sha256(system.encode()).hexdigest()
    identity = dict(recipe=RECIPE, source_request_sha256=digest(request),
                    prompt_sha256=prompt_sha, response_protocol_recipe=protocol)
    result.update(prompt_sha256=prompt_sha, caption_recipe=RECIPE,
        generation_recipe=RECIPE, fact_source_recipe=RECIPE,
        response_protocol_recipe=protocol, native_visual_model_protocol=False,
        recipe_identity=identity, recipe_identity_sha256=digest(identity),
        semantic_approval=False, human_review_completed=False,
        quality_approved=False, training_eligible=False)
    return result
