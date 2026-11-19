"""CPU-only new phase prompt; no dispatch, old-text splitting or support-gate changes."""
import copy
import hashlib

import caption_compact_visual_v1 as compact
from caption_motion_grounded_request_v1 import digest

RECIPE = 'caption_motion_phase_request_v1'
CONTRACT = """
SINGLE-SENTENCE MOTION-PHASE CONTRACT (same native JSON schema/support gates):
Each event description is exactly ONE complete sentence describing ONE
observed motion phase, normally no more than 40 English words. Describe the
actual torso/arm/leg change first, using concrete movement verbs and adjacent
supplied BODY-context frames to ground the transition. Reliable combined
body movements within the SAME phase may share that sentence. Preserve
minor changes and visible repetitions; never replace movement with static
pose lists or invent stationary actions because details are unknown.
Separate successive phases into independently evidenced events only when
actually observed; two or three real phases may use two or three events.
Never force that count, invent a phase or infer an unobserved transition.
Claim synchrony/order only when the supplied body-frame sequence supports it.
Do NOT append hands opening, palms, fingers, foot contact, gaze or visibility
explanations to a torso/arm/leg phase sentence. Reliable fine-hand shapes
belong in a SEPARATE event, using only their own at least TWO DISTINCT
source frames with distinguishable fingers/shapes and minimal necessary
supported parts. Preserve reliable fine-hand detail; do not blanket-drop it.
If cited frames cannot resolve hand detail, record it only in uncertainties;
never declare that same detail unknown while asserting a definite hand shape.
V-shaped forearms are NOT finger shapes. Same-frame ROIs add no temporal
support; ROI labels and screen sides are NOT anatomical-side ground truth.
When anatomical side is uncertain, use neutral one arm/another arm or one
side/the other side, NOT guessed left/right. Use anatomical left/right only
when clear in at least two distinct BODY-context frames. The original PARTS
metadata still requires independently supported enum entries; neutral prose
never authorizes invented sides, parts or missing-part repair. Each required
part must keep its OWN two distinct cited reliable global evidence frames.
Do NOT infer persistence from matching endpoints or missing frames: avoid
always/throughout/steady/minimal change as substitutes for observed phases.
Descriptions contain motion only, no frame numbers, blur, visibility/support
notes or 'suggesting' uncertainty claims; use frame_indices/uncertainties.
Format illustration ONLY, not video/action prior: one BODY_PHASE_SENTENCE
event and, only if supported, a separate FINE_HAND_SENTENCE event. These are
placeholders, NOT descriptions to emit, motions to invent or mandatory events.
All original schema/frame ordering/mask/half-open-window/abstention gates
remain unchanged. This is NEW prompt generation, never automatic splitting
of old raw text, deletion of old facts, part filling or quality approval.
Do not reuse or reset any old generation attempt/call budget. New generation
still requires independently authorized execution and pixel-quality review.
"""


def prepare_request(request, manifest):
    """Return a new identity; content refs are digests, not invented artifact paths."""
    keys = {'model', 'messages', 'sampling', 'json_schema', 'prompt_sha256'}
    compact.need(isinstance(request, dict) and set(request) == keys,
                 'original native five-field request required; default deny previous recipes')
    messages = request['messages']
    compact.need(isinstance(messages, list) and len(messages) == 2 and
        all(isinstance(m, dict) and set(m) == {'role', 'content'} for m in messages) and
        messages[0]['role'] == 'system' and messages[1]['role'] == 'user' and
        isinstance(messages[0]['content'], str) and isinstance(messages[1]['content'], list),
        'original native message layout required')
    compact.compact_request(request, manifest)  # Original schema/system/SHA/1800 gate only.
    result = copy.deepcopy(request)
    system = request['messages'][0]['content'] + CONTRACT
    prompt_sha = hashlib.sha256(system.encode()).hexdigest()
    result['messages'][0]['content'] = system
    identity = dict(recipe=RECIPE, response_protocol_recipe=compact.native().OUTPUT_SCHEMA,
        source_request_ref={'sha256': digest(request)}, source_window_manifest_ref={'sha256': digest(manifest)},
        original_prompt_ref={'sha256': request['prompt_sha256']},
        response_schema_ref={'sha256': digest(request['json_schema'])}, new_system_prompt_ref={'sha256': prompt_sha})
    result.update(prompt_sha256=prompt_sha, caption_recipe=RECIPE, generation_recipe=RECIPE,
        fact_source_recipe=RECIPE, response_protocol_recipe=compact.native().OUTPUT_SCHEMA,
        native_visual_model_protocol=False, recipe_identity=identity, recipe_identity_sha256=digest(identity),
        semantic_approval=False, human_review_completed=False, quality_approved=False, training_eligible=False)
    return result
