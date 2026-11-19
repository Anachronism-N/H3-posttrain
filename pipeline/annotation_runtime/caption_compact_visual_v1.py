"""Independent compact response protocol; CPU-only expansion into frozen support gates."""
import ast
import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

RT = Path(__file__).resolve().parent
SOURCE = RT / 'source_curated_consumer_v1'
RECIPE = 'caption_compact_visual_response_v1'
FROZEN = {
    'ms_caption_rewrite.py': '6052fbb8f799c8a88833b8f564a935dad4562e16b8faf63f3f265143c0c43232',
    'ms_visual_observe.py': 'b83a048d885edb4f6ae4dc41bb5aeaabe24b8646f764bfea9308164423cf2e6d',
    'ms_bar_annotation.py': 'df29139651345ebd4e743657f3553f2e5165c8ab327bfd66ac84f225dc63005e',
}
CONTRACT = """
COMPACT RESPONSE CONTRACT (JSON fields only; all observation rules above remain):
Return exactly events, part_support, uncertainties, abstain. Each event has
exactly description, parts, frame_indices; do NOT output event id. Preserve
all supported combined motion, repeated motion/counts, minor adjustments and
actual distinguishable finger/hand shapes; never compress actions into static
poses. Event array order defines unique e1..e32, not measured temporal order.
part_support has exactly ten rows in the fixed order given below. Each row is
exactly {"frame_indices": [...]}; do NOT output part or reason. Each index is
an original supplied global frame index, distinct and chronological; never
renumber frames or treat a same-frame ROI as another temporal observation.
For every required event part, at least two distinct cited frames must be in
that part's own reliable support. Empty support means unknown/unobservable,
NOT stationary. Keep any necessary visibility/anatomical-side qualifiers and
other uncertainty details in uncertainties; no support-reason prose is needed
for obvious visible/unknown tags. Existing masks, anatomical left/right and
actual two-frame hand-detail requirements remain unchanged. Preserve all
uncertainty text; abstain is true exactly when no supported events remain.
"""


def need(ok, message):
    if not ok:
        raise ValueError(message)


@lru_cache(maxsize=1)
def native():
    """Execute only frozen pure functions, with no model/resource package imports."""
    namespace = dict(copy=copy, re=re, json=json, math=math)
    inventories = (
        ('ms_caption_rewrite.py', {'FLAGS', 'window_bounds', 'neutral_fact_text'}),
        ('ms_visual_observe.py', {'strict_json'}),
        ('ms_bar_annotation.py', {'PARTS', 'OUTPUT_SCHEMA', 'SYSTEM', 'need',
          'observation_schema', 'check_indices', 'claimed_part_groups', 'validate_observation',
          'excluded_semantic_claim', 'evidence_only_note', 'filter_supported_observation',
          'facts_from_observations'}))
    refs = []
    for filename, names in inventories:
        path = SOURCE / filename
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        need(actual == FROZEN[filename], 'frozen native source changed: ' + filename)
        nodes = [n for n in ast.parse(raw).body if
                 (isinstance(n, ast.FunctionDef) and n.name in names) or
                 (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
        found = {n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id for n in nodes}
        need(found == names, 'frozen pure-function inventory changed')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
        refs.append(dict(path=str(path), sha256=actual))
        if filename == 'ms_caption_rewrite.py':
            namespace['core'] = SimpleNamespace(window_bounds=namespace['window_bounds'],
                                               neutral_fact_text=namespace['neutral_fact_text'])
    return SimpleNamespace(**namespace, code_refs=refs)


def observation_schema(indices):
    schema = copy.deepcopy(native().observation_schema(indices))
    schema['title'] = RECIPE
    event = schema['properties']['events']['items']
    del event['properties']['id']
    event['required'].remove('id')
    event['properties']['parts']['uniqueItems'] = True
    event['properties']['frame_indices']['uniqueItems'] = True
    for support in schema['properties']['part_support']['prefixItems']:
        for field in ('part', 'reason'):
            del support['properties'][field]
            support['required'].remove(field)
        support['properties']['frame_indices']['uniqueItems'] = True
    return schema


def compact_request(request, manifest):
    """New recipe only: preserve original pixels/user content/model/sampling budget."""
    bar = native()
    indices = [f['index'] for f in manifest['frames']]
    need(set(request) == {'model', 'messages', 'sampling', 'json_schema', 'prompt_sha256'},
         'original native request keys required; no compact/resumed recipe')
    need(request['json_schema'] == bar.observation_schema(indices), 'original native schema/frame binding differs')
    need(request['sampling'] == dict(temperature=0.0, seed=0, max_tokens=1800), 'original 1800 sampling budget required')
    messages = request['messages']
    need(isinstance(messages, list) and len(messages) == 2 and
         messages[0] == dict(role='system', content=bar.SYSTEM) and messages[1]['role'] == 'user',
         'original native system/user layout required')
    need(request['prompt_sha256'] == hashlib.sha256(bar.SYSTEM.encode()).hexdigest(), 'original system SHA differs')
    result = copy.deepcopy(request)
    system = bar.SYSTEM + CONTRACT + '\nFixed part_support order: ' + ', '.join(bar.PARTS) + '.\n'
    result['messages'][0]['content'] = system
    result['json_schema'] = observation_schema(indices)
    result['prompt_sha256'] = hashlib.sha256(system.encode()).hexdigest()
    result.update(caption_recipe=RECIPE, native_visual_model_protocol=False)
    return result


def expand_observation(value):
    """Never rewrite descriptions, infer motion, deduplicate events or repair indices."""
    parts = native().PARTS
    need(isinstance(value, dict) and set(value) == {'events', 'part_support', 'uncertainties', 'abstain'},
         'invalid compact observation schema keys')
    need(isinstance(value['events'], list) and len(value['events']) <= 32, 'invalid compact events')
    need(isinstance(value['part_support'], list) and len(value['part_support']) == len(parts),
         'exactly ten ordered compact part_support rows required')
    need(isinstance(value['uncertainties'], list) and len(value['uncertainties']) <= 20 and
         all(isinstance(x, str) and len(x) <= 600 for x in value['uncertainties']), 'invalid compact uncertainties')
    need(type(value['abstain']) is bool, 'invalid compact abstain')
    events, supports = [], []
    for i, event in enumerate(value['events'], 1):
        need(isinstance(event, dict) and set(event) == {'description', 'parts', 'frame_indices'},
             'invalid compact event schema keys')
        need(isinstance(event['description'], str) and 1 <= len(event['description']) <= 1600,
             'invalid compact description')
        events.append(dict(id='e' + str(i), **copy.deepcopy(event)))
    for part, support in zip(parts, value['part_support']):
        need(isinstance(support, dict) and set(support) == {'frame_indices'} and
             isinstance(support['frame_indices'], list), 'invalid compact support schema keys/indices')
        supports.append(dict(part=part, frame_indices=copy.deepcopy(support['frame_indices']),
                             reason='derivedtag:nonempty_support' if support['frame_indices']
                             else 'derivedtag:empty_support_unknown'))
    return dict(events=events, part_support=supports,
                uncertainties=copy.deepcopy(value['uncertainties']), abstain=value['abstain'])


def expand_and_filter(value, manifest, forbidden=None):
    bar = native()
    if isinstance(value, (str, bytes)):
        value = bar.strict_json(value)
    expanded = expand_observation(value)
    checked = bar.filter_supported_observation(expanded, manifest, forbidden)
    return dict(schema=RECIPE, caption_recipe=RECIPE, native_visual_model_protocol=False,
                status='abstain' if checked['abstain'] else 'candidate', json=checked,
                expanded_native_value=expanded,
                derivation=dict(native_value_is_adapter_derivation_not_model_output=True,
                    native_filter_raw_response_is_expanded_value=True,
                    event_ids='unique_one_based_response_array_order_not_measured_times',
                    part_support_order=list(bar.PARTS), support_reason='derivedtag_from_empty_or_nonempty_array',
                    support_reason_is_VLM_original_reason=False, support_reason_is_visibility_ground_truth=False,
                    unknown_support_is_not_stationary=True, event_description_rewritten=False,
                    original_complete_support_filter_called=True),
                training_eligible=False, quality_approved=False, **bar.FLAGS)


def facts_from_observations(windows, observations):
    """Keep original admitted fact text/IDs, but label the new protocol honestly."""
    need(len(windows) == len(observations) and all(o.get('schema') == RECIPE and
         o.get('native_visual_model_protocol') is False for o in observations), 'compact recipe results required')
    facts = native().facts_from_observations(windows, observations)
    for fact in facts:
        fact.update(source=RECIPE, caption_recipe=RECIPE, native_visual_model_protocol=False,
                    training_eligible=False, quality_approved=False)
    return facts
