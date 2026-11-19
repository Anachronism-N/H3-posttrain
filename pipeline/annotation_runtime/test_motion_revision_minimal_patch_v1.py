"""Minimal visual patches preserve source events; no whole-window VLM rewriting."""
import copy,json
import motion_revision_codec_minimal_patch_v1 as c
ctx=dict(row=dict(id='patchFixture',split='train'),style_hint=dict(candidates=['contemporary','urban']),plan=dict(windows=[dict(window_id=0,start=0.,end=1.)]),draft=dict(events=[dict(event_id='e0',start=0.,end=.9,text='一侧手腕相对肩部向画面右侧移动。')],observation={}),window_frames=[dict(window_id=0,start=0.,end=1.,frame_indices=[0,1,2],frames=[dict(frame_index=0,time=0.),dict(frame_index=1,time=.5),dict(frame_index=2,time=.95)])])
w=dict(window_id=0,start=0.,end=1.,visibility={p:'visible' for p in c.BODY_PARTS},edits=[],evidence_limits=[])
a=dict(id='patchFixture',split='train',dance_style='contemporary / urban',windows=[w])
c.validate(a,ctx,'visual_audit');assert c.apply_event_patches(ctx,a)==c.timed_events(ctx['draft'],ctx['plan'])
edit=dict(event_id='e0',action='replace_phrase',old_phrase='右侧',new_phrase='左侧',reason='direction_contradicted',evidence_frame_indices=[0,1],explanation='The wrist moves toward image-left relative to the shoulder in both samples.')
b=copy.deepcopy(a);b['windows'][0]['edits']=[edit];c.validate(b,ctx,'visual_audit');patched=c.apply_event_patches(ctx,b)[0]['events'][0];assert patched['text']=='一侧手腕相对肩部向画面左侧移动。' and patched['start']==0. and patched['end']==.9 and patched['event_id']=='e0'
rejected=0
for changes in [dict(old_phrase=ctx['draft']['events'][0]['text'],new_phrase='The dancer performs a wide sweep.'),dict(evidence_frame_indices=[0,2]),dict(event_id='missing'),dict(new_phrase='左侧，同时跳跃并跨步转身，重复数次后伸出手臂。')]:
    bad=copy.deepcopy(b);bad['windows'][0]['edits'][0].update(changes)
    try:c.validate(bad,ctx,'visual_audit');raise AssertionError('Nonminimal/unsupported patch accepted')
    except ValueError:rejected+=1
ctx['audit']=a
req=c.request_for(ctx,'caption_fusion',a);p=json.loads(req[1]['content']);assert 'corrected_original_timed_events' in p and 'motion_description' not in str(p) and 'original_timed_events' not in p
schema=c.response_schema(ctx,'visual_audit');fields=schema['properties']['windows']['prefixItems'][0]['properties'];assert 'edits' in fields and 'motion_description' not in fields and len(c.FEW_SHOT)==6
print(json.dumps(dict(unchanged_events_identical=True,phrase_only_patch=True,unsupported_edits_rejected=rejected,VLM_cannot_emit_window_caption=True,LLM_uses_patched_original_events=True,mixed_rich_style_preserved=True,few_shot_examples=6)))
