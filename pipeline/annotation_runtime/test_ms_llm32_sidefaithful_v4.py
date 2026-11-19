"""Candidate side labels and metadata-only observation reasons, not semantic approval."""
import json
import ms_llm32_codec_sidefaithful_candidate_v4 as c
ctx=dict(row=dict(id='sideFixture',split='train'),style_hint=dict(candidates=['urban']),plan=dict(windows=[dict(window_id=0,start=0.,end=1.)]),draft=dict(events=[dict(event_id='e0',start=.1,end=.5,text='一侧肘部的二维投影夹角增大。'),dict(event_id='e1',start=.2,end=.5,text='另一侧手腕相对肩部向画面右侧移动。'),dict(event_id='e2',start=.5,end=.8,text='一侧肘部的二维投影夹角减小。')],observation=dict(unknown_intervals=[dict(start=.8,end=1.,reason='invalid_tracking_points_or_scale')])))
e=c.conservative_wording_events(ctx)[0]['events'];assert [x['side_identity'] for x in e]==['A','B','A'] and [x['event_id'] for x in e]==['e0','e1','e2']
p=json.loads(c.request_for(ctx)[1]['content']);assert p['original_unknown_intervals']==[dict(start=.8,end=1.)] and 'side_identity_contract' in p
value=dict(id='sideFixture',split='train',dance_style='urban',windows=[dict(window_id=0,start=0.,end=1.,caption='One arm extends at the elbow while the other moves its wrist toward image-right. The first arm then bends at the elbow.')],continuity_notes=[]);c.validate(value,ctx)
value['windows'][0]['caption']='These movements occur after a period of missing tracking data.'
try:c.validate(value,ctx);raise AssertionError('Technical metadata leakage accepted')
except ValueError:pass
print(json.dumps(dict(original_event_identity_and_times_preserved=True,side_A_B_preserved=True,unknown_diagnostics_not_forwarded=True,tracking_text_rejected=True,semantic_approval=False)))
