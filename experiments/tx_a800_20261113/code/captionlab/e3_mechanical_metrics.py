"""Registered whole-song final-state diagnostics; no aesthetic or beat-score invention."""
import argparse,json,re
from pathlib import Path
from collections import Counter,defaultdict
from statistics import mean
from choreobench.schema import validate_plan
from choreobench.evaluation import evaluate_plan
from choreobench.io import file_sha


def plan_diagnostics(plan,song):
    validate_plan(plan,song)
    report=evaluate_plan(plan,song)
    events=plan['events'];phrases=[]
    for event in events:
        phrases.extend(' '.join(s.casefold().split()) for s in re.split(r'[.!?;]+',event['action']) if s.strip())
    counts=Counter(phrases)
    duplicates=sum(n-1 for n in counts.values())
    edges=[]
    for left,right in zip(events,events[1:]):
        edges.append(dict(left_event=left['event_id'],right_event=right['event_id'],left_action=left['action'],right_action=right['action'],
            scheduled_gap_samples=right['start']-left['end'],semantic_contradiction=None))
    return dict(valid=report['valid'],coverage=report['coverage'],events=len(events),boundary_inventory=edges,
        literal_duplicate_sentence_units=duplicates,literal_sentence_units=len(phrases),literal_duplicate_fraction=duplicates/len(phrases) if phrases else None,
        duration_seconds=[(e['end']-e['start'])/plan['sample_rate'] for e in events],
        music_fit=None,semantic_boundary_conflicts=None,invalid_semantic_repetition=None,
        note='Literal repetition can be a valid motif. Schedules are not observed musical/action boundaries. Coverage is a shared structural constraint, not quality advantage.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
    dataset=json.loads(a.dataset.read_text());songs={x['song_id']:x for x in dataset['songs']};methods=('workflow','self_refine','react','choreography')
    if len(songs)!=24:raise ValueError('Require independently frozen24 songs')
    results=[]
    for sid,song in songs.items():
        for method in methods:
            root=a.run/'runs'/sid/method;row=dict(song_id=sid,method=method,status='missing',legal_complete=False,quality_accepted=False)
            path=root/'results.json'
            if not path.exists():results.append(row);continue
            outcome=json.loads(path.read_text());row.update(status='complete_attempt' if outcome['complete'] else 'infrastructure_failed',calls_used=outcome['calls_used'],input_tokens=outcome['input_tokens'],output_tokens=outcome['output_tokens'],stop_reason=outcome['stop_reason'],committed_units=outcome['committed_units'],registered_units=outcome['total_units'])
            history=[]
            for step in sorted((root/'steps').glob('*.json')):
                v=json.loads(step.read_text());history.append(v)
            row['tool_status_counts']=dict(Counter(s['observation']['status'] for s in history))
            row['rejected_tool_calls']=sum(s['observation']['status']=='rejected' for s in history)
            row['no_change_steps']=sum(s['observation']['status']=='draft_retained_no_content_change' for s in history)
            if outcome['full_song_succeeded']:
                plan_path=root/'plan.json'
                if not plan_path.exists() or file_sha(plan_path)!=outcome['artifacts'].get('plan.json'):raise ValueError('Final plan absent or changed')
                try:
                    row['plan_metrics']=plan_diagnostics(json.loads(plan_path.read_text()),song)
                    row['legal_complete']=row['plan_metrics']['valid'] and row['plan_metrics']['coverage']['fraction']==1
                except (ValueError,TypeError,KeyError) as error:row['error']=str(error)
            results.append(row)
    if len(results)!=96:raise ValueError('Registered result denominator changed')
    summary={}
    for method in methods:
        group=[x for x in results if x['method']==method];valid=[x for x in group if x['legal_complete']]
        summary[method]=dict(registered_songs=24,statuses=dict(Counter(x['status'] for x in group)),legal_complete_songs=len(valid),legal_completion_rate=len(valid)/24,
            mean_calls_per_registered_song=mean(x.get('calls_used',0) for x in group),total_calls=sum(x.get('calls_used',0) for x in group),
            mean_input_tokens_per_registered_song=mean(x.get('input_tokens',0) for x in group),mean_output_tokens_per_registered_song=mean(x.get('output_tokens',0) for x in group),
            rejected_tool_calls=sum(x.get('rejected_tool_calls',0) for x in group),no_content_change_steps=sum(x.get('no_change_steps',0) for x in group),
            music_fit=None,long_range_organization=None,semantic_boundary_conflict_rate=None,semantic_repetition_rate=None)
    (a.out/'summary.json').write_text(json.dumps(dict(methods=summary,registered_cells=96,scope='actual final-state structure/cost/trajectory metrics, not semantic or musical quality',human=False),indent=2)+chr(10))
    with (a.out/'records.jsonl').open('x') as f:
        for row in results:f.write(json.dumps(row)+chr(10))
    print(json.dumps(dict(completed=True,registered=96,legal=sum(x['legal_complete'] for x in results))),flush=True)


if __name__=='__main__':main()
