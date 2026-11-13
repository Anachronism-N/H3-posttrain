"""Check registered final measurement inventories and add equal-song paired statistics."""
import argparse,collections,hashlib,itertools,json,statistics,time
from pathlib import Path
import numpy as np
from choreobench.planning_units import build_planning_units

METHODS=('workflow','self_refine','react','choreography')
EDIT_METHODS=('no_op','generic','choreography')


def load(path):
 return json.loads(Path(path).read_text())


def records(path):
 return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def inventory(rows,expected,fields,name):
 keys=[tuple(r[k] for k in fields) for r in rows]
 if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError(name+' has missing, duplicated or unexpected registered cells')


def bootstrap(values):
 if not values:return dict(n_songs=0,mean=None,interval=None)
 a=np.asarray(values,dtype=float)
 if not np.isfinite(a).all():raise ValueError('Nonfinite metric')
 rng=np.random.default_rng(42);samples=a[rng.integers(0,len(a),size=(5000,len(a)))].mean(axis=1)
 return dict(n_songs=len(a),mean=float(a.mean()),interval=np.quantile(samples,[.025,.975]).tolist())


def main():
 p=argparse.ArgumentParser();p.add_argument('--metrics',type=Path,required=True);p.add_argument('--formal',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 start=time.monotonic()
 while not (a.metrics/'complete.json').exists() or not (a.formal/'clap_metrics/summary.json').exists():
  if time.monotonic()-start>172800:raise TimeoutError('Original final measurements not terminal; no reruns')
  time.sleep(30)
 paths=[a.metrics/'complete.json',a.formal/'execution_results.json',a.dataset,a.observations/'observations/complete.json']
 for folder in ('mechanical','long_metrics','music_metrics','repeat_metrics','edit_metrics'):
  paths.extend([a.formal/folder/'records.jsonl',a.formal/folder/'summary.json'])
 paths.extend([a.formal/'long_metrics/boundaries.jsonl',a.formal/'music_metrics/registry.json',a.formal/'music_metrics/scores/ratings.jsonl',a.formal/'clap_metrics/records.jsonl',a.formal/'clap_metrics/summary.json',a.formal/'e4/registry.json',a.formal/'e4/results.json'])
 hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
 songs={s['song_id']:s for s in load(a.dataset)['songs']}
 if len(songs)!=24:raise ValueError('Original24 song denominator changed')
 expected={(sid,m) for sid in songs for m in METHODS};execution=load(a.formal/'execution_results.json')['rows'];inventory(execution,expected,('song_id','method'),'E3 generations')
 source=records(a.observations/'observations/observations.jsonl');binding=load(a.observations/'bindings.json')['bindings']
 if len(source)!=1127 or len({r['item_sha256'] for r in source})!=1127 or {r['item_sha256'] for r in source}!={r['audio_sha256'] for r in binding}:raise ValueError('Independent1127 audio inventory changed')
 mechanical=records(a.formal/'mechanical/records.jsonl');long=records(a.formal/'long_metrics/records.jsonl');repeat=records(a.formal/'repeat_metrics/records.jsonl')
 for name,rows in [('mechanical',mechanical),('longterm',long),('repetition',repeat)]:inventory(rows,expected,('song_id','method'),name)
 units={sid:build_planning_units(song,config=dict(bars_per_block=4,fallback_seconds=10.125,max_unit_seconds=20,meter_status='unverified',min_time_block_seconds=2))['units'] for sid,song in songs.items()}
 expected_boundaries={(sid,m,i) for sid,u in units.items() for m in METHODS for i in range(1,len(u))};boundaries=records(a.formal/'long_metrics/boundaries.jsonl');inventory(boundaries,expected_boundaries,('song_id','method','index'),'all registered adjacent planning boundaries')
 music_registry=load(a.formal/'music_metrics/registry.json')['blocks'];expected_blocks={(sid,m,u['unit_id']) for sid,rows in units.items() for m in METHODS for u in rows};inventory(music_registry,expected_blocks,('song_id','method','unit_id'),'music blocks')
 music_scores=records(a.formal/'music_metrics/scores/ratings.jsonl');expected_scores={(r['item_sha256'],) for r in music_registry if r['status']=='registered_for_scoring'};inventory(music_scores,expected_scores,('item_sha256',),'music scoring')
 clap=records(a.formal/'clap_metrics/records.jsonl');expected_windows={(r['song_id'],m,r['window']['index']) for r in binding for m in METHODS};inventory(clap,expected_windows,('song_id','method','window'),'CLAP original1127 windows x4 methods')
 registry=load(a.formal/'e4/registry.json');slots=registry['slots'];categories=collections.Counter(s['registered_category'] for s in slots)
 if len(slots)!=48 or categories['already_satisfied']!=4 or categories['ambiguous_or_conflicting']!=4:raise ValueError('Original48 edit categories changed')
 editors=load(a.formal/'e4/results.json')['cells'];inventory(editors,{(s['slot_id'],m) for s in slots for m in ('generic','choreography')},('slot_id','method'),'96 editor cells')
 edits=records(a.formal/'edit_metrics/records.jsonl');inventory(edits,{(s['slot_id'],m) for s in slots for m in EDIT_METHODS},('slot_id','method'),'144 edit evaluation cells')
 metrics={}
 def add(name,method,sid,value):
  if value is not None:metrics.setdefault(name,{}).setdefault(method,{})[sid]=float(value)
 for row in execution:
  m=row['method'];sid=row['song_id'];r=row.get('result') or {}
  add('legal_completion_registered24',m,sid,row['full_song_succeeded'])
  for name,key in [('outer_calls','calls_used'),('reported_input_tokens','input_tokens'),('reported_output_tokens','output_tokens')]:add(name,m,sid,r.get(key))
  add('wallclock_seconds',m,sid,row.get('wallclock_seconds'))
 for row in long:
  if row['status']!='reviewed':continue
  m=row['method'];sid=row['song_id'];add('model_longterm_score',m,sid,row['longterm_score'])
  for k,g in row['longterm']['gates'].items():add('model_gate_'+k,m,sid,g['grade'])
  labels=[row['longterm'][k] for k in ('solo_motion','motion_only')]
  add('public_five_conditions_complete_lower_bound',m,sid,(sum(x=='yes' for x in labels)+sum(row['longterm']['gates'][k]['grade']==2 for k in ('theme_development','contrast','connective_progression')))/5)
 for m in METHODS:
  for sid in songs:
   rows=[r for r in boundaries if r['method']==m and r['song_id']==sid];known=[r for r in rows if r['label']!='unknown']
   if rows:
    add('boundary_conflict_registered_lower_bound',m,sid,sum(r['label']=='conflict' for r in rows)/len(rows));add('boundary_unknown_registered_fraction',m,sid,sum(r['label']=='unknown' for r in rows)/len(rows))
   if known:add('boundary_conflict_known_fraction',m,sid,sum(r['label']=='conflict' for r in known)/len(known))
   values=[r['matching_minus_other'] for r in clap if r['method']==m and r['song_id']==sid and r['status']=='computed']
   if values:add('CLAP_matching_minus_other',m,sid,statistics.mean(values))
 scores={r['item_sha256']:r for r in music_scores}
 for m in METHODS:
  for sid in songs:
   values=[]
   for r in music_registry:
    if r['method']!=m or r['song_id']!=sid:continue
    score=scores.get(r.get('item_sha256'))
    if score and score['status']=='valid' and score.get('rating',{}).get('music_fit') is not None:values.append(score['rating']['music_fit'])
   if values:add('model_music_evidence_score',m,sid,statistics.mean(values))
 for row in repeat:
  if row['status']!='reviewed':continue
  n=row['registered_events'];m=row['method'];sid=row['song_id']
  add('model_unproductive_repetition_lower_bound',m,sid,row['unproductive_events']/n);add('model_repetition_unknown_fraction',m,sid,row['unknown_events']/n)
  if n>row['unknown_events']:add('model_unproductive_repetition_known_fraction',m,sid,row['unproductive_events']/(n-row['unknown_events']))
 for method in EDIT_METHODS:
  by_song=collections.defaultdict(list)
  for r in edits:
   if r['method']==method and r['category'] not in ('already_satisfied','ambiguous_or_conflicting'):by_song[r['song_id']].append(float(r['strict_joint_success']))
  for sid,values in by_song.items():add('edit_strict_success_registered_required',method,sid,statistics.mean(values))
 method_intervals={};pairs=[]
 for name,group in metrics.items():
  registered_songs=len({s['song_id'] for s in slots if s['registered_category'] not in ('already_satisfied','ambiguous_or_conflicting')}) if name.startswith('edit_') else 24
  method_intervals[name]={m:dict(bootstrap([v for _,v in sorted(values.items())]),registered_songs=registered_songs,missing_songs=registered_songs-len(values)) for m,values in group.items()}
  methods=EDIT_METHODS if name.startswith('edit_') else METHODS
  for left,right in itertools.combinations(methods,2):
   a_values=group.get(left,{});b_values=group.get(right,{});common=sorted(set(a_values)&set(b_values));pairs.append(dict(metric=name,a=left,b=right,common_songs=common,statistics=bootstrap([a_values[s]-b_values[s] for s in common]),registered_songs=registered_songs,missing_songs=registered_songs-len(common)))
 if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):raise ValueError('Final measurement inputs changed during aggregation')
 a.out.mkdir(exist_ok=False)
 result=dict(platform='TX A800 cluster',registered_E3_cells=96,registered_E4_tasks=48,registered_E4_edit_cells=96,registered_E4_evaluation_cells=144,registered_audio_observations=1127,registered_CLAP_windows=len(clap),registered_music_blocks=len(music_registry),registered_planning_boundaries=len(boundaries),all_required_inventories_present=True,measurement_completion_not_quality_acceptance=True,human=False,model_reviews_fallible=True,all_original_generation_and_edit_calls_preserved=True,new_generation_or_review_calls=0,failed_or_unknown_not_imputed_quality_scores=True,lower_bound_metrics_explicitly_named=True,paired_statistics_scope='equal-song paired percentile bootstrap, common measured songs; declared required-edit lower bound retains unknown as not-success, not imputed quality',seed=42,bootstrap_samples=5000,confidence=.95,rng='numpy.default_rng PCG64',multiplicity_adjusted=False,input_hashes=hashes,observation_statuses=dict(collections.Counter(r['status'] for r in source)),edit_categories=dict(categories),method_intervals=method_intervals,paired=pairs,original_summaries={name:load(a.formal/name/'summary.json') for name in ('mechanical','long_metrics','music_metrics','repeat_metrics','clap_metrics','edit_metrics')})
 (a.out/'summary.json').write_text(json.dumps(result,indent=2)+chr(10));print(json.dumps(dict(final_inventory_verified=True,E3=96,E4_tasks=48,E4_evaluation_cells=144,paired_comparisons=len(pairs),quality_or_human_acceptance=False)),flush=True)


if __name__=='__main__':main()
