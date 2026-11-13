"""Score every final planned block using candidate-independent actual-audio observations."""
import argparse,json,hashlib,subprocess,sys,collections,statistics
from pathlib import Path
from choreobench.planning_units import build_planning_units


def main():
 p=argparse.ArgumentParser();p.add_argument('--formal',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--provider',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
 binding=json.loads((a.observations/'bindings.json').read_text())['bindings'];obs={r['item_sha256']:r for r in map(json.loads,(a.observations/'observations/observations.jsonl').open())};dataset=json.loads(a.dataset.read_text());songs={s['song_id']:s for s in dataset['songs']};items=[];registry=[]
 cells=json.loads((a.formal/'execution_results.json').read_text())['rows']
 for cell in cells:
  sid=cell['song_id'];song=songs[sid];units=build_planning_units(song,config=dict(bars_per_block=4,fallback_seconds=10.125,max_unit_seconds=20,meter_status='unverified',min_time_block_seconds=2))['units']
  plan=json.loads((Path(cell['run_dir'])/'plan.json').read_text()) if cell['full_song_succeeded'] else None
  for unit in units:
   entry=dict(song_id=sid,method=cell['method'],unit_id=unit['unit_id'],status='failed_generation')
   if plan is None:registry.append(entry);continue
   relevant=[b for b in binding if b['song_id']==sid and b['window']['observation_start']<unit['end'] and b['window']['observation_end']>unit['start']];observations=[obs[b['audio_sha256']] for b in relevant]
   if any(r['status']!='valid' for r in observations) or not observations:entry['status']='audio_observation_unknown';registry.append(entry);continue
   evidence={}
   for field in ('pulse_rate','energy','articulation','development'):
    count=collections.Counter(r['observation'][field] for r in observations);top=count.most_common();evidence[field]=top[0][0] if len(top)==1 or top[0][1]>top[1][1] else 'uncertain'
   evidence['audio_evidence']='; '.join(r['observation']['audio_evidence'] for r in observations);evidence['limitations']='Fallible independent actual-audio descriptions, categorical aggregation over overlapping windows; no verified BPM or human synchronization. '+ '; '.join(r['observation']['limitations'] for r in observations);evidence['origin']='independent_actual_audio_model_observations_not_ground_truth'
   events=[e for e in plan['events'] if e['start']<unit['end'] and e['end']>unit['start']];text=' '.join(e['action'] for e in events);tid=hashlib.sha256((sid+'/'+cell['method']+'/'+unit['unit_id']+'/'+text).encode()).hexdigest();entry.update(status='registered_for_scoring',item_sha256=tid)
   items.append(dict(item_sha256=tid,case_id=sid,method_id=cell['method'],evidence_id=sid+'/'+unit['unit_id'],text=text,audio_observation=evidence));registry.append(entry)
 (a.out/'registry.json').write_text(json.dumps(dict(blocks=registry),indent=2));(a.out/'items.json').write_text(json.dumps(dict(items=items),indent=2));subprocess.run([sys.executable,'-m','captionlab.evidence_music_fit','--items',str(a.out/'items.json'),'--provider',str(a.provider),'--out',str(a.out/'scores')],check=True)
 scores={r['item_sha256']:r for r in map(json.loads,(a.out/'scores/ratings.jsonl').open())};summary={}
 for method in ('workflow','self_refine','react','choreography'):
  group=[r for r in registry if r['method']==method];per_song=collections.defaultdict(list);valid=0;unknown=0
  for row in group:
   rating=scores.get(row.get('item_sha256'),{});value=rating.get('rating',{}).get('music_fit') if rating.get('status')=='valid' else None
   if value is None:unknown+=1
   else:valid+=1;per_song[row['song_id']].append(value)
  summary[method]=dict(registered_songs=24,registered_blocks=len(group),valid_block_scores=valid,unscored_blocks=unknown,valid_songs=len(per_song),song_equal_mean=statistics.mean(statistics.mean(v) for v in per_song.values()) if per_song else None,song_means={s:statistics.mean(v) for s,v in per_song.items()})
 (a.out/'summary.json').write_text(json.dumps(dict(methods=summary,origin='actual-audio observer then frozen text evidence scorer; fallible model-assisted, not human music-fit validation',raw_audio_seen_by_scorer=False,no_candidate_in_observer=True,unknown_not_imputed=True),indent=2)+chr(10))


if __name__=='__main__':main()
