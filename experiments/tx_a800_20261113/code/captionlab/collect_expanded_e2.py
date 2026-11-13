"""Join immutable same-protocol nested E2 runs; never re-run or replace prior96."""
import argparse,json,shlex,subprocess,time,statistics
from pathlib import Path
from captionlab.review_bundle import collect_review_bundle


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--use-collected',action='store_true');a=p.parse_args()
    jobs=json.loads((a.root/'jobs.json').read_text());out=a.root/'collected'
    if not a.use_collected:out.mkdir(exist_ok=False)
    done={j['name'] for j in jobs if (out/j['name']/('additional288-'+j['name'])/'results.json').is_file()} if a.use_collected else set()
    old={
      'omni':a.previous/'collected_formal/test96-qwen3/results.json',
      'moss_music':a.previous/'collected_expanded/moss_music/test96-moss_music/results.json',
      'music_flamingo':a.previous/'collected_expanded/music_flamingo/test96-music_flamingo/results.json',
      'moss_audio':a.previous/'collected_secondwave/moss_audio/test96-moss_audio/results.json',
      'audio_flamingo3':a.previous/'collected_secondwave/audio_flamingo3/test96-audio_flamingo3-v2/results.json',
      'qwen25_omni':a.previous/'collected_qwen25_v3/test96-qwen25_omni-v3/results.json'}
    while len(done)<len(jobs):
        for j in jobs:
            name=j['name']
            if name in done:continue
            remote='test -f '+j['local']+'/'+name+'_manager/resources_restored.json'
            if subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',j['host'],remote],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:continue
            dest=out/name;dest.mkdir()
            remote='tar -C '+j['local']+' -cf - '+name+'_manager additional288-'+name
            cmd='set -o pipefail; ssh '+shlex.quote(j['host'])+' '+shlex.quote(remote)+' | tar -C '+shlex.quote(str(dest))+' -xf -'
            subprocess.run(['bash','-c',cmd],check=True)
            if not (dest/('additional288-'+name)/'results.json').is_file():raise RuntimeError('Expanded run missing final inventory; preserve failure')
            done.add(name);print('Collected additional288 '+name,flush=True)
        if len(done)<len(jobs):time.sleep(20)
    new={j['name']:out/j['name']/('additional288-'+j['name'])/'results.json' for j in jobs}
    rospy=json.loads((a.previous/'reviewers.json').read_text())
    # Restore only canonical WAV locations referenced by sealed node runs on the review host.
    import shutil
    for c in json.loads((a.root/'additional288/manifest.public.json').read_text())['cases']:
        dest=Path(jobs[0]['local'])/'audio'/(c['case_id']+'.wav');dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.is_file():shutil.copyfile(c['audio']['path'],dest)
    bundle=collect_review_bundle([p for name in old for p in (old[name],new[name])],
        a.root/'test384-review-6methods',reviewers=rospy,seed=42,allow_segments=True)
    manifest=json.loads((a.root/'test384/manifest.public.json').read_text());expected={r['case_id'] for r in manifest['cases']};stats={};inventory=[]
    for name in old:
        original=json.loads(old[name].read_text());added=json.loads(new[name].read_text());rows=original['rows']+added['rows']
        assert len(rows)==384 and {r['case_id'] for r in rows}==expected
        costs=original['costs']+added['costs'];latencies=[x['latency_seconds'] for x in costs if isinstance(x.get('latency_seconds'),(int,float))]
        stats[original['method_id']]=dict(cases=384,valid_generations=sum(r['status']=='complete' for r in rows),
            failed=sum(r['status']=='failed' for r in rows),unknown=sum(r['status']=='unknown' for r in rows),
            truncated=sum(r['stop_reason'] in ('length','budget_truncation','max_tokens') for r in rows),
            unique_whole_texts=len({r['text'] for r in rows}),
            mean_inference_latency_seconds=statistics.mean(latencies) if latencies else None,
            quality_verified=False,music_fit='pending audio evaluation; no score imputation')
        inventory.append(dict(method=original['method_id'],original_result=str(old[name]),additional_result=str(new[name]),
                              original_cases=96,additional_cases=288,total_cases=384))
    report=dict(total_cases=384,methods=6,formal_outputs=2304,new_model_calls=1728,original_model_calls_reused=576,
                independent_songs_verified=False,metric_calibration_complete=False,methods_statistics=stats)
    (a.root/'results_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.root/'combined_inventory.json').write_text(json.dumps(dict(segmented_results=inventory,benchmark_manifest=str(a.root/'test384/manifest.public.json')),indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
