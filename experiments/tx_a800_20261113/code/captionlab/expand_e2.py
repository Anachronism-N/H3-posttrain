"""Freeze a nested E2 expansion without changing the original96 or picking outputs."""
import argparse,copy,json
from pathlib import Path
from collections import Counter
from captionlab import sol61
from choreobench.io import object_sha,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(exist_ok=False)
    old=json.loads((a.previous/'test96/manifest.public.json').read_text())
    report=old['dataset']['audit']
    # Existing pinned data and its stage audit are reused; selected WAVs are checked at binding.
    source=a.previous/'data/sol61_captions'
    assert report['cross_split_id_count']==0 and report['cross_split_group_count']==0
    metadata={'test':[{k:r[k] for k in ('id','source_group_id','gate')} for r in sol61._rows(source/'test_captions.jsonl')]}
    sol61.audit_sources=lambda *args,**kwargs:(copy.deepcopy(report),metadata,set(),set())
    manifest,refs,audit=sol61.freeze_cases(source,split='test',limit=768,seed=42)
    oldids=[c['case_id'] for c in old['cases']]
    assert [c['case_id'] for c in manifest['cases'][:96]]==oldids
    for before,after in zip(old['cases'],manifest['cases']):
        assert before['source_group_id']==after['source_group_id']
        after['audio']=copy.deepcopy(before['audio'])
    import hashlib
    from choreobench.audio import inspect_audio
    media_path=a.previous/'data/ms_original_planner_dataset_v1/test.jsonl'
    media={r['id']:r for r in sol61._rows(media_path)}
    audio_root=Path('/apdcephfs_gy4/share_302533218/cedricnie/OmniDance_processed/pose_preprocess_outputs/mainline/normalized_704x1280/main/audio')
    chosen=[];chosen_refs=[];rejections=[];seen_pcm=set()
    ref_byid={r['case_id']:r for r in refs['records']}
    for case in manifest['cases']:
        cid=case['case_id']
        try:
            declared=media[cid]
            if declared['source_group_id']!=case['source_group_id'] or declared['split']!='test':
                raise ValueError('Source or split mismatch')
            if declared['audio_window_seconds']!=[0.0,5.0625] or declared['audio_duration_seconds']!=5.0625:
                raise ValueError('Source does not declare original full window')
            actual=inspect_audio(audio_root/(cid+'.wav'))
            if actual['sha256']!=declared['audio_sha256']:raise ValueError('Actual original WAV SHA differs')
            if actual['num_samples']!=int(5.0625*actual['sample_rate']+.5):
                raise ValueError('Original WAV is not exact full-window geometry; no crop/pad performed')
            pcm=(actual['sample_rate'],hashlib.sha256(actual['pcm']).hexdigest())
            if pcm in seen_pcm:raise ValueError('Duplicate exact PCM')
            seen_pcm.add(pcm)
            case['audio']={k:actual[k] for k in ('path','sha256','sample_rate','num_samples')}
            chosen.append(case);chosen_refs.append(ref_byid[cid])
        except (OSError,ValueError,KeyError) as error:
            if cid in oldids:raise
            rejections.append(dict(case_id=cid,error=str(error),selection_only=True,no_model_output_examined=True))
        if len(chosen)==384:break
    assert len(chosen)==384
    bound=copy.deepcopy(manifest);bound['cases']=chosen
    bound['selection'].update(limit=384,policy='nested seed42 SHA order; one per declared video group; exact original WAV required; reject recorded unreadable/misbound/short/duplicate PCM inputs only')
    bound['dataset'].update(media_bound=True,selected_pcm_duplicates=0)
    refs['records']=chosen_refs;refs['selection_sha256']=object_sha(bound['selection'])
    write_json(a.out/'input_selection_rejections.json',rejections)
    dev=json.loads((a.previous/'val12/manifest.public.json').read_text())['cases']
    assert not {c['source_group_id'] for c in dev}&{c['source_group_id'] for c in bound['cases']}
    for before,after in zip(old['cases'],bound['cases']):
        assert before==after
    new=copy.deepcopy(bound);new['cases']=new['cases'][96:]
    new['selection']=dict(split='test',seed=42,limit=288,parent_limit=384,original96_excluded=True,
        parent_ordered_ids_sha256=object_sha([c['case_id'] for c in bound['cases']]),
        policy='last288 of nested frozen SHA-ordered384; no output filtering')
    newrefs=copy.deepcopy(refs);newrefs['records']=newrefs['records'][96:];newrefs['selection_sha256']=object_sha(new['selection'])
    for folder,m,r in [('test384',bound,refs),('additional288',new,newrefs)]:
        out=a.out/folder;out.mkdir();write_json(out/'manifest.public.json',m);write_json(out/'references.private.json',r)
    summary=dict(input_rejections=len(rejections),original_cases=96,total_cases=384,new_cases=288,seed=42,
        original96_preserved_exactly=True,declared_video_groups=384,
        development_overlap=0,exact_PCM_duplicates=bound['dataset']['selected_pcm_duplicates'],
        style_counts=dict(Counter(r['dance_style'] for r in refs['records'])),
        gate_counts=dict(Counter(str(r['gate']) for r in refs['records'])),
        nominal_worst_case_binomial_95_halfwidth={'n96':1.96*(.25/96)**.5,'n384':1.96*(.25/384)**.5},
        independent_songs_verified=False,training_membership_verified=False,
        sampling='same pinned test pool and seed; nested expansion, no balancing by observed outputs',
        evaluation='source-paired statistics; original96 and total384 reported separately; music-judge validation still pending')
    write_json(a.out/'selection_summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
