"""Read-only contract and deterministic rule audit of committed new outputs.
No video/model access, generation writes, claims changes, or semantic approval.
"""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time
import traceback
from fractions import Fraction
import numpy as np
import ms_fullframe_pose2d_draft_v2 as v2

FLAGS={'diagnostic_not_production':True,'semantic_approval':False,'quality_approved':False,'training_eligible':False,'needs_visual_revision':True}
FILES=['pose.npz','fullframe_manifest.json','beats.json','draft.json','receipt.json']
B=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1')


def read_json_bytes(raw,label):
    if not raw.endswith(bytes([10])) or bytes([0]) in raw:raise ValueError('NUL/incomplete JSON '+label+'; retained, not skipped')
    return json.loads(raw)


def check_batch(path,ticket,assignment,global_seen):
    commit=read_json_bytes(path.read_bytes(),str(path))
    if commit['worker']!=ticket['worker_id'] or commit['edition']!=ticket['edition'] or commit['assignment_sha256']!=ticket['assignment_sha256']:raise ValueError('Commit ticket identity mismatch')
    if any(commit.get(k)!=v for k,v in FLAGS.items()):raise ValueError('Commit approval flags wrong')
    records=commit['receipts'];gpu=commit['gpu']
    if not 1<=len(records)<=20 or commit['unique_sources_attempted']!=len(records):raise ValueError('Batch source count outside1..20')
    archive=path.parent/'batch.tar'
    with archive.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    if sha!=commit['batch_sha256']:raise ValueError('Package/commit SHA mismatch')
    hashes={Path(r['path']).relative_to(path.parent).as_posix():r['sha256'] for r in commit['files']}
    seen={};summary=dict(sources=0,successful=0,sealed_failed=0,partial=0,needs_regeneration=0,frames=0,events=0,emptybeats=0,raw_effective_outside_points=0,exact_rules_recomputed=0)
    with tarfile.open(archive,'r:') as pack:
        members={m.name:m for m in pack.getmembers()}
        for name,m in members.items():
            if not m.isfile() or name.startswith('/') or '..' in Path(name).parts or any(s in name for s in ['inputs/','.mp4','.wav']):raise ValueError('Package contains forbidden input/unsafe member')
        for result in records:
            row=result['source_row'];key=row['split']+'/'+row['id']
            if key in global_seen or key in seen:raise ValueError('Duplicate source in committed queue')
            if key not in assignment or any(row.get(k)!=v for k,v in assignment[key].items()):raise ValueError('Source row not exact authorized assignment')
            if row['assignment_index']%4!=gpu:raise ValueError('Wrong zero-based row physical-GPU partition')
            prefix='cases/'+key+'/'
            contents={}
            for name in FILES:
                relative=prefix+name
                if relative not in members:
                    if result['status']=='failed_sealed_no_retry' and name!='receipt.json':continue
                    raise ValueError('Missing packaged '+relative)
                with pack.extractfile(members[relative]) as f:raw=f.read()
                if hashlib.sha256(raw).hexdigest()!=hashes.get(relative):raise ValueError('Actual package member/publication digest mismatch')
                contents[name]=raw
            case=read_json_bytes(contents['receipt.json'],prefix+'receipt.json')
            if any(case.get(k)!=v for k,v in FLAGS.items()):raise ValueError('Case approval flags wrong')
            if result['status']=='failed_sealed_no_retry':
                if case['status']!='failed_sealed_no_retry':raise ValueError('Sealed failed receipt mismatch')
                summary['sealed_failed']+=1;summary['sources']+=1;seen[key]={'status':'sealed_failed','commit':str(path)};continue
            if result['status']!='extracted_rule_draft_unapproved' or case['status']!=result['status']:raise ValueError('Unexpected source execution status')
            manifest=read_json_bytes(contents['fullframe_manifest.json'],prefix+'manifest')
            draft=read_json_bytes(contents['draft.json'],prefix+'draft');beats=read_json_bytes(contents['beats.json'],prefix+'beats')
            bound={'id':row['id'],'split':row['split'],'npz_path':str(path.parent/prefix/'pose.npz'),'fullframe_manifest_path':str(path.parent/prefix/'fullframe_manifest.json')}
            if manifest['npz_path']!=bound['npz_path'] or manifest['fullframe_manifest_path']!=bound['fullframe_manifest_path']:raise ValueError('Actual bound artifact paths differ')
            if manifest['video_path']!=row['video_path'] or beats['audio_path']!=row['audio_path']:raise ValueError('Original source media paths changed')
            if manifest['npz_sha256']!=hashlib.sha256(contents['pose.npz']).hexdigest():raise ValueError('Cache/manifest byte binding mismatch')
            if any(draft.get(k)!=v or manifest.get(k)!=v for k,v in FLAGS.items()):raise ValueError('Draft/manifest approval flags wrong')
            if draft['id']!=row['id'] or beats['id']!=row['id'] or draft['split']!=row['split'] or beats['split']!=row['split']:raise ValueError('Source id/split disagreement')
            with np.load(io.BytesIO(contents['pose.npz']),allow_pickle=False) as archive_data:data={k:archive_data[k] for k in archive_data.files}
            if any(x.dtype.hasobject for x in data.values()):raise ValueError('Object/pickle arrays forbidden')
            n=manifest['frame_count']
            if data['schema'].item()!='ms_fullvideo_pose2d_cache_v1' or data['bbox'].shape!=(n,4) or data['xy'].shape!=(n,134,2):raise ValueError('New fullframe schema/bbox/xy mismatch')
            if data['id'].item()!=row['id'] or data['split'].item()!=row['split']:raise ValueError('Cache identity wrong')
            if not np.array_equal(data['valid'],data['tracked'][:,None] & (data['score']>.3)):raise ValueError('Original raw valid rewritten')
            actual=manifest['metadata_provenance']['actual_probe'];stream=actual['streams'][0];frames=actual['frames']
            if len(frames)!=n or int(stream['width'])!=manifest['width'] or int(stream['height'])!=manifest['height'] or float(stream['duration'])!=manifest['duration_seconds']:raise ValueError('Video probe/frame metadata mismatch')
            if stream.get('nb_frames') not in [None,'N/A'] and int(stream['nb_frames'])!=n:raise ValueError('Probe native frame count wrong')
            ticks=np.asarray([int(f['best_effort_timestamp']) for f in frames],dtype=np.int64)
            pts=np.asarray([float(f['best_effort_timestamp_time']) for f in frames]);base=float(Fraction(stream['time_base']))
            if not np.allclose(ticks*base,pts,atol=1e-6) or not np.array_equal(data['frame_indices'],np.arange(n)) or not np.allclose(data['source_timestamps'],pts,atol=1e-6):raise ValueError('Native source PTS/frame coverage mismatch')
            if not manifest['metadata_provenance']['decoder'].startswith('ffmpeg native all frames BGR24'):raise ValueError('BGR fullframe provenance missing')
            if result['frame_count']!=n or case['frame_count']!=n or case['vlm_called'] or case['rgb_or_crops_written']:raise ValueError('Receipt native frame/noVLM policy mismatch')
            valid_before=data['valid'].copy();xy_before=data['xy'].copy()
            mask=v2.validate_observations(bound,manifest,data)
            inside=(data['xy'][:,:,0]>=0)&(data['xy'][:,:,0]<manifest['width'])&(data['xy'][:,:,1]>=0)&(data['xy'][:,:,1]<manifest['height'])&np.isfinite(data['xy']).all(axis=2)
            expected_mask=data['valid'] & data['tracked'][:,None] & inside
            if not np.array_equal(mask,expected_mask):raise ValueError('Rule geometry mask includes image-external/invalid points')
            recomputed=v2.generate_draft(bound,manifest,data,mask)
            raw_body_missing=int((~mask[:,:18].all(axis=1)).sum())
            if raw_body_missing or not data['tracked'].all():
                recomputed['status']='partial' if recomputed['events'] else 'needs_regeneration'
                recomputed['limitations'].append('Missing raw body/track evidence stays partial/unknown; never imputed or quality-complete.')
            if not np.array_equal(valid_before,data['valid']) or not np.array_equal(xy_before,data['xy'],equal_nan=True):raise ValueError('CPU rule validation mutated raw cache in memory')
            if json.dumps(recomputed,sort_keys=True,ensure_ascii=False,allow_nan=False)!=json.dumps(draft,sort_keys=True,ensure_ascii=False,allow_nan=False):raise ValueError('Stored full draft differs from deterministic frozen-rule recomputation')
            if case['draft_status']!=draft['status'] or result['draft_status']!=draft['status']:raise ValueError('Partial/failure status was rewritten')
            for event in draft['events']:
                support=event['support'];indices=np.asarray(support['frame_indices'],dtype=np.int64);joints=np.asarray(support['joint_indices'],dtype=np.int64)
                if not len(indices) or not event['text'] or np.any(indices<0) or np.any(indices>=n):raise ValueError('Missing actual supported action')
                if not mask[np.ix_(indices,joints)].all():raise ValueError('Event support includes invalid or image-external point')
                if not np.allclose(support['times'],data['times'][indices],atol=1e-6) or np.any(np.diff(indices)<=0):raise ValueError('Event support frame/time not actual observations')
                if event['start']<0 or event['end']>float(data['times'][-1])+1e-6:raise ValueError('Event extrapolated outside native observations')
            for name in ['beats','downbeats']:
                values=np.asarray(beats[name],dtype=float)
                if values.ndim!=1 or not np.isfinite(values).all() or np.any(np.diff(values)<=0) or np.any(values<0) or np.any(values>beats['actual_audio_metadata']['duration_seconds']+.02):raise ValueError('Invalid actual audio beat clock')
            if beats['float16'] is not False or beats['dbn'] is not False or beats['audio_video_alignment_verified'] is not False or beats['motions_on_beats_verified'] is not False:raise ValueError('Incorrect beat configuration/alignment claim')
            if beats['emptybeats']!=(not bool(beats['beats'])):raise ValueError('Empty beats falsely populated')
            summary['sources']+=1;summary['successful']+=1;summary['partial']+=int(draft['status']=='partial');summary['needs_regeneration']+=int(draft['status']=='needs_regeneration');summary['frames']+=n;summary['events']+=len(draft['events']);summary['emptybeats']+=int(beats['emptybeats']);summary['raw_effective_outside_points']+=int((data['valid'] & ~inside).sum());summary['exact_rules_recomputed']+=1
            seen[key]={'status':draft['status'],'events':len(draft['events']),'frames':n,'commit':str(path)}
    return summary,seen


def main(args):
    root=Path(args.root);auditroot=root/('correctness_audit_'+args.worker);auditroot.mkdir(exist_ok=True)
    audit_lock=(auditroot/'writer.lock').open('a')
    fcntl.flock(audit_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sysstate=auditroot/'state.json';saved=json.loads(sysstate.read_text()) if sysstate.is_file() else {};seen=saved.get('seen',{});done=set(saved.get('checked_commits',[]));totals=saved.get('totals',{})
    log=(auditroot/'progress.jsonl').open('a',encoding='utf-8');start=time.time();deadline=start+args.duration;last_report=0;last_checkpoint=0
    tickets=[]
    for edition in ['external_ms_unified_rollover_20261027_v1','external_ms_unified_clusters80k_20261027_v1']:
        t=json.loads((B/edition/'assignments'/(args.worker+'.ticket.json')).read_text());raw=Path(t['assignment_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=t['assignment_sha256']:raise ValueError('Audit central assignment SHA mismatch')
        assignment={}
        for i,line in enumerate(raw.splitlines()):r=json.loads(line);r['assignment_index']=i;assignment[r['split']+'/'+r['id']]=r
        tickets.append((t,assignment))
    def emit(row):
        row.update(worker=args.worker,unix=time.time());text=json.dumps(row,ensure_ascii=False);log.write(text+chr(10));log.flush();print(text,flush=True)
    def checkpoint():
        tmp=sysstate.with_name('state.'+str(os.getpid())+'.tmp')
        with tmp.open('x') as f:json.dump({'seen':seen,'checked_commits':sorted(done),'totals':totals},f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,sysstate)
        fd=os.open(auditroot,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    while time.time()<deadline:
        try:
            for t,assignment in tickets:
                folder=Path(t['output_root'])/'batches'
                if not folder.is_dir():continue
                with os.scandir(folder) as entries:paths=[Path(e.path)/'commit.json' for e in entries if e.is_dir()]
                for path in sorted(paths):
                    if str(path) in done or not path.is_file():continue
                    summary,added=check_batch(path,t,assignment,seen);seen.update(added);done.add(str(path))
                    for k,v in summary.items():totals[k]=totals.get(k,0)+v
                    emit({'event':'batch_rule_audit_passed','commit':str(path),'summary':summary})
                    if time.monotonic()-last_checkpoint>=30:
                        checkpoint();last_checkpoint=time.monotonic()
            checkpoint()
            if time.time()-last_report>=60:
                emit({'event':'correctness_audit_progress','verified_unique_sources':len(seen),'verified_commits':len(done),'totals':totals,'claim_or_artifact_modified':False,'semantic_visual_approval':False});last_report=time.time()
            if args.once:break
            time.sleep(30)
        except OSError as e:
            emit({'event':'audit_storage_read_alarm','error':str(e),'unverified_commit_not_counted':True,'original_artifacts_unchanged':True});checkpoint();time.sleep(15)
        except Exception as e:
            emit({'event':'correctness_contract_failure','error':str(e),'traceback':traceback.format_exc(),'original_artifacts_unchanged':True,'no_gpu_retry':True});checkpoint();raise
    checkpoint();emit({'event':'audit_pass_snapshot','verified_unique_sources':len(seen),'verified_commits':len(done),'totals':totals,'semantic_approval':False,'quality_approved':False,'training_eligible':False,'future_new_results_not_yet_audited':True,'seconds':time.time()-start})
    log.flush();os.fsync(log.fileno());log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);p.add_argument('--root',required=True);p.add_argument('--duration',type=int,default=1800);p.add_argument('--once',action='store_true');main(p.parse_args())
