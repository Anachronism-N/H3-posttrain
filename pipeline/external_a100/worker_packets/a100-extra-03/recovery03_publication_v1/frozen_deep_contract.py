"""Exact original audit v2 case contract, isolated from its looping/writing main."""
import hashlib
import io
from pathlib import Path
import numpy as np

def validate(payloads, target, receipt, row, rules, getjson):
    FLAGS = rules.FLAGS
    path = Path(target) / "commit.json"
    key = str(path)
    sourcekey = row["split"] + "/" + row["id"]
    relative = "cases/" + sourcekey
    frames = events = partial = 0
    staged = {}
    names=[relative+'/'+n for n in ['pose.npz','fullframe_manifest.json','beats.json','draft.json','receipt.json']]
    if any(n not in payloads for n in names):raise ValueError('source lacks five real output payloads')
    pose_raw=payloads[names[0]];manifest=getjson(payloads[names[1]]);beats=getjson(payloads[names[2]]);draft=getjson(payloads[names[3]]);case_receipt=getjson(payloads[names[4]])
    bound={'id':row['id'],'split':row['split'],'npz_path':str(path.parent/names[0]),'fullframe_manifest_path':str(path.parent/names[1])}
    with np.load(io.BytesIO(pose_raw),allow_pickle=False) as z:data={k:z[k] for k in z.files}
    if data['schema'].item()!='ms_fullvideo_pose2d_cache_v1' or data['bbox'].shape!=(len(data['times']),4):raise ValueError('new cache/bbox schema mismatch')
    if hashlib.sha256(pose_raw).hexdigest()!=manifest['npz_sha256']:raise ValueError('manifest pose SHA mismatch')
    mask=rules.validate_observations(bound,manifest,data)
    if rules.fullframe_binding(manifest,data)!=draft['source']['fullframe_binding']:raise ValueError('rule draft not bound to actual new fullframes')
    if case_receipt['source_row']!=row or case_receipt['frame_count']!=len(data['times']):raise ValueError('receipt source/frames mismatch')
    if manifest['video_path']!=row['video_path'] or not manifest['duration_seconds']>0 or manifest['duration_seconds']!=float(manifest['metadata_provenance']['actual_probe']['streams'][0]['duration']):raise ValueError('video duration not bound to actual video probe')
    if any(draft.get(k)!=v or case_receipt.get(k)!=v or manifest.get(k)!=v for k,v in FLAGS.items()):raise ValueError('approval/diagnostic flags mismatch')
    if draft['status'] not in ['draft_generated','partial','needs_regeneration'] or (draft['status']!='needs_regeneration' and not draft['events']) or not draft['ms_text'].strip():raise ValueError('no real rule events/text or wrong draft status')
    for event in draft['events']:
     support=event['support'];indices=np.array(support['frame_indices']);joints=np.array(support['joint_indices'])
     if not event['text'].strip() or not len(indices) or np.any(indices<0) or np.any(indices>=len(mask)) or np.any(joints<0) or np.any(joints>=134) or not mask[np.ix_(indices,joints)].all():raise ValueError('rule support outside valid native observations')
    audio=beats['actual_audio_metadata'];duration=float(audio['duration_seconds'])
    if beats['audio_path']!=row['audio_path'] or beats['device']!='cuda:0' or beats['float16'] is not False or beats['dbn'] is not False or beats['audio_video_alignment_verified'] is not False:raise ValueError('BeatThis clock/config mismatch')
    for name in ['beats','downbeats']:
     values=np.asarray(beats[name]);
     if not np.isfinite(values).all() or np.any(values<0) or np.any(np.diff(values)<=0) or np.any(values>duration+.02):raise ValueError('invalid actual beat times')
    n=len(data['times']);e=len(draft['events']);frames+=n;events+=e;partial+=int(draft['status']=='partial');staged[sourcekey]={'status':draft['status'],'frames':n,'events':e,'commit':key,'actual_artifacts_deep_verified':True}
    return staged[sourcekey]
