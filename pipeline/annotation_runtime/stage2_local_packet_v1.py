"""Pure local prepared-input validation; original logical bindings are never rewritten."""
import hashlib
import json
from pathlib import Path
import re
from stage2_local_publication_v1 import checked,require


def validate_packet(packet,local_root,owner):
    root=Path(local_root);require(root.is_absolute() and root.resolve()==root,'Stable node-local input root required')
    require(packet.get('schema')=='stage2_prepared_packet_v1' and packet.get('owner')==owner,'Original dynamic pool owner required')
    claim=packet['claim'];require(claim.get('schema')=='stage2_dynamic_claim_receipt_v1' and claim.get('owner')==owner and claim.get('automatic_retry') is False and claim.get('maximum_model_calls_per_source')==2,'Original permanent 2-call claim required')
    rows=claim['rows'];require(type(claim.get('source_count')) is int and 1<=len(rows)<=8 and len(rows)==claim['source_count'] and [[r['split'],r['id']] for r in rows]==claim['source_keys'],'Exact dynamic source batch mapping required')
    entries=packet['sources'];require(len(entries)==len(rows),'Prepared source count differs')
    result=[]
    for original,item in zip(rows,entries):
        id_=original['id'];split=original['split'];require(re.fullmatch('[A-Za-z0-9_-]+',id_) and re.fullmatch('[A-Za-z0-9_-]+',split) and item['id']==id_ and item['split']==split,'Prepared source identity differs')
        for name in ('draft_ref','manifest_ref'):
            p=Path(item[name]['path']);require(p.is_relative_to(root) and p.resolve()==p,'Owned local metadata reference required')
        draft=checked(item['draft_ref']);manifest=checked(item['manifest_ref'])
        require(item['draft_ref']['sha256']==original['rule_draft_ref']['sha256'] and item['manifest_ref']['sha256']==original['fullframe_manifest_ref']['sha256'],'Original source JSON bytes changed')
        for value in (draft,manifest):require(value['id']==id_ and value['split']==split and value.get('quality_approved') is False and value.get('semantic_approval') is False and value.get('training_eligible') is False and value.get('needs_visual_revision') is True,'Prepared metadata identity/approval differs')
        require(draft['schema'] in ('ms_fullframe_pose2d_rule_draft_v1','ms_fullframe_pose2d_rule_draft_v2') and manifest['schema']=='ms_fullframe_pose2d_manifest_v1' and isinstance(draft.get('ms_text'),str),'Exact native input schemas and original caption text required')
        source=draft['source'];origin=original.get('fullframe_origin_path',original['fullframe_manifest_ref']['path']);require(source['fullframe_manifest_path']==origin and source['video_path']==manifest['video_path']==original['video_path'] and source['video_sha256']==manifest['video_sha256'],'Original logical manifest/video binding differs')
        binding=source['fullframe_binding'];body=dict(schema=binding['schema'],fullframe_manifest=manifest,observations_sha256=binding['observations_sha256']);digest=hashlib.sha256(json.dumps(body,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':')).encode()).hexdigest();require(binding['schema']=='ms_fullframe_pose2d_binding_v1' and digest==binding['sha256'],'Original fullframe binding digest changed')
        frames=item['selected_frames'];n=manifest['frame_count'];require(type(n) is int and n>=2,'At least two native frames required');indices=sorted({round(i*(n-1)/(min(12,n)-1)) for i in range(min(12,n))});expected=[dict(frame_index=i,time=manifest['times'][i],source_pts=manifest['source_timestamps'][i]) for i in indices];require(frames==expected and len(item['image_refs'])==len(frames),'Original twelve native snapshot selection differs')
        video=Path(item['local_video_ref']['path']);require(video.is_relative_to(root) and video.resolve()==video and video.stem==id_ and item['local_video_ref']['sha256']==manifest['video_sha256'] and item.get('single_source_video_copy') is True,'Prepared video must be the original once-copied local source')
        require(video.is_file() and not video.is_symlink() and video.stat().st_size<=2**30,'Owned regular bounded local video required')
        digest=hashlib.sha256()
        with video.open('rb') as stream:
            for block in iter(lambda:stream.read(8*2**20),b''):digest.update(block)
        require(digest.hexdigest()==manifest['video_sha256'],'Prepared local video bytes differ from original source SHA')
        images=[]
        for ref in item['image_refs']:
            p=Path(ref['path']);require(p.is_relative_to(root) and p.resolve()==p and not p.is_symlink(),'Owned local JPEG required');raw=p.read_bytes();require(hashlib.sha256(raw).hexdigest()==ref['sha256'] and raw.startswith(bytes([255,216])) and raw.endswith(bytes([255,217])),'Prepared JPEG evidence changed');images.append(p)
        result.append(dict(original_row=original,draft=draft,manifest=manifest,selected_frames=frames,images=images,local_video_ref=item['local_video_ref'],claim_batch_id=claim['batch_id'],owner=owner))
    return result
