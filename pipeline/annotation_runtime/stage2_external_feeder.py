"""External CPU-only claim and03 input relay; exact original local prepared packets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import time
from urllib.parse import urlencode,urlsplit
import urllib.request
from stage2_local_publication_v1 import write_once,checked,require
from stage2_feeder_media_v1 import select_frames


def local_frames(row,manifest,local,ffmpeg_path):
    dst=local/(row['id']+Path(row['video_path']).suffix)
    frames = select_frames(manifest)
    expression = "+".join("eq(n\,%d)" % x["frame_index"] for x in frames)
    scale = min(1., math.sqrt(200704 / (manifest["width"] * manifest["height"])))
    width, height = max(1, int(manifest["width"] * scale)), max(1, int(manifest["height"] * scale))
    env = os.environ.copy()
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = ":".join(x for x in env["LD_LIBRARY_PATH"].split(":") if x)
    subprocess.run([ffmpeg_path, "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "1", "-noautorotate",
        "-i", str(dst), "-an", "-sn", "-filter_threads", "1", "-vf", "select=" + expression + ",scale=%d:%d" % (width, height),
        "-vsync", "0", "-threads", "1", "-q:v", "3", str(local / "frame-%03d.jpg")],
        check=True, cwd=local, env=env, capture_output=True)
    images = sorted(local.glob("frame-*.jpg"))
    require(len(images) == len(frames) <= 16, "Actual JPEG count differs from selected native frames")
    for p in images:
        raw = p.read_bytes()
        require(raw.startswith(b"\xff\xd8") and raw.endswith(b"\xff\xd9"), "Incomplete JPEG")
    return frames, images



def fetch(cfg,owner,id_,kind):
    u=cfg['relay_url'].rstrip('/')+'/input?'+urlencode(dict(owner=owner,id=id_,kind=kind));return urllib.request.urlopen(urllib.request.Request(u,headers={'Authorization':'Bearer '+Path(cfg['token_path']).read_text().strip()}),timeout=120)


def prepare(row,cfg):
    root=Path(cfg['out']);d=root/'prepared'/row['id'];d.mkdir(parents=True,exist_ok=False);refs={}
    for key,name,kind in (('rule_draft_ref','draft.json','draft'),('fullframe_manifest_ref','manifest.json','manifest')):
        with fetch(cfg,cfg['owner'],row['id'],kind) as f:raw=f.read(2**20+1)
        require(0<len(raw)<=2**20 and hashlib.sha256(raw).hexdigest()==row[key]['sha256'],'Frozen original JSON differs');p=d/name;fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        refs[key]=dict(path=str(p),sha256=row[key]['sha256'])
    manifest=checked(refs['fullframe_manifest_ref']);draft=checked(refs['rule_draft_ref']);require(manifest['id']==draft['id']==row['id'] and manifest['split']==draft['split']==row['split'],'Original identity differs');video=d/(row['id']+Path(row['video_path']).suffix);digest=hashlib.sha256();total=0
    with fetch(cfg,cfg['owner'],row['id'],'video') as incoming:
        expected=int(incoming.headers.get('Content-Length','0'));require(0<expected<=2**30,'Bounded single-copy video required');fd=os.open(video,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:
            while True:
                b=incoming.read(8*2**20)
                if not b:break
                total+=len(b);require(total<=expected,'Relay source grew');digest.update(b);f.write(b)
            f.flush();os.fsync(f.fileno())
        require(total==expected and digest.hexdigest()==manifest['video_sha256'],'Original once-copied video SHA differs')
    frames,images=local_frames(row,manifest,d,cfg['ffmpeg_path']);fd=os.open(d,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    return dict(id=row['id'],split=row['split'],draft_ref=refs['rule_draft_ref'],manifest_ref=refs['fullframe_manifest_ref'],selected_frames=frames,image_refs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in images],local_video_ref=dict(path=str(video),sha256=manifest['video_sha256']),single_source_video_copy=True)


def run(path):
    cfg=json.loads(Path(path).read_bytes());root=Path(cfg['out']);owner=cfg['owner'];q=root/'ready'/owner;q.mkdir(parents=True,exist_ok=True);claims=root/'feeder_claims'/owner;claims.mkdir(parents=True,exist_ok=True);token=Path(cfg['token_path']).read_text().strip();require(urlsplit(cfg['relay_url']).scheme in ('http','https'),'Authorized explicit relay URL required')
    while not (root/'FEEDER_STOP').exists():
        pending=[p for p in q.glob('*.ready.json') if not (root/'consumed_ready'/owner/p.name).exists()]
        if len(pending)>=8:time.sleep(.5);continue
        request_id=owner+'-'+secrets.token_hex(16);data=json.dumps(dict(owner=owner,request_id=request_id,size=8)).encode();r=urllib.request.Request(cfg['relay_url'].rstrip('/')+'/claim',data=data,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(r,timeout=30) as f:raw=f.read(2*2**20+1)
            require(len(raw)<=2*2**20,'Bounded claim response required');claim=json.loads(raw)
        except Exception as e:
            write_once(claims/(request_id+'_uncertain.json'),dict(request_id=request_id,owner=owner,claim_may_have_been_committed=True,automatic_retry=False,models_called=0,error=repr(e)));write_once(q/'FEEDER_FAILED',dict(error=repr(e),claims_not_released=True));raise
        if claim is None:break
        require(claim['owner']==owner and claim['automatic_retry'] is False and 1<=claim['source_count']<=8,'Original source-group claim required');write_once(claims/(claim['batch_id']+'.json'),claim);started=time.time()
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:items=list(executor.map(lambda row:prepare(row,cfg),claim['rows']))
            packet=dict(schema='stage2_prepared_packet_v1',owner=owner,claim=claim,sources=items,prepared_unix=time.time());p=root/'packets'/owner/(claim['batch_id']+'.json');ref=write_once(p,packet);write_once(q/(p.name+'.ready.json'),dict(packet_ref=ref,source_count=claim['source_count'],owner=owner,prepared_seconds=time.time()-started));print(json.dumps(dict(event='external_packet_ready',owner=owner,sources=claim['source_count'],seconds=time.time()-started)),flush=True)
        except Exception as e:
            write_once(claims/(claim['batch_id']+'_prepare_failure.json'),dict(original_claim=claim,error=repr(e),sources_permanently_claimed=True,automatic_retry=False,models_called=0));write_once(q/'FEEDER_FAILED',dict(error=repr(e),claims_not_released=True));raise
    write_once(q/'FEEDER_DRAINED',dict(owner=owner,ended_unix=time.time(),models_called=0,source_claims_released=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(a.config)
