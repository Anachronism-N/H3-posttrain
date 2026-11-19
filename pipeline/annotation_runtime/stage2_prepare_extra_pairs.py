"""Prepare extra native pairs in existing node-local out; reuse one node uploader and ledger."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from stage2_local_publication_v1 import checked,write_once

RT=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime')


def run(node):
    if node not in ('N3','N6'):raise ValueError('Only exact naturally safe returned pairs allowed')
    root=Path('/root/stage2_dynamic_native37610v3_'+node+'_20261028_v1');sys.path.insert(0,str(root/'code'));d=json.loads((root/'deployment.json').read_bytes());base=checked(d['configs'][0]['config_ref']);p=root/'code/stage2_local_dynamic_native_manager_v4.py';raw=(RT/p.name).read_bytes()
    if p.exists():assert p.read_bytes()==raw
    else:
        fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    spec=importlib.util.spec_from_file_location('extra_manager',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    pairs=([2,3],[4,5]) if node=='N3' else ([4,5],[6,7]);phases=('full12k_visual_s0_v1','full12k_visual_s1_v1') if node=='N3' else ('full12k_visual_s2_v1','full12k_visual_s3_v1');entries=[]
    for pair,previous in zip(pairs,phases):
        q=RT.parent/'queues'/previous;safe=json.loads((q/'resources_safe.json').read_bytes());assert all(safe.get(k) is True for k in ('all_holders_safe','locks_released','leases_restored'));holders=[]
        for g in pair:
            uuid=m.b.uuid_for(g);apps=[a for a in m.b.apps() if a['uuid']==uuid];assert len(apps)==1,'Fresh exact restored holder required';i=m.b.identity(apps[0]['pid']);i.update(gpu=g,uuid=uuid);m.b.check_holder(i,m.b.identity(i['pid']),g,uuid,m.b.apps());holders.append(i)
        phase='stage2_dynamic_native37610v3_'+node+'_g'+''.join(map(str,pair))+'_extra';owner=node+'-g'+''.join(map(str,pair))+'-native37610v3';cp=root/(phase+'_config.json');cfg=dict(base);hashes=dict(base['source_sha256']);hashes[str(p)]=hashlib.sha256(raw).hexdigest();cfg.update(phase=phase,owner=owner,gpus=pair,config_path=str(cp),queue=str(root/'queues'/phase),holders=holders,worker_state_layout='by_phase',source_sha256=hashes,producer=dict(path=str(p),sha256=hashes[str(p)]),resource_manager_ref=dict(path=str(p),sha256=hashes[str(p)]),argv=['/opt/venv_vllm/bin/python','-B',str(p),'worker','--config',str(cp)],env=dict(base['env'],CUDA_VISIBLE_DEVICES=','.join(map(str,pair))),previous_naturally_completed_queue=str(q));cr=write_once(cp,cfg);m.check_sources(cfg,cp,cr['sha256'],m.ORIGINAL_BASE_SHA);m.b.check_model(cfg)
        feed=dict(out=cfg['out'],owner=owner,token_path=str(root/'token'),claim_endpoint='http://28.216.17.70:19100/claim',ffmpeg_path='/usr/local/bin/ffmpeg',prefetch_packets=8);fr=write_once(root/(phase+'_feeder.json'),feed);(root/'out'/'ready'/owner).mkdir(parents=True,exist_ok=True);entry=dict(config_ref=cr,feeder_ref=fr,gpus=pair,owner=owner,phase=phase);ref=write_once(root/'extra_deployments'/(owner+'.json'),entry);entries.append(dict(owner=owner,gpus=pair,entry_ref=ref,source_claims_created=0,models_started=0))
    print(json.dumps(dict(node=node,prepared_extra_pairs=entries,same_node_out_and_single_uploader=True,same_original_source_ledger=True)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',required=True);a=p.parse_args();run(a.node)
