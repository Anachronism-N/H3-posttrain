"""N7 GPU6/7 singleton TP1 review admission; original pidfd TERM/FLOCK/lease lifecycle."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import select
import stat
import subprocess
import sys
import time

import curated_collab3_review_adapter_v1 as a

RT=Path(__file__).resolve().parent
BAR=RT.parent
WATCH=RT/'curated_collab3_8b_review_watch_v1.py'
INPUT=BAR/'curated_collab3_independent_inputs_bounded_v1'
BASE_SHA='6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6'
AGGREGATION_SHA='fed5c8ce83b4cf0d003fe6cdb1c98b57cfe57448b7734c0b91b2b3ac810ab7fa'
spec=importlib.util.spec_from_file_location('collab3_review_lifecycle',RT/'resource_manager.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


def checked(ref):
    a.need(a.sha(ref['path'])==ref['sha256'],'explicit source/reference SHA changed')
    return a.strict(Path(ref['path']).read_bytes())


def lease_value(info):
    return None if info is None else [info.st_dev,info.st_ino,info.st_uid,info.st_mode,info.st_size,info.st_mtime_ns]


def node():
    # Real local interface inventory, never a node alias from environment or cfg alone.
    plan=checked({'path':str(RT/'curated_collaborative3_source_plan_v1.json'),'sha256':a.PLAN_SHA})
    p=RT/'curated_collaborative3_plan_v1.py'
    a.need(a.sha(p)=='33d0e151921a45c4b120e5ebcf6a3458f48c4573f477793a89a5ea4cb6aac513',
           'frozen real-node identity map changed')
    value={}
    import ast
    for n in ast.parse(p.read_bytes()).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='HOSTS' for t in n.targets):
            value=ast.literal_eval(n.value)
    a.need(value.get('N7') in subprocess.check_output(['hostname','-I'],text=True).split(),
           'not the actual N7 network interface; fake alias refused')
    return plan


def lock_inventory(gpu):
    path=Path('/tmp/ms12k_gpu_%d.lock'%gpu)
    try:info=path.lstat()
    except FileNotFoundError:return dict(path=str(path),exists=False,active_lock=False)
    a.need(stat.S_ISREG(info.st_mode) and info.st_uid==os.getuid() and info.st_size==0,'unsafe existing node GPU FLOCK file')
    key='%02x:%02x:%d'%(os.major(info.st_dev),os.minor(info.st_dev),info.st_ino)
    a.need(not any(key in line.split() for line in Path('/proc/locks').read_text().splitlines()),
           'active kernel FLOCK; no reclaim or stale-lock deletion')
    return dict(path=str(path),exists=True,stat=lease_value(info),active_lock=False)


def identity_digest(identity):
    files={name:{'path':entry['path'],'sha256':entry['sha256']} for name,entry in identity['files'].items()}
    return hashlib_digest({'root':identity['root'],'files':files})


def hashlib_digest(value):
    import hashlib
    return hashlib.sha256((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest()


def checkpoint(identity):
    weights=sorted(v['sha256'] for n,v in identity['files'].items() if Path(n).suffix in ('.safetensors','.bin','.pt','.gguf'))
    a.need(weights,'actual checkpoint weight SHA inventory required')
    return hashlib_digest({'weight_file_content_sha256':weights})


def model_preflight(modelref,teacherref):
    value=checked(modelref)
    a.need(value.get('node')=='N7' and value.get('complete') is True and value.get('full_copy_SHA_verified') is True,
           'complete local N7 8B full-copy SHA stage required')
    origin={'path':str(RT/'reviewer8b_stage_N6_extension_v2.json'),
            'sha256':'5607668eb5c476308f20ca20b2853c97bc846f5d43182df2a66729b3c9c4385f'}
    a.need(value.get('origin_ref')==origin,'N7 stage must bind exact complete N6 extension checkpoint origin')
    original=checked(origin)['model_identity']
    identity=value['model_identity'];root=Path(identity['root'])
    a.need(identity['sha256']=='220d9152229696caec4decebb89a160afc8ac9b6ba91ab1b9d790343a82d733a' and
           identity['sha256']==original['sha256'] and
           {k:v['sha256'] for k,v in identity['files'].items()}=={k:v['sha256'] for k,v in original['files'].items()},
           'N7 stage content differs from the exact originally approved full extension checkpoint')
    a.need(root==Path('/opt/models/Qwen3-VL-8B-Instruct-review-extension-v2')
           and not root.is_symlink() and identity['sha256']==identity_digest(identity),
           'actual local 8B root/content identity required')
    a.need(modelref.get('model_identity_sha256',identity['sha256'])==identity['sha256'],'review modelref identity mismatch')
    inventory={p.name for p in root.iterdir() if p.is_file() and not p.name.startswith('.')}
    a.need(inventory==set(identity['files']) and len([n for n in inventory if n.endswith('.safetensors')])==4,
           'exact four-shard 8B local model inventory required')
    for name,entry in identity['files'].items():
        path=root/name;s=path.stat()
        current=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,device=s.st_dev,inode=s.st_ino)
        a.need(not path.is_symlink() and str(path)==entry['path'] and current==entry['stat'],
               'staged 8B storage changed: '+name)
    for name in ('config.json','model.safetensors.index.json'):
        a.need(a.sha(root/name)==identity['files'][name]['sha256'],'8B config/index SHA changed')
    teacher=checked(teacherref);teacher=teacher.get('model_identity',teacher.get('model',teacher))
    a.need(teacher['sha256']==identity_digest(teacher) and checkpoint(teacher)!=checkpoint(identity),
           'reviewer actual weight identity equals annotator or teacher attestation invalid')
    return identity


def sources(modelref,teacherref):
    a.need(a.sha(RT/'resource_manager.py')==BASE_SHA and a.sha(RT/'curated_collab3_review_adapter_v1.py')==AGGREGATION_SHA,
           'published lifecycle/aggregation code changed')
    prior=a.strict((RT/'curated_review_watch_launch_v3.json').read_bytes())
    bindings={k:v for k,v in prior['source_sha256'].items() if str(RT/'source_curated_review_v2')+'/' in k}
    a.need(len(bindings)==18,'frozen native review code inventory changed')
    for path,wanted in bindings.items():
        a.need(a.sha(path)==wanted,'native frozen reviewer source SHA changed')
    for path in (WATCH,Path(__file__),RT/'resource_manager.py',RT/'curated_collab3_review_adapter_v1.py',
                 RT/'curated_collaborative3_plan_v1.py',RT/'curated_collaborative3_source_plan_v1.json',
                 RT/'curated_review_watch_launch_v3.json',RT/'reviewer8b_stage_N6_extension_v2.json'):
        bindings[str(path)]=a.sha(path)
    for ref in (modelref,teacherref):
        checked(ref);bindings[ref['path']]=ref['sha256']
    return bindings


def capture(gpu,modelref,teacherref):
    a.need(gpu in (6,7),'only authorized N7 physical GPU6 or GPU7')
    # All code/model/source preflight BEFORE obtaining a fresh holder snapshot.
    sources(modelref,teacherref);model_preflight(modelref,teacherref);node()
    lock=lock_inventory(gpu)
    lease=base.lease_snapshot(Path('/tmp/od_reserve_%d'%gpu))
    uuid=base.uuid_for(gpu);running=base.apps()
    owners=[v for v in running if v['uuid']==uuid]
    a.need(len(owners)==1,'physical GPU must contain exactly one known ordinary holder')
    holder=dict(base.identity(owners[0]['pid']),gpu=gpu,uuid=uuid)
    base.check_holder(holder,base.identity(holder['pid']),gpu,uuid,running)
    budget=base.headroom();a.need(budget['headroom']>=base.SAFETY+base.LOAD,'insufficient fresh 160GiB node headroom')
    return dict(schema='curated_collab3_N7_review_capture_v1',node='N7',mode='review',gpus=[gpu],holders=[holder],
                leases=[lease_value(lease)],readonly_lock=lock,budget=budget,
                captured_unix_ns=time.time_ns(),signals_sent=False,gpu_started=False,lease_modified=False,
                modelref=modelref,annotator_model_reference=teacherref)


def config(gpu,capture_ref,modelref,teacherref,authorize):
    a.need(authorize is True,'explicit N7 GPU6/7 holder normalTERM authorization required')
    identity=model_preflight(modelref,teacherref)
    bindings=sources(modelref,teacherref)
    snap=checked(capture_ref)
    a.need(snap['node']=='N7' and snap['gpus']==[gpu] and snap['mode']=='review' and snap['signals_sent'] is False and
           all(snap['modelref'][k]==modelref[k] for k in ('path','sha256')) and
           snap['annotator_model_reference']==teacherref,'capture/model/GPU binding mismatch')
    phase='curated_collab3_review_n7g%d_v1'%gpu
    out=BAR/'curated_collab3_independent_8b_v1'/('gpu%d'%gpu)
    queue=BAR/'queues'/phase
    a.need(not queue.exists(),'already dispatched queue; use no destructive reset/repeated start')
    bindings[capture_ref['path']]=capture_ref['sha256']
    plan={'path':str(RT/'curated_collaborative3_source_plan_v1.json'),'sha256':a.PLAN_SHA}
    env=dict(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',
             TRANSFORMERS_OFFLINE='1',MS_VISUAL_CPU_SITEPATH='/opt/venv_eval/lib/python3.11/site-packages')
    cfg=dict(node='N7',mode='review',gpus=[gpu],holders=snap['holders'],release_authorized=True,
             phase=phase,queue=str(queue),out=str(out),cwd=str(RT),producer={'path':str(WATCH),'sha256':a.sha(WATCH)},
             plan=plan,input_count=222939,targetcount=15000,source_selection={'shard':gpu-6,'nshard':2,'limit':0},
             source_sha256=bindings,modelref={**modelref,'model_identity_sha256':identity['sha256']},
             annotator_model_reference=teacherref,holder_capture=capture_ref,reuse_existing_results=True,
             sampling_policy={'original_batch_size':500,'random_count':25,'risk_count':5,'seed':0},
             original_scope={'start':32000,'end':62000,'full_batch_count':448},env=env,semantic_approval=False,
             argv=['/opt/venv_vllm/bin/python','-B','-u',str(WATCH),'--gpu',str(gpu),'--plan',plan['path'],
                   '--plan-sha256',a.PLAN_SHA,'--input-root',str(INPUT),'--out',str(out),'--model',identity['root'],
                   '--model-reference',modelref['path'],'--annotator-model-reference',teacherref['path'],
                   '--ms-source',str(RT/'source_curated_review_v2'),'--poll-seconds','30',
                   '--initial-source-review-limit',str(2 if gpu==6 else 1),
                   '--continuation-authority',str(RT/('curated_collab3_review_n7g%d_continue_v1.json'%gpu)),
                   '--run','--watch'])
    return cfg


def selected(cfg):
    plan=checked(cfg['plan']);gpu=cfg['gpus'][0]
    ids=[]
    for member,info in plan['members'].items():
        if info['mode']=='consumer':
            for row in checked(plan['selected_sources'][member]):
                if row['source_ordinal']%2==gpu-6:ids.append(row['id'])
    a.need(len(ids)==15000 and len(set(ids))==15000,'exact original ordinal%2 review owner inventory changed')
    return ids,222939


def verify_capture(cfg):
    snap=checked(cfg['holder_capture']);gpu=cfg['gpus'][0]
    age=time.time_ns()-snap['captured_unix_ns']
    a.need(0<=age<=1200*10**9 and snap['holders']==cfg['holders'] and snap['gpus']==[gpu] and
           snap['node']=='N7' and snap['mode']=='review' and snap['signals_sent'] is False,
           'capture stale/not exact authorized holder')
    a.need(lease_value(base.lease_snapshot(Path('/tmp/od_reserve_%d'%gpu)))==snap['leases'][0],'lease changed since exact capture')
    lock_inventory(gpu)
    owner=cfg['holders'][0]
    base.check_holder(owner,base.identity(owner['pid']),gpu,base.uuid_for(gpu),base.apps())
    fd=os.pidfd_open(owner['pid'])
    try:a.need(not select.select([fd],[],[],0)[0],'captured holder exited before admission')
    finally:os.close(fd)
    a.need(base.headroom()['headroom']>=base.required_budget(cfg),'fresh headroom below 160GiB review admission')


ADMITTED_CAPTURE=None
original_acquire_lease=base.acquire_lease


def acquire_captured_lease(entry):
    a.need(ADMITTED_CAPTURE is not None and ADMITTED_CAPTURE['gpus']==[entry['gpu']], 'no exact admitted review capture')
    a.need(lease_value(entry['old'])==ADMITTED_CAPTURE['leases'][0], 'captured lease changed under physical node FLOCK')
    return original_acquire_lease(entry)


def validate(cfg):
    global ADMITTED_CAPTURE
    a.need(cfg['node']=='N7' and cfg['gpus'] in ([6],[7]) and cfg.get('release_authorized') is True,
           'explicit N7 physical GPU6/7 authorization required')
    gpu=cfg['gpus'][0]
    node()
    a.need(len(cfg['holders'])==1 and cfg['holders'][0]['gpu']==gpu and cfg['env']['CUDA_VISIBLE_DEVICES']==str(gpu),
           'exact single holder/CVD physical owner mismatch')
    # Reconstruct deterministic argv/out/phase/input/sampling policy from the admitted refs.
    reconstructed=config(gpu,cfg['holder_capture'],cfg['modelref'],cfg['annotator_model_reference'],True)
    a.need(cfg==reconstructed,'immutable launch differs from exact build; no path/model/alias override')
    selected(cfg);verify_capture(cfg)
    ADMITTED_CAPTURE=checked(cfg['holder_capture'])
    progress=a.strict((INPUT/'progress.json').read_bytes())
    a.need(progress['plan_ref']==cfg['plan'] and progress['full_source_denominator']==222939 and
           progress['full_batch_denominator']==448,'independent review input not bound to current collab3 plan')


def summaries(cfg,out):
    # Resident watcher exit never pretends whole448-batch review completion.
    return {}


base.acquire_lease=acquire_captured_lease
base.AUTHORIZED_PAIRS={'N7':((6,),(7,))}
base.validate_pair=validate
base.selected_sources=selected
base.check_model=lambda cfg:model_preflight(cfg['modelref'],cfg['annotator_model_reference'])
base.stage_summaries=summaries


def main():
    if '--capture-current' not in sys.argv and '--build' not in sys.argv and '--readonly-admission' not in sys.argv:
        a.need(a.sha(RT/'resource_manager.py')==BASE_SHA,'frozen base lifecycle SHA changed')
        return base.main()
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--capture-current',action='store_true')
    mode.add_argument('--build',action='store_true')
    mode.add_argument('--readonly-admission',action='store_true')
    parser.add_argument('--gpu',type=int,choices=(6,7))
    parser.add_argument('--model-reference')
    parser.add_argument('--annotator-model-reference')
    parser.add_argument('--capture')
    parser.add_argument('--capture-sha')
    parser.add_argument('--authorize-release',action='store_true')
    parser.add_argument('--config')
    parser.add_argument('--config-sha')
    parser.add_argument('--manager-sha')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.readonly_admission:
        cfg=checked({'path':args.config,'sha256':args.config_sha})
        validate(cfg);base.check_sources(cfg,args.config,args.config_sha,args.manager_sha)
        value=dict(status='readonly_admitted',node='N7',gpus=cfg['gpus'],config=a.reference(args.config),
                   manager=a.reference(__file__),headroom=base.headroom(),signals_sent=False,
                   resource_locks_acquired=0,leases_created=False,model_calls=0)
    else:
        a.need(args.gpu in (6,7) and args.model_reference and args.annotator_model_reference,'explicit GPU/model/teacher refs required')
        modelref=a.reference(args.model_reference);teacherref=a.reference(args.annotator_model_reference)
        if args.capture_current:value=capture(args.gpu,modelref,teacherref)
        else:
            a.need(args.capture and args.capture_sha,'exact fresh capture path/SHA required')
            value=config(args.gpu,{'path':args.capture,'sha256':args.capture_sha},modelref,teacherref,args.authorize_release)
            validate(value)
    with Path(args.out).open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'path':args.out,'sha256':a.sha(args.out),'signals_sent':False,'model_calls':0}),flush=True)


if __name__=='__main__':
    main()
