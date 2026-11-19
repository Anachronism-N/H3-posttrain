"""Resume frozen rich/MS fusion on three disjoint N1 shards; never rewrite v2."""
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import time

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
MS = ROOT / 'diff/logs_omnidance/exp/MS'
SCRIPT = Path(__file__).absolute()
LEGACY = MS / 'dance_caption_fusion6k_manager_v2.py'
LEGACY_SHA = '3840aa1df7c5e331d8aea5e5470acf8d8bcfc6cbe4c4f295b07a680b3b8365be'
SOURCE = MS / 'dance_caption_fusion6k_source_v2/records.jsonl'
SOURCE_SHA = 'd3711dcce17cc0ce9ced76f0a213de115b973f3558ccda431e5cb4764d3ddb77'
OLD = MS / 'dance_caption_fusion6k_rewrite_v2'
OUT = MS / 'fusion_parallel_rewrite_v3'
QUEUE = MS / 'fusion_parallel_queue_v3'
EXPORT = MS / 'fusion_source_assumed_full6000_v3'
VERSION = 'rich-ms-fusion6k-same-recipe-parallel-v3'
UUIDS = {0: 'GPU-305f1213-85ff-38b5-7295-c6839023bd20',
         1: 'GPU-2e6ff906-0d91-c994-92e7-9bedb0947be8',
         2: 'GPU-1a898afb-ce06-d40b-043a-45679f17c945'}
PYTHON = '/opt/venv_h3/bin/python'
OLD_PROCESSES = [(2039786, 517786074, False), (2039798, 517786262, True)]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load_module(name, path, expected=None):
    if expected:
        assert sha(path) == expected, str(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def save(path, value):
    # Exclusive creation preserves failures and immutable upstream outputs.
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

def event(kind, **data):
    row = dict(event=kind, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **data)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    with (QUEUE / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()
    return row

def identity(pid):
    p = Path('/proc') / str(pid)
    stat = (p / 'stat').read_text()
    env = dict(item.decode().split('=', 1) for item in (p / 'environ').read_bytes().split(b'\0') if item)
    return dict(pid=pid, uid=p.stat().st_uid,
        uids=[int(v) for line in (p / 'status').read_text().splitlines() if line.startswith('Uid:') for v in line.split()[1:]],
        start=int(stat[stat.rfind(')') + 2:].split()[19]),
        argv=[v.decode() for v in (p / 'cmdline').read_bytes().split(b'\0') if v],
        cgroup=(p / 'cgroup').read_text(), cuda=env.get('CUDA_VISIBLE_DEVICES'),
        owner_cgroup=(Path('/proc/self/cgroup')).read_text())

def stop_old():
    for pid, start, worker in OLD_PROCESSES:
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            event('authorized_old_already_exited', pid=pid)
            continue
        try:
            value = identity(pid)
            argv = [PYTHON, '-B', '-u', str(LEGACY)] + (['--worker'] if worker else []) + ['--source-sha', LEGACY_SHA]
            assert value['uid'] == 0 and value['uids'] == [0]*4 and value['start'] == start and value['argv'] == argv
            assert value['cgroup'] == value['owner_cgroup'] and identity(pid) == value
            event('authorized_old_exact_SIGTERM', identity=value)
            signal.pidfd_send_signal(fd, signal.SIGTERM)
            assert select.select([fd], [], [], 30)[0], 'No escalation allowed'
            event('authorized_old_exited', pid=pid)
        finally:
            os.close(fd)

def rows_and_snapshot():
    legacy = load_module('legacy', LEGACY, LEGACY_SHA)
    helper = legacy.helper()
    assert sha(SOURCE) == SOURCE_SHA
    rows = [json.loads(line, object_pairs_hook=helper.unique_object) for line in SOURCE.read_text().splitlines()]
    assert len(rows) == len({r['id'] for r in rows}) == 6000
    assert all(r['split'] == 'train' and re.fullmatch(r'[A-Za-z0-9_-]+', r['id']) for r in rows)
    assert json.loads((SOURCE.parent / 'source_complete.json').read_text())['records_sha256'] == SOURCE_SHA
    def inspect(item):
        index, row = item
        path = OLD / (row['id'] + '.json')
        if not path.exists():
            return index, row, None, dict(id=row['id'], reason='missing')
        try:
            data = path.read_bytes()
            rec = json.loads(data, object_pairs_hook=helper.unique_object)
            assert rec['id'] == row['id'] and rec['original_source_record'] == row
            assert rec['split'] == row['split'] and rec['source_group'] == row['source_group']
            assert rec['source_sha256'] == LEGACY_SHA and rec['source_records_sha256'] == SOURCE_SHA
            assert rec['actual_text_input'] == legacy.actual_material(row)
            assert rec['actual_text_input_sha256'] == legacy.object_sha(legacy.actual_material(row))
            assert rec['raw_source_text_input_sha256'] == row['actual_text_input_sha256']
            assert rec['semantic_approval'] is False and rec['human_review_completed'] is False
            structural = [e for e in legacy.validate(rec.get('target'), rec['actual_text_input'], helper) if e not in helper.FORBIDDEN]
            assert not structural and rec['generated_tokens'] < legacy.MAX_TOKENS, structural or 'generation limit'
            return index, row, dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=data), None
        except (ValueError, TypeError, KeyError, AssertionError, OSError) as exc:
            return index, row, None, dict(id=row['id'], path=str(path), reason=str(exc), retained=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        inspected = list(pool.map(inspect, enumerate(rows)))
    accepted = [dict(id=row['id'], source_path=result['path'], sha256=result['sha256']) for _,row,result,_ in inspected if result]
    missing = [dict(index=i, id=row['id'], shard=i%3, reason=error) for i,row,result,error in inspected if not result]
    return rows, inspected, dict(total=6000, reused=len(accepted), missing=len(missing), accepted=accepted, tasks=missing,
        shard_counts={str(g):sum(t['shard']==g for t in missing) for g in UUIDS}, source_records_sha256=SOURCE_SHA,
        legacy_source_sha256=LEGACY_SHA, new_source_sha256=sha(SCRIPT), recipe_changed=False,
        semantic_review_performed=False, human_review_completed=False, val500_deferred=True)

def plan():
    QUEUE.mkdir(exist_ok=True)
    _, _, report = rows_and_snapshot()
    save(QUEUE / 'preflight_plan.json', report)
    event('CPU_plan_ready', reused=report['reused'], missing=report['missing'], shard_counts=report['shard_counts'])

def gpu_snapshot():
    gpus = {}
    text = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
    for line in text.splitlines():
        gpu,uuid,total,used=[v.strip() for v in line.split(',')]
        if int(gpu) in UUIDS:
            assert uuid == UUIDS[int(gpu)]
            gpus[int(gpu)] = dict(uuid=uuid,total=int(total),used=int(used),processes=[])
    assert set(gpus)==set(UUIDS)
    text = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    for line in text.splitlines():
        uuid,pid,used=[v.strip() for v in line.split(',')]
        for gpu in gpus.values():
            if gpu['uuid']==uuid:
                gpu['processes'].append(dict(pid=int(pid),used=int(used)))
    return gpus

def admit():
    mem = {line.split()[0].rstrip(':'):int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    cg = Path('/sys/fs/cgroup/memory')
    headroom = int((cg/'memory.limit_in_bytes').read_text())-int((cg/'memory.usage_in_bytes').read_text())
    assert mem['MemAvailable'] >= 180*2**20 and headroom >= 300*2**30
    snapshot=gpu_snapshot()
    event('resource_preflight', host_available=mem['MemAvailable']*1024, cgroup_headroom=headroom, gpus=snapshot)
    expected_holder=['/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/envs/longlive/bin/python',str(ROOT/'diff/scripts_omnidance/exp/hold_gpu.py')]
    for gpu,state in snapshot.items():
        assert len(state['processes'])<=1
        reserve=Path('/tmp/od_reserve_'+str(gpu))
        assert not reserve.exists() or (reserve.is_file() and not reserve.is_symlink() and reserve.stat().st_uid==0)
        reserve.touch()
        if not state['processes']:
            continue
        proc=state['processes'][0]
        fd=os.pidfd_open(proc['pid'])
        try:
            value=identity(proc['pid'])
            assert value['uids']==[0]*4 and value['argv']==expected_holder and value['cuda']==str(gpu)
            assert value['cgroup']==value['owner_cgroup'] and proc['used']<=1100
            assert gpu_snapshot()[gpu]==state and identity(proc['pid'])==value
            event('verified_project_holder_SIGTERM',gpu=gpu,identity=value)
            signal.pidfd_send_signal(fd,signal.SIGTERM)
            assert select.select([fd],[],[],15)[0]
        finally:
            os.close(fd)
    fresh=gpu_snapshot()
    assert all(not g['processes'] and g['used']<=1024 and g['total']>=80000 for g in fresh.values())
    event('three_gpu_admission_passed',gpus=fresh)

def materialize(inspected, report):
    OUT.mkdir(exist_ok=False)
    (OUT/'aggregate').mkdir()
    for gpu in UUIDS:
        (OUT/('shard'+str(gpu))).mkdir()
    tasks={g:[] for g in UUIDS}
    for index,row,result,error in inspected:
        if result:
            with (OUT/'aggregate'/(row['id']+'.json')).open('xb') as stream:
                stream.write(result['bytes'])
        else:
            tasks[index%3].append(row)
    for gpu, shard in tasks.items():
        path=OUT/('shard'+str(gpu))/'tasks.jsonl'
        with path.open('x',encoding='utf-8') as stream:
            for row in shard:
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        report.setdefault('manifest_sha256',{})[str(gpu)]=sha(path)
    sets=[{r['id'] for r in v} for v in tasks.values()]
    assert not sets[0]&sets[1] and not sets[1]&sets[2] and not sets[0]&sets[2]
    assert sum(map(len,sets))+report['reused']==6000
    save(OUT/'plan.json',report)

def worker(gpu, expected_sha):
    import torch
    import transformers
    assert sha(SCRIPT)==expected_sha
    legacy=load_module('legacy_worker',LEGACY,LEGACY_SHA)
    helper=legacy.helper()
    plan_record=json.loads((OUT/'plan.json').read_text())
    folder=OUT/('shard'+str(gpu))
    assert sha(folder/'tasks.jsonl')==plan_record['manifest_sha256'][str(gpu)]
    rows=[json.loads(line) for line in (folder/'tasks.jsonl').read_text().splitlines()]
    assert len(rows)==plan_record['shard_counts'][str(gpu)]
    staging=json.loads((legacy.MODEL/'local_staging_complete.json').read_text())
    assert sha(legacy.MODEL/'local_staging_complete.json')==legacy.OFFICIAL_SHA
    assert staging['complete'] and staging['official_content_verified'] and staging['revision']==legacy.REVISION
    for f in staging['files']:
        p=legacy.MODEL/f['name']
        assert p.is_file() and not p.is_symlink() and p.stat().st_size==f['bytes'] and f['official_content_match']
        if not f['name'].endswith('.safetensors'):
            assert sha(p)==f['sha256']
    names=helper.teacher_family(json.loads((legacy.MODEL/'config.json').read_text()))
    torch.manual_seed(42)
    torch.set_num_threads(2)
    processor=getattr(transformers,names[0]).from_pretrained(legacy.MODEL,local_files_only=True)
    config=getattr(transformers,names[2]).from_pretrained(legacy.MODEL,local_files_only=True)
    model,loading=getattr(transformers,names[1]).from_pretrained(legacy.MODEL,config=config,
        torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
    checks=helper.thinker_loading_checks(loading)
    model.eval().requires_grad_(False)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert not any(hasattr(model,n) for n in ('talker','code2wav','token2wav'))
    run=dict(version=VERSION,source_sha256=expected_sha,legacy_source_sha256=LEGACY_SHA,source_records_sha256=SOURCE_SHA,
        helper_sha256=legacy.HELPER_SHA,official_staging_sha256=legacy.OFFICIAL_SHA,official_revision=legacy.REVISION,
        model_local_preverified=True,model_byte_verification='prior immutable 25-file official SHA verification; current sizes and nonweight SHAs checked',
        loading_checks=checks,text_only=True,frozen=True,lora_source=None,instruction=legacy.SYSTEM,
        seed=42,do_sample=False,max_new_tokens=legacy.MAX_TOKENS,gpu=gpu,gpu_uuid=UUIDS[gpu],host='28.216.19.69',
        semantic_approval=False,human_review_completed=False,original_source_window_mapping='unknown')
    save(folder/('run_started_'+str(time.time_ns())+'.json'),run)
    event('shard_model_loaded',gpu=gpu,pid=os.getpid(),assigned=len(rows))
    failures=[]
    for index,row in enumerate(rows):
        path=folder/(row['id']+'.json')
        material=legacy.actual_material(row)
        if path.exists():
            result=json.loads(path.read_text())
            assert result['id']==row['id'] and result['source_sha256']==expected_sha
            assert result['actual_text_input_sha256']==legacy.object_sha(material)
        else:
            conversation=[dict(role='system',content=[dict(type='text',text=legacy.SYSTEM)]),
                dict(role='user',content=[dict(type='text',text=json.dumps(material,ensure_ascii=False))]),
                dict(role='assistant',content=[dict(type='text',text='{')])]
            inputs=helper.text_generation_inputs(processor,conversation,True).to(model.device)
            with torch.inference_mode():
                generated=model.generate(**inputs,max_new_tokens=legacy.MAX_TOKENS,do_sample=False,
                    eos_token_id=processor.tokenizer.eos_token_id,pad_token_id=processor.tokenizer.pad_token_id)
            tokens=generated[0,inputs['input_ids'].shape[1]:].tolist()
            raw='{'+processor.tokenizer.decode(tokens,skip_special_tokens=True)
            try:
                target=json.loads(raw,object_pairs_hook=helper.unique_object)
                errors=legacy.validate(target,material,helper)
            except (ValueError,TypeError,KeyError) as exc:
                target=None
                errors=['JSON parse: '+str(exc)]
            if len(tokens)>=legacy.MAX_TOKENS:
                errors.append('generation_reached_limit')
            structural=[e for e in errors if e not in helper.FORBIDDEN]
            result=dict(id=row['id'],split=row['split'],source_group=row['source_group'],genre=row['genre'],
                caption_version=VERSION,original_source_record=row,actual_text_input=material,
                actual_text_input_sha256=legacy.object_sha(material),raw_source_text_input_sha256=row['actual_text_input_sha256'],
                raw_generation=raw,raw_generated_token_ids=tokens,target=target,format_valid=not errors,
                validation_errors=errors,structural_errors=structural,source_assumed_exportable=not structural,
                completion_status='candidate' if not errors else 'failed_format',prompt_tokens=int(inputs['input_ids'].shape[1]),
                generated_tokens=len(tokens),source_sha256=expected_sha,source_records_sha256=SOURCE_SHA,
                legacy_source_sha256=LEGACY_SHA,official_staging_sha256=legacy.OFFICIAL_SHA,
                semantic_approval=False,human_review_completed=False,review_status='weak_unreviewed',window_mapping='unknown',
                created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
            save(path,result)
            del inputs, generated
        if result.get('structural_errors'):
            failures.append(row['id'])
        if index<3 or (index+1)%50==0 or index+1==len(rows):
            event('actual_shard_forward_progress',gpu=gpu,completed=index+1,assigned=len(rows),id=row['id'],structural_failures=len(failures))
    save(folder/('run_complete_'+str(time.time_ns())+'.json'),dict(run,processed=len(rows),complete=True,structural_failed_ids=failures))

def finalize():
    legacy=load_module('legacy_final',LEGACY,LEGACY_SHA)
    helper=legacy.helper()
    exporter=load_module('source_assumed_exporter',SCRIPT.parent/'export_fusion_source_assumed_v1.py')
    exporter.self_test()
    plan_record=json.loads((OUT/'plan.json').read_text())
    rows=[json.loads(line) for line in SOURCE.read_text().splitlines()]
    assert sha(SOURCE)==SOURCE_SHA
    records=[]
    unavailable=[]
    for row in rows:
        sid=row['id']
        paths=[p for p in [OUT/'aggregate'/(sid+'.json')]+[OUT/('shard'+str(g))/(sid+'.json') for g in UUIDS] if p.exists()]
        if len(paths)!=1:
            unavailable.append(dict(id=sid,reason='expected_one_result',paths=[str(p) for p in paths]))
            continue
        p=paths[0]
        data=p.read_bytes()
        try:
            rec=json.loads(data,object_pairs_hook=helper.unique_object)
            assert rec['original_source_record']==row and rec['id']==sid
            assert rec['source_records_sha256']==SOURCE_SHA
            assert rec['source_sha256'] in (LEGACY_SHA,plan_record['new_source_sha256'])
            assert rec['actual_text_input_sha256']==legacy.object_sha(legacy.actual_material(row))
            structural=[e for e in legacy.validate(rec.get('target'),rec['actual_text_input'],helper) if e not in helper.FORBIDDEN]
            assert not structural and rec['generated_tokens']<legacy.MAX_TOKENS,structural
            records.append((row,p,data,rec))
        except (AssertionError,ValueError,TypeError,KeyError) as exc:
            unavailable.append(dict(id=sid,path=str(p),reason=str(exc),raw_preserved=True))
    if unavailable:
        save(OUT/('unavailable_'+str(time.time_ns())+'.json'),unavailable)
        event('finalize_blocked',valid=len(records),unavailable=len(unavailable),ids=[r['id'] for r in unavailable])
        raise RuntimeError('Must repair only unavailable IDs; never claim batch complete')
    assert len(records)==len({r[0]['id'] for r in records})==6000
    EXPORT.mkdir(exist_ok=False)
    changed=0
    with (EXPORT/'captions.jsonl').open('x',encoding='utf-8') as caps, (EXPORT/'lineage.jsonl').open('x',encoding='utf-8') as lineage:
        for row,path,data,rec in records:
            body=rec['target']['motion_caption']
            normalized=exporter.clean(body)
            changed+=body!=normalized
            common=dict(id=row['id'],split=row['split'],source_group=row['source_group'],genre=row['genre'],
                caption_version='rich_ms_fusion_source_assumed_full6000_v3',source_policy=exporter.POLICY,
                source_action_correctness_assumed=True,semantic_review_performed=False,human_review_completed=False,
                original_source_window_mapping='unknown',faithfulness_status='not_individually_verified',
                source_record_path=str(path),source_record_sha256=hashlib.sha256(data).hexdigest())
            caps.write(json.dumps(dict(common,motion_caption=normalized),ensure_ascii=False)+'\n')
            lineage.write(json.dumps(dict(common,raw_fused_caption=body,exported_caption=normalized,text_changed=body!=normalized,
                original_source_record=row,actual_text_input=rec['actual_text_input'],actual_text_input_sha256=rec['actual_text_input_sha256'],
                raw_source_text_input_sha256=rec['raw_source_text_input_sha256'],raw_rich_text_sha256=row['raw_rich_text_sha256'],
                raw_MS_text_sha256=row['raw_MS_text_sha256'],source_records_sha256=SOURCE_SHA,
                generation_source_code_sha256=rec['source_sha256'],legacy_source_code_sha256=LEGACY_SHA,
                raw_generation_sha256=hashlib.sha256(rec['raw_generation'].encode()).hexdigest(),
                raw_generated_token_ids_sha256=legacy.object_sha(rec['raw_generated_token_ids']),
                original_completion_status=rec['completion_status'],original_rule_flags=rec['validation_errors'],
                original_flags_do_not_block_export=True,source_refs=rec['target']['source_refs'],conflicts=rec['target']['conflicts'],
                uncertainty=rec['target']['uncertainty']),ensure_ascii=False)+'\n')
        for stream in (caps,lineage):
            stream.flush()
            os.fsync(stream.fileno())
    report=dict(complete=True,batch6000_complete=True,included=6000,unique_ids=6000,unavailable=0,
        source_policy=exporter.POLICY,source_action_correctness_assumed=True,semantic_review_performed=False,human_review_completed=False,
        source_records_sha256=SOURCE_SHA,new_source_code_sha256=sha(SCRIPT),legacy_source_code_sha256=LEGACY_SHA,
        reused_old=plan_record['reused'],new_sharded=6000-plan_record['reused'],shard_counts=plan_record['shard_counts'],
        deterministic_cleaned=changed,raw_outputs_unchanged=True,val500_deferred=True,split='train',
        cleanup='original exporter clean: pronouns/listed synthetic timing only; body directions/degrees/amplitude/repetition unchanged',
        exporter_source_sha256=sha(SCRIPT.parent/'export_fusion_source_assumed_v1.py'),
        files={name:sha(EXPORT/name) for name in ('captions.jsonl','lineage.jsonl')})
    save(EXPORT/'complete.json',report)
    save(OUT/'run_complete.json',report)
    event('all6000_source_assumed_export_complete',report=report)

def manager():
    assert os.getuid()==0 and QUEUE.is_dir() and (QUEUE/'preflight_plan.json').is_file()
    lock=(QUEUE/'manager.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pre=json.loads((QUEUE/'preflight_plan.json').read_text())
    assert pre['new_source_sha256']==sha(SCRIPT)
    if pre['missing']:
        stop_old()
    _,inspected,report=rows_and_snapshot()
    materialize(inspected,report)
    if not report['missing']:
        finalize()
        return
    admit()
    running=[]
    for gpu in UUIDS:
        if report['shard_counts'][str(gpu)]==0:
            continue
        log=(QUEUE/('shard'+str(gpu)+'.log')).open('xb')
        argv=['env','CUDA_VISIBLE_DEVICES='+str(gpu),'OMP_NUM_THREADS=2','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
            PYTHON,'-B','-u',str(SCRIPT),'--worker',str(gpu),'--source-sha',sha(SCRIPT)]
        child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
        running.append((gpu,child,log))
        event('managed_shard_started',gpu=gpu,identity=identity(child.pid),argv=argv,log=str(log.name))
    while any(child.poll() is None for _,child,_ in running):
        for gpu,child,_ in running:
            if child.poll() is None:
                Path('/tmp/od_reserve_'+str(gpu)).touch()
        time.sleep(60)
    for gpu,child,log in running:
        log.close()
        event('managed_shard_exited',gpu=gpu,pid=child.pid,exit_code=child.returncode)
    assert all(child.returncode==0 for _,child,_ in running), 'Partials retained; resume exact shard only'
    finalize()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--worker',type=int,choices=[0,1,2])
    parser.add_argument('--source-sha')
    parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    if args.plan:
        plan()
    elif args.worker is not None:
        worker(args.worker,args.source_sha)
    elif args.finalize:
        finalize()
    else:
        manager()
