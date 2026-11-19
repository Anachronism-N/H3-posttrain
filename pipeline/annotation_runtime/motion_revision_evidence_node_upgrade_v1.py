"""Safe evidence-policy amendment; native drain/ACK, immutable new code, same source ledger."""
import argparse,fcntl,hashlib,importlib.util,json,os,shlex,subprocess,sys,time
from pathlib import Path
RT=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime')
HOSTS={'N1':'28.216.19.69','N2':'28.216.17.70','N3':'28.216.19.213','N6':'28.216.19.225','N8':'28.216.18.136'}
POLICY='window_visibility_boundary_evidence_v1'


def live_script(script):
    found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:a=[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x]
        except (OSError,UnicodeError):continue
        if len(a)>2 and a[2]==str(script):found.append(int(proc.name))
    return found


def run(node,pilot):
    assert HOSTS[node] in subprocess.check_output(['hostname','-I'],text=True).split()
    root=Path('/root/stage2_motion_revision_final_main_'+node+'_20261028_v1')
    sys.path.insert(0,str(root/'code'))
    from stage2_local_publication_v1 import checked,write_once
    lock=os.open(root/'evidence_upgrade_v1.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deployment=json.loads((root/'deployment.json').read_bytes());old_entries=[]
    for e in deployment['configs']:
        bp=root/'runtime_bindings'/(e['owner']+'.json');binding=json.loads(bp.read_bytes()) if bp.exists() else dict(config_ref=e['config_ref']);c=checked(binding['config_ref']);assert c['revision']=='english_merged_edges_motion_revision_v3'
        old_entries.append(dict(config_ref=binding['config_ref'],feeder_ref=e['feeder_ref'],config=c,binding=binding,binding_path=str(bp)))
    receipt=dict(node=node,evidence_policy=POLICY,old_entries=old_entries,original_outputs_budgets_preserved=True,no_source_reassignment=True,no_unknown_process_signals=True,created_unix=time.time())
    write_once(root/'evidence_upgrade_v1_before_drain.json',receipt)
    print(json.dumps(dict(event='evidence_before_drain_recorded',node=node)),flush=True)
    stop_names=set()
    uploader_scripts=set()
    for entry in old_entries:
        c=entry['config'];count=c.get('max_active_conversations',8)
        stop_names.add('FEEDER_STOP_BATCH16' if count>8 else 'FEEDER_STOP')
        uploader_scripts.add(Path(c['producer']['path']).parent/('motion_revision_uploader_batch16_v1.py' if count>8 else 'stage2_node_uploader_v1.py'))
    for name in stop_names:
        p=root/'out'/name
        if not p.exists():write_once(p,dict(reason='Authorized evidence-policy correction; drain original claims without model retry',evidence_policy=POLICY,models_called=0))
    deadline=time.monotonic()+3600
    while True:
        ready=True
        for entry in old_entries:
            q=Path(entry['config']['queue'])
            if (q/'failure.json').exists():raise RuntimeError('Original manager failed; preserve and investigate '+str(q))
            p=q/'resources_safe.json'
            if not p.exists():ready=False;continue
            safe=json.loads(p.read_bytes());assert all(safe.get(k) is True for k in ('all_holders_safe','locks_released','leases_restored'))
            assert json.loads((q/'worker_exit.json').read_bytes())['exit_code']==0
        if ready:break
        assert time.monotonic()<deadline,'Original live models still draining; no new model budget'
        time.sleep(3)
    write_once(root/'evidence_upgrade_v1_native_safe.json',dict(node=node,old_configs=[e['config_ref'] for e in old_entries],safe_flags_all_true=True,observed_unix=time.time()))
    print(json.dumps(dict(event='evidence_old_native_safe',node=node)),flush=True)
    while True:
        keys={tuple(p.relative_to(root/'out/sources').parts[:2]) for p in (root/'out/sources').glob('*/*/commit.json')};acked=set();pending=False
        for directory in (root/'out/outbox').iterdir():
            if not (directory/'sealed.json').exists():continue
            seal=json.loads((directory/'sealed.json').read_bytes());ap=directory/'upload_receipt.json'
            if not ap.exists():pending=True;continue
            ack=json.loads(ap.read_bytes());assert ack['batch_id']==seal['batch_id'] and ack['archive_sha256']==seal['archive_sha256'];acked.update(map(tuple,seal['source_keys']))
        if not pending and keys==acked:break
        assert time.monotonic()<deadline,'Old local results still await shared ACK; never retry model'
        time.sleep(3)
    write_once(root/'evidence_upgrade_v1_ACK_closed.json',dict(source_count=len(keys),all_old_local_results_ACKed=True,models_called=0,observed_unix=time.time()))
    for name in ('UPLOAD_STOP','UPLOAD_STOP_BATCH16'):
        p=root/'out'/name
        if not p.exists():write_once(p,dict(reason='Old native-safe and all ACKed; single-node uploader evidence-policy handoff',models_called=0))
    until=time.monotonic()+30
    while any(live_script(p) for p in uploader_scripts):
        assert time.monotonic()<until,'Original uploader still live; do not start duplicate uploader';time.sleep(.5)
    code=root/'evidencecode_v1';code.mkdir(exist_ok=False);hashes={}
    files={'motion_revision_codec_v1.py':'motion_revision_codec_evidence_v1.py','motion_revision_model_worker_evidence_v1.py':'motion_revision_model_worker_evidence_v1.py','motion_revision_feeder_evidence_v1.py':'motion_revision_feeder_evidence_v1.py','motion_revision_uploader_evidence_v1.py':'motion_revision_uploader_evidence_v1.py','stage2_local_publication_v1.py':'stage2_local_publication_v1.py','resource_manager.py':'resource_manager.py'}
    for count in {e['config'].get('max_active_conversations',8) for e in old_entries}:files['motion_revision_native_manager_evidence'+str(count)+'_v1.py']='motion_revision_native_manager_evidence'+str(count)+'_v1.py'
    for target,source in files.items():
        raw=(RT/source).read_bytes()
        if target=='motion_revision_codec_v1.py':raw=raw.replace(b'max_num_seqs=8,max_model_len',b'max_num_seqs='+str(old_entries[0]['config'].get('max_active_conversations',8)).encode()+b',max_model_len')
        compile(raw,str(code/target),'exec');p=code/target;fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        hashes[str(p)]=hashlib.sha256(raw).hexdigest()
    # Each scheduler size gets its own immutable codec directory if a node is mixed.
    entries=[];acceptance=root/'evidence_acceptance_v1.json'
    for entry in old_entries:
        old=entry['config'];count=old.get('max_active_conversations',8);paircode=code
        if count!=old_entries[0]['config'].get('max_active_conversations',8):
            paircode=root/('evidencecode'+str(count)+'_v1');paircode.mkdir(exist_ok=False)
            for target,source in files.items():
                raw=(RT/source).read_bytes()
                if target=='motion_revision_codec_v1.py':raw=raw.replace(b'max_num_seqs=8,max_model_len',b'max_num_seqs='+str(count).encode()+b',max_model_len')
                compile(raw,str(paircode/target),'exec');p=paircode/target;fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
                with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
                hashes[str(p)]=hashlib.sha256(raw).hexdigest()
        path=paircode/('motion_revision_native_manager_evidence'+str(count)+'_v1.py');spec=importlib.util.spec_from_file_location('evidence_manager_'+str(count),path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        holders=[]
        for gpu in old['gpus']:
            uuid=m.b.uuid_for(gpu);apps=[x for x in m.b.apps() if x['uuid']==uuid];assert len(apps)==1,'Restored authorized holder only';holder=m.b.identity(apps[0]['pid']);holder.update(gpu=gpu,uuid=uuid);m.b.check_holder(holder,m.b.identity(holder['pid']),gpu,uuid,m.b.apps());holders.append(holder)
        phase=old['phase']+'_evidence1';cp=root/(phase+'_config.json');producer=dict(path=str(path),sha256=hashes[str(path)]);c=dict(old);is_pilot=pilot and old['gpus']==[4,5]
        c.update(phase=phase,queue=str(root/'queues'/phase),config_path=str(cp),holders=holders,producer=producer,resource_manager_ref=producer,source_sha256=dict(old['source_sha256'],**hashes),argv=['/opt/venv_vllm/bin/python','-B',str(path),'worker','--config',str(cp)],evidence_policy=POLICY,evidence_pilot=is_pilot,evidence_acceptance_path=str(acceptance),max_active_conversations=count)
        cr=write_once(cp,c);m.check_sources(c,cp,cr['sha256'],m.ORIGINAL_BASE_SHA);m.b.check_model(c)
        f=checked(entry['feeder_ref']);f.update(evidence_policy=POLICY,evidence_pilot=is_pilot,evidence_acceptance_path=str(acceptance));fr=write_once(root/(phase+'_feeder.json'),f)
        entries.append(dict(config_ref=cr,feeder_ref=fr,owner=c['owner'],gpus=c['gpus'],phase=phase))
    write_once(root/'evidence_deployment_v1.json',dict(node=node,evidence_policy=POLICY,configs=entries,source_ledger_unchanged=True,old_results_budgets_preserved=True,created_unix=time.time()))
    sock=root/'evidence_tmux_v1.sock';logs=root/'evidencelogs_v1';logs.mkdir();session='evidence-v1'
    def window(name,args,first=False):
        command=shlex.join(args)+' > '+shlex.quote(str(logs/(name+'.log')))+' 2>&1'
        subprocess.run(['tmux','-S',str(sock),'new-session','-d','-s',session,'-n',name,'-c','/tmp',command] if first else ['tmux','-S',str(sock),'new-window','-d','-t',session,'-n',name,'-c','/tmp',command],check=True)
    window('uploader',['/opt/venv_eval/bin/python','-B',str(code/'motion_revision_uploader_evidence_v1.py'),'--local-root',str(root/'out'),'--shared-root',str(RT.parent/'motion_revision_english_merged_edges_final_20261028_v1/production_results'),'--node',node,'--seconds','86400'],True)
    previous=None
    for entry in sorted(entries,key=lambda e:not (pilot and e['gpus']==[4,5])):
        if previous:
            prev=checked(previous['config_ref']);until=time.monotonic()+420;state=root/'worker_state'/prev['owner']/prev['phase']
            while not (state/'load.receipt.json').exists():assert time.monotonic()<until and not (Path(prev['queue'])/'failure.json').exists();time.sleep(1)
        c=checked(entry['config_ref']);path=Path(c['producer']['path']);spec=importlib.util.spec_from_file_location('evidence_admit_'+c['owner'],path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.validate(c)
        bp=root/'runtime_bindings'/(c['owner']+'.json');original=next(e for e in old_entries if e['config']['owner']==c['owner']);write_once(root/'evidence_binding_history_v1'/(c['owner']+'.json'),original['binding'])
        new_binding=dict(config_ref=entry['config_ref'],previous_config_ref=original['config_ref'],state_layout='by_phase',batch_size=c['max_active_conversations'],evidence_policy=POLICY,source_claim_reassignment=False);temp=bp.with_name(bp.name+'.evidence1.tmp');write_once(temp,new_binding);os.replace(temp,bp);fd=os.open(bp.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
        window(c['phase'],['/opt/venv_vllm/bin/python','-B',str(path),'--config',c['config_path'],'--config-sha',entry['config_ref']['sha256'],'--manager-sha',m.ORIGINAL_BASE_SHA]);until=time.monotonic()+40
        while not (Path(c['queue'])/'live.json').exists():assert time.monotonic()<until and not (Path(c['queue'])/'failure.json').exists();time.sleep(.2)
        window(c['phase']+'-feeder',['/opt/venv_eval/bin/python','-B',str(path.parent/'motion_revision_feeder_evidence_v1.py'),'--config',entry['feeder_ref']['path']]);previous=entry
        print(json.dumps(dict(event='evidence_native_admitted',node=node,owner=c['owner'],batch=c['max_active_conversations'],pilot=c['evidence_pilot'])),flush=True)
    os.close(lock)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',required=True,choices=tuple(HOSTS));p.add_argument('--pilot',action='store_true');a=p.parse_args();run(a.node,a.pilot)
