"""Finite source/resource handoff; no models, claims, leases, or production edits."""
from pathlib import Path
import concurrent.futures,datetime,hashlib,io,json,os,shlex,stat,subprocess,tarfile,tempfile,uuid
ROOT=Path('/tmp/ms_pose2d_draft_20261026T1427Z/launcher');BASE=ROOT.parent
P=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/external_ms_unified_clusters80k_20261027_v1')
SOURCES=['unified_lane_rollover_v1.py','unified_access_transport_rollover_v1.py','unified_access_transport_v3.py','fullvideo_validated_helpers_ed283d684.py']+['unified_code_snapshot_v1/'+n for n in ['ms_unified_novlm_v1.py','ms_fullframe_pose2d_draft_v2.py','ms_fullframe_pose2d_draft_v1.py','ms_pose2d_draft.py','dwpose_full_extract.py']]
EXPECTED={'unified_lane_rollover_v1.py':'95eff3b6af6104d090b318647bb6442aa95f6b536c6beb4349ff33f8138af1b9','unified_access_transport_rollover_v1.py':'c8d36e48e930b5dbd235bb21541fe05adc4b6307060bf2bb67658f5f7ff35406','unified_access_transport_v3.py':'c946aec60e1498bbd48755d1ae783bbfc0955ec50b444434def841498c2a9f0d','fullvideo_validated_helpers_ed283d684.py':'ed283d6844830a1c2da109361391dbc44858a16c0e0262ca3ace117b3c48cb78'}
NODES=[('a100-extra-03','launcher',None),('a100-extra-04','worker-0','11.254.248.52'),('a100-extra-05','worker-1','11.239.106.11'),('a100-extra-06','worker-2','11.239.84.43')]
PROBE=r'''
from pathlib import Path
import datetime,hashlib,json,os,pwd,stat,subprocess,sys,time,xml.etree.ElementTree as ET
root=Path(sys.argv[1]);sources=json.loads(sys.argv[2]);worker=sys.argv[3]
def jsonfile(p):
 b=p.read_bytes()
 if len(b)>1024**2 or bytes([0]) in b or not b.endswith(bytes([10])):raise ValueError('bounded/NUL/partial metadata retained')
 return json.loads(b)
def identity(pid):
 p=Path('/proc')/str(pid)
 try:
  f=(p/'stat').read_text().rsplit(') ',1)[1].split();uid=p.stat().st_uid
  try:owner=pwd.getpwuid(uid).pw_name
  except KeyError:owner='uid-only'
  return dict(pid=pid,uid=uid,owner=owner,start_ticks=int(f[19]),state=f[0],comm=(p/'comm').read_text().strip())
 except OSError:return dict(pid=pid,owner='not_resolvable_in_container_pid_namespace',ownership_unverified=True)
q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
if q.returncode:raise RuntimeError('GPU query failed')
gpus={int(a):dict(uuid=b,pci_bus_id=c,name=d,total_MiB=int(e),used_MiB=int(f),utilization_pct=int(g)) for a,b,c,d,e,f,g in ([v.strip() for v in line.split(',')] for line in q.stdout.splitlines())}
if set(gpus)!=set(range(8)):raise ValueError('expected physical GPU inventory changed')
x=subprocess.run(['nvidia-smi','-q','-x'],capture_output=True,text=True,timeout=20)
if x.returncode:raise RuntimeError('GPU process XML query failed')
byuuid={g.findtext('uuid'):g for g in ET.fromstring(x.stdout).findall('gpu')};locklines=Path('/proc/locks').read_text().splitlines();devices=[]
for index in range(4,8):
 device=dict(index=index,**gpus[index]);device['processes']=[]
 for info in byuuid[device['uuid']].findall('processes/process_info'):
  pid=int(info.findtext('pid'));device['processes'].append(dict(type=info.findtext('type'),used_memory=info.findtext('used_memory'),identity=identity(pid)))
 files=[];scopedlocks=[]
 for p in sorted((root/'gpu_locks').glob('physical_gpu'+str(index)+'*')):
  s=p.lstat();entry=dict(path=str(p),inode=s.st_ino,device=s.st_dev,is_symlink=stat.S_ISLNK(s.st_mode))
  if p.name.endswith('.lease.json') and stat.S_ISREG(s.st_mode):
   value=jsonfile(p);entry['lease']={k:value.get(k) for k in ['worker','physical_gpu_index','gpu_uuid','pid','start_ticks','edition','heartbeat_unix']};entry['recorded_PID_identity']=identity(value['pid']) if type(value.get('pid')) is int else None
  files.append(entry)
  if p.name.endswith('.lock'):
   for row in locklines:
    fields=row.split()
    if 'FLOCK' not in row or len(fields)<6:continue
    parts=fields[5].split(':')
    if len(parts)==3 and int(parts[0],16)==os.major(s.st_dev) and int(parts[1],16)==os.minor(s.st_dev) and int(parts[2])==s.st_ino:scopedlocks.append(dict(lock=row,owner=identity(int(fields[4]))))
 device['own_runner_lock_lease_files']=files;device['actual_kernel_locks_on_these_files']=scopedlocks;device['resource_admission_performed']=False;device['worker_start_authorized']=False;devices.append(device)
mem={}
for maximum,current,label in [('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes','v1'),('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current','v2')]:
 if Path(maximum).exists():
  limit=Path(maximum).read_text().strip();usage=int(Path(current).read_text());mem.update(cgroup_version=label,cgroup_limit_bytes=None if limit=='max' else int(limit),cgroup_usage_bytes=usage);break
info=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines());available=int(info['MemAvailable'].split()[0])*1024;headroom=None if mem.get('cgroup_limit_bytes') is None else max(0,mem['cgroup_limit_bytes']-mem['cgroup_usage_bytes']);mem['host_MemAvailable_bytes']=available;mem['effective_shared_headroom_bytes']=available if headroom is None else min(available,headroom)
s=os.statvfs('/dev/shm');mem['dev_shm']={'total_bytes':s.f_blocks*s.f_frsize,'available_bytes':s.f_bavail*s.f_frsize}
hashes={}
for rel in sources:
 p=root/rel
 with p.open('rb') as f:
  size=os.fstat(f.fileno()).st_size
  if size>2**20:raise ValueError('code file cap exceeded')
  raw=f.read(2**20+1)
 if len(raw)!=size:raise ValueError('source size changed')
 hashes[rel]=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
active=[]
for gpu in range(4):
 v=jsonfile(root/'logs'/('rollover_gpu'+str(gpu)+'_progress.json'));i=identity(v['pid']);active.append(dict(gpu=gpu,pid=v['pid'],start_ticks_match=i.get('start_ticks')==v['start_ticks'],status=v['status'],heartbeat_age_seconds=round(time.time()-v['heartbeat_unix'],1),published=v['published'],processed=v['unique_processed']))
print(json.dumps(dict(node=worker,role=root.name,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU4_to_7=devices,memory=mem,source_hashes=hashes,active_GPU0_to_3=active,lock_scope='only the established own runner gpu_locks directory; absence is not proof of no external leases',no_GPU_worker_started=True,no_claim_lease_or_config_writes=True)),flush=True)
'''
def encode(value):return (json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)+chr(10)).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path,cap=2**20):
 with path.open('rb') as f:
  size=os.fstat(f.fileno()).st_size
  if size>cap:raise ValueError('bounded handoff input too large:'+str(path))
  raw=f.read(cap+1)
 if len(raw)!=size:raise ValueError('handoff input size mismatch')
 return raw
def run(node):
 worker,role,host=node;args=['python3','-B','-c',PROBE,str(BASE/role),json.dumps(SOURCES),worker]
 cmd=args if host is None else ['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pose2d_resource_known_hosts_20261026','-o','ConnectTimeout=8',host,shlex.join(args)]
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=70)
 if r.returncode:raise RuntimeError(worker+':readonly probe failed:'+r.stderr[-1000:])
 if '\x00' in r.stdout or not r.stdout.endswith(chr(10)):raise ValueError('partial/NUL node evidence retained')
 return json.loads(r.stdout)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:nodes=list(pool.map(run,NODES))
files={};entries=[]
for rel in SOURCES:
 raw=read(ROOT/rel);digest=sha(raw);compile(raw,str(ROOT/rel),'exec')
 if rel in EXPECTED and digest!=EXPECTED[rel]:raise ValueError('known frozen source changed:'+rel)
 if any(n['source_hashes'][rel]['sha256']!=digest or n['source_hashes'][rel]['bytes']!=len(raw) for n in nodes):raise ValueError('worker production sources differ:'+rel)
 member='production/'+rel;files[member]=raw;entries.append(dict(member=member,source_path=str(ROOT/rel),sha256=digest,bytes=len(raw),frozen_capture=True,all_four_nodes_byte_identity_verified=True))
exp=read(ROOT/'batch_rpc_publication_experiment_v1.py');current='e794817d585ddc6e4449553de760593d0087081326c6d7d7fa873b553148ece6';historical='c5a394b1b50755c6e8e819ed31b53499aa910dd0d477ebcb5b6f47b989ac689f'
if sha(exp)!=current:raise ValueError('experiment current SHA changed')
old=b"LOCAL_ROOT = Path('/tmp/ms_pose2d_draft_20261026T1427Z/worker-0/publication_experiments') / RUN";new=b"LOCAL_ROOT = Path('/tmp/ms_pose2d_draft_20261026T1427Z/publication_experiments') / RUN"
if exp.count(old)!=1:raise ValueError('sole LOCAL_ROOT revision not identifiable')
reconstructed=exp.replace(old,new)
if sha(reconstructed)!=historical:raise ValueError('recorded execution SHA reconstruction differs')
for name,raw,kind in [('experiments/batch_rpc_publication_experiment_current_e794817d.py',exp,'captured_current_LOCAL_ROOT_relocated_not_reexecuted'),('experiments/batch_rpc_publication_experiment_reconstructed_c5a394b1.py',reconstructed,'one_line_reconstruction_SHA_matches_recorded_execution_not_recovered_original_file')]:
 compile(raw,name,'exec');files[name]=raw;entries.append(dict(member=name,sha256=sha(raw),bytes=len(raw),provenance=kind,frozen_capture=True))
report=read(ROOT/'logs/batch_rpc_publication_validated_result.json');value=json.loads(report);result=encode(value['result'])
if sha(result)!=value['report_SHA256']:raise ValueError('saved experiment result SHA differs')
files['evidence/batch_rpc_publication_validated_result.json']=report;files['evidence/experiment_result_reencoded_verified.json']=result
requirements='''# 主集群下一冻结版本验收要求\n\n此包仅为源码和只读资源证据交接，不是可直接部署的完整环境。不含权重、输入媒体、claims或租约文件；仅包含锁租约观察摘要。当前16卡继续原80000来源票据；禁止修改运行文件、旧claims或向现runner追加第三票。\n\n## 生产兼容20来源有界批RPC\n- 保持原worker、assignment SHA、零基行号%4 GPU、source_row、输入/输出路径和算法/模型身份；同一原授权输出scope、唯一run_id。\n- 输出保留成功/partial/needs_regeneration的五文件；封存失败为receipt-only；existing_claim不得重做。混合20来源批次必须验收。\n- 接收端仍核中央原ticket/queue/assignment SHA与路径白名单；成员集合、SHA、字节数和流式总容量严格绑定，拒绝路径穿越、重复、链接及越界成员。禁止实验固定worker/4源/8MiB整块内存方案直接生产化。\n- 仍使用ms_unified_novlm_batch_commit_v1、batch_sha256、完整{path,sha256,receipt:{path,bytes,sha256}}。fsync、write-once，commit最后写；expectedSHA完整回读相等后才计published和回收缓存。\n- 独立test-only输出先验收：全成功、成功与封存失败混合、partial/needs_regeneration、已有claim跳过、容量边界、断流/错误SHA、不确定commit及成员部分写入。禁止使用真实新source claim/GPU生成作为publication测试。\n\n## publication-only回退\n只能复用已完整生成且SHA绑定的同一批产物、同target。检查已存在成员/commit并拒绝不同字节；不得GPU重做、source claim重置、删partial或换output根绕冲突。回退应有明确的幂等状态与独立测试；现runner与实验没有此自动回退，不得冒称已有。\n\n## BeatThis拒绝诊断\n新冻结版本在任何合法性拒绝前，独立write-once保存模型实际返回的beats/downbeats数组（保原dtype/shape，不以NaN或任意对象注入非法JSON）及具体失败predicate列表、实际audio duration、source身份、model/版本SHA和cuda/FP32/dbn=false配置。明确检查一维、finite、非负、递增、audio duration+.02范围及downbeat属于beats(atol=1e-6,rtol=0)。不裁剪、排序、插值或放宽原条件。诊断产物须有界、独立schema与发布/失败验收契约，不能改变原五文件或将失败当成功。旧3条未保存被拒序列，精确predicate未知，原失败不重跑。\n\n## 安全epoch和GPU4–7\n只在明确的drain/可靠commit/旧PID-start实际结束边界使用新签名runner/transport；保全部旧progress/lease/claims/results，沿用真实物理锁，新epoch独立write-once lease/progress。现runner没有此接口，不靠修改文件或信号热切换。GPU4–7本次仅只读资源快照，未做准入或新分区；即使无PID/lease也不得启动worker。需共享cgroup有效headroom、GPU UUID和实际PID/锁归属资源准入以及中央新有限任务分区契约。\n\n## 版本与实测速率\n当前生产runner/transport已在四节点逐文件SHA一致核验。实验当前e794版本仅改LOCAL_ROOT，未再执行；c5a文件是sole-line恢复并匹配记录执行SHA的重建，不是从旧执行进程恢复的源文件。原实验只证明4源发布阶段约4.28倍，production_pipeline_speedup_proven=false，未接入生产。\n'''
files['NEXT_VERSION_REQUIREMENTS.md']=requirements.encode().replace(bytes([92,110]),bytes([10]))
resource=dict(schema='external_GPU4to7_readonly_and_frozen_source_handoff_v1',created=datetime.datetime.now(datetime.timezone.utc).isoformat(),nodes=nodes,resource_admission_performed=False,new_workers_started=False,claim_lease_or_config_writes=False)
files['resource_report.json']=encode(resource)
name='integration_handoff_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
local=ROOT/name;target=P/'results/a100-extra-03'/name
if not ROOT.is_dir() or not target.parent.is_dir():raise ValueError('authorized handoff parent unavailable')
local.mkdir(exist_ok=False)
manifest=dict(schema='ms_external_frozen_code_handoff_v1',created=resource['created'],scope='code/resource evidence only; not deployment; no model weights or input media',production_sources=entries,member_SHA256={n:sha(b) for n,b in files.items()},member_bytes={n:len(b) for n,b in files.items()},current_production_unchanged=True,production_pipeline_speedup_proven=False,historical_experiment_original_file_recovered=False,historical_reconstruction_SHA_matches_recorded_execution=True,local_root=str(local),shared_root=str(target))
manifest_raw=encode(manifest);files['manifest.json']=manifest_raw
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:') as tar:
 for member,raw in sorted(files.items()):
  t=tarfile.TarInfo(member);t.size=len(raw);t.mode=0o444;t.mtime=0;tar.addfile(t,io.BytesIO(raw))
archive=buffer.getvalue()
if len(archive)>8*2**20:raise ValueError('source-only archive exceeds handoff cap')
archive_sha=sha(archive)
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:') as tar:
 members=tar.getmembers()
 if len(members)!=len(files) or {m.name for m in members}!=set(files):raise ValueError('archive member mismatch')
 for m in members:
  b=tar.extractfile(m).read()
  if b!=files[m.name]:raise ValueError('archive frozen source differs')
target.mkdir(exist_ok=False)
def write_once(path,raw):
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.chmod(path,0o444)
 with path.open('rb') as f:
  size=os.fstat(f.fileno()).st_size;b=f.read(len(raw)+1)
 if size!=len(raw) or len(b)!=len(raw) or sha(b)!=sha(raw):raise ValueError('handoff publication readback differs:'+str(path))
 fd=os.open(path.parent,os.O_DIRECTORY|os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
publications=[]
for filename,raw in [('frozen_code.tar',archive),('manifest.json',manifest_raw),('resource_report.json',files['resource_report.json'])]:
 write_once(local/filename,raw);write_once(target/filename,raw);publications.append(dict(path=str(target/filename),sha256=sha(raw),bytes=len(raw)))
commit=dict(schema='ms_frozen_code_handoff_commit_v1',created=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive_sha256=archive_sha,archive_bytes=len(archive),publications=publications,code_members=len(entries),all_four_node_production_SHA_match=True,no_GPU_models_called=True,no_new_GPU_workers=True,current_production_files_modified=False,claims_leases_modified=False,resource_admission_performed=False)
commit_raw=encode(commit);write_once(local/'commit.json',commit_raw);write_once(target/'commit.json',commit_raw)
print(json.dumps(dict(shared_root=str(target),archive=str(target/'frozen_code.tar'),archive_SHA256=archive_sha,archive_bytes=len(archive),manifest=str(target/'manifest.json'),manifest_SHA256=sha(manifest_raw),resource_report=str(target/'resource_report.json'),commit=str(target/'commit.json'),commit_SHA256=sha(commit_raw),production_source_files=entries,nodes=[dict(node=n['node'],GPU4_to_7=n['GPU4_to_7'],memory=n['memory'],active_GPU0_to_3=n['active_GPU0_to_3']) for n in nodes],all_four_node_production_SHA_match=True,no_production_restart=True)),flush=True)
