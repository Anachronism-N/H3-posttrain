import concurrent.futures,datetime,hashlib,json,os,pathlib,shlex,shutil,subprocess,sys,time
BASE=pathlib.Path('/tmp/ms_pose2d_draft_20261026T1427Z');ROOT=BASE/'launcher';T=pathlib.Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/external_a100_validation_20261026_v1');LF=chr(10)
NODES=[('a100-extra-03','launcher',None),('a100-extra-04','worker-0','11.254.248.52'),('a100-extra-05','worker-1','11.239.106.11'),('a100-extra-06','worker-2','11.239.84.43')]
PROJECT=pathlib.Path('/apdcephfs_gy4/share_302533218/cedricnie/H3-posttrain/scripts_omnidance/exp/ms');DW=pathlib.Path('/apdcephfs_gy4/share_302533218/cedricnie/pose-preprocess/third_party/scail_pose/DWPoseProcess/dwpose')
CODE=[(PROJECT/name,'code/'+name) for name in ['ms_hand_roi.py','dwpose_full_extract.py','ms_pose2d_draft.py']]+[(DW/name,'code/dwpose/'+name) for name in ['wholebody.py','onnxdet.py','onnxpose.py','__init__.py']]
KEYS=['video_path','audio_path','npz_path','source_manifest_path','roi_manifest_path','bar_plan_path']
def emit(state,**data):
 record={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':'bounded_validation_relay','state':state,'pid':os.getpid(),**data}
 with (ROOT/'logs/preparation_events.jsonl').open('a') as f:f.write(json.dumps(record)+LF);f.flush();os.fsync(f.fileno())
 print(json.dumps(record),flush=True)
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
REMOTE='''
import os,pathlib,sys,hashlib,json
path=pathlib.Path(sys.argv[1]);base=pathlib.Path(sys.argv[2]);expected=sys.argv[3];size=int(sys.argv[4])
if base not in path.parents:raise ValueError('unsafe target')
if not base.parent.is_dir():raise ValueError('owned current cache missing')
path.parent.mkdir(parents=True,exist_ok=True)
h=hashlib.sha256();n=0
with path.open('xb') as f:
 while b:=sys.stdin.buffer.read(4*1024*1024):f.write(b);h.update(b);n+=len(b)
 f.flush();os.fsync(f.fileno())
if n!=size or h.hexdigest()!=expected:raise ValueError('relay byte mismatch;preserve partial')
print(json.dumps({'local_path':str(path),'bytes':n,'sha256':h.hexdigest(),'verified':True}))
'''
def transfer(host,source,target,packet):
 source=pathlib.Path(source);size=source.stat().st_size;hashvalue=sha(source)
 if host is None:
  target.parent.mkdir(parents=True,exist_ok=True)
  with source.open('rb') as inp,target.open('xb') as out:shutil.copyfileobj(inp,out,4*1024*1024);out.flush();os.fsync(out.fileno())
  if sha(target)!=hashvalue:raise ValueError('local copied code differs')
 else:
  args=['python3','-c',REMOTE,str(target),str(packet),hashvalue,str(size)]
  command=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pose2d_resource_known_hosts_20261026','-o','ConnectTimeout=8',host,shlex.join(args)]
  with source.open('rb') as inp:p=subprocess.run(command,stdin=inp,capture_output=True,timeout=180)
  if p.returncode:raise RuntimeError('relay transfer failed rc='+str(p.returncode)+';no retry')
  if bytes([0]) in p.stdout or not p.stdout.endswith(bytes([10])):raise ValueError('partial or NUL receipt')
  receipt=json.loads(p.stdout)
  if not receipt.get('verified'):raise ValueError('relay unverified')
 return {'original_path':str(source),'local_path':str(target),'bytes':size,'sha256':hashvalue}
def prepare(node):
 server,role,host=node;start=time.monotonic();root=BASE/role;packet=root/'cache/current/assigned_validation_packet_v2';ticketpath=T/(server+'.json')
 try:
  with ticketpath.open() as f:ticket=json.load(f)
  expected=3 if role=='launcher' else 1
  if ticket['worker_id']!=server or ticket['source_count']!=expected or len(ticket['sources'])!=expected or ticket['allowed_gpu_indices']!=[0] or not ticket['diagnostic_not_production']:raise ValueError('ticket mismatch')
  files=[(ticketpath,'ticket.json')]+CODE
  for source in ticket['sources']:
   for key in KEYS:
    path=pathlib.Path(source[key])
    if not path.is_file():raise ValueError('ticket input unavailable '+key)
    if host:files.append((path,'assets/'+source['id']+'/'+path.name))
  total=sum(pathlib.Path(p).stat().st_size for p,_ in files)
  if host and total>2*2**30:raise ValueError('input packet exceeds 2GiB '+str(total))
  receipts=[]
  for source,relative in files:
   receipts.append(transfer(host,source,packet/relative,packet))
   emit('progress',server=server,file=relative,bytes=receipts[-1]['bytes'])
  sources=[]
  for source in ticket['sources']:
   view={**source,'original_paths':{k:source[k] for k in KEYS},'assets':[]}
   for key in KEYS:
    original=pathlib.Path(source[key]);local=packet/'assets'/source['id']/original.name if host else original
    view[key]=str(local);view['assets'].append({'key':key,'original_path':str(original),'local_path':str(local),'bytes':original.stat().st_size,'sha256':sha(original)})
   sources.append(view)
  plan={'schema':'bounded_external_validation_plan_v2','worker_id':server,'role':role,'root':str(root),'ticket_path':str(packet/'ticket.json'),'ticket_original_path':str(ticketpath),'ticket_sha256':sha(ticketpath),'code_root':str(packet/'code'),'sources':sources,'source_count':expected,'diagnostic_not_production':True,'output_namespace':ticket['output_namespace'],'packet_bytes':total,'packet_limit_bytes':2*2**30,'code_assets':receipts[1:8],'copy_receipts':receipts,'relay_seconds':time.monotonic()-start,'weight_copy_bytes':0}
  localplan=ROOT/'logs'/(server+'_validation_plan_v2.json')
  with localplan.open('x') as f:json.dump(plan,f,indent=2);f.write(LF)
  transfer(host,localplan,packet/'plan.json',packet)
  receipt={'server':server,'role':role,'host':host,'remote_plan_path':str(packet/'plan.json'),'central_plan_path':str(localplan),'packet_bytes':total,'weight_copy_bytes':0,'all_files_verified':True,'source_count':expected,'relay_seconds':time.monotonic()-start}
  with (ROOT/'logs'/('relay_'+role+'_receipt_v2.json')).open('x') as f:json.dump(receipt,f,indent=2);f.write(LF)
  emit('progress',**receipt);return receipt
 except Exception as e:
  result={'server':server,'state':'failed','error':type(e).__name__+': '+str(e),'automatic_retry':False};emit('progress',**result);return result
emit('started')
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(prepare,NODES))
with (ROOT/'logs/relay_summary_v2.json').open('x') as f:json.dump({'results':results,'diagnostic_not_production':True,'formal_sources':0},f,indent=2);f.write(LF)
failed=any(r.get('state')=='failed' for r in results)
emit('failed' if failed else 'completed',results=results)
sys.exit(1 if failed else 0)
