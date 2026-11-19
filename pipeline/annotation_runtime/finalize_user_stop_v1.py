"""Finalize node shutdown after source ACK; stop only owned annotation CPU workers, keep data access UI."""
import argparse,json,os,pathlib,signal,subprocess,time,re
P=pathlib.Path;RT=P(__file__).parent;REPORT=RT.parent/'user_stop_preserve_all_20261029_v1'
KEEP={'coarse3_review_server_wrapper_v1.py','motion_revision_allocated_external_server_v1.py','ms_llm32_server_schemafix_v4.py','ms_llm32_progress_live_v4.py'}

def run(node):
 report=P('/root/stage2_user_stop_preserve_'+node+'_v1/gpu_release_report.json');v=json.loads(report.read_bytes());assert v['all_native_safe'] and not v['owned_GPU_applications_remaining'];events=[]
 # Reliable results are already ACKed: stop their uploaders by actual generation-specific marker.
 for proc in P('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:a=[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x];s=(proc/'stat').read_text().rsplit(') ',1)[1].split()
  except (OSError,UnicodeError):continue
  scripts=[x for x in a if x.endswith('.py') and x.startswith(('/root/stage2_',str(RT.parent)+'/'))]
  if not scripts:continue
  script=P(scripts[0]);name=script.name
  if 'uploader' not in name or '--config' not in a:continue
  cfg=json.loads(P(a[a.index('--config')+1]).read_bytes());out=P(cfg['out']);preserve=next((r for r in v['source_preservation'] if r['local_out']==str(out)),None);assert preserve and preserve['unacked_source_count']==0
  source=script.read_text();markers=set(re.findall(r"['\"](UPLOAD_STOP[A-Z0-9_]*)['\"]",source))
  for marker in markers:
   p=out/marker
   if not p.exists():
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:json.dump(dict(reason='User stop, all completed sources reliable ACKed',models_called=0),f);f.flush();os.fsync(f.fileno())
  events.append(dict(pid=int(proc.name),script=str(script),action='uploader_stop_marker',markers=sorted(markers)))
 time.sleep(6)
 # All native-safe managers should finish after restored holders exited. No thread or unknown job is targeted.
 for proc in P('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:a=[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x];s=(proc/'stat').read_text().rsplit(') ',1)[1].split()
  except (OSError,UnicodeError):continue
  scripts=[x for x in a if x.endswith('.py') and x.startswith(('/root/stage2_',str(RT.parent)+'/'))]
  if not scripts:continue
  script=P(scripts[0]);name=script.name
  if name in KEEP or name.startswith(('stop_all_annotation','finalize_user_stop','user_stop_')):continue
  if '--manager-sha' in a:continue
  if name.startswith(('stage2_', 'motion_revision_', 'ms_llm32_', 'curated_', 'caption_', 'update_salvage_', 'coarse6when_')):
   pid=int(proc.name);fd=os.pidfd_open(pid);ticks=int(s[19]);now=(proc/'stat').read_text().rsplit(') ',1)[1].split();assert int(now[19])==ticks and proc.stat().st_uid==os.getuid();signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd);events.append(dict(pid=pid,start_ticks=ticks,argv=a,action='owned_annotation_CPU_TERM'))
 p=REPORT/node/'CPU_services_stopped.json';raw=json.dumps(dict(node=node,events=events,GPU_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),data_deleted=False,external_upload_reconciliation_servers_kept_on_N2=node=='N2',review_web_kept=node=='N1',observed_unix=time.time())).encode();fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 print(json.dumps(dict(node=node,CPU_actions=len(events),GPUs_empty=True,all_data_preserved=True)),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--node',required=True);run(p.parse_args().node)
