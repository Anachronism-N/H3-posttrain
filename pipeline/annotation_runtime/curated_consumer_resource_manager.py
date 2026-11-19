import importlib.util,json
from pathlib import Path
RT=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('verified_curated_resource',RT/'curated_resource_manager.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
def summaries(cfg,out):
 selected,total=c.selected(cfg);wanted=set(selected);results={}
 for p in (out/'sources').glob('*/*/consumer.terminal.json'):
  v=json.loads(p.read_text())
  if v['id'] not in wanted:continue
  if v.get('joint_candidate_ref'):
   a=v['joint_candidate_ref']
   if c.base.sha(a['path'])!=a['sha256']:raise RuntimeError('consumer output changed')
  results[v['id']]=v
 markers=[p for p in (out/'runs').glob('v'+str(cfg['source_selection']['shard'])+'-*/completed.json')]
 if set(results)!=wanted or not markers:raise RuntimeError('not all assigned sources have terminal captions')
 return {str(markers[-1]):{'summary':{'stage':'run','total_source_count':total,'selected_source_count':len(wanted),'processed_source_count':len(results),'plan_sha256':cfg['plan']['sha256']},'selection':{'selected_ids':selected}}}
c.base.stage_summaries=summaries
if __name__=='__main__':c.base.main()
