"""Explicit source-range local ROI producer, using verified exact GPU lifecycle."""
import importlib.util,json
from pathlib import Path
RT=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('verified_io_curated_base',RT/'curated_resource_manager.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
def selected(cfg):
 a=cfg['argv'];start=int(c.base.argv_option(cfg,'--ordinal-start'));end=int(c.base.argv_option(cfg,'--ordinal-end'));shard=int(c.base.argv_option(cfg,'--shard',0));nshard=int(c.base.argv_option(cfg,'--nshard',1));limit=int(c.base.argv_option(cfg,'--limit',0))
 if not 0<=start<end<=222939 or not 0<=shard<nshard or limit<0:raise RuntimeError('invalid explicit IO selection')
 selected=[];total=0
 with Path(cfg['plan']['path']).open() as stream:
  for line in stream:
   if not line.strip():continue
   r=json.loads(line);ordinal=r['source_ordinal']
   if ordinal!=total:raise RuntimeError('source index ordinal order changed')
   if start<=ordinal<end and ordinal%nshard==shard and (not limit or len(selected)<limit):selected.append(r['id'])
   total+=1
 if total!=222939 or len(selected)!=cfg['targetcount']:raise RuntimeError('explicit source-range selected count changed')
 return selected,total
c.selected=selected;c.base.selected_sources=selected
if __name__=='__main__':c.base.main()
