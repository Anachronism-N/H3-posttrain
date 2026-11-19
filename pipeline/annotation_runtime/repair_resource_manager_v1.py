"""One explicitly authorized visual repair runtime on N3 physical GPUs6,7."""
import importlib.util,json,subprocess
from pathlib import Path
RT=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('verified_repair_resource',RT/'resource_manager.py');base=importlib.util.module_from_spec(s);s.loader.exec_module(base)
old_validate=base.validate_pair
def validate(cfg):
 if cfg['node']!='N3' or cfg['gpus']!=[6,7]:raise RuntimeError('repair only N3 physical6,7 authorized')
 if '28.216.19.213' not in subprocess.check_output(['hostname','-I'],text=True).split():raise RuntimeError('repair executing on wrong physical node')
 old_validate(cfg)
 if '--run-job' not in cfg['argv']:raise RuntimeError('exact repair source job required')
def summaries(cfg,out):
 p=Path(cfg['result_path']);v=json.loads(p.read_text());job=json.loads(Path(cfg['job_ref']['path']).read_text())
 if base.sha(cfg['job_ref']['path'])!=cfg['job_ref']['sha256']:raise RuntimeError('repair source job changed')
 if v['id']!=job['id'] or v['original_result_ref']!=job['original_result_ref'] or v['extra_source_attempts']!=1:raise RuntimeError('repair original evidence/budget differs')
 if v['original_raw_or_terminal_modified']:raise RuntimeError('repair overwrote original outputs')
 return {str(p):{'summary':{'source_count':1,'completed_source_attempt_count':1,'structural_status':v.get('final',{}).get('status')},'semantic_approval':False,'independent_review_required':True}}
base.AUTHORIZED_PAIRS={'N3':((6,7),)};base.validate_pair=validate;base.stage_summaries=summaries
if __name__=='__main__':base.main()
