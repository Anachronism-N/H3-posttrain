"""Finalize existing96 E3 generations and streaming48 E4 tasks, without new edits."""
import json,time,shutil,subprocess,sys,argparse
from pathlib import Path
from choreobench.baseline import _verify


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
 source=Path('/opt/music_action_runs/20261113_e34_formal_v1/formal');study=Path('/opt/music_action_runs/20261113_e4_stream_v1/study');provider=root/'provider.json';dataset=source.parent/'dataset.json';observations=Path('/opt/music_action_runs/20261113_e3_observe_catalog_v1')
 start=time.monotonic()
 while not (source/'execution_results.json').exists() or not (study/'results.json').exists():
  if time.monotonic()-start>172800:raise TimeoutError('Registered E3/E4 original outputs not terminal; no resampling')
  time.sleep(30)
 execution=json.loads((source/'execution_results.json').read_text());edits=_verify(json.loads((study/'results.json').read_text()))
 if len(execution['rows'])!=96 or edits['expected_registered_tasks']!=48 or len(edits['cells'])!=96:raise ValueError('Registered denominators changed')
 published=Path('/apdcephfs_gy4/share_302533218/cedricnie/choreobench_runs/20261113_sol61_e3_formal_v1')
 if published.exists():
  if not (published/'execution_results.json').exists() or json.loads((published/'execution_results.json').read_text())!=execution:raise ValueError('Unexpected shared final state, not overwritten')
 else:
  published.mkdir();shutil.copytree(source/'runs',published/'runs');shutil.copyfile(source/'execution_results.json',published/'execution_results.json')
 from captionlab.e4_final_metrics import calibrate as edit_calibrate
 from captionlab.e3_final_metrics import calibrate as long_calibrate
 e=edit_calibrate(provider,root/'edit_controls');l=long_calibrate(provider,root/'long_controls')
 if not e['accepted'] or not l['accepted']:raise RuntimeError('Evaluation controls failed; no formal score accepted, raw outputs retained')
 def run(module,args):subprocess.run([sys.executable,'-m',module,*map(str,args)],check=True)
 run('captionlab.e3_mechanical_metrics',['--run',source,'--dataset',dataset,'--out',root/'mechanical'])
 run('captionlab.e4_final_metrics',['--registry',study/'registry.json','--study',study,'--provider',provider,'--out',root/'edit_metrics'])
 run('captionlab.e3_final_metrics',['--formal',source,'--dataset',dataset,'--provider',provider,'--out',root/'long_metrics'])
 start=time.monotonic()
 while not (observations/'observations/complete.json').exists():
  if time.monotonic()-start>172800:raise TimeoutError('Independent original-audio observations not terminal')
  time.sleep(30)
 run('captionlab.e3_music_metrics',['--formal',source,'--dataset',dataset,'--observations',observations,'--provider',provider,'--out',root/'music_metrics'])
 from captionlab.e3_repetition_metrics import calibrate as repetition_calibrate
 q=repetition_calibrate(provider,root/'repeat_controls')
 if not q['accepted']:raise RuntimeError('Repetition metric control failed; no quality acceptance')
 run('captionlab.e3_repetition_metrics',['--formal',source,'--dataset',dataset,'--observations',observations,'--provider',provider,'--out',root/'repeat_metrics'])
 for folder in ('edit_controls','long_controls','repeat_controls','mechanical','edit_metrics','long_metrics','music_metrics','repeat_metrics'):shutil.copytree(root/folder,published/folder)
 shutil.copytree(study,published/'e4')
 (root/'complete.json').write_text(json.dumps(dict(platform='TX A800-SXM4-80GB cluster',registered_E3=96,registered_E4_tasks=48,registered_E4_evaluation_cells=144,model_assisted_reviews=True,human=False,original_generated_artifacts_not_overwritten=True,new_E3_calls=0,new_editor_calls=0,all_requested_model_and_literal_metrics_completed=True),indent=2))
 print('Registered E3/E4 terminal outputs and measured metrics published; human/visual/music truth not claimed',flush=True)


if __name__=='__main__':main()
