"""One native expert chosen before E3 testing; same proposals for all four methods."""
import argparse,json
from pathlib import Path
from choreobench.untimed_catalog import run_catalog
from choreobench.providers import create_provider


def factory(spec,role,base_dir):
 if spec.get('family') in ('qwen25_omni','audio_flamingo3','moss_audio'):
  from captionlab.secondwave_baseline_v3 import SecondWaveProvider
  return SecondWaveProvider(spec,role,base_dir)
 return create_provider(spec,role,base_dir)


def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--provider',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 v=run_catalog(a.dataset,a.provider,a.out,seed=42,candidates_per_window=2,max_output_tokens=1024,max_calls=3000,resume=a.resume,provider_factory=factory)
 print(json.dumps(dict(completed=v['complete'],calls=v['calls_used'],complete_proposals=v['complete_proposals'],failed=v['failed_proposals'],quality_accepted=False)),flush=True)


if __name__=='__main__':main()
