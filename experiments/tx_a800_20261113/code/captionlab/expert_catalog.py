"""Collect K2 original-audio proposals from the frozen E2-selected base expert."""
import argparse,json
from pathlib import Path
from choreobench.untimed_catalog import run_catalog
from choreobench.catalog_song import export_song
from choreobench.providers import create_provider
from captionlab.secondwave_baseline_v3 import SecondWaveProvider


def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--provider',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=4);a=p.parse_args()
 if not 0<=a.shard<a.shards:p.error('Invalid shard')
 a.out.mkdir(exist_ok=False);dataset=json.loads(a.dataset.read_text());dataset['songs']=dataset['songs'][a.shard::a.shards]
 path=a.out/'dataset.json';path.write_text(json.dumps(dataset,indent=2)+'\n');spec=json.loads(a.provider.read_text())
 def factory(spec,role,base_dir):
  return SecondWaveProvider(spec,role,base_dir) if spec['family'] in ('moss_audio','qwen25_omni','audio_flamingo3') else create_provider(spec,role,base_dir)
 result=run_catalog(path,a.provider,a.out/'catalog',seed=42,candidates_per_window=2,max_output_tokens=1024,max_calls=3000,provider_factory=factory)
 views=a.out/'views';views.mkdir()
 exports=[]
 for song in dataset['songs']:
  try:
   export_song(a.out/'catalog',song['song_id'],views/(song['song_id']+'.json'));exports.append(dict(song_id=song['song_id'],status='complete'))
  except (ValueError,TypeError,KeyError) as error:exports.append(dict(song_id=song['song_id'],status='failed',error=str(error)))
 (a.out/'complete.json').write_text(json.dumps(dict(completed=True,songs=len(dataset['songs']),result=result,exports=exports,quality_verified=False,new_training_updates=0))+'\n')


if __name__=='__main__':main()
