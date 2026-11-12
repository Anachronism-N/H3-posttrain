"""Completed song views never imply other songs or quality are complete."""
import copy
import pytest
from choreobench import catalog_song as view
from choreobench import untimed_catalog as catalog
from choreobench.io import read_json,write_json,object_sha
from test_choreobench_untimed_catalog import material


def source(tmp_path,monkeypatch):
 dataset,song,provider,factory,calls=material(tmp_path,monkeypatch)
 out=tmp_path/'catalog'
 result=catalog.run_catalog(dataset,provider,out,max_calls=10,provider_factory=factory)
 return song,out,result,calls


def test_export_complete_song_without_global_results_no_new_calls(tmp_path,monkeypatch):
 song,root,result,calls=source(tmp_path,monkeypatch)
 (root/'results.json').unlink()
 path=tmp_path/'song.json';value=view.export_song(root,song['song_id'],path)
 assert value['complete_song'] and not value['full_catalog_complete'] and value['new_model_calls']==0
 assert len(value['call_records'])==6 and len(calls)==6
 original,options,synthetic,digest=view.load_song_view(path,song['song_id'],{})
 assert original==song and len(options['windows'])==3 and synthetic
 assert len(calls)==6 and digest==value['record_sha256']


@pytest.mark.parametrize('change',['raw_missing','raw_text','audio_bytes','sealed_options'])
def test_changed_complete_song_evidence_rejected(tmp_path,monkeypatch,change):
 song,root,result,calls=source(tmp_path,monkeypatch);path=tmp_path/'song.json'
 view.export_song(root,song['song_id'],path)
 if change=='raw_missing':next((root/'calls').glob('*.raw.json')).unlink()
 elif change=='raw_text':
  p=next((root/'calls').glob('*.raw.json'));v=read_json(p);v['result']['text']='Forged';write_json(p,v,overwrite=True)
 elif change=='audio_bytes':
  p=next((root/'audio').rglob('*.wav'));p.write_bytes(p.read_bytes()+b' ')
 else:
  v=read_json(path);v['options']['windows'][0]['candidates'][0]['raw_text']='Forged';v.pop('record_sha256');v['record_sha256']=object_sha(v);write_json(path,v,overwrite=True)
 with pytest.raises((ValueError,FileNotFoundError)):view.load_song_view(path,song['song_id'],{})
 assert len(calls)==6


def test_incomplete_k_inventory_cannot_export_song(tmp_path,monkeypatch):
 song,root,result,calls=source(tmp_path,monkeypatch)
 next((root/'calls').glob('*.raw.json')).unlink()
 with pytest.raises(FileNotFoundError):view.export_song(root,song['song_id'],tmp_path/'song.json')
 assert not (tmp_path/'song.json').exists()


def test_completed_song_view_consumed_by_composition_source(tmp_path,monkeypatch):
 from choreobench.composition_workflows import _source
 song,root,result,calls=source(tmp_path,monkeypatch);path=tmp_path/'song.json';view.export_song(root,song['song_id'],path)
 data=_source(path,song['song_id'],dict(unit_config={}),{})
 assert data['original']==song and data['source_contract_sha256']==read_json(path)['record_sha256']
 assert data['options']['windows'][0]['candidates'][0]['raw_text']==result['entries'][0]['text']
 assert len(calls)==6
