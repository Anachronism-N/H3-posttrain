"""Read-only sealed complete-song views from a catalog still collecting other songs."""
import copy
import hashlib
from pathlib import Path

from .audio import inspect_audio,make_windows
from .baseline import _request,_verify,_recorded_result,_sealed
from .io import read_json_snapshot,read_json,file_sha,object_sha,write_json
from .schema import validate_dataset


def _snapshot(path,sources,expected=None):
    path=Path(path).resolve();value,digest=read_json_snapshot(path)
    if expected is not None and digest!=expected:raise ValueError('Complete-song catalog source changed')
    sources[str(path)]=digest
    return value


def _read_song(root,song_id,sources):
    contract=_verify(_snapshot(root/'contract.json',sources))
    if contract['schema']!='choreobench.untimed_catalog_contract.v1':raise ValueError('Require original untimed catalog contract')
    dataset=contract['dataset_snapshot'];validate_dataset(dataset)
    original=next(s for s in dataset['songs'] if s['song_id']==song_id)
    declared=next(s['windows'] for s in contract['song_windows'] if s['song_id']==song_id)
    windows=make_windows(original['audio']['num_samples'],original['audio']['sample_rate'])
    if declared!=windows:raise ValueError('Original catalog window partition differs')
    full_path=Path(original['audio']['path'])
    if not full_path.is_absolute():full_path=Path(contract['dataset_path']).parent/full_path
    full=inspect_audio(full_path)
    if any(full[k]!=original['audio'][k] for k in ('sha256','sample_rate','num_samples')):raise ValueError('Original full-song audio differs')
    sources[str(full_path.resolve())]=full['sha256'];options=[];costs=[];call_records={}
    for window in windows:
        path=root/'audio'/song_id/('%04d.wav'%window['index']);audio=inspect_audio(path)
        if (audio['sample_rate']!=full['sample_rate'] or audio['num_samples']!=window['observation_end']-window['observation_start']
                or audio['pcm']!=full['pcm'][2*window['observation_start']:2*window['observation_end']]):
            raise ValueError('Observation waveform is not exact full-song PCM slice')
        sources[str(path.resolve())]=audio['sha256'];audio.pop('pcm')
        audio.update(parent_audio_sha256=full['sha256'],parent_start=window['observation_start'],parent_end=window['observation_end'],
            conversion='lossless PCM16 sample slice; no padding, resampling or mixing')
        candidates=[]
        for k in range(contract['candidates_per_window']):
            call=song_id+'/untimed/%04d/%02d'%(window['index'],k);name=hashlib.sha256(call.encode()).hexdigest()
            start=_verify(_snapshot(root/'calls'/(name+'.started.json'),sources))
            raw=_verify(_snapshot(root/'calls'/(name+'.raw.json'),sources))
            expected=_request(call,dict(seed=contract['seed'],max_output_tokens=contract['max_output_tokens']),response_kind='local_caption',audio=audio)
            if (start['request']!=expected or start['call_id']!=call or raw['call_id']!=call
                    or start['contract_sha256']!=object_sha(contract) or raw['contract_sha256']!=object_sha(contract)
                    or start['request_sha256']!=object_sha(expected) or raw['started_sha256']!=start['record_sha256']
                    or start['provider_identity']!=contract['provider_identity']):raise ValueError('Original proposal raw/request binding differs')
            response=_recorded_result(raw,contract['provider_identity'],expected)
            candidates.append(dict(call_id=call,raw_text=response['text']));costs.append(dict(response['costs'],call_id=call));call_records[call]=raw['record_sha256']
        options.append(dict(window=copy.deepcopy(window),candidates=candidates))
    for p,digest in sources.items():
        if file_sha(p)!=digest:raise ValueError('Original complete song changed during reading')
    return copy.deepcopy(original),dict(windows=options),contract['synthetic'],costs,call_records,object_sha(contract)


def export_song(catalog_dir,song_id,out_path):
    root=Path(catalog_dir).resolve();sources={}
    original,options,synthetic,costs,calls,contract_sha=_read_song(root,song_id,sources)
    value=_sealed(dict(schema='choreobench.complete_catalog_song.v1',catalog_dir=str(root),song_id=song_id,sources=sources,
        original=original,options=options,synthetic=synthetic,costs=costs,call_records=calls,original_contract_sha256=contract_sha,
        complete_song=True,full_catalog_complete=(root/'results.json').is_file(),new_model_calls=0,
        note='All registered K proposals of this predeclared song are complete and bound. Other songs may still be collecting; no overall completion or quality selection is inferred.'))
    write_json(out_path,value);return value


def load_song_view(path,song_id,sources):
    value=_verify(_snapshot(path,sources))
    if value['schema']!='choreobench.complete_catalog_song.v1' or value['song_id']!=song_id or not value['complete_song']:
        raise ValueError('Invalid completed song view')
    for p,digest in value['sources'].items():
        if file_sha(p)!=digest:raise ValueError('Sealed complete-song evidence changed')
    originals={}
    original,options,synthetic,costs,calls,contract_sha=_read_song(Path(value['catalog_dir']),song_id,originals)
    if (original!=value['original'] or options!=value['options'] or synthetic!=value['synthetic']
            or costs!=value['costs'] or calls!=value['call_records'] or contract_sha!=value['original_contract_sha256']
            or originals!=value['sources']):raise ValueError('Sealed song view differs from actual original raw')
    sources.update(originals)
    return original,options,synthetic,value['record_sha256']
