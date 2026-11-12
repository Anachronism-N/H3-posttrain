"""Whole-window untimed local proposals with a complete immutable call inventory.

Ownership windows are schedules, not detected action boundaries. Text is never
split into artificial timestamps or filtered by an internal quality judgment.
"""
import argparse
import copy
from pathlib import Path

from .audio import inspect_audio,make_windows,write_window
from .baseline import _Journal,_request,_sealed,_verify,_recorded_result,_journal_accounting
from .dataset import resolve_audio
from .io import file_lock,file_sha,object_sha,read_json,read_json_snapshot,write_json
from .providers import create_provider
from .schema import validate_dataset,integer


def run_catalog(dataset_path,provider_path,out_dir,*,seed=42,candidates_per_window=2,max_output_tokens=1024,max_calls=3000,resume=False,provider_factory=None):
    dataset_path=Path(dataset_path).resolve();provider_path=Path(provider_path).resolve();output=Path(out_dir).resolve()
    dataset,sha=read_json_snapshot(dataset_path);validate_dataset(dataset)
    spec,spec_sha=read_json_snapshot(provider_path)
    for name,value,low,high in [('seed',seed,0,2**63-1),('candidates_per_window',candidates_per_window,1,4),('max_output_tokens',max_output_tokens,1,8192),('max_calls',max_calls,1,10000)]:
        integer(value,name,low)
        if value>high:raise ValueError('Catalog budget outside bounds')
    if spec.get('kind') not in ('native','native_service','replay'):raise ValueError('Catalog requires explicit original-audio provider')
    if spec.get('kind')=='native' and (spec.get('checkpoint') or not spec.get('task_prompt')):
        raise ValueError('Untimed base-only catalog requires a frozen public task prompt')
    songs=[];requests=[];prepared=[]
    for song in dataset['songs']:
        path=resolve_audio(song,dataset_path);audio=inspect_audio(path)
        if any(audio[k]!=song['audio'][k] for k in ('sha256','sample_rate','num_samples')):raise ValueError('Full-song waveform differs from dataset')
        windows=make_windows(audio['num_samples'],audio['sample_rate']);songs.append(dict(song_id=song['song_id'],windows=windows))
        prepared.append((song,audio,windows))
    expected=sum(len(s['windows'])*candidates_per_window for s in songs)
    if expected>max_calls:raise ValueError('Catalog cannot fit complete independent source inventory')
    code={p.name:file_sha(p) for p in Path(__file__).parent.glob('*.py')}
    base=dict(schema='choreobench.untimed_catalog_contract.v1',dataset_path=str(dataset_path),dataset_sha256=sha,dataset_snapshot=dataset,
        provider_path=str(provider_path),provider_config_sha256=spec_sha,provider_config=spec,seed=seed,
        candidates_per_window=candidates_per_window,max_output_tokens=max_output_tokens,max_calls=max_calls,
        expected_calls=expected,song_windows=songs,code_sha256=code,untimed_whole_window=True)
    if output.exists() and not resume:raise FileExistsError(str(output))
    if resume and not (output/'contract.json').is_file():raise ValueError('Resume requires original catalog contract')
    output.mkdir(parents=True,exist_ok=True)
    with file_lock(output/'.run.lock'):
        provider=None
        if resume:
            contract=_verify(read_json(output/'contract.json'))
            if set(contract)!=set(base)|{'provider_identity','synthetic','record_sha256'} or any(contract[k]!=v for k,v in base.items()):raise ValueError('Catalog input/code/config changed')
        else:
            provider=(provider_factory or create_provider)(spec,role='local',base_dir=provider_path.parent)
            contract=_sealed(dict(base,provider_identity=copy.deepcopy(provider.identity),synthetic=dataset['synthetic'] or provider.identity['synthetic']))
            write_json(output/'contract.json',contract)
        identity=contract['provider_identity'];journal=_Journal(output,object_sha(contract),max_calls)
        previous=_verify(read_json(output/'results.json')) if (output/'results.json').exists() else None
        audio_signatures={audio['path']:(Path(audio['path']).stat().st_ino,Path(audio['path']).stat().st_size,Path(audio['path']).stat().st_mtime_ns,Path(audio['path']).stat().st_ctime_ns) for song,audio,windows in prepared}
        def unchanged():
            if file_sha(dataset_path)!=sha or file_sha(provider_path)!=spec_sha:raise ValueError('Catalog source changed')
            if read_json(output/'contract.json')!=contract:raise ValueError('Frozen catalog contract changed')
            if any(file_sha(Path(__file__).parent/name)!=digest for name,digest in code.items()):raise ValueError('Catalog execution code changed')
            for song,audio,windows in prepared:
                stat=Path(audio['path']).stat()
                if (stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)!=audio_signatures[audio['path']]:raise ValueError('Full-song source audio changed')
            if provider is not None and provider.identity!=identity:raise ValueError('Catalog provider identity drift')
        entries=[];artifacts={}
        for song,audio,windows in prepared:
            for window in windows:
                relative='audio/'+song['song_id']+'/%04d.wav'%window['index']
                if previous is not None and (not (output/relative).is_file() or file_sha(output/relative)!=previous['artifacts'].get(relative)):
                    raise ValueError('Completed catalog waveform missing/changed; no replacement')
                write_window(audio,window,output/relative);observed=inspect_audio(output/relative);observed.pop('pcm')
                artifacts[relative]=file_sha(output/relative)
                original_audio=dict(observed,parent_audio_sha256=audio['sha256'],parent_start=window['observation_start'],parent_end=window['observation_end'],
                    conversion='lossless PCM16 sample slice; no padding, resampling or mixing')
                for candidate in range(candidates_per_window):
                    call=song['song_id']+'/untimed/%04d/%02d'%(window['index'],candidate)
                    request=_request(call,dict(seed=seed,max_output_tokens=max_output_tokens),response_kind='local_caption',audio=original_audio)
                    requests.append(request)
                    if previous is not None and call not in journal.records:raise ValueError('Completed catalog call missing; no resampling')
                    if call in journal.started and journal.started[call]['request']!=request:raise ValueError('Catalog request differs')
                    unchanged()
                    if call not in journal.records:
                        if provider is None:provider=(provider_factory or create_provider)(spec,role='local',base_dir=provider_path.parent)
                        if provider.identity!=identity:raise ValueError('Catalog provider differs from original')
                        try:journal.call(provider,request)
                        except (ValueError,TypeError,KeyError):
                            if call not in journal.records:raise
                    raw=journal.records[call];start=journal.started[call]
                    if start['request']!=request or start['provider_identity']!=identity:raise ValueError('Catalog raw/request identity changed')
                    value=raw.get('result')
                    if raw.get('success') and (not isinstance(value,dict) or value.get('identity')!=identity or value.get('synthetic') is not identity['synthetic']):
                        raise ValueError('Catalog returned provider identity differs')
                    if (not raw.get('success') or not isinstance(value,dict)
                            or value.get('stop_reason') in (None,'http_error','infrastructure_error','malformed_api_response')):
                        raise RuntimeError('Catalog infrastructure outcome failed; stop without automatic retry')
                    entry=dict(song_id=song['song_id'],window=copy.deepcopy(window),candidate_index=candidate,call_id=call,
                        started_record_sha256=start['record_sha256'],raw_record_sha256=raw['record_sha256'],status='failed',text='',error=None)
                    try:
                        value=_recorded_result(raw,identity,request);entry.update(status='complete',text=value['text'],error=None)
                    except (ValueError,TypeError,KeyError) as error:
                        entry['error']=dict(type=type(error).__name__,message=str(error))
                        value=raw.get('result')
                        if isinstance(value,dict) and isinstance(value.get('text'),str):entry['text']=value['text']
                    entries.append(entry);unchanged()
        expected_ids={r['call_id'] for r in requests}
        if set(journal.started)!=set(journal.records) or set(journal.records)!=expected_ids:raise ValueError('Catalog call inventory incomplete or unexpected')
        accounting=_journal_accounting(journal)
        accounting['costs'].sort(key=lambda row:row['call_id'])
        for key in ('unknown_outcome_call_ids','unknown_usage_call_ids'):accounting[key].sort()
        for song,audio,windows in prepared:
            if file_sha(audio['path'])!=audio['sha256']:raise ValueError('Full-song source audio changed at publication')
        result=_sealed(dict(schema='choreobench.untimed_catalog.v1',complete=True,contract_sha256=object_sha(contract),dataset_sha256=sha,
            dataset_snapshot=dataset,provider_identity=identity,entries=entries,artifacts=artifacts,expected_calls=expected,calls_used=len(journal.started),
            call_records={k:v['record_sha256'] for k,v in journal.records.items()},**accounting,
            complete_proposals=sum(e['status']=='complete' for e in entries),failed_proposals=sum(e['status']=='failed' for e in entries),
            synthetic=contract['synthetic'],quality_verified=False,formal_E3_completed=False,
            note='Complete fixed K whole-audio catalog. No synthetic actions, timing split, candidate quality selection or observed action boundaries. Failures retained for every requested cell.'))
        if previous is not None:
            if result!=previous:raise ValueError('Completed catalog reconstructed evidence differs')
        else:write_json(output/'results.json',result)
        return result


def load_catalog_song(catalog_path,song_id,sources):
    catalog_path=Path(catalog_path).resolve();value,sha=read_json_snapshot(catalog_path);_verify(value)
    sources[str(catalog_path)]=sha
    if value['schema']!='choreobench.untimed_catalog.v1' or not value['complete']:raise ValueError('Complete raw catalog required')
    contract_path=catalog_path.parent/'contract.json';contract,digest=read_json_snapshot(contract_path);_verify(contract);sources[str(contract_path)]=digest
    if (object_sha(contract)!=value['contract_sha256'] or contract['dataset_sha256']!=value['dataset_sha256']
            or contract['dataset_snapshot']!=value['dataset_snapshot'] or contract['provider_identity']!=value['provider_identity']):raise ValueError('Catalog source binding differs')
    validate_dataset(value['dataset_snapshot'])
    original=next(s for s in value['dataset_snapshot']['songs'] if s['song_id']==song_id)
    selected=[e for e in value['entries'] if e['song_id']==song_id]
    if not selected or any(e['status']!='complete' for e in selected):raise ValueError('Song has failed catalog proposals; no partial success')
    journal=_Journal(catalog_path.parent,object_sha(contract),contract['max_calls'])
    if value['call_records']!={k:v['record_sha256'] for k,v in journal.records.items()} or value['calls_used']!=len(journal.started):raise ValueError('Catalog calls missing/changed')
    windows=next(s['windows'] for s in contract['song_windows'] if s['song_id']==song_id);options=[]
    if windows!=make_windows(original['audio']['num_samples'],original['audio']['sample_rate']):raise ValueError('Catalog window clock differs')
    if len(selected)!=len(windows)*contract['candidates_per_window']:raise ValueError('Catalog proposal inventory differs')
    for window in windows:
        relative='audio/'+song_id+'/%04d.wav'%window['index']
        audio_path=catalog_path.parent/relative
        if not audio_path.is_file() or file_sha(audio_path)!=value['artifacts'].get(relative):
            raise ValueError('Catalog source waveform missing or changed')
        sources[str(audio_path.resolve())]=file_sha(audio_path)
        candidates=[]
        window_entries=[e for e in selected if e['window']==window]
        if {e['candidate_index'] for e in window_entries}!=set(range(contract['candidates_per_window'])):
            raise ValueError('Catalog candidate identities are incomplete or duplicated')
        for entry in window_entries:
            request=journal.started[entry['call_id']]['request'];raw=journal.records[entry['call_id']]
            if raw['record_sha256']!=entry['raw_record_sha256'] or journal.started[entry['call_id']]['record_sha256']!=entry['started_record_sha256']:raise ValueError('Catalog proposal binding differs')
            expected_id=song_id+'/untimed/%04d/%02d'%(window['index'],entry['candidate_index'])
            audio=request['audio']
            if (entry['call_id']!=expected_id or type(entry['candidate_index']) is not int or not 0<=entry['candidate_index']<contract['candidates_per_window']
                    or audio['parent_audio_sha256']!=original['audio']['sha256'] or audio['parent_start']!=window['observation_start']
                    or audio['parent_end']!=window['observation_end'] or audio['num_samples']!=window['observation_end']-window['observation_start']
                    or request!=_request(expected_id,dict(seed=contract['seed'],max_output_tokens=contract['max_output_tokens']),response_kind='local_caption',audio=audio)):
                raise ValueError('Catalog original request/audio clock differs')
            response=_recorded_result(raw,value['provider_identity'],request)
            if response['text']!=entry['text']:raise ValueError('Catalog text differs from raw')
            candidates.append(dict(call_id=entry['call_id'],raw_text=entry['text']))
        if len(candidates)!=contract['candidates_per_window']:raise ValueError('Catalog K changed')
        options.append(dict(window=copy.deepcopy(window),candidates=candidates))
    for p in journal.directory.glob('*.json'):sources[str(p.resolve())]=file_sha(p)
    return copy.deepcopy(original),dict(windows=options),value['synthetic'],value['record_sha256']


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','provider','out-dir'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--max-calls',type=int,default=3000);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv);result=run_catalog(args.dataset,args.provider,args.out_dir,max_calls=args.max_calls,resume=args.resume)
    print({k:result[k] for k in ('calls_used','complete_proposals','failed_proposals','formal_E3_completed')})


if __name__=='__main__':main()
