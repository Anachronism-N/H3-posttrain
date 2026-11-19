"""Single explicitly authorized N5 GPU6 ROI run, using the verified resource lifecycle."""
import importlib.util,json,os
from pathlib import Path

spec=importlib.util.spec_from_file_location('verified_roi_resource',Path(__file__).with_name('resource_manager.py'))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def validate_roi(cfg):
    if cfg.get('node')!='N5' or cfg.get('gpus')!=[6] or cfg.get('release_authorized') is not True:
        raise RuntimeError('Only explicitly authorized N5 GPU6 ROI extraction is allowed')
    if len(cfg.get('holders',[]))!=1 or cfg['holders'][0]['gpu']!=6:
        raise RuntimeError('Exactly one current GPU6 holder identity is required')
    if cfg['env'].get('CUDA_VISIBLE_DEVICES')!='6':
        raise RuntimeError('ROI physical GPU binding differs')
    if not isinstance(cfg['targetcount'],int) or cfg['targetcount']!=3:
        raise RuntimeError('Only fixed three-source ROI pilot is authorized')
    if not cfg.get('argv') or cfg['producer']['path'] not in cfg['argv'] or 'extract' not in cfg['argv']:
        raise RuntimeError('Explicit frozen ROI extraction argv required')
    if cfg.get('phase')!='hand_roi_fixed3_v1' or not Path(cfg['cwd']).is_absolute():
        raise RuntimeError('Unexpected ROI phase')

def check_roi_model(cfg):
    ref=cfg['modelref']
    if base.sha(ref['path'])!=ref['sha256']:
        raise RuntimeError('ROI model reference changed')
    identity=json.loads(Path(ref['path']).read_text())
    if identity['sha256']!=ref['model_identity_sha256']:
        raise RuntimeError('ROI model identity differs')
    for file in identity['files'].values():
        if base.sha(file['path'])!=file['sha256']:
            raise RuntimeError('ROI detector/pose/import bytes changed')

base.AUTHORIZED_PAIRS={'N5':((6,),)}
base.validate_pair=validate_roi
base.check_model=check_roi_model

def checked_roi_summaries(cfg,out):
    selected,total=base.selected_sources(cfg)
    source=Path(cfg['producer']['path'])
    import sys
    sys.path.insert(0,str(source.parent))
    spec=importlib.util.spec_from_file_location('roi_stage_validator',source)
    roi=importlib.util.module_from_spec(spec);spec.loader.exec_module(roi)
    cases=[]
    for sid in selected:
        directory=out/sid
        record=json.loads((directory/'source_manifest.json').read_text())
        if record['id']!=sid or record['source_manifest']['sha256']!=cfg['plan']['sha256']:
            raise RuntimeError('ROI source does not match admitted plan')
        roi.verify_identity(record['source'])
        roi.verify_model(record['model_ref'])
        manifest=roi.validate_completed(directory,record)
        if manifest['selected_frames'][-1]['time']!=record['probe']['frame_times'][-1]:
            raise RuntimeError('ROI sampling does not include the last video frame')
        cases.append({'id':sid,'selected_frames':len(manifest['selected_frames']),'crops':len(manifest['crops']),
                      'manifest_sha256':base.sha(directory/'roi_manifest.json')})
    profile=out/'runtime_profile/cuda_execution_receipt.json'
    evidence=json.loads(profile.read_text())
    if evidence.get('real_CUDA_nodes_executed') is not True:
        raise RuntimeError('ROI lacks real CUDA node execution')
    for model in evidence['models'].values():
        if base.sha(model['profile_path'])!=model['profile_sha256'] or model['executed_node_counts'].get('CUDAExecutionProvider',0)<=0:
            raise RuntimeError('ROI CUDA profile content changed or has no CUDA work')
    return {str(profile):{'summary':{'stage':'extract','plan_sha256':cfg['plan']['sha256'],
                                     'total_source_count':total,'selected_source_count':len(selected),
                                     'processed_source_count':len(cases)},
                         'selection':{'selected_ids':selected},'cases':cases,
                         'CUDA_profile_sha256':base.sha(profile),'semantic_approval':False}}

base.stage_summaries=checked_roi_summaries

if __name__=='__main__':base.main()
