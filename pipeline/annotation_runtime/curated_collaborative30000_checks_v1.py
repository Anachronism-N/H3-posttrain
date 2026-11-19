"""Targeted CPU-only rejection and source-selection checks; no GPU runtime loaded."""
import copy
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

RT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('checks_manager', RT / 'curated_collaborative30000_resource_manager_v1.py')
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)
worker = manager.worker
passed = []


def reject(name, function, phrase):
    try:
        function()
    except (RuntimeError, ValueError, json.JSONDecodeError) as error:
        if phrase not in str(error):
            raise AssertionError(name + ': unexpected rejection ' + str(error))
        passed.append(name)
    else:
        raise AssertionError(name + ': not rejected')


def queue_case(payloads):
    pipeline = worker.load_pipeline()
    worker.install(pipeline)
    rows = json.loads((RT / 'curated_collaborative30000_n3g67_selected_v1.json').read_text())[:2]
    pipeline.SELECTED_SOURCE_ROWS = {r['id']: r for r in rows}
    with tempfile.TemporaryDirectory(prefix='curated30000-check-') as tmp:
        out = Path(tmp)
        summary = out / 'summary.json'
        summary.write_text(json.dumps({'total': 222939, 'counts': {'train': 200568, 'val': 11110, 'test': 11261}, 'batches': []}))
        bar = SimpleNamespace(__file__=str(out / 'bar.py'), code_identity=lambda: {},
                              params=lambda args, code: {}, runtime_loader=lambda *args: None)
        visual = SimpleNamespace(digest=lambda value: 'cpu-fixture')
        pipeline.modules = lambda *args: (bar, None, visual)
        pipeline.load_style_reuse = lambda paths: {}
        pipeline.notify_batch_terminal = lambda *args: None
        hint = {**rows[0], 'status': 'failed', 'manifest_record': rows[0], 'formal_training': False}
        directory = out / 'sources' / hint['split'] / hint['id']
        directory.mkdir(parents=True)
        (directory / 'producer.terminal.json').write_text(json.dumps(hint))
        lines = payloads(hint)
        (out / 'ready.jsonl').write_bytes(b''.join(x if isinstance(x, bytes) else (json.dumps(x) + '\n').encode() for x in lines))
        args = SimpleNamespace(manifest=str(summary), out=str(out), tp=2, batch_size=2, max_num_seqs=2,
                               shard=0, nshard=1, poll_seconds=.01, repo='/', ms_source='/', ids=[r['id'] for r in rows],
                               limit=0, watch=False, style_reuse=[], stage='run')
        return pipeline.run_consumer(args)


def main():
    configs = [json.loads((RT / ('curated_collaborative30000_' + label + '_launch_v1.json')).read_text())
               for label in ['n5g0', 'n5g1', 'n3g67']]
    selections = [manager.selected_rows(cfg) for cfg in configs]
    left, right, combined = [{r['id'] for r in rows} for rows in selections]
    assert len(left) == len(right) == 1000 and not left & right and left | right == combined and len(combined) == 2000
    assert [r['source_ordinal'] for r in selections[2]] == list(range(30000, 32000))
    assert {r['split'] for r in selections[2]} == {'train'}
    passed.append('exact2000_two_disjoint1000_original_four500_batches')
    for cfg in configs:
        bad = copy.deepcopy(cfg)
        bad['targetcount'] -= 1
        reject('targetcount_rejected_' + cfg['mode'], lambda: manager.selected_rows(bad), 'range/count/selection')
    bad = copy.deepcopy(configs[2])
    bad.update(gpus=[0, 1])
    reject('wrong_physical_consumer_gpu_rejected', lambda: manager.validate(bad), 'GPU assignment')
    with tempfile.TemporaryDirectory(prefix='curated30000-split-') as tmp:
        bad = copy.deepcopy(configs[2])
        rows = copy.deepcopy(selections[2])
        rows[0]['split'] = 'val'
        target = Path(tmp) / 'selection.json'
        target.write_text(json.dumps(rows))
        bad['selected_sources'] = {'path': str(target), 'sha256': manager.c.base.sha(target)}
        bad['argv'][bad['argv'].index('--selected-sources') + 1] = str(target)
        reject('original_split_change_rejected', lambda: manager.selected_rows(bad), 'Selected paths differ')
        lease = Path(tmp) / 'lease'
        lease.touch()
        reject('active_lease_rejected', lambda: manager.c.base.lease_snapshot(lease), 'active lease')
    reject('public_ready_wrong_split_rejected', lambda: queue_case(lambda h: [{**h, 'split': 'val'}]), 'outside exact independent selected binding')
    reject('public_ready_wrong_ordinal_rejected', lambda: queue_case(lambda h: [{**h, 'source_ordinal': 20000}]), 'outside exact independent selected binding')
    reject('public_ready_formal_training_true_rejected', lambda: queue_case(lambda h: [{**h, 'formal_training': True}]), 'outside exact independent selected binding')
    reject('public_ready_duplicate_rejected', lambda: queue_case(lambda h: [h, h]), 'Duplicate public ready')
    reject('public_ready_bad_json_rejected', lambda: queue_case(lambda h: [b'{bad-json}\n']), 'Expecting property name')
    evidence = json.loads((RT / 'curated_collaborative30000_params_evidence_v1.json').read_text())
    assert evidence['changed_fields'] == ['hand_roi_root']
    assert evidence['bar_parameters_and_source_unchanged_except_hand_roi_root'] is True
    assert evidence['new_configuration']['hand_roi_root'] == str(worker.OUT / 'roi')
    assert worker.load_pipeline().FLAGS['formal_training'] is False
    for batch in configs[2]['selected_original_batches']:
        values = [json.loads(line) for line in Path(batch['path']).read_text().splitlines()]
        assert len(values) == 500 and all(r['formal_training'] is False and r['split'] == 'train' for r in values)
    passed.append('native_params_only_hand_roi_path_changed_all2000_formalfalse')
    result = {'status': 'targeted_checks_passed', 'passed': passed, 'gpu_started': False, 'signals_sent': False,
              'mocked_resource_rejections': True, 'full_cpu_regression': False}
    with (RT / 'curated_collaborative30000_checks_receipt_v1.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
