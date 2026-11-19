"""Minimal v8 source builder adaptation; never load GPU or send a signal."""
from pathlib import Path

from main_review_watch_v9 import RT, module, need

BUILDER_V8_SHA = '80183807dfaf21b4f15256cae768368c4c00ce7a6d2a220baa0f800e3b6bc85c'
v8 = module('private_frozen_source_builder_v8_for_v9', RT / 'build_main_review_v8_fresh.py', BUILDER_V8_SHA)
BUILDER_V7_SHA, save_new = v8.BUILDER_V7_SHA, v8.save_new


def cpu_smoke(worker, binding, reader, cache, out):
    proof = v8.cpu_smoke(worker, binding, reader, cache, out)
    proof.update(schema='main_review_v9_cpu_smoke_v1', worker_sha256=cache.sha(RT / 'main_review_watch_v9.py'),
        historical_load_receipts_exact_source_count=2, original_historical_load_refs_preserved=True,
        new_load_used_to_prove_old_call=False, original_watch_v8_sha256=worker.V8_SHA)
    return proof


source = v8.source


def replace_once(old, new):
    global source
    need(source.count(old) == 1, 'Frozen v9 builder anchor differs: ' + old)
    source = source.replace(old, new, 1)


replace_once("    incident = worker.Incident(production, manifest_path, binding['incident'], empty, binding['offset2053650'])",
    "    binding['incident']['historical_load_receipts'] = worker.capture_historical_load_binding(binding['incident'])\n    binding['incident']['historical_reconciliation_load_receipts'] = worker.historical.capture_reconciliation()\n    incident = worker.Incident(production, manifest_path, binding['incident'], empty, binding['offset2053650'])")
source = source.replace('main_review_watch_v8', 'main_review_watch_v9').replace('main_review_resource_manager_v8', 'main_review_resource_manager_v9')
source = source.replace('build_main_review_v8_fresh.py', 'build_main_review_v9_fresh.py').replace('test_main_review_v8.py', 'test_main_review_v9.py')
source = source.replace('main_review_v8_source_only_v1', 'main_review_v9_source_only_v1')
replace_once("    for ref in immutable_refs:",
    "    immutable_refs.extend(worker.historical.immutable_refs(binding['incident']['historical_load_receipts']))\n    immutable_refs.extend(worker.historical.immutable_refs(binding['incident']['historical_reconciliation_load_receipts']))\n    for name in ('main_review_watch_v8.py', 'main_review_resource_manager_v8.py', 'build_main_review_v8_fresh.py', 'test_main_review_v8.py', 'main_old_load_receipts_v9.py'):\n        cfg['source_sha256'][str(RT / name)] = cache.sha(RT / name)\n    for ref in immutable_refs:")
namespace = {**v8.main.__globals__, 'cpu_smoke': cpu_smoke, 'BUILDER_V7_SHA': BUILDER_V7_SHA}
exec(compile(source, str(Path(__file__)), 'exec'), namespace)
main = namespace['main']

if __name__ == '__main__':
    main()
