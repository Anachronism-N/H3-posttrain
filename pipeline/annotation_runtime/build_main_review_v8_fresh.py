"""Planned CPU source-only builder; resource capture follows the slow native history check."""
import inspect
from pathlib import Path
import textwrap

from main_review_watch_v8 import RT, module, need

BUILDER_V7_SHA = '5f86dba83821d34d461f35eedf6ba616733ed30eeca3d671547daeeb58280458'
v7 = module('private_frozen_source_builder_v7_for_v8', RT / 'build_main_review_v7_fresh.py', BUILDER_V7_SHA)
save_new = v7.save_new


def cpu_smoke(worker, binding, reader, cache, out):
    original = worker.ROOT / 'curated_independent_reviews_v1'
    protected = {str(p): cache.sha(p) for p in original.rglob('*.json')
        if p.name != 'progress.json' and p.name != 'run.summary.json'}
    proof = v7.cpu_smoke(worker, binding, reader, cache, out)
    for path, digest in protected.items():
        worker.need(cache.sha(path) == digest, 'Original full score/history/cache changed during CPU source preparation')
    proof.update(schema='main_review_v8_cpu_smoke_v1', worker_sha256=cache.sha(RT / 'main_review_watch_v8.py'),
        all_original_score_cache_history_verified_unchanged=True, protected_original_JSON_count=len(protected),
        original_watch_v7_sha256=worker.V7_SHA, fragment_row_offsets=reader.incident.fragment_row_offsets,
        fragment_bytes_contained_complete_JSON_sources=0, fragment_source_association='unknown',
        independent_offset2053650_reconciliation_sources=4, original_source_offsets_assigned=False)
    return proof


source = textwrap.dedent(inspect.getsource(v7.main))

def replace_once(old, new):
    global source
    need(source.count(old) == 1, 'Frozen v8 builder anchor differs: ' + old)
    source = source.replace(old, new, 1)


replace_once("ap.add_argument('--reconciliation-manifest')", "ap.add_argument('--reconciliation-manifest', required=True)\n    ap.add_argument('--incident-snapshot', required=True)\n    ap.add_argument('--collector-sha', required=True)")
replace_once("incident = worker.Incident(production, manifest_path, binding['incident'], empty)",
    "binding['offset2053650'] = worker.capture_offset_binding(a.incident_snapshot, a.collector_sha)\n    incident = worker.Incident(production, manifest_path, binding['incident'], empty, binding['offset2053650'])")
start = source.index('    uuid = base.uuid_for(7)')
end = source.index("    stem = 'main_review_watch_v7-'", start)
resource_capture = source[start:end]
source = source[:start] + source[end:]
replace_once("    cfg['holders'] = [expected]", resource_capture + "    cfg['holders'] = [expected]")
# New execution identities only; all original source/historical namespaces remain untouched.
source = source.replace('main_review_watch_v7', 'main_review_watch_v8').replace('main_review_resource_manager_v7', 'main_review_resource_manager_v8')
source = source.replace('build_main_review_v7_fresh.py', 'build_main_review_v8_fresh.py').replace('test_main_review_v7.py', 'test_main_review_v8.py')
source = source.replace('main_review_v7_source_only_v1', 'main_review_v8_source_only_v1')
replace_once("    for ref in immutable_refs:", "    offset = binding['offset2053650']\n    immutable_refs.extend([offset['snapshot_ref'], offset['collector_ref'], offset['raw_ref'], *offset['record_refs'], *offset['receipt_refs']])\n    immutable_refs.extend(x['raw_prefix_ref'] for x in offset['mount_bindings'].values())\n    cfg['source_sha256'][str(RT / 'main_review_watch_v7.py')] = worker.V7_SHA\n    cfg['source_sha256'][str(RT / 'main_review_resource_manager_v7.py')] = wrapper.OLD_MANAGER_SHA\n    cfg['source_sha256'][str(RT / 'build_main_review_v7_fresh.py')] = BUILDER_V7_SHA\n    for ref in immutable_refs:")
namespace = {**vars(v7), 'cpu_smoke': cpu_smoke, 'BUILDER_V7_SHA': BUILDER_V7_SHA}
exec(compile(source, str(Path(__file__)), 'exec'), namespace)
main = namespace['main']

if __name__ == '__main__':
    main()
