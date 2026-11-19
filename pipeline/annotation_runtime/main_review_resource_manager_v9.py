"""Exact history bridge; unchanged authorized N5G7 pidfd/lease transaction."""
from main_review_watch_v9 import RT, ROOT, module, derive, need, checked, historical

MANAGER_V8_SHA = '2273e1fafca0fd3be82aa1b6b6c98a6ac4590106cb84f286781b25d4d6f95e3d'
v8 = module('private_frozen_resource_v8_for_v9', RT / 'main_review_resource_manager_v8.py', MANAGER_V8_SHA)
base, BASE_MANAGER_SHA = v8.base, v8.BASE_MANAGER_SHA
OLD_MANAGER_SHA, check_model = v8.OLD_MANAGER_SHA, v8.check_model
release_holder, _release = v8.release_holder, v8._release
_validate = derive(v8.v7.validate, [
    ("'main_review_watch_v7.py'", "'main_review_watch_v9.py'"),
    ("cfg['phase'] != 'main_review_watch_v7'", "cfg['phase'] != 'main_review_watch_v9'"),
    ("startswith('main_review_watch_v7-')", "startswith('main_review_watch_v9-')")], vars(v8.v7))
_summaries = derive(v8.v7.summaries,
    [("'curated_independent_review_watch_v7'", "'curated_independent_review_watch_v9'")], vars(v8.v7))
_offset_validate = derive(v8.validate,
    [("'main_review_resource_manager_v8.py'", "'main_review_resource_manager_v9.py'")], {**vars(v8), '_validate': _validate, '__file__': __file__})


def validate(cfg):
    _offset_validate(cfg)
    binding = checked(cfg['source_binding'])
    old_loads = historical.validate(binding['incident']['historical_load_receipts'])
    old_four = historical.validate_reconciliation(binding['incident']['historical_reconciliation_load_receipts'])
    refs = historical.immutable_refs(old_loads) + historical.immutable_refs(old_four)
    need(old_four['admission_ref'] == binding['reconciliation_manifest_ref'], 'Original four-source exact load admission differs')
    need(all(cfg['source_sha256'].get(r['path']) == r['sha256'] for r in refs),
         'Exact historical actual-load/model/code/fullsource refs must remain frozen')
    need(cfg['source_sha256'].get(str(RT / 'main_old_load_receipts_v9.py')) == base.sha(historical.__file__),
         'Historical bridge execution SHA changed')


base.validate_pair = validate
base.check_model = check_model
base.stage_summaries = _summaries
base.release_holder = release_holder
base.AUTHORIZED_PAIRS = {'N5': ((7,),)}
main = derive(v8.main,
    [("'main_review_v8_source_only_v1'", "'main_review_v9_source_only_v1'")], {**vars(v8), 'base': base, '__file__': __file__})

if __name__ == '__main__':
    main()
