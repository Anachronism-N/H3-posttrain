"""Frozen v8 plus explicit exact two-source historical-load content bridge."""
import hashlib
from pathlib import Path

RT = Path(__file__).resolve().parent
V8_SHA = '65972a15098bf838522fa79fe0e5b30dd139664a30dfceb7d1daad94d709079b'
if hashlib.sha256((RT / 'main_review_watch_v8.py').read_bytes()).hexdigest() != V8_SHA:
    raise RuntimeError('Frozen v8 watch changed')
import main_review_watch_v8 as v8
import main_old_load_receipts_v9 as historical

ROOT, module, derive = v8.ROOT, v8.module, v8.derive
need, sha, checked, recovery = v8.need, v8.sha, v8.checked, v8.recovery
V7_SHA, EMPTY_OFFSET, OFFSET, RAW, RAW_SHA = v8.V7_SHA, v8.EMPTY_OFFSET, v8.OFFSET, v8.RAW, v8.RAW_SHA
Membership, ByteCache = v8.Membership, v8.ByteCache
NormalizedReader = v8.NormalizedReader
historical_ids, v5 = v8.historical_ids, v8.v5
capture_offset_binding = v8.capture_offset_binding
capture_reconciliation_mount_bindings = v8.capture_reconciliation_mount_bindings
retained_selection, install_history_selection = v8.retained_selection, v8.install_history_selection
reviewer_from_verified_local_model = v8.reviewer_from_verified_local_model


_reconciliation_init = derive(v8.v7.Reconciliation.__init__,
    [("recovery.Evidence(production, index.membership.ref['path'])",
      "historical.ReconciliationEvidence(production, index.membership.ref['path'], index.incident.reconciliation_load_binding, manifest, expected_ref)")],
    {**vars(v8.v7), 'historical': historical})


class Reconciliation(v8.v7.Reconciliation):
    __init__ = _reconciliation_init


_original_reconcile = derive(v8.v7.RawIndex.reconcile,
    [('Reconciliation(self, manifest, expected_ref)', 'HistoricalReconciliation(self, manifest, expected_ref)')],
    {**vars(v8.v7), 'HistoricalReconciliation': Reconciliation})


class RawIndex(v8.RawIndex):
    def reconcile(self, manifest, expected_ref=None):
        dirty = _original_reconcile(self, manifest, expected_ref)
        if self.incident and self.incident.preparation:
            dirty |= self.reconcile_offset()
        return dirty


def capture_historical_load_binding(incident_binding):
    binding = historical.capture()
    need([e['original_hint_ref'] for e in binding['entries'].values()] == incident_binding['hint_refs'],
         'Exact two original source hints and historical bridge disagree')
    return binding


class Incident(v8.Incident):
    def __init__(self, production, manifest, binding, empty_binding=None, offset_binding=None):
        old_loads = binding.get('historical_load_receipts')
        need(old_loads is not None, 'Explicit exact historical receipt bridge required')
        evidence = historical.Evidence(production, manifest, old_loads)
        need([e['original_hint_ref'] for e in old_loads['entries'].values()] == binding['hint_refs'],
             'Historical bridge cannot admit foreign source hints')
        super().__init__(production, manifest, binding, empty_binding, offset_binding)
        self.evidence = evidence
        self.reconciliation_load_binding = binding.get('historical_reconciliation_load_receipts')
        if self.reconciliation_load_binding is not None:
            historical.validate_reconciliation(self.reconciliation_load_binding)


def progress(index, out, edition, bids, total):
    value = v8.progress(index, out, edition, bids, total)
    value['schema'] = 'curated_independent_review_watch_v9'
    value['execution_watch_v9_ref'] = {'path': str(Path(__file__)), 'sha256': sha(__file__)}
    value['execution_watch_v8_ref'] = {'path': str(RT / 'main_review_watch_v8.py'), 'sha256': V8_SHA}
    value['historical_load_receipts_exact_sources'] = 2 if isinstance(index.incident, Incident) else 0
    value['historical_reconciliation_load_receipts_exact_sources'] = 4 if isinstance(index.incident, Incident) and index.incident.reconciliation_load_binding else 0
    value['historical_load_receipt_original_refs_preserved'] = True
    value['new_load_used_to_prove_old_call'] = False
    return value


def validate_binding(binding, args, membership):
    bids = v8.validate_binding(binding, args, membership)
    historical.validate(binding['incident']['historical_load_receipts'])
    historical.validate_reconciliation(binding['incident']['historical_reconciliation_load_receipts'])
    return bids


main = derive(v8.v7.main,
    [("binding['incident'], binding['empty_incident'])", "binding['incident'], binding['empty_incident'], binding['offset2053650'])")],
    {**vars(v8.v7), 'Incident': Incident, 'RawIndex': RawIndex, 'progress': progress, 'validate_binding': validate_binding})

if __name__ == '__main__':
    main()
