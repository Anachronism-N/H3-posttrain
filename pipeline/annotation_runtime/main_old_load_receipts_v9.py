"""Exact old load receipt preimages; no current load or fabricated runtime fields."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import main_review_watch_v8 as v8

RT, ROOT, recovery, need = v8.RT, v8.ROOT, v8.recovery, v8.need
SCHEMA = 'main_review_exact_historical_load_receipts_v9'
STORE = ROOT / 'curated222939_repair_v1/main-old-load-receipts-v9/receipts'
EXPECTED = {
    'MVuipekTKQDQprcC': ('shard-0000', '2642c7ddba43a6973670ebeec370902e22e7d99541e93956543e7583d575385e'),
    'ytykxzHnSDAecSEP': ('shard-0003', '5f00958c9d1abcd7637e8575ee3382e0c2d3e841b4cc15d577750fbcce6df6bb'),
}


def native_bytes(receipt):
    code = receipt['code_sha256']
    for path, digest in code.items():
        recovery.q.verify_bytes({'path': path, 'sha256': digest})
    path = RT / 'source_curated_consumer_v1/ms_optimize.py'
    need(str(path) in code, 'Native JSON serializer not receipt-code bound')
    tree = ast.parse(path.read_bytes())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'json_bytes')
    namespace = {'json': json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['json_bytes']


def compose(actual_ref, actual, encode):
    return encode({**actual, 'actual_load_ref': actual_ref, 'model_load_completed': True})


def model_binding(path, receipt, encode):
    model, ref = recovery.document(path)
    files = {name: {k: row[k] for k in ('path', 'sha256')} for name, row in model['files'].items()}
    digest = hashlib.sha256(encode({'root': model['root'], 'files': files})).hexdigest()
    need(model['full_content_hash_computed_once_per_process'] is True
         and digest == model['sha256'] == receipt['model_identity_sha256'],
         'Historical full model inventory/weights identity differs')
    need(any(name.endswith('.safetensors') for name in files), 'Historical model has no bound weights')
    return ref, {'root': model['root'], 'files': files, 'sha256': digest}


def capture(save=False):
    entries = {}
    production = ROOT / 'curated222939_production_v1'
    repair = ROOT / 'curated222939_repair_v1/collector-recovery-v1/hints/train'
    for sid, (shard, digest) in EXPECTED.items():
        hint, hint_ref = recovery.document(repair / (sid + '.json'))
        historical_ref = {'path': str(production / 'visual' / shard / 'load.receipt.json'), 'sha256': digest}
        need(hint['evidence']['load_receipt_ref'] == historical_ref and hint['source_key'] == 'train/' + sid,
             'Exact original hint historical receipt identity differs')
        matches = []
        for path in sorted((production / 'visual' / shard / 'loads').glob('*.json')):
            need(path.stem.isdigit(), 'Immutable native loads candidate is not a PID file')
            actual, actual_ref = recovery.document(path)
            encode = native_bytes(actual)
            raw = compose(actual_ref, actual, encode)
            if hashlib.sha256(raw).hexdigest() == digest:
                matches.append((actual, actual_ref, raw, encode))
        need(len(matches) == 1, 'Exact old receipt preimage missing/ambiguous; retain pending')
        actual, actual_ref, raw, encode = matches[0]
        recovered_ref = {'path': str(STORE / (digest + '.json')), 'sha256': digest}
        if save:
            recovery.save_raw_once(Path(recovered_ref['path']), raw)
        recovered, recovered_actual_ref = recovery.document(recovered_ref['path'], digest)
        need(recovered_actual_ref == recovered_ref and Path(recovered_ref['path']).read_bytes() == raw,
             'Recovered immutable exact receipt bytes differ')
        model_ref, inventory = model_binding(production / 'visual' / shard / 'model.identity.json', actual, encode)
        entries[sid] = {'source_key': 'train/' + sid, 'original_hint_ref': hint_ref,
            'historical_load_receipt_ref': historical_ref, 'recovered_receipt_ref': recovered_ref,
            'actual_load_ref': actual_ref, 'model_identity_ref': model_ref, 'historical_model_inventory': inventory,
            'native_code_refs': [{'path': p, 'sha256': h} for p, h in actual['code_sha256'].items()],
            'reconstructed_SHA_exact': True, 'original_load_receipt_modified': False,
            'new_load_used_to_prove_old_call': False, 'new_model_load_performed': False,
            'retry_budget_reset': False, 'quality_approved': False}
    return {'schema': SCHEMA, 'original_byte_begin': recovery.BAD_OFFSET,
            'source_count': 2, 'entries': entries}


def validate(binding):
    need(binding == capture(), 'Frozen exact historical load bridge differs')
    return binding


def immutable_refs(binding):
    refs = []
    for entry in binding['entries'].values():
        refs.extend(entry[name] for name in ('original_hint_ref', 'recovered_receipt_ref', 'actual_load_ref', 'model_identity_ref'))
        refs.extend(entry['native_code_refs'])
    return refs


_params = v8.derive(recovery.Evidence.params,
    [("document(visual_root.parent.parent / 'load.receipt.json')", "self.historical_receipt(visual_root.parent.parent / 'load.receipt.json')")], vars(recovery))
_terminal = v8.derive(recovery.Evidence.terminal,
    [("if name.endswith('_ref'):\n            q.verify_bytes(ref)", "if name.endswith('_ref'):\n            self.verify_evidence_ref(ref)")], vars(recovery))


class Evidence(recovery.Evidence):
    params = _params
    terminal = _terminal

    def __init__(self, production, manifest, binding):
        super().__init__(production, manifest)
        self.bridge = validate(binding)
        need(self.production == ROOT / 'curated222939_production_v1', 'Historical bridge foreign production')

    def historical_receipt(self, path):
        matches = [e for e in self.bridge['entries'].values() if e['historical_load_receipt_ref']['path'] == str(path)]
        need(len(matches) == 1, 'No explicit exact historical receipt admission')
        entry = matches[0]
        self.verify_evidence_ref(entry['historical_load_receipt_ref'])
        value, _ = recovery.document(entry['recovered_receipt_ref']['path'], entry['recovered_receipt_ref']['sha256'])
        return value, entry['historical_load_receipt_ref']

    def verify_evidence_ref(self, ref):
        matches = [e for e in self.bridge['entries'].values() if e['historical_load_receipt_ref'] == ref]
        if not matches:
            recovery.q.verify_bytes(ref)
            return
        need(len(matches) == 1, 'Ambiguous historical ref bridge')
        entry = matches[0]
        actual, actual_ref = recovery.document(entry['actual_load_ref']['path'], entry['actual_load_ref']['sha256'])
        encode = native_bytes(actual)
        raw, recovered_ref = recovery.stable_read(entry['recovered_receipt_ref']['path'])
        need(recovered_ref == entry['recovered_receipt_ref'] and recovered_ref['sha256'] == ref['sha256']
             and raw == compose(actual_ref, actual, encode), 'Exact historical load bytes/actual process changed')
        model_ref, inventory = model_binding(entry['model_identity_ref']['path'], actual, encode)
        need(model_ref == entry['model_identity_ref'] and inventory == entry['historical_model_inventory'],
             'Original complete weights/code identity bridge changed')


RECONCILIATION_MANIFEST = ROOT / 'curated_terminal_reconciliation_v1/admission-main-review-v7-v1.json'
RECONCILIATION_EXPECTED = {
    'BBucGIEtwcrDIngP': ('shard-0002', '3599e28c36a80182c9aafc044dc122cc2bbf3dd87d2418a3d6562670a5318051'),
    'dBloPhzUQlCooRtD': ('shard-0002', '3599e28c36a80182c9aafc044dc122cc2bbf3dd87d2418a3d6562670a5318051'),
    'pbUXtdkTIWrvjeXw': ('shard-0000', '2642c7ddba43a6973670ebeec370902e22e7d99541e93956543e7583d575385e'),
    'tYhkFFYDNlaEJimA': ('shard-0002', '3599e28c36a80182c9aafc044dc122cc2bbf3dd87d2418a3d6562670a5318051'),
}


def capture_reconciliation(save=False):
    admission, admission_ref = recovery.document(RECONCILIATION_MANIFEST)
    production = ROOT / 'curated222939_production_v1'
    need(admission['source_keys'] == ['train/' + sid for sid in RECONCILIATION_EXPECTED]
         and admission['production'] == str(production), 'Exact original four-source admission differs')
    entries = {}
    recovered = {}
    for record_ref in admission['record_refs']:
        record, actual_record_ref = recovery.document(record_ref['path'], record_ref['sha256'])
        sid = record['hint']['id']
        need(sid in RECONCILIATION_EXPECTED and actual_record_ref == record_ref
             and record_ref['path'] == str(ROOT / 'curated_terminal_reconciliation_v1/records/train' / (sid + '.json')),
             'Foreign historical four-source record')
        shard, digest = RECONCILIATION_EXPECTED[sid]
        original_ref = {'path': str(production / 'visual' / shard / 'load.receipt.json'), 'sha256': digest}
        need(record['evidence']['load_receipt_ref'] == original_ref and record['original_byte_offset'] is None,
             'Original four-source historical load/null-offset differs')
        if digest not in recovered:
            matches = []
            for path in sorted((production / 'visual' / shard / 'loads').glob('*.json')):
                need(path.stem.isdigit(), 'Native load candidate is not immutable PID namespace')
                actual, actual_ref = recovery.document(path)
                encode = native_bytes(actual)
                raw = compose(actual_ref, actual, encode)
                if hashlib.sha256(raw).hexdigest() == digest:
                    matches.append((actual, actual_ref, raw, encode))
            need(len(matches) == 1, 'Exact old four-source load preimage missing/ambiguous; retain pending')
            actual, actual_ref, raw, encode = matches[0]
            receipt_ref = {'path': str(STORE / (digest + '.json')), 'sha256': digest}
            if save:
                recovery.save_raw_once(Path(receipt_ref['path']), raw)
            stored, stored_ref = recovery.stable_read(receipt_ref['path'])
            need(stored == raw and stored_ref == receipt_ref, 'Original four-source exact load bytes differ')
            model_ref, inventory = model_binding(production / 'visual' / shard / 'model.identity.json', actual, encode)
            recovered[digest] = {'historical_load_receipt_ref': original_ref, 'recovered_receipt_ref': receipt_ref,
                'actual_load_ref': actual_ref, 'model_identity_ref': model_ref, 'historical_model_inventory': inventory,
                'native_code_refs': [{'path': path, 'sha256': sha} for path, sha in actual['code_sha256'].items()],
                'reconstructed_SHA_exact': True, 'original_load_receipt_modified': False,
                'new_load_used_to_prove_old_call': False, 'new_model_load_performed': False,
                'retry_budget_reset': False, 'quality_approved': False}
        entries[sid] = {'source_key': record['source_key'], 'batch_id': record['hint']['batch_id'],
                        'original_hint_ref': record_ref, **recovered[digest]}
    need(set(entries) == set(RECONCILIATION_EXPECTED), 'Exact four-source load bridge incomplete')
    return {'schema': SCHEMA + '_original_four_reconciliation', 'admission_ref': admission_ref,
            'source_count': 4, 'entries': entries}


def validate_reconciliation(binding):
    need(binding == capture_reconciliation(), 'Frozen original four-source load bridge differs')
    return binding


class ReconciliationEvidence(Evidence):
    def __init__(self, production, manifest, binding, admission, expected_ref=None):
        recovery.Evidence.__init__(self, production, manifest)
        self.bridge = validate_reconciliation(binding)
        need(self.production == ROOT / 'curated222939_production_v1'
             and str(Path(admission).resolve()) == str(RECONCILIATION_MANIFEST)
             and (expected_ref is None or expected_ref == binding['admission_ref']),
             'Four-source bridge foreign production/admission')
        self.active_entry = None

    def terminal(self, sid, split, batch):
        need(sid in self.bridge['entries'] and split == 'train'
             and batch == self.bridge['entries'][sid]['batch_id'], 'Foreign source cannot borrow historical load proof')
        self.active_entry = self.bridge['entries'][sid]
        try:
            return _terminal(self, sid, split, batch)
        finally:
            self.active_entry = None

    def historical_receipt(self, path):
        entry = self.active_entry
        need(entry is not None and str(path) == entry['historical_load_receipt_ref']['path'],
             'Historical receipt not bound to active original source')
        self.verify_evidence_ref(entry['historical_load_receipt_ref'])
        value, _ = recovery.document(entry['recovered_receipt_ref']['path'], entry['recovered_receipt_ref']['sha256'])
        return value, entry['historical_load_receipt_ref']

    def verify_evidence_ref(self, ref):
        entry = self.active_entry
        if entry is None or ref != entry['historical_load_receipt_ref']:
            recovery.q.verify_bytes(ref)
            return
        # Only the currently admitted original source may use this exact receipt bridge.
        full_bridge = self.bridge
        try:
            self.bridge = {'entries': {entry['source_key']: entry}}
            Evidence.verify_evidence_ref(self, ref)
        finally:
            self.bridge = full_bridge


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare-exact', action='store_true', required=True)
    ap.parse_args()
    print(json.dumps(capture(save=True), indent=2))
