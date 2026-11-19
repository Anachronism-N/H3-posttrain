"""CPU-only explicit import from original result serialization, never sorted export order."""
import hashlib
import importlib.util
from pathlib import Path

RT = Path(__file__).resolve().parent
BASE = RT / 'oldbar_explicit_import_v1.py'
BASE_SHA = 'aadc2f7578c1e9d986c7dcee86f7f84297dcb34190c81dee1233a375fe53d66b'
if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA:
    raise RuntimeError('Frozen original importer changed')
spec = importlib.util.spec_from_file_location('oldbar_original_import_refrepair_v2', BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
OriginalImporter = base.Importer


class ExactResultImporter(OriginalImporter):
    def verify_evidence(self, result, plan, raw_line, source, current):
        ref = result['result_ref']
        self.files.verify(ref)
        path = Path(ref['path']).resolve(strict=True)
        base.need(path.is_relative_to(source.resolve() / 'results'), 'Original result escaped source')
        original = base.load(path)
        base.need(all(result.get(k) == v for k, v in original.items()),
                  'Completed source differs from original result payload')
        # Old transaction digests preserve dictionary insertion order. The completed
        # export is recursively key-sorted, while the original result remains bound
        # to the exact request/receipt serialization used during generation.
        ordered = {**original, **{k:v for k,v in result.items() if k not in original}}
        provenance, model = super().verify_evidence(ordered, plan, raw_line, source, current)
        self.files.verify(ref)
        provenance['original_result_key_order_preserved_for_legacy_digests'] = True
        provenance['legacy_digest_source_result_ref'] = ref
        provenance['exact_result_import_wrapper_ref'] = self.files.ref(__file__)
        provenance['no_original_result_request_or_receipt_bytes_rewritten'] = True
        return provenance, model


base.Importer = ExactResultImporter
if __name__ == '__main__':
    raise SystemExit(base.main())
