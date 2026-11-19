"""Preserve one verified empty physical row; all source rows use frozen recovery."""
import hashlib
import importlib.util
from pathlib import Path

RT = Path(__file__).resolve().parent
BASE = RT / 'curated_repair_collector_recovery_v1.py'
BASE_SHA = 'f85efb1441930cfc31bfe4cafacf25e0fdb18aeb9da6ad80fb540b58519237cd'
EMPTY_OFFSET = 1192186
EMPTY_SHA = '01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b'
if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA:
    raise RuntimeError('Frozen source recovery code changed')
spec = importlib.util.spec_from_file_location('frozen_collector_recovery_empty_v2', BASE)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
original_recover_line = recovery.recover_line


def recover_line(production, out, begin, line, evidence, snapshots, rules=recovery.KNOWN):
    if begin != EMPTY_OFFSET:
        return original_recover_line(production, out, begin, line, evidence, snapshots, rules)
    if line != b'\n' or hashlib.sha256(line).hexdigest() != EMPTY_SHA:
        raise recovery.UnrecoverableLine('Verified empty physical row changed; retain cursor')
    if len(snapshots) != 3 or any(s['line_sha256'] != EMPTY_SHA for s in snapshots):
        raise recovery.StreamBusy('Empty physical row lacks three stable byte snapshots')
    if any(s['identity'] != snapshots[0]['identity'] for s in snapshots):
        raise recovery.StreamBusy('Empty physical row changed during snapshot verification')
    production, out = Path(production).resolve(), Path(out).resolve()
    if production != RT.parent / 'curated222939_production_v1' or out != RT.parent / 'curated222939_repair_v1':
        raise recovery.UnrecoverableLine('Empty physical row is outside its verified original stream')
    raw_path = out / 'collector-recovery-v1/badlines' / (str(begin) + '-' + EMPTY_SHA + '.raw')
    recovery.save_raw_once(raw_path, line)
    receipt_path = out / 'collector-emptyline-v2' / (str(begin) + '.json')
    receipt = dict(schema='verified_empty_physical_jsonl_row_v2',
        original_stream=str(production / 'completed.jsonl'), byte_begin=begin, byte_end=begin + 1,
        raw_ref=recovery.q.identity(raw_path), raw_size=1, raw_sha256=EMPTY_SHA,
        contained_source_records=0, reconstructed_source_records=0,
        source_terminal_or_label_generated=False, source_records_skipped=False,
        original_log_modified=False, original_decode_error_preserved=True,
        source_population_completeness_inferred=False,
        unseen_or_lost_terminals_require_separate_reconciliation=True,
        stable_badline_snapshots=snapshots, collector_base_ref=dict(path=str(BASE), sha256=BASE_SHA),
        emptyline_handler_ref=recovery.q.identity(__file__), **recovery.q.FLAGS)
    if receipt_path.exists():
        old = recovery.q.load(receipt_path)
        recovery.same({k:v for k,v in old.items() if k != 'stable_badline_snapshots'},
            {k:v for k,v in receipt.items() if k != 'stable_badline_snapshots'},
            'Verified empty physical row receipt changed')
        receipt = old
    recovery.q.save_once(receipt_path, receipt)
    return []


recovery.recover_line = recover_line
if __name__ == '__main__':
    recovery.main()
