"""Negative tests only mutate fresh node-local fixtures, never original artifacts."""
from dataclasses import replace
from pathlib import Path
import tempfile


def run_negative_tests(pilot, local_root):
    results = {}
    root = Path(tempfile.mkdtemp(prefix='negative-fixtures-', dir=local_root))

    def fixture(name):
        original = root / (name + '.json')
        original.write_bytes(b'{"safe":1}\n')
        raw, s = pilot.stable_bytes(original)
        source = {'job': {'id': name, 'source_key': 'fixture/' + name},
                  'refs': {str(original): (pilot.digest(raw), s, str(original.resolve(strict=True)))}}
        entries, unused_build_stats = pilot.build_bundle(source, root)
        reader = pilot.Reader(source['job']['source_key'], entries, True)
        assert reader.read(original) == raw
        return original, entries, reader

    def rejected(name, action):
        try:
            action()
        except ValueError as error:
            results[name] = {'rejected': True, 'error': str(error)}
        else:
            raise AssertionError('negative fixture unexpectedly accepted: ' + name)

    path, entries, reader = fixture('original_byte')
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    rejected('original_one_byte', lambda: reader.read(path))

    path, entries, reader = fixture('local_byte')
    local = Path(entries[str(path)].local_path)
    local.chmod(0o600)
    raw = local.read_bytes()
    local.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    rejected('local_one_byte_identity', lambda: reader.read(path))
    # Even a forged refreshed local stat cannot excuse the expected content SHA.
    entries[str(path)] = replace(entries[str(path)], local_stat=pilot.stable_bytes(local)[1])
    reader = pilot.Reader('fixture/local_byte', entries, True)
    rejected('local_one_byte_sha_with_refreshed_stat', lambda: reader.read(path))

    path, entries, reader = fixture('source_swap')
    rejected('source_mapping_swap', lambda: pilot.Reader('fixture/other_source', entries, True).read(path))
    path, entries, reader = fixture('logical_swap')
    entries[str(path)] = replace(entries[str(path)], logical_path=str(root / 'other.json'))
    rejected('logical_path_swap', lambda: pilot.Reader('fixture/logical_swap', entries, True).read(path))
    path, entries, reader = fixture('expected_swap')
    rejected('expected_sha_swap', lambda: reader.read(path, '0' * 64))
    rejected('unknown_logical_path', lambda: reader.read(root / 'not-in-closure.json'))
    return results


if __name__ == '__main__':
    import curated_local_input_bundle_pilot3_v1 as pilot
    import json
    root = Path(tempfile.mkdtemp(prefix='pilot3-tests-', dir='/tmp'))
    print(json.dumps(run_negative_tests(pilot, root), indent=2, sort_keys=True))
