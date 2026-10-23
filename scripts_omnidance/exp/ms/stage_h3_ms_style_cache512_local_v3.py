import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SOURCE = Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/h3_ms_style_pilot_cache512_v2')
DESTINATION = Path('/opt/h3_ms_style_cache512_v3')
RECORD = Path('/opt/h3_ms_style_cache512_v3_staging.json')
EXPECTED_MARKER = '6eba363545c7717155a141755d1c47ed03f1659d028d12305f27d8db6f0e30dc'


def main():
    assert SOURCE.is_dir() and DESTINATION.parent.is_dir()
    assert not DESTINATION.exists() and not RECORD.exists()
    raw_marker = (SOURCE / 'complete.json').read_bytes()
    assert hashlib.sha256(raw_marker).hexdigest() == EXPECTED_MARKER
    marker = json.loads(raw_marker)
    assert marker['scope'] == 'source_label_train' and marker['signature']['dataset_size'] == 512
    assert len(marker['files']) == 512
    DESTINATION.mkdir(exist_ok=False)
    for rank in range(4):
        (DESTINATION / str(rank)).mkdir(exist_ok=False)
    def copy_member(item):
        name, expected = item
        relative = Path(name)
        assert not relative.is_absolute() and len(relative.parts) == 2
        assert relative.parts[0] in ('0', '1', '2', '3') and relative.suffix == '.pth'
        digest = hashlib.sha256()
        count = 0
        with (SOURCE / relative).open('rb') as src, (DESTINATION / relative).open('xb') as dst:
            for data in iter(lambda: src.read(8 * 1024 * 1024), b''):
                digest.update(data)
                dst.write(data)
                count += len(data)
            dst.flush()
            os.fsync(dst.fileno())
        actual = digest.hexdigest()
        if isinstance(expected, dict):
            assert actual == expected['sha256'] and count == expected['bytes']
        else:
            assert actual == expected
        return dict(name=name, sha256=actual, bytes=count)
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, result in enumerate(pool.map(copy_member, sorted(marker['files'].items())), 1):
            results.append(result)
            if index % 64 == 0:
                print(json.dumps(dict(event='existing_cache_copy_progress', members=index)), flush=True)
    assert (SOURCE / 'complete.json').read_bytes() == raw_marker
    with (DESTINATION / 'complete.json').open('xb') as stream:
        stream.write(raw_marker)
        stream.flush()
        os.fsync(stream.fileno())
    with RECORD.open('x') as stream:
        json.dump(dict(complete=True, source=str(SOURCE), destination=str(DESTINATION),
            source_marker_sha256=EXPECTED_MARKER, destination_marker_sha256=EXPECTED_MARKER,
            members=results, copied_bytes=sum(row['bytes'] for row in results),
            original_producer_relabelled=False, cache_reencoded=False,
            data_scope='source_label_train', semantic_approval=False), stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(event='existing_cache_local_staging_complete', members=512,
        destination=str(DESTINATION), marker_sha256=EXPECTED_MARKER,
        copied_bytes=sum(row['bytes'] for row in results), reencoded=False)), flush=True)


if __name__ == '__main__':
    main()
