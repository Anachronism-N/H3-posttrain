#!/usr/bin/env python3
"""Download every evaluation run for a checkpoint set and print one comparison table.

Usage: collect_results.py <prefix> [prefix ...]
Each prefix is matched against m2d_train_v1/runs/<prefix>*/out/<name>/gen*.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

OSSUTIL = '/ssd/2/niezichao.nzc/.secrets/ossutilconfig'
ENDPOINT = 'oss-cn-hangzhou.aliyuncs.com'
RUNS = 'oss://futurelab-dpro-hz/niezichao.nzc/m2d_train_v1/runs/'
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_generations as audit  # noqa: E402


def oss(*args):
    out = subprocess.run(['ossutil', '-c', OSSUTIL, '-e', ENDPOINT, *args],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return out.stdout.decode('utf-8', 'replace')


def listing(uri):
    return [line.split()[-1] for line in oss('ls', uri).splitlines() if 'oss://' in line]


def main():
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for uri in listing(RUNS):
            name = uri.rstrip('/').rsplit('/', 1)[-1]
            if not any(name.startswith(p) for p in sys.argv[1:]):
                continue
            for key in listing(uri + 'out/'):
                gen_uri = key + 'generations.jsonl'
                if '/gen' not in gen_uri:
                    continue
                local = Path(tmp) / (key.rstrip('/').replace('/', '_') + '.jsonl')
                oss('cp', gen_uri, str(local), '--force')
                if not local.is_file() or local.stat().st_size == 0:
                    continue
                m = audit.audit(local)
                v = audit.verdict(m)
                rows.append((name, gen_uri.split('/out/')[1].rstrip('/'), m, v))
    rows.sort(key=lambda r: (r[0], r[1]))
    print('%-34s %-14s %6s %5s %5s %6s %5s %6s %s' % (
        'run', 'gen-set', 'KL', 'dur', 'cl/w', 'lenRatio', 'open', 'recall', 'verdict'))
    for name, genset, m, v in rows:
        print('%-34s %-14s %6.3f %5.2f %5.2f %6.2f %5d %6.3f %s' % (
            name[:34], genset[:14], m['window_kl'], m['mean_window_duration'],
            m['mean_clauses_per_window'], m['mean_length_ratio'], m['distinct_openers'],
            m['mean_ref_3gram_recall'],
            'PASS' if v['passed'] else 'fail:' + ','.join(k for k, ok in v['checks'].items() if not ok)))


if __name__ == '__main__':
    main()
