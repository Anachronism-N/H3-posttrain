"""Import binding checks; no synthetic source is called a paper replication."""
import subprocess
from pathlib import Path
import sys

import pytest

MS = Path(__file__).resolve().parents[1] / "scripts_omnidance/exp/ms"
sys.path.insert(0, str(MS))
from ms_original_rules import load_original_rules


def test_missing_original_source_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="captioning.py"):
        load_original_rules(tmp_path)


def test_flat_source_import_is_bound_and_different_sources_are_rejected(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "captioning.py").write_text("marker = 'synthetic_import_contract_only'\n")
    (second / "captioning.py").write_text("marker = 'different_synthetic_source'\n")
    script = """
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from ms_original_rules import load_original_rules
first=load_original_rules(sys.argv[2])
assert first.marker=='synthetic_import_contract_only'
assert Path(first.__file__).resolve()==Path(sys.argv[2]).resolve()/'captioning.py'
assert load_original_rules(sys.argv[2]) is first
try:
    load_original_rules(sys.argv[3])
except ValueError:
    pass
else:
    raise AssertionError('different upstream was admitted')
"""
    subprocess.run([sys.executable, "-B", "-c", script, str(MS), str(first), str(second)], check=True)
