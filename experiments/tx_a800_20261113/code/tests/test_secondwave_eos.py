from types import SimpleNamespace
import pytest
from captionlab.secondwave_baseline_v3 import terminal_ids

class Tokenizer:
    eos_token_id=7
    def __len__(self):return 20

def model(generation=None,config=7):
    return SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=generation),config=SimpleNamespace(eos_token_id=config))

def test_native_thinker_eos_when_generation_config_missing():
    assert terminal_ids(model(),Tokenizer())==[7]
    assert terminal_ids(model([7,8]),Tokenizer())==[7,8]

def test_missing_or_mismatched_eos_refused():
    for m in (model(config=None),model(8),model([None]),model(40)):
        with pytest.raises(ValueError):terminal_ids(m,Tokenizer())
