"""Selection locks must survive a JSON write/read round trip (integer-keyed iterative records)."""
import json
from pathlib import Path
import runpy
import sys

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from prune_lab import campaign, multiseed
from test_multiseed import tiny_support_config

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'relock_selection.py'


def test_fingerprint_is_round_trip_stable_and_unchanged_without_int_keys():
    value = {'layers': {i: [i, 0.1 * i] for i in range(12)}, 'name': 'x'}
    assert campaign._fingerprint(value) == campaign._fingerprint(json.loads(json.dumps(value)))
    assert campaign._fingerprint(value) != campaign.legacy_fingerprint(value)  # the old bug: '10' vs 2 ordering
    plain = {'b': [1, 2.5, None, True, {'z': 1e-17, 'a': 'é'}], 'a': (3, 4)}
    assert campaign._fingerprint(plain) == campaign.legacy_fingerprint(plain)  # existing locks stay byte-identical


@pytest.fixture
def iterative_run(tmp_path, monkeypatch):
    config = tiny_support_config(tmp_path, monkeypatch)
    torch.manual_seed(5)  # 12 layers: integer layer keys >= 10 sort differently as strings
    Qwen2ForCausalLM(Qwen2Config(vocab_size=5, hidden_size=16, intermediate_size=24, num_hidden_layers=12,
                                 num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
                                 pad_token_id=0, eos_token_id=2)).save_pretrained(config['model'], safe_serialization=True)
    config.update(targets=[.10, .20], uniform_per_target=True,
                  iterative=[{'id': 'iter-20', 'stages': [.10, .20], 'steps': [1, 1]}],
                  train_candidates=['unpruned-recovery', 'uniform-20', 'iter-20'])
    config['recovery'].update(max_steps=2, checkpoints=[1, 2], gradient_accumulation=1)
    root = tmp_path / 'runs' / 'iter'
    multiseed.run_seeds(config, root, [5])
    return root


def test_iterative_selection_reads_back_and_aggregates(iterative_run):
    seed = iterative_run / 'seed-5'
    selection = campaign._selection(seed)  # read back from disk; previously "Frozen development selection changed"
    record = next(c for c in selection['candidates'] if c['id'] == 'iter-20')
    assert len(record['iterative']['per_layer_differing']) == 12
    summary = multiseed.aggregate_run(iterative_run, samples=10)
    assert 'iter-20' in summary['variants']


def run_script(monkeypatch, *args):
    monkeypatch.setattr(sys, 'argv', ['relock_selection.py', *args])
    runpy.run_path(str(SCRIPT), run_name='__main__')


@pytest.mark.parametrize('resumed', [False, True])
def test_relock_script_verifies_old_lock_then_relocks(iterative_run, monkeypatch, resumed):
    sys.path.insert(0, str(SCRIPT.parent))
    from relock_selection import integer_key_positions, restore_integer_keys
    seed = iterative_run / 'seed-5'
    selection = json.loads((seed / 'selection.json').read_text())
    positions = integer_key_positions(selection)
    if resumed:  # a seed that resumed after stage 1 held only stage-2 fields with integer keys
        positions = [p for p in positions if 0 not in p[1][2:3]]
    old = campaign.legacy_fingerprint(restore_integer_keys(selection, positions))
    (seed / 'selection-lock.json').write_text(json.dumps({'sha256': old}))
    with pytest.raises(ValueError, match='Frozen development selection changed'):
        campaign._selection(seed)
    run_script(monkeypatch, '--run', str(iterative_run), '--dry-run')
    assert json.loads((seed / 'selection-lock.json').read_text())['sha256'] == old
    run_script(monkeypatch, '--run', str(iterative_run))
    assert campaign._selection(seed)
    run_script(monkeypatch, '--run', str(iterative_run))  # idempotent: already stable


def test_relock_script_refuses_changed_selection(iterative_run, monkeypatch):
    seed = iterative_run / 'seed-5'
    selection = json.loads((seed / 'selection.json').read_text())
    sys.path.insert(0, str(SCRIPT.parent))
    from relock_selection import restore_integer_keys
    (seed / 'selection-lock.json').write_text(json.dumps({'sha256': campaign.legacy_fingerprint(restore_integer_keys(selection))}))
    selection['candidates'][0]['dev']['token_f1'] += 0.01
    (seed / 'selection.json').write_text(json.dumps(selection))
    before = (seed / 'selection-lock.json').read_text()
    with pytest.raises(SystemExit, match='nothing written'):
        run_script(monkeypatch, '--run', str(iterative_run))
    assert (seed / 'selection-lock.json').read_text() == before
