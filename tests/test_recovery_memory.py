"""MPS allocator housekeeping during recovery: cache release only, peak driver memory recorded."""
from types import SimpleNamespace

import torch

from prune_lab import recovery
from test_recovery import TinyModel, batch


def test_housekeeping_empties_every_n_updates_and_after_assessments(monkeypatch):
    calls, memory = [], iter(range(100, 10000, 100))
    monkeypatch.setattr(torch.mps, 'driver_allocated_memory', lambda: next(memory))
    monkeypatch.setattr(torch.mps, 'empty_cache', lambda: calls.append('empty'))
    monkeypatch.setattr(recovery, 'MPS_EMPTY_CACHE_EVERY', 3)
    stats, mps = {}, SimpleNamespace(type='mps')
    for completed in range(1, 8):
        recovery._mps_housekeeping(stats, mps, completed)
    assert len(calls) == 2  # updates 3 and 6
    recovery._mps_housekeeping(stats, mps)  # after an assessment: always
    assert len(calls) == 3 and stats['peak_mps_driver_bytes'] == 800
    recovery._mps_housekeeping(stats, SimpleNamespace(type='cpu'), 3)
    assert len(calls) == 3


def test_cpu_recovery_unaffected_and_reports_no_mps_peak():
    torch.manual_seed(3)
    result = recovery.recover_model(TinyModel(), [batch(2), batch(3)], lambda m: {'accuracy': .5, 'loss': 1.0},
                                    {'rank': 2, 'alpha': 4, 'max_steps': 2, 'gradient_accumulation': 1, 'checkpoints': [2]})
    assert result['peak_mps_driver_bytes'] is None and result['mps_empty_cache_every'] == recovery.MPS_EMPTY_CACHE_EVERY
