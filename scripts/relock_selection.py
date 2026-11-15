#!/usr/bin/env python3
"""Re-lock selection.json files whose lock was computed with integer dict keys (iterative candidates).

Before the fingerprint fix, iterative candidates carried integer-keyed dicts in memory
(recovery.stages[].keep_indices, iterative.stage_keep_indices[], iterative.per_layer_differing),
so the stored lock hashed int-sorted keys while the file holds string keys. For every seed
directory this script restores integer keys on those fields (all of them, or the stage-2-only subset
of a seed that resumed after stage 1), requires the OLD lock to equal the old-style fingerprint of
that restored value (proof that selection.json is unchanged apart from key types),
and only then writes the new, round-trip-stable lock. Anything else is refused.

  .venv/bin/python scripts/relock_selection.py --run runs/support-v2-allocator [--dry-run]
"""
import argparse
import copy
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def integer_key_positions(selection):
    """Every known field that may have carried integer keys in memory: (candidate, path) pairs."""
    positions = []
    for c, candidate in enumerate(selection.get('candidates', [])):
        for k, stage in enumerate((candidate.get('recovery') or {}).get('stages', [])):
            if isinstance(stage.get('keep_indices'), dict):
                positions.append((c, ('recovery', 'stages', k, 'keep_indices')))
        iterative = candidate.get('iterative') or {}
        for k, mapping in enumerate(iterative.get('stage_keep_indices', [])):
            if isinstance(mapping, dict):
                positions.append((c, ('iterative', 'stage_keep_indices', k)))
        if isinstance(iterative.get('per_layer_differing'), dict):
            positions.append((c, ('iterative', 'per_layer_differing')))
    return positions


def restore_integer_keys(selection, positions=None):
    """Copy of ``selection`` with integer keys restored at ``positions`` (default: all known ones)."""
    restored = copy.deepcopy(selection)
    for c, path in integer_key_positions(selection) if positions is None else positions:
        parent = restored['candidates'][c]
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = {int(k): v for k, v in parent[path[-1]].items()}
    return restored


def old_lock_matches(selection, lock):
    """True if some subset of the known fields, with integer keys restored, reproduces the old lock.

    A seed that resumed after stage 1 read that stage back from disk (string keys) and kept only the
    later stages in memory (integer keys), so the matching subset differs between seeds. Only key
    types at known positions vary; every value is taken from selection.json unchanged.
    """
    from prune_lab.campaign import legacy_fingerprint
    positions = integer_key_positions(selection)
    if len(positions) > 16:
        raise ValueError('Too many integer-keyed fields to verify exhaustively')
    for subset in itertools.product([False, True], repeat=len(positions)):
        chosen = [p for p, keep in zip(positions, subset) if keep]
        if legacy_fingerprint(restore_integer_keys(selection, chosen)) == lock:
            return chosen
    return None


def relock(seed_dir, dry_run=False):
    from prune_lab.campaign import _fingerprint
    from prune_lab.reporting import write_json
    selection = json.loads((seed_dir / 'selection.json').read_text())
    lock = json.loads((seed_dir / 'selection-lock.json').read_text())['sha256']
    new = _fingerprint(selection)
    if lock == new:
        return 'already stable'
    if old_lock_matches(selection, lock) is None:
        raise ValueError(f'{seed_dir}: old lock does not match selection.json even with integer keys restored; refusing')
    if not dry_run:
        write_json(seed_dir / 'selection-lock.json', {'sha256': new})
    return 'verified; ' + ('would re-lock' if dry_run else 're-locked')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    seeds = sorted(p for p in Path(args.run).glob('seed-*') if (p / 'selection-lock.json').exists())
    if not seeds:
        parser.error(f'No locked seed directories under {args.run}')
    # Verify every seed before writing any lock.
    from prune_lab.campaign import _fingerprint
    for seed_dir in seeds:
        selection = json.loads((seed_dir / 'selection.json').read_text())
        lock = json.loads((seed_dir / 'selection-lock.json').read_text())['sha256']
        if lock != _fingerprint(selection) and old_lock_matches(selection, lock) is None:
            raise SystemExit(f'{seed_dir}: old lock does not match selection.json even with integer keys restored; nothing written')
    for seed_dir in seeds:
        print(f'{seed_dir.name}: {relock(seed_dir, args.dry_run)}')


if __name__ == '__main__':
    main()
