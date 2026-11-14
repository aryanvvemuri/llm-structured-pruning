"""Independent checks of the seed x document-family hierarchical bootstrap and aggregation.

Written against the behaviour that matters for research conclusions, using synthetic
per-example records whose true answers are known. Does not reuse the writer's helpers.
"""
import json
import random
import statistics

import pytest

from prune_lab import campaign, multiseed


def make_variant(rows, loss=1.0, parameters=100, reduction=0.0):
    """rows: list of (key, group, token_f1, rouge_l)."""
    examples = {k: (g, {'token_f1': f, 'rouge_l': r}) for k, g, f, r in rows}
    n = len(rows)
    return {'examples': examples, 'parameters': parameters, 'actual_reduction': reduction,
            'aggregate': {'token_f1': 100 * sum(r[2] for r in rows) / n,
                          'rouge_l': 100 * sum(r[3] for r in rows) / n,
                          'answer_token_loss': loss, 'generation_limit_fraction': 0.0}}


def from_scores(scores, groups):
    return make_variant([(f'ex{i}', g, s, s) for i, (s, g) in enumerate(zip(scores, groups))])


# --- exact estimates ---------------------------------------------------------------------

def test_estimate_is_exact_grand_mean_with_unequal_family_sizes():
    groups = ['f0'] * 5 + ['f1'] + ['f2'] * 2
    a = {1: from_scores([.1, .2, .3, .4, .5, .9, .0, .6], groups),
         2: from_scores([.2, .2, .2, .2, .2, .2, .2, .2], groups)}
    b = {1: from_scores([.5, .5, .5, .5, .5, .5, .5, .5], groups),
         2: from_scores([1., 0., 1., 0., 1., 0., 1., 0.], groups)}
    own = multiseed.hierarchical_bootstrap(None, b, samples=50, seed=1)
    # Example-weighted (not family-weighted) mean, averaged over seeds.
    assert own['estimate_points'] == pytest.approx(100 * (0.5 + 0.5) / 2)
    mean_a = (sum([.1, .2, .3, .4, .5, .9, .0, .6]) / 8 + .2) / 2
    diff = multiseed.hierarchical_bootstrap(a, b, samples=50, seed=1)
    assert diff['estimate_points'] == pytest.approx(100 * (0.5 - mean_a))
    reverse = multiseed.hierarchical_bootstrap(b, a, samples=50, seed=1)
    assert reverse['estimate_points'] == pytest.approx(-diff['estimate_points'])
    lo, hi = diff['interval_95_points']
    rlo, rhi = reverse['interval_95_points']
    assert rlo == pytest.approx(-hi) and rhi == pytest.approx(-lo)
    assert diff['examples'] == 8 and diff['document_families'] == 3


def test_metric_selection_uses_rouge_not_f1():
    groups = ['g0', 'g1']
    a = {1: make_variant([('x', 'g0', .2, .9), ('y', 'g1', .2, .9)])}
    b = {1: make_variant([('x', 'g0', .4, .1), ('y', 'g1', .4, .1)])}
    assert multiseed.hierarchical_bootstrap(a, b, 'token_f1', samples=20)['estimate_points'] == pytest.approx(20)
    assert multiseed.hierarchical_bootstrap(a, b, 'rouge_l', samples=20)['estimate_points'] == pytest.approx(-80)
    with pytest.raises(ValueError):
        multiseed.hierarchical_bootstrap(a, b, 'exact_match', samples=20)
    for bad in (1, 0, 2.0, True):
        with pytest.raises(ValueError):
            multiseed.hierarchical_bootstrap(a, b, samples=bad)
    assert groups


def test_determinism_and_seed_sensitivity():
    rng = random.Random(3)
    groups = [f'g{i // 3}' for i in range(30)]
    a = {s: from_scores([rng.random() for _ in groups], groups) for s in (1, 2, 3)}
    b = {s: from_scores([rng.random() for _ in groups], groups) for s in (1, 2, 3)}
    first = multiseed.hierarchical_bootstrap(a, b, samples=300, seed=11)
    assert first == multiseed.hierarchical_bootstrap(a, b, samples=300, seed=11)
    other = multiseed.hierarchical_bootstrap(a, b, samples=300, seed=12)
    assert other['estimate_points'] == first['estimate_points']
    assert other['interval_95_points'] != first['interval_95_points']
    # Insertion order of examples and seeds must not matter.
    shuffled_b = {}
    for s in (3, 1, 2):
        items = list(b[s]['examples'].items())
        rng.shuffle(items)
        shuffled_b[s] = dict(b[s], examples=dict(items))
    again = multiseed.hierarchical_bootstrap(a, shuffled_b, samples=300, seed=11)
    assert again['estimate_points'] == pytest.approx(first['estimate_points'])
    assert again['interval_95_points'] == pytest.approx(first['interval_95_points'])


# --- pairing errors ----------------------------------------------------------------------

def test_mismatched_ids_groups_and_seeds_are_errors():
    groups = ['g0', 'g0', 'g1', 'g1']
    a = {1: from_scores([.1, .2, .3, .4], groups)}
    renamed = {1: make_variant([(f'other{i}', g, .5, .5) for i, g in enumerate(groups)])}
    with pytest.raises(ValueError, match='identical example ids'):
        multiseed.hierarchical_bootstrap(a, renamed, samples=10)
    regrouped = {1: from_scores([.1, .2, .3, .4], ['g0', 'g1', 'g1', 'g1'])}
    with pytest.raises(ValueError, match='identical example ids'):
        multiseed.hierarchical_bootstrap(a, regrouped, samples=10)
    missing = {1: make_variant([('ex0', 'g0', .1, .1), ('ex1', 'g0', .1, .1), ('ex2', 'g1', .1, .1)])}
    with pytest.raises(ValueError, match='identical example ids'):
        multiseed.hierarchical_bootstrap(a, missing, samples=10)
    # Within one variant, seeds must also cover identical examples.
    inconsistent = {1: a[1], 2: regrouped[1]}
    with pytest.raises(ValueError, match='identical example ids'):
        multiseed.hierarchical_bootstrap(None, inconsistent, samples=10)
    with pytest.raises(ValueError, match='shared recovery seeds'):
        multiseed.hierarchical_bootstrap({1: a[1]}, {2: a[1]}, samples=10)
    # Unpaired (cross-run) comparisons legitimately allow different seed values.
    result = multiseed.hierarchical_bootstrap({1: a[1]}, {2: a[1]}, paired_seeds=False, samples=10)
    assert result['seeds_a'] == [1] and result['seeds_b'] == [2] and result['estimate_points'] == 0


def test_paired_comparison_with_unequal_seed_sets_is_refused():
    groups = ['g0', 'g1']
    a = {s: from_scores([.5, .5], groups) for s in (1, 2, 3)}
    b = {1: from_scores([.6, .6], groups), 2: from_scores([.6, .6], groups)}
    with pytest.raises(ValueError):
        multiseed.hierarchical_bootstrap(a, b, samples=10)


# --- degenerate shapes -------------------------------------------------------------------

def test_single_seed_and_single_family_degenerate_correctly():
    rng = random.Random(5)
    groups = [f'g{i % 6}' for i in range(24)]
    a = {7: from_scores([rng.random() for _ in groups], groups)}
    b = {7: from_scores([rng.random() for _ in groups], groups)}
    one_seed = multiseed.hierarchical_bootstrap(a, b, samples=400, seed=2)
    est = one_seed['estimate_points']
    assert one_seed['seeds_only_interval'] == pytest.approx([est, est])
    lo, hi = one_seed['interval_95_points']
    assert lo < est < hi
    # With one seed the full interval is a pure family bootstrap: same rng stream positions differ,
    # so compare against an explicit family-only computation of the same width scale.
    assert hi - lo == pytest.approx(one_seed['families_only_interval'][1] - one_seed['families_only_interval'][0], rel=.35)

    one_family = ['only'] * 10
    a2 = {s: from_scores([.5] * 10, one_family) for s in (1, 2, 3, 4)}
    b2 = {s: from_scores([.5 + .05 * s] * 10, one_family) for s in (1, 2, 3, 4)}
    res = multiseed.hierarchical_bootstrap(a2, b2, samples=400, seed=2)
    assert res['document_families'] == 1
    assert res['estimate_points'] == pytest.approx(12.5)
    assert res['families_only_interval'] == pytest.approx([12.5, 12.5])
    assert res['interval_95_points'][0] >= 5 - 1e-9 and res['interval_95_points'][1] <= 20 + 1e-9
    assert res['interval_95_points'][0] < 12.5 < res['interval_95_points'][1]


def test_families_not_examples_are_the_resampling_unit():
    rng = random.Random(9)
    n_families, per = 12, 3
    base_groups = [f'g{i // per}' for i in range(n_families * per)]
    family_effect = {f'g{f}': rng.gauss(0, .15) for f in range(n_families)}
    diffs = [min(1, max(0, .5 + family_effect[g])) for g in base_groups]
    a = {1: from_scores([.5] * len(base_groups), base_groups)}
    b = {1: from_scores(diffs, base_groups)}
    base = multiseed.hierarchical_bootstrap(a, b, samples=1000, seed=4)
    # Duplicate every example inside its own family: no new information, interval must not shrink.
    dup_groups = base_groups * 2
    a_dup = {1: from_scores([.5] * len(dup_groups), dup_groups)}
    b_dup = {1: from_scores(diffs * 2, dup_groups)}
    dup = multiseed.hierarchical_bootstrap(a_dup, b_dup, samples=1000, seed=4)
    assert dup['estimate_points'] == pytest.approx(base['estimate_points'])
    assert dup['interval_95_points'] == pytest.approx(base['interval_95_points'])
    # Treating each duplicated example as its own family (the wrong unit) would narrow it.
    wrong = {1: make_variant([(f'ex{i}', f'solo{i}', s, s) for i, s in enumerate(diffs * 2)])}
    wrong_a = {1: make_variant([(f'ex{i}', f'solo{i}', .5, .5) for i in range(len(dup_groups))])}
    narrow = multiseed.hierarchical_bootstrap(wrong_a, wrong, samples=1000, seed=4)
    width = lambda r: r['interval_95_points'][1] - r['interval_95_points'][0]
    assert width(narrow) < 0.75 * width(dup)


def test_paired_seeds_remove_shared_seed_variation():
    groups = [f'g{i}' for i in range(10)]
    seed_effect = {1: -.1, 2: 0, 3: .1, 4: .05}
    a = {s: from_scores([.4 + e] * 10, groups) for s, e in seed_effect.items()}
    b = {s: from_scores([.45 + e] * 10, groups) for s, e in seed_effect.items()}
    paired = multiseed.hierarchical_bootstrap(a, b, samples=500, seed=1)
    assert paired['interval_95_points'] == pytest.approx([5, 5])
    unpaired = multiseed.hierarchical_bootstrap(a, b, paired_seeds=False, samples=500, seed=1)
    assert unpaired['interval_95_points'][0] < 5 < unpaired['interval_95_points'][1]


# --- calibration by simulation ----------------------------------------------------------

def simulate(delta, n_seeds, n_families, per_family, rng):
    """Paired a/b per seed; true population difference is exactly ``delta``."""
    groups = [f'f{i // per_family}' for i in range(n_families * per_family)]
    fam_a = {f'f{i}': rng.gauss(0, .06) for i in range(n_families)}
    fam_d = {f'f{i}': rng.gauss(0, .03) for i in range(n_families)}
    a, b = {}, {}
    for s in range(n_seeds):
        seed_a, seed_d = rng.gauss(0, .02), rng.gauss(0, .015)
        sa, sb = [], []
        for g in groups:
            x = .5 + seed_a + fam_a[g] + rng.gauss(0, .04)
            y = x + delta + seed_d + fam_d[g] + rng.gauss(0, .03)
            sa.append(min(1, max(0, x)))
            sb.append(min(1, max(0, y)))
        a[s], b[s] = from_scores(sa, groups), from_scores(sb, groups)
    return a, b


def test_interval_coverage_near_nominal_and_false_positive_rate():
    rng = random.Random(20261113)
    sims, covered, false_positive = 60, 0, 0
    for i in range(sims):
        a, b = simulate(.02, 6, 30, 3, rng)
        result = multiseed.hierarchical_bootstrap(a, b, samples=300, seed=i)
        lo, hi = result['interval_95_points']
        covered += lo <= 2.0 <= hi
        a0, b0 = simulate(0.0, 6, 30, 3, rng)
        false_positive += multiseed.hierarchical_bootstrap(a0, b0, samples=300, seed=i)['excludes_zero']
    # Few seeds make percentile intervals somewhat narrow; demand roughly nominal behaviour.
    assert covered / sims >= .82, covered
    assert false_positive / sims <= .15, false_positive


def test_large_effect_excludes_zero_and_null_effect_does_not():
    rng = random.Random(1)
    a, b = simulate(.15, 3, 40, 4, rng)
    big = multiseed.hierarchical_bootstrap(a, b, samples=500, seed=3)
    assert big['excludes_zero'] and big['interval_95_points'][0] > 10
    same = {s: v for s, v in a.items()}
    null = multiseed.hierarchical_bootstrap(a, same, samples=500, seed=3)
    assert null['estimate_points'] == 0 and null['interval_95_points'] == [0, 0] and not null['excludes_zero']


# --- aggregation over synthetic campaign directories ---------------------------------------

def predictions(rows):
    return {'token_f1': sum(r[2] for r in rows) / len(rows), 'rouge_l': sum(r[3] for r in rows) / len(rows),
            'selection_score': sum(r[2] for r in rows) / len(rows), 'score_name': 'token_f1',
            'answer_token_loss': 1.5, 'generation_limit_fraction': 0.0,
            'predictions': [{'key': k, 'group_id': g, 'score': f, 'rouge_l': r} for k, g, f, r in rows]}


def write_seed(root, seed, dev_rows, test_rows=None, dataset=None, finalized=(), started=False,
               candidates=('per-layer-01',), parameters=None):
    directory = root / f'seed-{seed}'
    directory.mkdir(parents=True)
    dataset = dataset or {'dev': {'count': len(dev_rows), 'sha256': 'dev'},
                          'test': {'count': len(test_rows or [1]), 'sha256': 'test'}}
    (directory / 'config.json').write_text(json.dumps({'seed': 1, 'recovery': {'seed': seed}}))
    (directory / 'state.json').write_text(json.dumps({'config_sha256': 'x', 'dataset': dataset, 'test_started': started}))
    records = []
    for i, name in enumerate(candidates):
        rows = [(k, g, min(1, f + .1 * (i + 1) + .01 * seed), r) for k, g, f, r in dev_rows]
        records.append({'id': name, 'status': 'complete', 'parameters': (parameters or {}).get(name, 80),
                        'actual_reduction': .2, 'checkpoint': str(directory / name), 'dev': predictions(rows)})
    selection = {'baseline': {'dev': predictions(dev_rows), 'parameters': 100}, 'candidates': records, 'tiers': []}
    (directory / 'selection.json').write_text(json.dumps(selection))
    (directory / 'selection-lock.json').write_text(json.dumps({'sha256': campaign._fingerprint(selection)}))
    if test_rows is not None:
        (directory / 'final-test').mkdir()
        for name in finalized:
            rows = test_rows if name == 'original' else [(k, g, min(1, f + .05), r) for k, g, f, r in test_rows]
            (directory / 'final-test' / f'{name}.json').write_text(json.dumps({'metrics': predictions(rows)}))
        if set(finalized) >= {'original', *candidates}:
            (directory / 'summary.json').write_text('{}')
    return directory


DEV = [(f'd{i}', f'fam{i % 4}', .1 * (i % 7), .05 * (i % 5)) for i in range(12)]
TEST = [(f't{i}', f'tf{i % 2}', .3, .3) for i in range(4)]


def test_aggregate_means_sd_and_pairing_from_saved_records(tmp_path):
    root = tmp_path / 'run'
    for seed in (3, 4, 5):
        write_seed(root, seed, DEV)
    summary = multiseed.aggregate_run(root, samples=100, seed=2)
    assert summary['seeds'] == [3, 4, 5]
    base_mean = 100 * sum(r[2] for r in DEV) / len(DEV)
    original = summary['variants']['original']['metrics']['token_f1']
    assert original['mean'] == pytest.approx(base_mean) and original['sd'] == pytest.approx(0)
    per_seed = [100 * sum(min(1, r[2] + .1 + .01 * s) for r in DEV) / len(DEV) for s in (3, 4, 5)]
    cand = summary['variants']['per-layer-01']
    assert cand['metrics']['token_f1']['mean'] == pytest.approx(statistics.fmean(per_seed))
    assert cand['metrics']['token_f1']['sd'] == pytest.approx(statistics.stdev(per_seed))
    assert cand['token_f1_interval']['estimate_points'] == pytest.approx(statistics.fmean(per_seed))
    (comparison,) = summary['comparisons']
    assert comparison['a'] == 'original' and comparison['b'] == 'per-layer-01'
    assert comparison['token_f1']['estimate_points'] == pytest.approx(statistics.fmean(per_seed) - base_mean)
    assert comparison['token_f1']['seed_pairing'] == 'paired'
    assert comparison['token_f1']['document_families'] == 4
    assert comparison['answer_token_loss']['mean_difference'] == pytest.approx(0)


def test_aggregate_rejects_incomplete_or_inconsistent_records(tmp_path):
    root = tmp_path / 'short'
    write_seed(root, 1, DEV)
    directory = write_seed(root, 2, DEV)
    selection = json.loads((directory / 'selection.json').read_text())
    selection['candidates'][0]['dev']['predictions'].pop()
    (directory / 'selection.json').write_text(json.dumps(selection))
    with pytest.raises(ValueError, match='Frozen development selection changed'):
        multiseed.aggregate_run(root, samples=10)
    (directory / 'selection-lock.json').write_text(json.dumps({'sha256': campaign._fingerprint(selection)}))
    with pytest.raises(ValueError, match='cover 11 of 12'):
        multiseed.aggregate_run(root, samples=10)

    other = tmp_path / 'datasets'
    write_seed(other, 1, DEV)
    write_seed(other, 2, DEV, dataset={'dev': {'count': 12, 'sha256': 'different'}, 'test': {'count': 1, 'sha256': 't'}})
    with pytest.raises(ValueError, match='datasets differ'):
        multiseed.collect_run(other)

    params = tmp_path / 'params'
    write_seed(params, 1, DEV)
    write_seed(params, 2, DEV, parameters={'per-layer-01': 81})
    with pytest.raises(ValueError, match='Parameter count'):
        multiseed.aggregate_run(params, samples=10)

    unlocked = tmp_path / 'unlocked'
    write_seed(unlocked, 1, DEV)
    (write_seed(unlocked, 2, DEV) / 'selection-lock.json').unlink()
    run = multiseed.collect_run(unlocked)
    assert sorted(run['seeds']) == [1] and run['incomplete'] == ['seed-2']


def test_compare_runs_requires_same_split_data_and_reports_difference(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    for seed in (1, 2):
        write_seed(a, seed, DEV)
    shifted = [(k, g, min(1, f + .2), r) for k, g, f, r in DEV]
    for seed in (8, 9):
        write_seed(b, seed, shifted)
    result = multiseed.compare_runs(a, b, samples=200, seed=1)
    original = next(c for c in result['comparisons'] if c['a'].endswith(':original'))
    expected = 100 * (sum(min(1, r[2] + .2) for r in DEV) - sum(r[2] for r in DEV)) / len(DEV)
    assert original['token_f1']['estimate_points'] == pytest.approx(expected)
    assert original['token_f1']['seed_pairing'] == 'independent' and original['token_f1']['excludes_zero']
    with pytest.raises(ValueError, match='shared recovery seeds'):
        multiseed.compare_runs(a, b, samples=20, paired_seeds=True)
    c = tmp_path / 'c'
    write_seed(c, 1, DEV, dataset={'dev': {'count': 12, 'sha256': 'other'}, 'test': {'count': 1, 'sha256': 'test'}})
    with pytest.raises(ValueError, match='different dev data'):
        multiseed.compare_runs(a, c, samples=20)
    with pytest.raises(ValueError, match='Unknown variant pair'):
        multiseed.compare_runs(a, b, pairs=[('original', 'nope')], samples=20)


# --- test-split sealing in aggregation ----------------------------------------------------

def test_test_split_aggregation_ignores_unopened_seeds_and_dev_never_reads_test(tmp_path):
    root = tmp_path / 'run'
    write_seed(root, 1, DEV, TEST, finalized=('original', 'per-layer-01'), started=True)
    write_seed(root, 2, DEV, TEST, finalized=(), started=False)
    run = multiseed.collect_run(root, 'test')
    assert sorted(run['seeds']) == [1] and run['incomplete'] == ['seed-2']
    assert all(len(v['examples']) == len(TEST) for v in run['seeds'][1].values())
    dev = multiseed.collect_run(root, 'dev')
    assert all(set(v['examples']) == {r[0] for r in DEV} for s in dev['seeds'].values() for v in s.values())
    with pytest.raises(ValueError, match='split must be'):
        multiseed.collect_run(root, 'recovery')


def test_partially_finalized_seed_is_not_aggregated_on_test(tmp_path):
    root = tmp_path / 'run'
    write_seed(root, 1, DEV, TEST, finalized=('original', 'per-layer-01'), started=True)
    write_seed(root, 2, DEV, TEST, finalized=('original',), started=True)  # interrupted finalize
    run = multiseed.collect_run(root, 'test')
    assert sorted(run['seeds']) == [1]


def test_three_seed_false_positive_rate_when_seed_variation_dominates():
    rng = random.Random(1)
    groups = [f'f{i // 3}' for i in range(90)]
    false_positive, sims = 0, 100
    for i in range(sims):
        family = {g: rng.gauss(0, .005) for g in set(groups)}
        a = {s: from_scores([.5] * 90, groups) for s in range(3)}
        b = {}
        for s in range(3):
            shift = rng.gauss(0, .03)
            b[s] = from_scores([.5 + shift + family[g] + rng.gauss(0, .01) for g in groups], groups)
        false_positive += multiseed.hierarchical_bootstrap(a, b, samples=300, seed=i)['decision'] == 'confident'
    assert false_positive / sims <= .10, false_positive
