"""Independently verify complete strict outputs, retained-pool identity and archives."""
from __future__ import annotations
if not __debug__:
    raise RuntimeError("Strict audit cannot run in optimized Python (-O or PYTHONOPTIMIZE); validation assertions must remain enabled.")
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from .engine import DEFAULT_DATA_DIR, DEFAULT_FROZEN_DIR, PROTOCOLS, SEEDS, digest, keep, rank_metrics
from .run import grid, save_json


SENSITIVITY_PROTOCOLS = ['H0', 'H2', 'H4', 'H8']
PRIMARY_METHODS = ['fusion_strict', 'projection', 'cosine']
LEGACY_METHODS = ['fusion_legacy_unfiltered', 'fusion_filtered_open_topology']
METRIC_NAMES = ['top1_pct', 'top3_pct', 'mrr_pct']


def verify_metric_record(r):
    assert len(r['ranking']) == len(set(r['ranking'])), 'Duplicate label in returned ranking.'
    assert rank_metrics(r['ranking'], r['truth']) == [r['top1'], r['top3'], r['rr']], 'Stored metric differs from returned ranking.'
    if 'rr_top10' in r:
        assert rank_metrics(r['ranking'][:10], r['truth'])[2] == r['rr_top10'], 'Stored top-10 metric differs from ranking.'


def verify_prediction_grid(counts, n_seeds, n_queries):
    assert n_seeds > 0 and n_queries > 0, 'Empty prediction grid.'
    n = n_seeds*n_queries
    expected = dict(fusion_strict=10*n, projection=10*n, cosine=10*n, fusion_open_topology=n)
    assert dict(counts) == expected, f'Incomplete prediction grid: {dict(counts)} != {expected}'


def compare_stream(actual, frozen, selected_ids):
    """Compare nonempty original-order records, filtering the archive to this run's queries."""
    assert selected_ids, 'Empty selected query intersection for archive comparison.'
    assert Path(actual).is_file() and Path(frozen).is_file(), 'Missing required archive comparison file.'
    count = expected_count = mismatches = 0
    with gzip.open(actual, 'rt', encoding='utf-8') as fa, gzip.open(frozen, 'rt', encoding='utf-8') as ff:
        expected = (r for r in map(json.loads, ff) if r['query_id'] in selected_ids)
        for r in map(json.loads, fa):
            e = next(expected, None); count += 1; expected_count += e is not None
            if r != e: mismatches += 1
        extra = next(expected, None)
        if extra is not None: mismatches += 1; expected_count += 1
    assert count > 0, 'Empty actual records in archive comparison.'
    assert expected_count > 0, 'No selected-query intersection with archived records.'
    return dict(records=count, mismatches=mismatches, exact_records_equal=mismatches == 0)


def required_outputs(study, legacy):
    assert study in ['holdout', 'sensitivity', 'all'], 'Unknown requested study in frozen_design.json.'
    stems = (['holdout'] if study == 'holdout' else ['sensitivity'] if study == 'sensitivity' else ['holdout', 'sensitivity'])
    if legacy: stems.append('legacy_attribution')
    predictions = {'holdout': 'predictions.jsonl.gz', 'sensitivity': 'sensitivity_predictions.jsonl.gz',
                   'legacy_attribution': 'legacy_attribution_predictions.jsonl.gz'}
    return stems, predictions, ['retained_provenance.jsonl.gz'] + [name for stem in stems for name in
           [predictions[stem], stem+'_by_seed.csv', stem+'_summary.csv']]


def verify_cells(raw, expected_cells, n_queries, label):
    assert raw, f'Empty {label} grid.'
    assert set(raw) == expected_cells, f'Incomplete {label} seed/protocol/method grid.'
    assert {len(v) for v in raw.values()} == {n_queries}, f'Incomplete {label} query grid.'


def verify_tables(out, stem, raw, groupname, n_queries):
    """Recalculate both aggregate CSVs from independently validated query records."""
    cell_metrics = {key: 100*np.asarray(vals).mean(axis=0) for key, vals in raw.items()}
    with (out/(stem+'_by_seed.csv')).open(encoding='utf-8') as f: rows = list(csv.DictReader(f))
    assert rows, f'Empty {stem} by-seed CSV grid.'
    seen = set()
    for r in rows:
        key = int(r['seed']), r['protocol'], r[groupname]
        assert key in raw and key not in seen, f'Invalid or duplicate {stem} CSV grid cell.'
        seen.add(key)
        assert int(r['n']) == n_queries, f'Wrong {stem} CSV query count.'
        assert np.allclose(cell_metrics[key], [float(r[k]) for k in METRIC_NAMES], rtol=0, atol=1e-10), f'{stem} CSV metrics differ.'
    assert seen == set(raw), f'Incomplete {stem} by-seed CSV grid.'
    with (out/(stem+'_summary.csv')).open(encoding='utf-8') as f: rows = list(csv.DictReader(f))
    assert rows, f'Empty {stem} summary CSV grid.'
    groups = {(proto, method) for _, proto, method in raw}; seen = set()
    for r in rows:
        key = r['protocol'], r[groupname]
        assert key in groups and key not in seen, f'Invalid or duplicate {stem} summary grid cell.'
        seen.add(key)
        vals = np.array([m for (_, proto, method), m in cell_metrics.items() if (proto, method) == key])
        assert int(r['n_noise_seeds']) == len(vals) and int(r['n_queries_each']) == n_queries, f'Wrong {stem} summary grid dimensions.'
        for i, metric in enumerate(METRIC_NAMES):
            assert np.isclose(float(r[metric+'_mean']), vals[:,i].mean(), rtol=0, atol=1e-10), f'{stem} summary mean differs.'
            assert np.isclose(float(r[metric+'_sd']), vals[:,i].std(ddof=1), rtol=0, atol=1e-10), f'{stem} summary deviation differs.'
    assert seen == groups, f'Incomplete {stem} summary CSV grid.'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=Path('outputs/strict_engine'))
    p.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument('--frozen-dir', type=Path, default=DEFAULT_FROZEN_DIR)
    p.add_argument('--compare-frozen', action='store_true')
    args = p.parse_args(argv); out = args.output_dir
    design = json.loads((out/'frozen_design.json').read_text(encoding='utf-8'))
    stems, prediction_names, required = required_outputs(design['study'], bool(design.get('legacy_attribution', False)))
    for name in required: assert (out/name).is_file(), f'Missing required {design["study"]} output: {name}'
    fps = json.loads((args.data_dir/'sensor_fingerprints.json').read_text(encoding='utf-8'))['fingerprints']
    assert fps, 'Empty reference scenario grid.'
    n_queries = design['n_queries_per_seed']
    assert type(n_queries) is int and 0 < n_queries <= len(fps), 'Invalid or empty requested query grid.'
    assert design['seeds'] == SEEDS, 'Incompatible fixed noise-seed grid.'
    assert design['protocols'] == PROTOCOLS, 'Incompatible protocol grid.'
    if 'sensitivity' in stems:
        assert design['sensitivity_protocols'] == SENSITIVITY_PROTOCOLS, 'Incompatible sensitivity protocol grid.'
        assert set(design['sensitivity_grid']) == set(grid()), 'Incompatible 34-setting sensitivity grid.'
    byid = {f['scenario_id']: f for f in fps}; index_by_id = {f['scenario_id']: i for i, f in enumerate(fps)}
    assert len(byid) == len(fps), 'Duplicate reference scenario identity.'
    selected = fps[:n_queries]; selected_ids = {f['scenario_id'] for f in selected}
    with gzip.open(out/'retained_provenance.jsonl.gz', 'rt', encoding='utf-8') as f: provenance = list(map(json.loads, f))
    prov = {r['pool_id']: r for r in provenance}
    assert prov and len(prov) == len(provenance), 'Empty or duplicate retained provenance grid.'
    expected_pool = {}; pool_checks = 0
    for q in selected:
        for protocol in PROTOCOLS:
            ids = [f['scenario_id'] for f in fps if keep(f, q, protocol)]; key = digest(ids)
            assert key in prov, 'Missing expected retained pool.'
            meta = prov[key]; expected_pool[q['scenario_id'], protocol] = key
            assert meta['retained_ids'] == ids == meta['semantic_document_ids'] == meta['centroid_source_ids'], 'Wrong retained source IDs.'
            assert sorted(s for ss in meta['community_source_ids'].values() for s in ss) == sorted(ids), 'Wrong retained community source IDs.'
            assert meta['retained_labels'] == sorted({byid[s]['leak_partition'] for s in ids}), 'Wrong retained label domain.'
            pool_checks += 1
    norms = {s: np.linalg.norm(f['sensor_fingerprint']) for s, f in byid.items()}
    eligible = {(key, method): {byid[s]['leak_partition'] for s in meta['retained_ids'] if norms[s] > threshold}
                for key, meta in prov.items() for method, threshold in [('projection', 0), ('cosine', 1e-12)]}
    physical_labels = {f['leak_partition'] for f in fps}
    counts = Counter(); h9 = Counter(); sensitivity_cells = 0; sensitivity_records = 0
    validated_counts = {}
    for stem in stems:
        raw = defaultdict(list); seen = set()
        with gzip.open(out/prediction_names[stem], 'rt', encoding='utf-8') as f:
            for r in map(json.loads, f):
                assert r['seed'] in SEEDS and r['query_id'] in selected_ids and r['protocol'] in PROTOCOLS, 'Invalid seed/query/protocol identity grid.'
                if 'query_index' in r or stem == 'holdout':
                    assert r.get('query_index') == index_by_id[r['query_id']], 'Wrong query_index identity.'
                assert r['pool_id'] == expected_pool[r['query_id'], r['protocol']], 'Prediction pool differs from expected retained query/protocol pool.'
                if stem == 'sensitivity':
                    group = r['setting']; groupname = 'setting'
                    assert r['protocol'] in SENSITIVITY_PROTOCOLS and group in design['sensitivity_grid'], 'Invalid sensitivity setting/protocol grid.'
                    if 'truth' in r: assert r['truth'] == byid[r['query_id']]['leak_partition'], 'Wrong sensitivity truth identity.'
                    r = dict(r, truth=byid[r['query_id']]['leak_partition'])
                else:
                    group = r['method']; groupname = 'method'
                    methods = LEGACY_METHODS if stem == 'legacy_attribution' else PRIMARY_METHODS + (['fusion_open_topology'] if r['protocol'] == 'H9' else [])
                    assert group in methods, 'Invalid method/protocol grid.'
                    assert r['truth'] == byid[r['query_id']]['leak_partition'], 'Wrong truth identity.'
                key = r['seed'], r['query_id'], r['protocol'], group
                assert key not in seen, f'Duplicate {stem} query grid record.'
                seen.add(key); verify_metric_record(r); meta = prov[r['pool_id']]
                if stem == 'holdout':
                    assert r['n_retained'] == len(meta['retained_ids']), 'Wrong retained scenario count.'
                    if group != 'fusion_open_topology':
                        assert set(r['ranking']).issubset(meta['retained_labels']), 'Ranking outside retained label domain.'
                        if group in ['projection', 'cosine']: assert set(r['ranking']) == eligible[r['pool_id'], group], 'Wrong retained numeric eligibility.'
                        if r['protocol'] == 'H9':
                            assert [r['top1'],r['top3'],r['rr']] == [0,0,0.], 'H9 retained-label integrity failure.'
                            h9[group] += 1
                    else: assert set(r['ranking']).issubset(physical_labels), 'Invalid physical label.'
                    counts[group] += 1
                elif stem == 'sensitivity':
                    assert set(r['ranking']).issubset(meta['retained_labels']), 'Sensitivity ranking outside retained label domain.'
                else: assert set(r['ranking']).issubset(physical_labels), 'Invalid attribution label.'
                raw[r['seed'], r['protocol'], group].append([r['top1'], r['top3'], r['rr']])
        if stem == 'holdout':
            verify_prediction_grid(counts, len(SEEDS), n_queries)
            expected_cells = {(s, proto, method) for s in SEEDS for proto in PROTOCOLS for method in
                              PRIMARY_METHODS + (['fusion_open_topology'] if proto == 'H9' else [])}
        elif stem == 'sensitivity':
            expected_cells = {(s, proto, setting) for s in SEEDS for proto in SENSITIVITY_PROTOCOLS for setting in design['sensitivity_grid']}
            sensitivity_cells = len(raw); sensitivity_records = len(seen)
        else: expected_cells = {(s, proto, method) for s in SEEDS for proto in PROTOCOLS for method in LEGACY_METHODS}
        verify_cells(raw, expected_cells, n_queries, stem)
        verify_tables(out, stem, raw, 'setting' if stem == 'sensitivity' else 'method', n_queries)
        validated_counts[stem] = len(seen)
    assert validated_counts and all(n > 0 for n in validated_counts.values()), 'No nonempty requested output grid was verified.'
    report = dict(study=design['study'], n_queries_per_seed=n_queries, all_query_protocol_retention_checks=pool_checks,
                  unique_retained_pools=len(prov), prediction_counts=dict(counts), H9_zero_checks_by_method=dict(h9),
                  sensitivity_setting_seed_protocol_cells=sensitivity_cells, sensitivity_total_predictions=sensitivity_records,
                  validated_record_counts=validated_counts, all_output_metrics_recomputed_successfully=True,
                  numeric_baseline_ranks_exclude_all_ineligible_templates=True if 'holdout' in stems else None)
    if args.compare_frozen:
        parity = {prediction_names[stem]: compare_stream(out/prediction_names[stem], args.frozen_dir/prediction_names[stem], selected_ids) for stem in stems}
        assert parity and all(r['records'] > 0 and r['exact_records_equal'] for r in parity.values()), parity
        report['archive_parity'] = parity
    save_json(out/'independent_output_audit.json', report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__': main()
