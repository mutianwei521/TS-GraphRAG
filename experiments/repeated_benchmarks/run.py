"""Recalculate E4 statistics or refit its classical models from explicit local inputs."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import classical
from .primitives import projection_residual

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'data/frozen/repeated_benchmarks'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def summarize(rows, seeds=classical.SEEDS):
    """Compute complete-run Top-1 rates and their sample SD (ddof=1)."""
    cells = defaultdict(lambda: defaultdict(list))
    keys = set()
    for row in rows:
        seed, qi = int(row['seed']), int(row['query_index'])
        key = seed, row['protocol'], row['method'], qi
        if key in keys:
            raise ValueError(f'Duplicate prediction: {key}')
        keys.add(key)
        predicted = row['predicted_partition']
        if predicted is None or str(predicted) in ('', 'None'):
            raise ValueError(f'Prediction has a fitting failure: {key}')
        hit = int(int(predicted) == int(row['true_partition']))
        if hit != int(row['correct']):
            raise AssertionError(f'Saved correctness differs from predicted label: {key}')
        cells[row['protocol'], row['method']][seed].append(hit)
    summaries = []
    for (proto, method), by_seed in sorted(cells.items()):
        if set(by_seed) != set(seeds):
            raise ValueError(f'Incomplete repetition seeds: {proto}/{method}')
        counts = [sum(by_seed[s]) for s in seeds]
        ns = [len(by_seed[s]) for s in seeds]
        if len(set(ns)) != 1:
            raise ValueError(f'Inconsistent query counts: {proto}/{method}')
        rates = 100 * np.asarray(counts) / ns[0]
        summaries.append(dict(protocol=proto, method=method, counts=counts,
                              rates_percent=rates.tolist(), mean_percent=float(rates.mean()),
                              sample_std_percent=float(rates.std(ddof=1)),
                              n_queries_per_repeat=ns[0], n_repeats=len(seeds), sd_ddof=1))
    return summaries


def replay(scope, output_dir):
    """Recalculate all archived prediction metrics and the exact E4 projection labels."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(FROZEN/f'classical_{scope}_predictions.csv.gz', 'rt', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    stats = summarize(rows)
    old = json.loads((FROZEN/f'classical_{scope}_results.json').read_text(encoding='utf-8'))
    expected = {(r['protocol'], r['method']): r for r in old['summary']}
    if {(r['protocol'],r['method']) for r in stats} != set(expected):
        raise AssertionError('The E4 predictions do not cover every expected protocol/method cell')
    for s in stats:
        archive = expected[s['protocol'], s['method']]
        assert s['n_queries_per_repeat']==archive['n_queries_per_repeat']
        for key in ('mean_percent', 'sample_std_percent', 'rates_percent'):
            if not np.allclose(s[key], archive[key], atol=1e-12, rtol=0):
                raise AssertionError(f'Archived E4 statistic differs: {s["protocol"]}/{s["method"]}/{key}')
    fps, X, y, qidx = classical.load_data()
    qis = qidx if scope == 'main90' else list(range(len(fps)))
    protos = classical.MAIN_PROTOCOLS if scope == 'main90' else classical.ALL_PROTOCOLS
    projection_rows = {(int(r['seed']),r['protocol'],int(r['query_index'])):r
                       for r in rows if r['method']=='projection'}
    mismatches = 0
    for seed in classical.SEEDS:
        for proto in protos:
            for qi in qis:
                idx = classical.retain(qi,proto)
                prediction = int(y[idx[np.argmin(projection_residual(classical.query_value(seed,qi), X[idx]))]])
                mismatches += prediction != int(projection_rows[seed,proto,qi]['predicted_partition'])
    if mismatches:
        raise AssertionError(f'{mismatches} E4 projection predictions differ from the archive')
    gnn_stats = []
    if scope == 'main90':
        gnn = json.loads((FROZEN/'gnn_repeats_results.json').read_text(encoding='utf-8'))
        gnn_rows = [dict(seed=r['run_seed'],query_index=r['corpus_index'],protocol=r['protocol'],
                         method=r['method'],true_partition=r['true_label'],
                         predicted_partition=r['predicted_label'],correct=r['correct']) for r in gnn['predictions']]
        gnn_stats = summarize(gnn_rows)
        for s in gnn_stats:
            archive=gnn['summary'][s['protocol']][s['method']]
            assert np.allclose(s['rates_percent'],archive['top1_by_seed'],atol=1e-12,rtol=0)
            assert abs(s['sample_std_percent']-archive['sample_sd_percent'])<1e-12
    write_json(output_dir/f'classical_{scope}_recalculated.json',stats)
    if gnn_stats:
        write_json(output_dir/'gnn_recalculated.json',gnn_stats)
    with (output_dir/f'Table_S3_{scope}.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=['protocol','method','mean_percent','sample_std_percent','n_queries_per_repeat','n_repeats','sd_ddof'])
        writer.writeheader()
        writer.writerows({key:r[key] for key in writer.fieldnames} for r in stats+gnn_stats)
    report=dict(study='repeated_benchmarks',profile='replay',scope=scope,prediction_rows=len(rows),
                summary_cells=len(stats),gnn_summary_cells=len(gnn_stats),
                projection_queries_verified=len(projection_rows),projection_prediction_mismatches=mismatches,
                archive_statistics_match=True,noise_rule='RandomState(run_seed + original_query_index).normal(0, 0.02, 30)',
                seeds=classical.SEEDS,uncertainty='sample SD ddof=1 across five complete stochastic repetitions')
    write_json(output_dir/'replay_report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['replay','full'],default='replay')
    parser.add_argument('--scope',choices=['main90','all540'],default='main90')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--include-gnn',action='store_true')
    parser.add_argument('--max-pools',type=int,help='Limit classical fits and label every output as smoke; no complete-run claims')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/repeated_benchmarks')
    args=parser.parse_args()
    if args.workers<1: parser.error('--workers must be positive')
    if args.max_pools is not None and args.max_pools<1: parser.error('--max-pools must be positive')
    if args.max_pools is not None and (args.profile!='full' or args.include_gnn):
        parser.error('--max-pools is a classical fitting smoke mode; use --profile full without --include-gnn')
    if args.profile=='replay':
        report=replay(args.scope,args.output_dir)
    else:
        if args.include_gnn and args.scope!='main90':
            parser.error('GNN training uses the fixed main90 protocol')
        if args.include_gnn:
            import importlib.util
            if any(importlib.util.find_spec(name) is None for name in ['torch','torch_geometric']):
                parser.error('--include-gnn requires the optional torch/torch-geometric environment')
        smoke=classical.run(args.scope,args.workers,args.output_dir,args.max_pools)
        if args.include_gnn:
            from . import gnn
            gnn.run(args.output_dir)
        report=smoke if args.max_pools is not None else dict(study='repeated_benchmarks',profile='full',scope=args.scope,completed=True)
    print(json.dumps({k:v for k,v in report.items() if k!='metadata'},indent=2))


if __name__=='__main__':
    main()
