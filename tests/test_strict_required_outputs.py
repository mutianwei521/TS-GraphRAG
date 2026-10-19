"""Audit must reject absent, empty, misbound, or incomplete experiment outputs."""
import contextlib
import csv
import gzip
import io
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiments.strict_engine.engine import PROTOCOLS, SEEDS, build_pools, rank_metrics
from experiments.strict_engine.run import grid, save_json, write_tables
from experiments.strict_engine.verify import main, compare_stream


def fixture(base, study='all', limit=1):
    data, out, frozen = base/'data', base/'out', base/'frozen'
    for p in [data, out, frozen]: p.mkdir()
    fps = []
    for part in range(2):
        for node in ['a','b']:
            for rate in [1,2]:
                for period in ['day','night']:
                    i = len(fps)
                    fps.append(dict(scenario_id=f'q{i}', leak_node=f'{part}{node}', leak_rate_Ls=rate,
                                    demand_period=period, leak_partition=part, sensor_fingerprint=[i+1., -(i+1.)/2],
                                    max_pressure_drop=1., num_affected_nodes=1))
    save_json(data/'sensor_fingerprints.json', dict(sensor_nodes=['a','b'], fingerprints=fps))
    docs = {f['scenario_id']:dict(f, summary_text=f['scenario_id']) for f in fps}
    vectors = {f['scenario_id']:[1.,2.] for f in fps}
    pools,mapping = build_pools(fps, fps, docs, vectors, {'partitions':{}}, [[0,1],[1,0]])
    with gzip.open(out/'retained_provenance.jsonl.gz','wt',encoding='utf-8') as f:
        for p in pools.values(): f.write(json.dumps(p.provenance())+'\n')
    save_json(out/'frozen_design.json', dict(study=study,seeds=SEEDS,n_queries_per_seed=limit,protocols=PROTOCOLS,
              sensitivity_protocols=['H0','H2','H4','H8'], sensitivity_grid={s:{} for s in grid()},legacy_attribution=False))
    primary, sensitivity = [], []
    for seed in SEEDS:
        for qi,q in enumerate(fps):
            for proto in PROTOCOLS:
                p = pools[mapping[qi,proto]]
                methods = ['fusion_strict','projection','cosine'] + (['fusion_open_topology'] if proto=='H9' else [])
                for method in methods:
                    rank = [0,1] if method=='fusion_open_topology' else p.labels.tolist(); m=rank_metrics(rank,q['leak_partition'])
                    primary.append(dict(seed=seed,query_index=qi,query_id=q['scenario_id'],truth=q['leak_partition'],protocol=proto,
                                        method=method,pool_id=p.key,n_retained=len(p.ids),ranking=rank,top1=m[0],top3=m[1],rr=m[2]))
                if proto in ['H0','H2','H4','H8']:
                    for setting in grid():
                        rank=p.labels.tolist();m=rank_metrics(rank,q['leak_partition'])
                        sensitivity.append(dict(seed=seed,query_id=q['scenario_id'],protocol=proto,setting=setting,
                                                pool_id=p.key,ranking=rank,top1=m[0],top3=m[1],rr=m[2]))
    for name, records, needed in [('predictions.jsonl.gz',primary,study in ['holdout','all']),
                                  ('sensitivity_predictions.jsonl.gz',sensitivity,study in ['sensitivity','all'])]:
        with gzip.open(frozen/name,'wt',encoding='utf-8') as f:
            for r in records: f.write(json.dumps(r)+'\n')
        if needed:
            selected = {f['scenario_id'] for f in fps[:limit]}
            rows = [r for r in records if r['query_id'] in selected]
            with gzip.open(out/name,'wt',encoding='utf-8') as f:
                for r in rows: f.write(json.dumps(r)+'\n')
            raw=defaultdict(list); group='method' if name=='predictions.jsonl.gz' else 'setting'
            for r in rows: raw[r['seed'],r['protocol'],r[group]].append([r['top1'],r['top3'],r['rr']])
            write_tables(raw,out,'holdout' if group=='method' else 'sensitivity',group,limit)
    return data,out,frozen


def rewrite(path, change):
    with gzip.open(path,'rt',encoding='utf-8') as f: rows=list(map(json.loads,f))
    change(rows)
    with gzip.open(path,'wt',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r)+'\n')


class StrictRequiredOutputTests(unittest.TestCase):
    def audit(self, data, out, frozen, compare=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return main(['--data-dir',str(data),'--output-dir',str(out),'--frozen-dir',str(frozen)]+(['--compare-frozen'] if compare else []))

    def test_all_requires_primary_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            data,out,frozen=fixture(Path(td));(out/'predictions.jsonl.gz').unlink()
            with self.assertRaisesRegex(AssertionError,'[Mm]issing|required'):
                self.audit(data,out,frozen)

    def test_all_requires_sensitivity_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            data,out,frozen=fixture(Path(td));(out/'sensitivity_predictions.jsonl.gz').unlink()
            with self.assertRaisesRegex(AssertionError,'[Mm]issing|required'):
                self.audit(data,out,frozen)

    def test_empty_primary_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            data,out,frozen=fixture(Path(td),'holdout')
            (out/'holdout_by_seed.csv').write_text('seed,protocol,method,n,top1_pct,top3_pct,mrr_pct\n',encoding='utf-8')
            with self.assertRaisesRegex(AssertionError,'grid|[Ee]mpty'):
                self.audit(data,out,frozen)

    def test_empty_sensitivity_predictions_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            data,out,frozen=fixture(Path(td),'sensitivity')
            with gzip.open(out/'sensitivity_predictions.jsonl.gz','wt',encoding='utf-8'):pass
            with self.assertRaisesRegex(AssertionError,'grid|[Ee]mpty'):
                self.audit(data,out,frozen)

    def test_wrong_query_protocol_pool_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            data,out,frozen=fixture(Path(td),'holdout')
            def poison(rows):
                h0=next(r for r in rows if r['protocol']=='H0')
                for r in rows:
                    if r['protocol']=='H8':r.update(pool_id=h0['pool_id'],n_retained=h0['n_retained'])
            rewrite(out/'predictions.jsonl.gz',poison)
            with self.assertRaisesRegex(AssertionError,'pool|retained'):
                self.audit(data,out,frozen)

    def test_wrong_sensitivity_pool_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            data,out,frozen=fixture(Path(td),'sensitivity')
            def poison(rows):
                h0=next(r for r in rows if r['protocol']=='H0')
                for r in rows:
                    if r['protocol']=='H8':r['pool_id']=h0['pool_id']
            rewrite(out/'sensitivity_predictions.jsonl.gz',poison)
            with self.assertRaisesRegex(AssertionError,'pool|retained'):
                self.audit(data,out,frozen)

    def test_wrong_query_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            data,out,frozen=fixture(Path(td),'holdout')
            rewrite(out/'predictions.jsonl.gz',lambda rows:rows[0].update(query_index=1))
            with self.assertRaisesRegex(AssertionError,'index|identity'):
                self.audit(data,out,frozen)

    def test_empty_comparison_never_passes(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'empty.gz'
            with gzip.open(p,'wt',encoding='utf-8'):pass
            with self.assertRaisesRegex(AssertionError,'[Ee]mpty|intersection|records'):
                compare_stream(p,p,{'missing'})

    def test_empty_primary_predictions_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            data,out,frozen=fixture(Path(td),'holdout')
            with gzip.open(out/'predictions.jsonl.gz','wt',encoding='utf-8'):pass
            with self.assertRaisesRegex(AssertionError,'grid|[Ee]mpty'):
                self.audit(data,out,frozen)

    def test_empty_summary_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            data,out,frozen=fixture(Path(td),'holdout')
            (out/'holdout_summary.csv').write_text('protocol,method\n',encoding='utf-8')
            with self.assertRaisesRegex(AssertionError,'grid|[Ee]mpty'):
                self.audit(data,out,frozen)

    def test_zero_query_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            data,out,frozen=fixture(Path(td),'all')
            design=json.loads((out/'frozen_design.json').read_text());design['n_queries_per_seed']=0
            save_json(out/'frozen_design.json',design)
            with self.assertRaisesRegex(AssertionError,'query grid'):
                self.audit(data,out,frozen)

    def test_wrong_seed_cannot_replace_a_complete_cell(self):
        with tempfile.TemporaryDirectory() as td:
            data,out,frozen=fixture(Path(td),'holdout')
            rewrite(out/'predictions.jsonl.gz',lambda rows:rows[0].update(seed=42))
            with self.assertRaisesRegex(AssertionError,'identity grid'):
                self.audit(data,out,frozen)

    def test_archive_without_selected_queries_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            actual,frozen=Path(td)/'actual.gz',Path(td)/'frozen.gz'
            for path,query in [(actual,'selected'),(frozen,'different')]:
                with gzip.open(path,'wt',encoding='utf-8') as f:f.write(json.dumps({'query_id':query})+'\n')
            with self.assertRaisesRegex(AssertionError,'intersection'):
                compare_stream(actual,frozen,{'selected'})

    def test_limited_complete_each_study_passes_without_other_study_files(self):
        for study in ['holdout','sensitivity','all']:
            with self.subTest(study=study),tempfile.TemporaryDirectory() as td:
                data,out,frozen=fixture(Path(td),study,2)
                report=self.audit(data,out,frozen,compare=True)
                self.assertTrue(report['all_output_metrics_recomputed_successfully'])
                self.assertTrue(report['archive_parity'])


if __name__ == '__main__':unittest.main()
