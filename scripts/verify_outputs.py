"""Check frozen input integrity and compare regenerated scientific outputs."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def verify_assets(root=ROOT, manifest=None):
    root=Path(root).resolve()
    manifest=Path(manifest) if manifest else root/'reproducibility/assets.json'
    metadata=json.loads(manifest.read_text(encoding='utf-8'))
    entries=metadata['assets']
    if 'network_sha256' in metadata:
        network=root/'data/Exa7.inp'
        if not network.is_file() or hashlib.sha256(network.read_bytes()).hexdigest()!=metadata['network_sha256']:
            raise AssertionError('Network asset integrity mismatch: data/Exa7.inp')
    for row in entries:
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root):
            raise AssertionError(f'Asset path leaves repository: {row["path"]}')
        if not path.is_file():
            raise AssertionError(f'Missing asset: {row["path"]}')
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=row['sha256'] or path.stat().st_size!=row['bytes']:
            raise AssertionError(f'Asset integrity mismatch: {row["path"]}')
    return {'checked':len(entries),'status':'passed'}


def _number(value):
    if value is None or value=='':return None
    try:return float(value)
    except (ValueError,TypeError):return str(value)


def compare_csv(actual, expected, keys, atol=1e-8):
    def read(path):
        opener=gzip.open if Path(path).suffix=='.gz' else open
        with opener(path,'rt',newline='',encoding='utf-8-sig') as stream:
            reader=csv.DictReader(stream);columns=reader.fieldnames;rows=list(reader)
        result={}
        for row in rows:
            key=tuple(_number(row[k]) for k in keys)
            if key in result:raise AssertionError(f'Duplicate output key {key}: {path}')
            result[key]=row
        return columns,result
    ac,a=read(actual);ec,e=read(expected)
    if set(ac)!=set(ec):raise AssertionError(f'CSV columns: {set(ac)^set(ec)}')
    if set(a)!=set(e):raise AssertionError(f'CSV key mismatch: {len(a)} vs {len(e)}')
    count=0
    for key,row in a.items():
        for column in ec:
            av,ev=_number(row[column]),_number(e[key][column])
            if isinstance(av,float) and isinstance(ev,float):
                equal=(math.isnan(av) and math.isnan(ev)) or av==ev or math.isclose(av,ev,rel_tol=0,abs_tol=atol)
            else:equal=av==ev
            if not equal:raise AssertionError(f'{Path(actual).name}/{key}/{column}: {av} != {ev}')
            count+=1
    return {'status':'passed','rows':len(a),'cells':count,'absolute_tolerance':atol}


def compare_predictions(actual,expected):
    def read(path):
        opener=gzip.open if Path(path).suffix=='.gz' else open
        result={}
        with opener(path,'rt',newline='',encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                key=tuple(row[name] for name in ['seed','protocol','query_index','method'])
                if key in result:raise AssertionError(f'Duplicate prediction: {key}')
                truth=int(row['true_partition']);prediction=int(row['predicted_partition']);correct=int(row['correct'])
                if correct!=int(truth==prediction):raise AssertionError(f'Invalid prediction correctness: {key}')
                result[key]=(truth,prediction,correct)
        if not result:raise AssertionError(f'Empty predictions: {path}')
        return result
    a,e=read(actual),read(expected)
    if set(a)!=set(e):raise AssertionError(f'Incomplete prediction grid: {len(a)} vs {len(e)}')
    mismatches=sum(a[key]!=e[key] for key in a)
    if mismatches:raise AssertionError(f'{mismatches} query-level prediction mismatches')
    return {'status':'passed','predictions':len(a),'mismatches':mismatches}


def verify_run(run_dir):
    run_dir=Path(run_dir).resolve()
    report_path=run_dir/'run_report.json'
    if not report_path.exists():raise FileNotFoundError(f'Run report: {report_path}')
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if report['status']!='passed' or not report['steps']:
        raise AssertionError(f'Incomplete run: {report_path}')
    for step in report['steps']:
        if step['exit_code']!=0:raise AssertionError(f'Failed step: {step["name"]}')
        for artifact in step.get('outputs',[]):
            path=run_dir/artifact['path']
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=artifact['sha256']:
                raise AssertionError(f'Run output integrity mismatch: {path}')
    for artifact in report.get('result_files',[]):
        path=(run_dir/artifact['path']).resolve()
        if not path.is_relative_to(run_dir) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=artifact['sha256']:
            raise AssertionError(f'Scientific output integrity mismatch: {path}')
    return {'status':'passed','profile':report['profile'],'study':report['study'],
            'verified_steps':len(report['steps']),'assets':verify_assets()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--assets-only',action='store_true')
    parser.add_argument('--classical-refit',choices=['main90','all540'])
    args=parser.parse_args()
    if not args.run_dir and not args.assets_only:parser.error('Provide --run-dir or --assets-only')
    if args.classical_refit:
        if not args.run_dir:parser.error('--classical-refit requires --run-dir')
        result=compare_predictions(args.run_dir/f'classical_{args.classical_refit}_predictions.csv',
             ROOT/f'data/frozen/repeated_benchmarks/classical_{args.classical_refit}_predictions.csv.gz')
        (args.run_dir/'archive_prediction_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    else:result=verify_assets() if args.assets_only else verify_run(args.run_dir)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
