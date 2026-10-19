"""Recalculate hydraulic-study accuracy and uncertainty from fixed query outcomes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/hydraulic_replay')
    args=parser.parse_args();output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    frozen=ROOT/'data/frozen/hydraulic_robustness'
    query=pd.read_csv(frozen/'query_metrics.csv.gz')
    keys=['seed','scenario','family','strength','baseline','library_policy','method']
    if len(query)!=26*5*60*2*3*3:raise AssertionError('Hydraulic query grid is incomplete')
    if query.duplicated(keys+['query_id']).any():raise AssertionError('Duplicate hydraulic query records')
    np.testing.assert_array_equal(query.top1_raw.to_numpy(),(query.pred_partition==query.partition).astype(int).to_numpy())
    np.testing.assert_array_equal(query.top1_all.to_numpy(),(query.top1_raw.astype(bool)&query.admissible).astype(int).to_numpy())
    runs=[]
    for key,group in query.groupby(keys,sort=False):
        if len(group)!=60:raise AssertionError(f'Incomplete hydraulic cell: {key}')
        valid=group[group.admissible]
        runs.append(dict(zip(keys,key))|{'n_queries':len(group),'n_admissible':len(valid),
            'top1_all_pct':100*group.top1_all.mean(),'top3_all_pct':100*group.top3_all.mean(),'mrr_all':group.mrr_all.mean(),
            'top1_admissible_pct':100*valid.top1_raw.mean() if len(valid) else np.nan,
            'top3_admissible_pct':100*valid.top3_raw.mean() if len(valid) else np.nan,
            'median_coordinate_distance':valid.coordinate_distance.median() if len(valid) else np.nan,
            'failure_pct':100*(~group.admissible).mean(),'negative_pressure_pct':100*group.negative_pressure.mean(),
            'baseline_drift_norm_m':group.baseline_drift_norm_m.mean()})
    runs=pd.DataFrame(runs);expected=pd.read_csv(frozen/'seed_summary.csv')
    metrics=['n_queries','n_admissible','top1_all_pct','top3_all_pct','mrr_all','top1_admissible_pct',
             'top3_admissible_pct','median_coordinate_distance','failure_pct','negative_pressure_pct','baseline_drift_norm_m']
    joined=runs.merge(expected,on=keys,suffixes=('','_expected'),validate='one_to_one')
    if len(joined)!=len(expected):raise AssertionError('Hydraulic aggregate grid mismatch')
    for metric in metrics:np.testing.assert_allclose(joined[metric],joined[metric+'_expected'],atol=1e-8,rtol=0,equal_nan=True)
    output_rows=[]
    for key,group in runs.groupby(keys[1:],sort=False):
        if len(group)!=5:raise AssertionError(f'Expected five hydraulic realizations: {key}')
        row=dict(zip(keys[1:],key));row['n_realizations']=5
        for metric in metrics[2:]:
            row[metric+'_mean']=group[metric].mean();row[metric+'_sd']=group[metric].std(ddof=1)
        output_rows.append(row)
    summary=pd.DataFrame(output_rows);expected_summary=pd.read_csv(frozen/'summary.csv')
    joined_summary=summary.merge(expected_summary,on=keys[1:],suffixes=('','_expected'),validate='one_to_one')
    checked=0
    for name in summary.columns:
        if name in keys[1:]:continue
        np.testing.assert_allclose(joined_summary[name],joined_summary[name+'_expected'],atol=1e-8,rtol=0,equal_nan=True);checked+=len(summary)
    runs.to_csv(output/'seed_summary_recomputed.csv',index=False)
    summary.to_csv(output/'summary_recomputed.csv',index=False)
    report={'status':'passed','profile':'archived query-outcome aggregation','query_records':len(query),
            'seed_cells':len(runs),'aggregate_cells':len(summary),'checked_summary_values':checked,
            'all_query_denominators':60,'realizations_per_condition':5,'absolute_tolerance':1e-8}
    (output/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
