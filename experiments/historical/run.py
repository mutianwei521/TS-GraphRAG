"""Recalculate historical physics tables and archive-backed result statistics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[2]
FROZEN=ROOT/'data/frozen/historical'
PAIRS=[('H0','projection','Bayes-marginal'),('H2','projection','Bayes-marginal'),
       ('H2','projection','Mahalanobis'),('H4','projection','Bayes-marginal'),
       ('H6','projection','Mahalanobis'),('H8','projection','Mahalanobis'),
       *[(p,'graph-reg 1-NN','cosine 1-NN') for p in ['H0','H2','H4','H6','H8']]]
DISPLAY={'projection':'Projection residual','cosine 1-NN':'Cosine 1-NN','WLS/NNLS':'WLS/NNLS',
         'Mahalanobis':'Mahalanobis','Bayes-marginal':'Bayes-marginal','KDE':'KDE',
         'graph-reg 1-NN':'Graph-regularized 1-NN','label-prop':'Label propagation'}


def csv_out(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def replay(output_dir):
    output_dir=Path(output_dir).resolve();output_dir.mkdir(parents=True,exist_ok=True)
    raw=json.loads((FROZEN/'expH_physics_baselines_results.json').read_text(encoding='utf-8'))
    summaries=[];s5=[]
    for p,by_method in raw['raw'].items():
        row={'Protocol':p}
        for method,hits in by_method.items():
            assert len(hits)==540 and set(hits)<={0,1}
            rate=100*float(np.mean(hits))
            assert abs(rate-raw['summary'][p][method])<1e-12
            summaries.append(dict(protocol=p,method=method,n_queries=len(hits),correct=sum(hits),accuracy_percent=rate))
            row[DISPLAY[method]]=rate
        s5.append(row)
    comparisons=[]
    for p,a,b in PAIRS:
        x=np.asarray(raw['raw'][p][a]);y=np.asarray(raw['raw'][p][b])
        n10=int(((x==1)&(y==0)).sum());n01=int(((x==0)&(y==1)).sum())
        pv=float(binomtest(n10,n10+n01,.5).pvalue) if n10+n01 else 1.
        comparisons.append(dict(Protocol=p,Comparison=DISPLAY[a]+' vs '+DISPLAY[b],
                                first_percent=100*float(x.mean()),second_percent=100*float(y.mean()),n10=n10,n01=n01,
                                exact_p=pv,holm_adjusted_p=None))
    running=0.
    for i,r in enumerate(sorted(comparisons,key=lambda r:r['exact_p'])):
        running=min(1.,max(running,(len(comparisons)-i)*r['exact_p']));r['holm_adjusted_p']=running
    csv_out(output_dir/'Table_S5_physics_baselines.csv',s5)
    csv_out(output_dir/'Table_S6_paired_tests.csv',comparisons)
    llm=json.loads((FROZEN/'expH_llm_results.json').read_text(encoding='utf-8'))
    llm_rows=[]
    for name,r in llm['models'].items():
        recovery=r['recovered']/r['n_errors'];breakage=r['broken']/r['n_controls']
        net=100*(recovery*(1-llm['base_top1'])-breakage*llm['base_top1'])
        assert np.allclose([recovery,breakage,net],[r['recovery_rate'],r['breakage_rate'],r['net_pp']],atol=1e-12,rtol=0)
        llm_rows.append(dict(model=name,recovery_rate=recovery,breakage_rate=breakage,net_percentage_points=net,
                             malformed=r['malformed'],source='archived model-service output counts'))
    csv_out(output_dir/'llm_archived_statistics.csv',llm_rows)
    direction=json.loads((FROZEN/'exp3_results.json').read_text(encoding='utf-8'))
    direction_checks=0
    for key,r in direction['q2'].items():
        if 'ok' in r:
            assert abs(float(np.mean(r['ok']))-r['part'])<1e-12;direction_checks+=1
    (output_dir/'archive_catalog.json').write_text((Path(__file__).with_name('dependencies.json')).read_text(encoding='utf-8'),encoding='utf-8')
    report=dict(study='historical',profile='replay',physics_cells=len(summaries),physics_raw_accuracy_match=True,
                paired_comparisons=len(comparisons),paired_test='two-sided exact McNemar via binomial discordance counts; Holm across the original 11 comparisons',
                llm_mode='archived statistics only',llm_models=len(llm_rows),
                pressure_dependent_leak_accuracy_checks=direction_checks,
                municipal_inputs='not redistributed; archived statistics only')
    (output_dir/'replay_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['replay','full'],default='replay')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/historical')
    args=parser.parse_args()
    report=replay(args.output_dir)
    if args.profile=='full':
        from .physics import run
        report['physics_full']=run(args.output_dir/'physics')
        report['profile']='full'
        report['fresh_computation']='S5/S6 physics dictionary baselines from the 540 fixed reference vectors'
        report['archival_components']='S7–S12: archived statistics and explicit original script/input dependencies'
        (args.output_dir/'full_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
