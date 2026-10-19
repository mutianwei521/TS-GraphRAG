"""Replay E5 draw summaries or rerun projection retrieval on its frozen hydraulic tensor."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import random
import re
from pathlib import Path

import numpy as np

from experiments.repeated_benchmarks.primitives import projection_residual

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT/'data/frozen/allocation'
PERIODS = ['night_low','day_normal','evening_peak']
QUERY_MAGS = [5.,20.,50.]
PROTOCOLS = [f'H{i}' for i in range(10)]
DESIGNS = {
    'A':(30,[2.,5.,10.,20.,35.,50.]),
    'A_prime':(30,[3.,8.,15.,25.,30.,40.]),
    'C60':(60,[2.,10.,35.]),'C90':(90,[10.,35.]),'B':(180,[10.])}
METRICS = ['top1','top3','mrr','med_m']
DISTANCE_METRIC = 'median_coordinate_distance'
DISTANCE_UNITS = 'native_network_coordinate_units'
ARCHIVE_METRIC_ALIASES = {DISTANCE_METRIC:'med_m'}
ARCHIVE_CSV_ALIASES = {'median_error_coordinate_units':'e5_h0_draws.csv:median_error_m'}


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def load_inputs(tensor_path=None):
    z=np.load(tensor_path or FROZEN/'expH_robust_tensor.npz',allow_pickle=False)
    part=json.loads((ROOT/'data/reference/partition_summary.json').read_text(encoding='utf-8'))['15']['node_assignments']
    nodes=[str(n) for n in z['nodes'] if str(n) in part]
    sensors=z['sensors'].tolist()
    pool=[n for n in nodes if n not in set(sensors)]
    if z['T'].shape != (350,12,3,30) or not np.isfinite(z['T']).all():
        raise ValueError('Expected the finite 350 x 12 x 3 x 30 original E5 tensor')
    assert z['periods'].tolist()==PERIODS
    return z,part,nodes,sensors,pool


def reconstruct_splits(pool):
    """Reconstruct nested corpus nodes and paired query groups in original sampling order."""
    records=[]
    for s in range(15):
        rng=random.Random(41+s)
        b=rng.sample(pool,180)
        a,c60,c90=b[:30],b[:60],b[:90]
        g1=rng.sample(a,20)
        g2=rng.sample([n for n in pool if n not in set(b)],20)
        noise=np.random.RandomState(41+s).normal(0.,.02,30)
        records.append(dict(stored_seed=s,actual_rng_seed=41+s,A=a,A_prime=a,C60=c60,C90=c90,B=b,
                            G1_query_nodes=g1,G2_query_nodes=g2,
                            one_shared_noise_vector_sha256=hashlib.sha256(noise.tobytes()).hexdigest()))
    return records


def summarize(v):
    v=np.asarray(v,dtype=float)
    med,q1,q3=np.percentile(v,[50,25,75],method='linear')
    return dict(median=float(med),q1=float(q1),q3=float(q3),mean=float(np.mean(v)),
                sample_sd=float(np.std(v,ddof=1)),run_values=v.tolist())


def analyze(runs,names):
    label_for=dict(zip(names,DESIGNS))
    lookup={(r['seed'],label_for[r['design']],r['group'],r['protocol']):r for r in runs}
    expected=set(itertools.product(range(15),DESIGNS,['G1','G2'],PROTOCOLS))
    if len(runs)!=1500 or set(lookup)!=expected:
        raise ValueError('The E5 archive must contain all 15 x 5 x 2 x 10 draw-level records')
    for r in runs:
        assert 0<=r['top1']<=r['top3']<=1 and 0<=r['mrr']<=1 and r['med_m']>=0
        assert np.isfinite([r[k] for k in METRICS]).all()
        assert all(abs(r[k]*180-round(r[k]*180))<1e-10 for k in ['top1','top3'])
    def values(d,g,p,k):
        return np.array([lookup[s,d,g,p][k]*(1 if k=='med_m' else 100) for s in range(15)])
    def compare(a,b,g,p,k):
        v=values(a,g,p,k)-values(b,g,p,k)
        pos,neg=int((v>1e-9).sum()),int((v<-1e-9).sum())
        n=pos+neg
        pv=min(1.,2*sum(math.comb(n,j) for j in range(min(pos,neg)+1))/(2**n)) if n else 1.
        return dict(first=a,second=b,group=g,protocol=p,metric=k,**summarize(v),positive=pos,
                    negative=neg,ties=15-n,non_tied_n=n,p_two_sided_exact=pv,
                    test='paired exact sign test; ties omitted; unadjusted')
    summaries={g:{p:{d:{k:summarize(values(d,g,p,k)) for k in METRICS} for d in DESIGNS}
                  for p in PROTOCOLS} for g in ['G1','G2']}
    comps=[compare(b,a,g,p,k) for g,p in [('G2','H0'),('G1','H0'),('G1','H8'),('G2','H8')]
           for a,b in itertools.combinations(DESIGNS,2) for k in METRICS]
    family=[c for c in comps if (c['group'],c['protocol'],c['metric'])==('G2','H0','top1')]
    running=0.
    for i,c in enumerate(sorted(family,key=lambda c:c['p_two_sided_exact'])):
        running=min(1.,max(running,(len(family)-i)*c['p_two_sided_exact']))
        c['p_holm_sensitivity_family_10_G2_H0_top1']=running
    return lookup,summaries,comps,family


def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def replay(output_dir):
    output_dir=Path(output_dir).resolve();output_dir.mkdir(parents=True,exist_ok=True)
    raw=json.loads((FROZEN/'expH_robust_results.json').read_text(encoding='utf-8'))
    lookup,summaries,comparisons,family=analyze(raw['runs'],list(raw['designs']))
    z,part,nodes,sensors,pool=load_inputs()
    splits=reconstruct_splits(pool)
    h0=[dict(group=g,protocol='H0',design=d,metric=DISTANCE_METRIC if k=='med_m' else k,
             archived_metric=k,units=DISTANCE_UNITS if k=='med_m' else '%',n_draws=15,
             **{q:summaries[g]['H0'][d][k][q] for q in ['median','q1','q3','mean','sample_sd']})
        for g,d,k in itertools.product(['G1','G2'],DESIGNS,METRICS)]
    with (FROZEN/'e5_h0_summary.csv').open(encoding='utf-8-sig',newline='') as handle:
        archive=list(csv.DictReader(handle))
    for r,old in zip(h0,archive):
        for key in ('median','q1','q3','mean','sample_sd'):
            assert abs(r[key]-float(old[key]))<1e-10,(r,key,old)
    assert len(h0)==len(archive)
    pair_rows=[{k:v for k,v in c.items() if k!='run_values'} for c in family]
    with (FROZEN/'e5_g2_top1_pairwise.csv').open(encoding='utf-8-sig',newline='') as handle:
        pair_archive=list(csv.DictReader(handle))
    assert len(pair_rows)==len(pair_archive)
    for rec,old in zip(pair_rows,pair_archive):
        for key,value in rec.items():
            if isinstance(value,(int,float)):
                assert abs(value-float(old[key]))<1e-10,(rec,key,old)
            else:
                assert value==old[key],(rec,key,old)
    text=(FROZEN/'expH_robust_results.json').read_text(encoding='utf-8')
    source_lines=[i+1 for i,line in enumerate(text.splitlines()) if re.search(r'"seed"\s*:',line)]
    label_for=dict(zip(raw['designs'],DESIGNS))
    record_lines={(r['seed'],label_for[r['design']],r['group'],r['protocol']):line
                  for r,line in zip(raw['runs'],source_lines)}
    h0_draws=[dict(stored_seed=s,actual_seed=s+41,group=g,protocol='H0',design=d,n_queries=180,
                   top1_percent=lookup[s,d,g,'H0']['top1']*100,
                   top3_percent=lookup[s,d,g,'H0']['top3']*100,
                   mrr_at5_percent=lookup[s,d,g,'H0']['mrr']*100,
                   median_error_coordinate_units=lookup[s,d,g,'H0']['med_m'],
                   original_json_seed_line=record_lines[s,d,g,'H0'])
              for s,d,g in itertools.product(range(15),DESIGNS,['G1','G2'])]
    equal=all(lookup[s,d,'G2','H0'][k]==lookup[s,d,'G2','H2'][k]
              for s,d,k in itertools.product(range(15),DESIGNS,METRICS))
    assert equal
    write_csv(output_dir/'e5_h0_summary.csv',h0)
    write_csv(output_dir/'e5_g2_top1_pairwise.csv',pair_rows)
    write_csv(output_dir/'e5_h0_draws.csv',h0_draws)
    save(output_dir/'draws.json',splits)
    public_summaries={g:{p:{d:{DISTANCE_METRIC if k=='med_m' else k:v for k,v in values.items()}
                              for d,values in designs.items()} for p,designs in protocols.items()}
                      for g,protocols in summaries.items()}
    public_comparisons=[{**r,'metric':DISTANCE_METRIC if r['metric']=='med_m' else r['metric'],
                         'archived_metric':r['metric'],
                         'units':DISTANCE_UNITS if r['metric']=='med_m' else 'percentage_points'}
                        for r in comparisons]
    save(output_dir/'all_statistics.json',dict(summaries=public_summaries,pairwise_comparisons=public_comparisons,
                                             distance_units=DISTANCE_UNITS,archive_metric_aliases=ARCHIVE_METRIC_ALIASES))
    report=dict(study='allocation',profile='replay',run_count=len(raw['runs']),actual_seeds=list(range(41,56)),
                tensor_shape=list(z['T'].shape),sampling_pool_size=len(pool),archive_tables_match=True,
                G2_H0_equals_H2_all_metrics=equal,noise_rule='one shared RandomState(41 + stored_seed).normal(0, 0.02, 30) vector per draw',
                metrics=dict(top1='percent',top3='percent',mrr='MRR@5 percent; five distinct partitions',
                             median_coordinate_distance='within-draw median Euclidean distance in native network-coordinate units'),
                distance_units=DISTANCE_UNITS,archive_metric_aliases=ARCHIVE_METRIC_ALIASES,
                archive_csv_aliases=ARCHIVE_CSV_ALIASES,
                uncertainty='median [linear Q1,Q3]; sample SD ddof=1 over 15 draws')
    save(output_dir/'replay_report.json',report)
    return report


def coordinates():
    section=None;xy={}
    for line in (ROOT/'data/Exa7.inp').read_text(encoding='utf-8-sig').splitlines():
        line=line.split(';',1)[0].strip()
        if line.startswith('['): section=line.upper()
        elif section=='[COORDINATES]' and line:
            nd,x,y,*_=line.split();xy[nd]=np.array([float(x),float(y)])
    return xy


def full(output_dir,draws=None,protocols=None,groups=None,tensor_path=None):
    """Rerun the exact E5 atom ranking, shared query noise and partition metrics."""
    output_dir=Path(output_dir).resolve();output_dir.mkdir(parents=True,exist_ok=True)
    draws=list(range(15)) if draws is None else list(draws)
    protocols=PROTOCOLS if protocols is None else protocols
    groups=['G1','G2'] if groups is None else groups
    z,part,nodes,sensors,pool=load_inputs(tensor_path)
    T=z['T'];ni={str(n):i for i,n in enumerate(z['nodes'])};mi={float(q):i for i,q in enumerate(z['mags'])}
    pi={str(p):i for i,p in enumerate(z['periods'])};xy=coordinates()
    original=json.loads((FROZEN/'expH_robust_results.json').read_text(encoding='utf-8'))
    names=list(original['designs']);name_for=dict(zip(DESIGNS,names))
    archived={(r['seed'],r['design'],r['group'],r['protocol']):r for r in original['runs']}
    runs=[];maxdiff=0.;query_count=0
    with gzip.open(output_dir/'query_predictions.csv.gz','wt',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['stored_seed','actual_seed','design','group','protocol','leak_node','magnitude_Ls','demand_period','top1','top3','mrr_at5','distance_coordinate_units'])
        for s in draws:
            split=reconstruct_splits(pool)[s]
            noise=np.random.RandomState(41+s).normal(0,.02,30)
            for label,(nn,mags) in DESIGNS.items():
                atoms=[(n,q,t) for n in split[label] for q in mags for t in PERIODS]
                assert len(atoms)==540
                A=np.array([T[ni[n],mi[q],pi[t]] for n,q,t in atoms])
                an=np.array([a[0] for a in atoms]);aq=np.array([a[1] for a in atoms]);at=np.array([a[2] for a in atoms]);ap=np.array([part[n] for n in an])
                for group in groups:
                    for proto in protocols:
                        hits=[];top3=[];rr=[];dists=[]
                        for nd in split[group+'_query_nodes']:
                            for q in QUERY_MAGS:
                                for t in PERIODS:
                                    sn,sq,st=an==nd,aq==q,at==t
                                    mask={'H0':np.ones(540,bool),'H1':~(sn&sq&st),'H2':~sn,'H3':~sq,'H4':~st,
                                          'H5':~(sn|sq),'H6':~(sn|st),'H7':~(sq|st),'H8':~(sn|sq|st),'H9':ap!=part[nd]}[proto]
                                    idx=np.flatnonzero(mask)
                                    if not idx.size: continue
                                    y=T[ni[nd],mi[q],pi[t]]+noise
                                    order=idx[np.argsort(projection_residual(y,A[idx]))]
                                    seen=set();rank=[]
                                    for i in order:
                                        p=part[atoms[i][0]]
                                        if p not in seen: seen.add(p);rank.append(p)
                                        if len(rank)>=5: break
                                    h=int(rank[0]==part[nd]);h3=int(part[nd] in rank[:3])
                                    mr=next((1./(j+1) for j,p in enumerate(rank) if p==part[nd]),0.)
                                    distance=float(np.linalg.norm(xy[atoms[order[0]][0]]-xy[nd]))
                                    hits.append(h);top3.append(h3);rr.append(mr);dists.append(distance)
                                    writer.writerow([s,41+s,label,group,proto,nd,q,t,h,h3,mr,distance]);query_count+=1
                        rec=dict(seed=s,design=name_for[label],group=group,protocol=proto,
                                 top1=float(np.mean(hits)),top3=float(np.mean(top3)),mrr=float(np.mean(rr)),med_m=float(np.median(dists)))
                        old=archived[s,name_for[label],group,proto]
                        maxdiff=max(maxdiff,max(abs(rec[k]-old[k]) for k in METRICS))
                        runs.append({DISTANCE_METRIC if k=='med_m' else k:v for k,v in rec.items()})
            print(f'E5 draw {s+1}/15 finished',flush=True)
    save(output_dir/'expH_robust_results.json',dict(runs=runs,designs=original['designs'],n_seeds=len(draws),n_query_nodes=20,q_mags=QUERY_MAGS,noise=.02,seed0=41,n_admissible=len(nodes),
                                                  distance_units=DISTANCE_UNITS,archive_metric_aliases=ARCHIVE_METRIC_ALIASES))
    report=dict(study='allocation',profile='full',run_count=len(runs),query_count=query_count,
                max_absolute_metric_difference=maxdiff,archive_metrics_match=maxdiff<1e-10,
                distance_units=DISTANCE_UNITS,archive_metric_aliases=ARCHIVE_METRIC_ALIASES)
    save(output_dir/'full_report.json',report)
    if maxdiff>=1e-10: raise AssertionError(f'E5 metrics differ from original tensor calculation: {maxdiff}')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['replay','full'],default='replay')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/allocation')
    parser.add_argument('--draws',type=int,nargs='+',choices=range(15))
    parser.add_argument('--tensor',type=Path)
    args=parser.parse_args()
    result=replay(args.output_dir) if args.profile=='replay' else full(args.output_dir,args.draws,tensor_path=args.tensor)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
