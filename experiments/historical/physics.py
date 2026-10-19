"""Original eight physics-dictionary baselines using local matrices and no embedding model."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from threadpoolctl import threadpool_limits

from experiments.repeated_benchmarks import classical
from experiments.repeated_benchmarks.primitives import projection_residual

ROOT=Path(__file__).resolve().parents[2]
METHODS=['projection','cosine 1-NN','WLS/NNLS','Mahalanobis','Bayes-marginal','KDE','graph-reg 1-NN','label-prop']


def predict(y,D,yl,P,Adeg,methods,error_log):
    """Score one common retained dictionary using the original baseline formulas."""
    m=D.shape[1]
    Dn=D/np.maximum(np.linalg.norm(D,axis=1,keepdims=True),1e-12)
    yn=y/max(np.linalg.norm(y),1e-12)
    out={}
    def best(sc): return max(sc.items(),key=lambda kv:kv[1])[0] if sc else -1
    if 'projection' in methods:
        out['projection']=int(yl[int(np.argmin(projection_residual(y,D)))])
    cs=Dn@yn
    if 'cosine 1-NN' in methods:
        out['cosine 1-NN']=int(yl[int(np.argmax(cs))])
    if 'WLS/NNLS' in methods:
        try: coef,_=nnls(D.T,y)
        except Exception as exc:
            error_log.append(dict(method='WLS/NNLS',type=type(exc).__name__,message=str(exc)))
            coef=np.zeros(len(D))
        out['WLS/NNLS']=int(yl[int(np.argmax(coef))]) if coef.max()>0 else -1
    if 'Mahalanobis' in methods:
        mu,resid={},[]
        for p in set(yl.tolist()):
            sel=Dn[yl==p];mu[p]=sel.mean(0);resid.append(sel-mu[p])
        Rm=np.vstack(resid);C=(Rm.T@Rm)/max(len(Rm)-1,1)
        C=C+np.trace(C)/m*.1*np.eye(m);Ci=np.linalg.pinv(C)
        sc={}
        for p,v in mu.items():
            d=yn-v;sc[p]=-float(d@Ci@d)
        out['Mahalanobis']=int(best(sc))
    if 'Bayes-marginal' in methods:
        nf2=np.maximum((D**2).sum(1),1e-18)
        proj=(D@y)**2/nf2
        ll=-(float(y@y)-proj)/(2*.02**2)-.5*np.log(nf2);ll-=ll.max();w=np.exp(ll)
        post=defaultdict(float)
        for lab,wi in zip(yl,w): post[int(lab)]+=float(wi)
        out['Bayes-marginal']=int(best(post))
    if 'KDE' in methods:
        sc={}
        for p in set(yl.tolist()):
            Z=Dn[yl==p];d2=((Z-yn)**2).sum(1)
            sc[int(p)]=float(np.exp(-d2/(2*.25**2)).mean())
        out['KDE']=int(best(sc))
    if 'graph-reg 1-NN' in methods:
        top=np.argsort(-cs)[:10];votes=defaultdict(float)
        for rank,i in enumerate(top): votes[int(yl[i])]+=float(max(cs[i],0))/(rank+1)
        v=np.array([votes.get(p,0.) for p in P]);v=v/max(v.max(),1e-12)
        v2=.85*v+.15*(Adeg@v)
        out['graph-reg 1-NN']=int(best({p:float(v2[i]) for i,p in enumerate(P)}))
    if 'label-prop' in methods:
        X=np.vstack([Dn,yn[None,:]])
        S=np.exp((X@X.T-1.)/.05);np.fill_diagonal(S,0.)
        Wn=S/np.maximum(S.sum(1,keepdims=True),1e-12)
        Y=np.zeros((len(X),len(P)))
        for i,lab in enumerate(yl): Y[i,P.index(int(lab))]=1.
        Z=Y.copy()
        for _ in range(30):
            Z=.5*(Wn@Z)+.5*Y;Z[:len(yl)]=Y[:len(yl)]
        out['label-prop']=int(P[int(np.argmax(Z[-1]))])
    return out


@threadpool_limits.wrap(limits=1)
def run(output_dir,methods=None,protocols=None):
    output_dir=Path(output_dir).resolve();output_dir.mkdir(parents=True,exist_ok=True)
    methods=METHODS if methods is None else methods
    protocols=classical.ALL_PROTOCOLS if protocols is None else protocols
    fps,F,part,qidx=classical.load_data();P=sorted(set(part.tolist()))
    pag=json.loads((ROOT/'data/reference/partition_adjacency.json').read_text(encoding='utf-8'))
    A=np.zeros((len(P),len(P)))
    for a,ns in pag['adjacency'].items():
        for b in ns:
            ai,bi=int(a),int(b)
            if ai in P and bi in P: A[P.index(ai),P.index(bi)]=A[P.index(bi),P.index(ai)]=1.
    Adeg=A/np.maximum(A.sum(1,keepdims=True),1e-9)
    raw={p:{name:[] for name in methods} for p in protocols};errors=[];records=[]
    for qi,q in enumerate(fps):
        y=F[qi]+np.random.RandomState(41+qi).normal(0,.02,F.shape[1])
        for proto in protocols:
            idx=classical.retain(qi,proto)
            prior_errors=len(errors)
            predictions=predict(y,F[idx],part[idx],P,Adeg,methods,errors)
            query_errors=errors[prior_errors:]
            for error in query_errors:
                error.update(query_index=qi,protocol=proto,scenario_id=q['scenario_id'])
            failed_methods={error['method'] for error in query_errors}
            for name,pred in predictions.items():
                hit=int(pred==part[qi]);raw[proto][name].append(hit)
                records.append(dict(query_index=qi,protocol=proto,method=name,true_partition=int(part[qi]),predicted_partition=pred,correct=hit,
                                    fit_failed=name in failed_methods))
        if (qi+1)%90==0: print(f'Physics dictionary baselines: {qi+1}/540 queries',flush=True)
    old=json.loads((ROOT/'data/frozen/historical/expH_physics_baselines_results.json').read_text(encoding='utf-8'))
    mismatch=sum(a!=b for p in protocols for name in methods for a,b in zip(raw[p][name],old['raw'][p][name]))
    completed=not errors and mismatch==0
    status='complete' if completed else 'failed'
    summary={p:{name:100*float(np.mean(hits)) for name,hits in raw[p].items()} for p in protocols}
    payload=dict(summary=summary,raw=raw,methods=methods,n_queries=540,noise=.02,seed=41,protocols=protocols,fit_errors=errors,
                 status=status,completed=completed)
    (output_dir/'expH_physics_baselines_results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    with gzip.open(output_dir/'predictions.jsonl.gz','wt',encoding='utf-8') as handle:
        for rec in records: handle.write(json.dumps(rec)+'\n')
    report=dict(study='historical_physics',profile='full',compared_query_predictions=len(records),
                archived_prediction_mismatches=mismatch,fit_errors=len(errors),
                status=status,completed=completed,
                noise_rule='RandomState(41 + original_query_index).normal(0, 0.02, 30)')
    (output_dir/'full_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if errors:
        raise RuntimeError(f'{len(errors)} physics baseline fit errors; run failed. Inspect {output_dir/"expH_physics_baselines_results.json"}')
    if mismatch: raise AssertionError(f'{mismatch} historical baseline predictions differ from archive')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/historical/physics')
    parser.add_argument('--methods',choices=METHODS,nargs='+')
    parser.add_argument('--protocols',choices=classical.ALL_PROTOCOLS,nargs='+')
    args=parser.parse_args();print(json.dumps(run(args.output_dir,args.methods,args.protocols),indent=2))


if __name__=='__main__': main()
