"""Frozen EXA7 partition sensor evaluation."""
from pathlib import Path
import os, sys, json, time, hashlib, copy, warnings, itertools
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import networkx as nx
import wntr, igraph as ig, leidenalg
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.sparse.csgraph import shortest_path
from scipy import sparse

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('TS_STUDY_OUTPUT',ROOT/'outputs/partition_sensor')).resolve()
SEEDS=[101,202,303,404,505]
KGRID=[5,10,15,20,30,40]
BUDGETS=[10,15,20,30,45,60]
NOISES=[0,0.01,0.02,0.05,0.1]
DROPOUTS=[0,0.1,0.25,0.5]
RATES=[5.,15.]
SIGMA=0.02

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,x): (OUT/name).write_text(json.dumps(x,indent=2),encoding='utf-8')

def simulate(wn,prefix):
    errors=[]
    for attempt in range(3):
        try:
            return wntr.sim.EpanetSimulator(copy.deepcopy(wn)).run_sim(file_prefix=f'{prefix}_{os.getpid()}_{attempt}')
        except Exception as error:
            errors.append(str(error))
            with (OUT/'simulation_retries.jsonl').open('a') as f:
                f.write(json.dumps({'prefix':prefix,'attempt':attempt,'error':str(error)})+'\n')
    raise RuntimeError(f'EPANET failed after three attempts: {errors}')

def graph_and_pressure(wn):
    baseline=simulate(wn,str(OUT/'epanet_work'/'base'))
    avg=baseline.node['pressure'].mean(axis=0)
    G=nx.Graph(); G.add_nodes_from(wn.node_name_list)
    pipes=[]
    for name in wn.link_name_list:
        l=wn.get_link(name); u,v=l.start_node_name,l.end_node_name
        length=float(l.length) if name in wn.pipe_name_list else 0.
        raw=float((avg[u]+avg[v])/2)
        if G.has_edge(u,v):
            G[u][v]['length']=min(G[u][v]['length'],length)
        else: G.add_edge(u,v,length=length,pressure_raw=raw)
        if name in wn.pipe_name_list: pipes.append((name,u,v,length))
    vals=np.array([d['pressure_raw'] for u,v,d in G.edges(data=True)])
    for u,v,d in G.edges(data=True): d['weight']=float(max((d['pressure_raw']-vals.min())/(vals.max()-vals.min()),.001))
    return G,pipes,baseline,avg

def build_truth(wn,nodes):
    cache=OUT/'query_fingerprints_5_10_15_all_nodes.npz'
    if cache.exists():
        z=np.load(cache); assert list(z['nodes'])==nodes
        return z['F'][[0,2]],z['F'][1],z['baseline_pressure']
    t=time.time(); prefix=str(OUT/'epanet_work'/'truth')
    base=simulate(wn,prefix+'_base')
    bp=base.node['pressure'].mean(axis=0)
    primary_rates=[5.,10.,15.]
    partial=OUT/'query_fingerprints_primary_partial.npz'
    F=np.full((3,len(nodes),len(nodes)),np.nan)
    minimum=np.full((3,len(nodes)),np.nan)
    negcount=np.zeros((3,len(nodes)),dtype=int)
    if partial.exists():
        part=np.load(partial); F=part['F']; minimum=part['minimum']; negcount=part['negcount']
    elif (OUT/'query_fingerprints_partial.npz').exists():
        # The completed 5-L/s rows from the aborted 5/35 run remain valid.
        old=np.load(OUT/'query_fingerprints_partial.npz')['F']
        assert np.isfinite(old[0]).all(); F[0]=old[0]
        audit=pd.read_csv(OUT/'hydraulic_validity.csv',dtype={'node':str})
        lookup=audit[audit.rate_lps==5].set_index('node')
        minimum[0]=lookup.loc[nodes,'min_pressure_m'].values
        negcount[0]=lookup.loc[nodes,'negative_node_times'].values
    for ri,rate in enumerate(primary_rates):
        for ni,node in enumerate(nodes):
            if np.isfinite(F[ri,ni]).all(): continue
            w=copy.deepcopy(wn); w.get_node(node).add_demand(rate/1000.,'881')
            r=simulate(w,prefix+f'_q{rate:g}')
            p=r.node['pressure'].loc[:,nodes]
            F[ri,ni]=p.mean(axis=0).values-bp.loc[nodes].values
            minimum[ri,ni]=p.values.min(); negcount[ri,ni]=(p.values<0).sum()
            if (ni,rate)==(nodes.index('80'),15.): print('worst-node feasibility at 15 L/s:',minimum[ri,ni],flush=True)
            if (ni+1)%25==0: np.savez_compressed(partial,F=F,minimum=minimum,negcount=negcount)
            if (ni+1)%100==0: print('truth',rate,ni+1,'elapsed_s',round(time.time()-t),flush=True)
    assert np.isfinite(F).all() and np.min(minimum)>=0 and np.sum(negcount)==0
    auditrows=[dict(node=n,rate_lps=rate,min_pressure_m=float(minimum[ri,i]),negative_node_times=int(negcount[ri,i])) for ri,rate in enumerate(primary_rates) for i,n in enumerate(nodes)]
    pd.DataFrame(auditrows).to_csv(OUT/'primary_hydraulic_validity.csv',index=False)
    np.savez_compressed(cache,F=F,nodes=np.array(nodes),rates=np.array(primary_rates),baseline_pressure=bp.loc[nodes].values,minimum_pressure=minimum)
    return F[[0,2]],F[1],bp.loc[nodes].values

def stratified_split(nodes,historical,seed):
    rng=np.random.default_rng(seed); test=[]
    for part in sorted(set(historical.values())):
        inds=np.array([i for i,n in enumerate(nodes) if historical[n]==part])
        if len(inds)<3: continue
        count=min(len(inds)-1,max(2,int(round(.2*len(inds)))))
        test.extend(rng.choice(inds,count,replace=False).tolist())
    test=np.array(sorted(test)); train=np.setdiff1d(np.arange(len(nodes)),test)
    assert not set(train)&set(test)
    return train,test

def exact_leiden(G,k,seed,weighted):
    nodelist=list(G); ix={n:i for i,n in enumerate(nodelist)}
    edges=[(ix[u],ix[v]) for u,v in G.edges()]
    weights=[G[u][v]['weight'] if weighted else 1. for u,v in G.edges()]
    g=ig.Graph(n=len(nodelist),edges=edges,directed=False)
    candidates=[]
    # Resolution search is entirely structural, never based on held-out localization.
    for res in np.geomspace(.003,8,80):
        p=leidenalg.find_partition(g,leidenalg.RBConfigurationVertexPartition,weights=weights,resolution_parameter=float(res),seed=seed,n_iterations=-1)
        if len(p)>=k: candidates.append((len(p),float(res),np.array(p.membership)))
    candidates.sort(key=lambda t:(t[0],t[1])); count,res,lab=candidates[0]
    while len(set(lab))>k:
        conn={}
        for (u,v),w in zip(edges,weights):
            a,b=int(lab[u]),int(lab[v])
            if a!=b:
                pair=tuple(sorted((a,b))); conn[pair]=conn.get(pair,0)+w
        a,b=max(conn,key=lambda x:(conn[x],-x[0],-x[1])); lab[lab==b]=a
    vals=sorted(set(lab)); rem={a:i for i,a in enumerate(vals)}
    lab=np.array([rem[a] for a in lab])
    return dict(zip(nodelist,lab.tolist())),res,count

def hydraulic_cluster(G,S,train,nodes,k):
    # Each measurement node is described only by its response to TRAIN perturbations.
    # Held-out leak rows are not used in feature construction or clustering.
    feat=S[train].T.copy(); norm=np.linalg.norm(feat,axis=1,keepdims=True)
    feat=np.divide(feat,norm,out=np.zeros_like(feat),where=norm>1e-12)
    features={n:feat[i] for i,n in enumerate(nodes)}
    for n in G:
        if n not in features:
            neighbours=[features[v] for v in G[n] if v in features]
            features[n]=np.mean(neighbours,axis=0) if neighbours else np.zeros(len(train))
    names=list(G); conn=nx.to_scipy_sparse_array(G,nodelist=names,weight=None,format='csr')
    fit=AgglomerativeClustering(n_clusters=k,linkage='ward',connectivity=sparse.csr_matrix(conn))
    lab=fit.fit_predict(np.array([features[n] for n in names]))
    return dict(zip(names,lab.tolist()))

def partition_stats(mapping,G,pipes,nodes,method,k,seed):
    rows=[]
    for part in range(k):
        members={n for n,c in mapping.items() if c==part}; junctions=set(nodes)&members
        internal=sum(l for _,u,v,l in pipes if u in members and v in members)
        boundary=sum(l for _,u,v,l in pipes if (u in members)!=(v in members))
        rows.append(dict(method=method,k=k,seed=seed,partition=part,n_nodes=len(members),n_junctions=len(junctions),
                         internal_pipe_m=internal,boundary_pipe_m=boundary,inspection_pipe_m=internal+boundary,
                         connected=nx.is_connected(G.subgraph(members))))
    sets=[{n for n,c in mapping.items() if c==i} for i in range(k)]
    info=dict(method=method,k=k,seed=seed,n_partitions=k,boundary_pipe_count=sum(mapping[u]!=mapping[v] for _,u,v,l in pipes),
              boundary_pipe_m=sum(l for _,u,v,l in pipes if mapping[u]!=mapping[v]),
              modularity_unweighted=nx.community.modularity(G,sets,weight=None),
              modularity_pressure=nx.community.modularity(G,sets,weight='weight'),
              min_junctions=min(r['n_junctions'] for r in rows),max_junctions=max(r['n_junctions'] for r in rows),
              size_cv=float(np.std([r['n_junctions'] for r in rows])/np.mean([r['n_junctions'] for r in rows])),
              all_connected=all(r['connected'] for r in rows))
    return rows,info

def sensor_sets(S,train,D,seed):
    rng=np.random.default_rng(seed); n=S.shape[1]
    random_order=rng.permutation(n).tolist()
    first=int(rng.integers(n)); topo=[first]; mind=D[first].copy()
    for _ in range(max(BUDGETS)-1):
        mind[topo]=-1; best=int(np.argmax(mind+rng.uniform(0,1e-9,n)))
        topo.append(best); mind=np.minimum(mind,D[best])
    F=S[train].copy(); norm=np.linalg.norm(F,axis=1,keepdims=True)
    F=np.divide(F,norm,out=np.zeros_like(F),where=norm>1e-12)
    _,sing,V=np.linalg.svd(F,full_matrices=False)
    rank=min(20,int(np.sum(sing>1e-10))); X=V[:rank].T*sing[:rank]
    # Greedy log-determinant increments with prespecified ridge 0.1.
    Ainv=np.eye(rank)/.1; chosen=[]
    for _ in range(max(BUDGETS)):
        score=np.sum((X@Ainv)*X,axis=1); score[chosen]=-np.inf
        j=int(np.argmax(score+rng.uniform(0,1e-12,n))); chosen.append(j)
        z=Ainv@X[j]; Ainv-=np.outer(z,z)/(1+X[j]@z)
    return {'random':random_order[:max(BUDGETS)],'topological_maximin':topo,'train_sensitivity_doptimal':chosen}

def predict(Y,S,train,cols):
    F=S[np.ix_(train,cols)]; norm=np.linalg.norm(F,axis=1)
    unit=np.divide(F,norm[:,None],out=np.zeros_like(F),where=norm[:,None]>1e-12)
    # Min nonnegative projection residual == max positive unit-vector dot product.
    score=Y[:,cols]@unit.T; score[:,norm<1e-12]=-np.inf
    score=np.maximum(score,0); score[:,norm<1e-12]=-np.inf
    pred=train[np.argmax(score,axis=1)]
    abstain=np.max(score,axis=1)<=0
    pred[abstain]=-1
    return pred,abstain

def eval_rows(kind,method,k,budget,seed,queries,pred,mapping,partstat,D,nodes,sigma=SIGMA,dropout=0):
    rows=[]; stats={r['partition']:r for r in partstat}
    for q,p in zip(queries,pred):
        t=int(q['node_index']); tp=mapping[nodes[t]]; pp=mapping[nodes[int(p)]] if p>=0 else -1
        ps=stats.get(pp,{})
        rows.append(dict(experiment=kind,method=method,k=k,budget=budget,seed=seed,query_id=q['query_id'],
                         node=nodes[t],rate_lps=q['rate'],sigma_m=sigma,dropout=dropout,gt_partition=tp,
                         pred_partition=pp,pred_node=nodes[int(p)] if p>=0 else '',correct=int(tp==pp),
                         abstained=int(p<0),distance_m=float(D[t,int(p)]) if p>=0 else np.nan,
                         within_100m=int(p>=0 and D[t,int(p)]<=100),within_500m=int(p>=0 and D[t,int(p)]<=500),
                         inspection_pipe_m=ps.get('inspection_pipe_m',np.nan),internal_pipe_m=ps.get('internal_pipe_m',np.nan),
                         boundary_pipe_m=ps.get('boundary_pipe_m',np.nan),pred_partition_junctions=ps.get('n_junctions',np.nan)))
    return rows

def main():
    started=time.time(); (OUT/'epanet_work').mkdir(parents=True,exist_ok=True)
    tempdir=OUT/'epanet_work'/'main_process_temp'; tempdir.mkdir(exist_ok=True)
    os.environ['TMP']=str(tempdir); os.environ['TEMP']=str(tempdir)
    os.chdir(tempdir)
    wn=wntr.network.WaterNetworkModel(str(ROOT/'data/Exa7.inp'))
    z=np.load(ROOT/'data/reference/sensitivity_matrix.npz'); nodes=list(map(str,z['nodes'])); S=z['S']
    assert nodes==list(map(str,z['meas']))==list(wn.junction_name_list)
    old=json.loads((ROOT/'data/reference/partition_summary.json').read_text())['15']['node_assignments']
    historical={n:int(c) for n,c in old.items()}
    sens=json.loads((ROOT/'data/reference/sensor_summary.json').read_text())
    deployed=sorted([nodes.index(str(n)) for p in sens['partition_details'].values() for n in p['sensor_nodes']])
    assert len(deployed)==30 and len(set(deployed))==30
    G,pipes,baseline,avg=graph_and_pressure(wn)
    allnames=list(G); allix={n:i for i,n in enumerate(allnames)}
    lengthmap=dict(nx.all_pairs_dijkstra_path_length(G,weight='length'))
    distances=np.array([[lengthmap[u][v] for v in allnames] for u in allnames]); inds=[allix[n] for n in nodes]
    D=distances[np.ix_(inds,inds)]; assert np.isfinite(D).all()
    pd.DataFrame(pipes,columns=['pipe','start','end','length_m']).to_csv(OUT/'network_pipe_lengths.csv',index=False)
    truth,S,bp=build_truth(wn,nodes)
    allrows=[]; geometries=[]; profiles=[]; allocations=[]; assignments=[]; splits=[]; maps={}
    hp,hinfo=partition_stats(historical,G,pipes,nodes,'historical_pressure_leiden',15,0)
    for seed in SEEDS:
        train,test=stratified_split(nodes,historical,seed)
        for i in train: splits.append({'seed':seed,'node':nodes[i],'role':'train'})
        for i in test: splits.append({'seed':seed,'node':nodes[i],'role':'test'})
        queries=[dict(query_id=f'{nodes[i]}_q{rate:g}',node_index=int(i),rate=rate) for ri,rate in enumerate(RATES) for i in test]
        clean=np.concatenate([truth[ri,test] for ri in range(len(RATES))])
        # The full 381-channel noise field is paired across all design comparisons.
        noise=np.random.default_rng(seed+17000).normal(size=clean.shape)
        Y=clean+SIGMA*noise
        for method,order in sensor_sets(S,train,D,seed).items():
            for budget in BUDGETS:
                cols=order[:budget]
                allocations.extend([dict(seed=seed,method=method,budget=budget,sensor_rank=r+1,node=nodes[c]) for r,c in enumerate(cols)])
                pred,_=predict(Y,S,train,cols)
                allrows.extend(eval_rows('sensor',method,15,budget,seed,queries,pred,historical,hp,D,nodes))
        allocations.extend([dict(seed=seed,method='historical_deployed',budget=30,sensor_rank=r+1,node=nodes[c]) for r,c in enumerate(deployed)])
        pred,_=predict(Y,S,train,deployed)
        allrows.extend(eval_rows('sensor','historical_deployed',15,30,seed,queries,pred,historical,hp,D,nodes))
        for k in KGRID:
            for method in ['pressure_leiden','topology_leiden','train_hydraulic_ward']:
                if method=='train_hydraulic_ward':
                    mapping=hydraulic_cluster(G,S,train,nodes,k); res=None; pre=k
                else: mapping,res,pre=exact_leiden(G,k,seed,method=='pressure_leiden')
                maps[(method,k,seed)]=mapping
                ps,info=partition_stats(mapping,G,pipes,nodes,method,k,seed)
                info.update(resolution=res,pre_merge_k=pre)
                geometries.extend(ps); profiles.append(info)
                assignments.extend([dict(method=method,k=k,seed=seed,node=n,partition=c) for n,c in mapping.items()])
                allrows.extend(eval_rows('partition',method,k,30,seed,queries,pred,mapping,ps,D,nodes))
        # Complete prespecified noise-by-dropout grid, historical configuration.
        for sigma in NOISES:
            for drop in DROPOUTS:
                keep=np.random.default_rng(seed+23000).permutation(deployed)[:30-int(round(30*drop))].tolist()
                pr,_=predict(clean+sigma*noise,S,train,keep)
                allrows.extend(eval_rows('noise_dropout','historical_deployed',15,len(keep),seed,queries,pr,historical,hp,D,nodes,sigma,drop))
        print('finished design seed',seed,'queries',len(queries),'elapsed_s',round(time.time()-started),flush=True)
    pd.DataFrame(splits).to_csv(OUT/'train_test_splits.csv',index=False)
    pd.DataFrame(geometries).to_csv(OUT/'partition_geometry.csv',index=False)
    pd.DataFrame(profiles).to_csv(OUT/'partition_structure_summary.csv',index=False)
    pd.DataFrame(allocations).to_csv(OUT/'sensor_allocations.csv',index=False)
    pd.DataFrame(assignments).to_csv(OUT/'partition_assignments.csv',index=False)
    rows=pd.DataFrame(allrows); rows.to_csv(OUT/'query_level_results.csv.gz',index=False,compression='gzip')
    keys=['experiment','method','k','budget','seed','sigma_m','dropout']
    runs=rows.groupby(keys,dropna=False).agg(n_queries=('correct','size'),top1=('correct','mean'),
         median_distance_m=('distance_m','median'),mean_distance_m=('distance_m','mean'),within_100m=('within_100m','mean'),
         within_500m=('within_500m','mean'),mean_inspection_pipe_m=('inspection_pipe_m','mean'),
         median_inspection_pipe_m=('inspection_pipe_m','median'),mean_internal_pipe_m=('internal_pipe_m','mean'),
         mean_boundary_pipe_m=('boundary_pipe_m','mean'),mean_pred_partition_junctions=('pred_partition_junctions','mean'),
         abstained=('abstained','sum')).reset_index()
    runs.to_csv(OUT/'run_level_summary.csv',index=False)
    group=[k for k in keys if k!='seed']; metrics=['top1','median_distance_m','within_100m','within_500m','mean_inspection_pipe_m','mean_pred_partition_junctions']
    tidy=[]
    for key,sub in runs.groupby(group,dropna=False):
        for metric in metrics:
            tidy.append({**dict(zip(group,key)),'metric':metric,'mean':float(sub[metric].mean()),'sd':float(sub[metric].std(ddof=1)),
                         'min':float(sub[metric].min()),'max':float(sub[metric].max()),'n_seeds':len(sub),'min_n_queries':int(sub.n_queries.min()),'max_n_queries':int(sub.n_queries.max())})
    pd.DataFrame(tidy).to_csv(OUT/'summary_tidy.csv',index=False)
    stability=[]
    for method in ['pressure_leiden','topology_leiden','train_hydraulic_ward']:
        for k in KGRID:
            for a,b in itertools.combinations(SEEDS,2):
                stability.append(dict(method=method,k=k,seed_a=a,seed_b=b,ari=adjusted_rand_score([maps[method,k,a][n] for n in nodes],[maps[method,k,b][n] for n in nodes])))
    pd.DataFrame(stability).to_csv(OUT/'partition_stability.csv',index=False)
    inputs=['data/Exa7.inp','data/reference/sensitivity_matrix.npz','data/reference/partition_summary.json','data/reference/sensor_summary.json']
    manifest=dict(seeds=SEEDS,k_grid=KGRID,budgets=BUDGETS,noise_sd_m=NOISES,dropout_fractions=DROPOUTS,test_rates_lps=RATES,
         reference_rate_lps=10,reference_pattern='881',network_junctions=len(nodes),network_nodes=len(G),pipes=len(pipes),total_pipe_length_m=sum(p[3] for p in pipes),
         input_inp_units=wn.options.hydraulic.inpfile_units,reported_length_units='metres (WNTR converts EPANET input feet to SI)',
         primary_noise_sd_m=SIGMA,simulated_primary_response_scenarios=len(nodes)*3,primary_response_rates_lps=[5,10,15],
         query_level_rows=len(rows),seed_query_counts=runs.groupby('seed').n_queries.first().to_dict(),
         elapsed_seconds=time.time()-started,software={'python':sys.version,'wntr':wntr.__version__,'numpy':np.__version__},
         inputs={x:digest(ROOT/x) for x in inputs},script_sha256=digest(__file__),outputs={p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ['.csv','.gz','.npz']},
         design_scope=['Controlled physical retrieval experiment; the six retrieval channels plus graph aggregation are not retrained.','One benchmark network; five prescribed seed runs are Monte Carlo split/design/noise replicates, not independent physical networks.','All design grids fixed before outcomes; no best configuration chosen on test results.','Historical 30-sensor design was built previously; only new sensitivity designs enforce train-row restrictions.','Region pipe length is a candidate-set proxy, not measured field labour or travel.'])
    dump('run_manifest.json',manifest)
    print('COMPLETE',len(rows),'query rows',round(time.time()-started),'seconds',flush=True)

if __name__=='__main__': main()
