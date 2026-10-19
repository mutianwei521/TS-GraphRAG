"""Recompute train/test, sensor-budget and spatial-unit checks for design controls."""
from pathlib import Path
import json,hashlib,shutil,itertools,os
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(os.environ.get('TS_STUDY_OUTPUT',ROOT/'outputs/partition_sensor')).resolve()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    rows=pd.read_csv(OUT/'query_level_results.csv.gz',dtype={'node':str,'pred_node':str})
    mappings=pd.read_csv(OUT/'partition_assignments.csv',dtype={'node':str})
    pipes=pd.read_csv(OUT/'network_pipe_lengths.csv',dtype={'start':str,'end':str,'pipe':str})
    splits=pd.read_csv(OUT/'train_test_splits.csv',dtype={'node':str})
    geo=pd.read_csv(OUT/'partition_geometry.csv')
    structure=pd.read_csv(OUT/'partition_structure_summary.csv')
    sensors=pd.read_csv(OUT/'sensor_allocations.csv',dtype={'node':str})
    historical=json.loads((ROOT/'data/reference/partition_summary.json').read_text())['15']['node_assignments']
    historical={n:int(c) for n,c in historical.items()}
    test_boundaries={}; facts=[]
    def boundary_set(mapping):
        return {n for p in pipes.itertuples() if mapping[p.start]!=mapping[p.end] for n in [p.start,p.end]}
    historic_boundary=boundary_set(historical)
    for (method,k,seed), sub in mappings.groupby(['method','k','seed']):
        mp=dict(zip(sub.node,sub.partition)); test_boundaries[method,k,seed]=boundary_set(mp)
    flags=[]
    for r in rows.itertuples():
        boundaries=test_boundaries[r.method,r.k,r.seed] if r.experiment=='partition' else historic_boundary
        flags.append('boundary' if r.node in boundaries else 'interior')
    rows['node_location']=flags
    rows.to_csv(OUT/'query_level_results.csv.gz',index=False,compression='gzip')
    group=['experiment','method','k','budget','seed','sigma_m','dropout','node_location']
    byboundary=rows.groupby(group,dropna=False).agg(n_queries=('correct','size'),n_unique_leak_nodes=('node','nunique'),
        top1=('correct','mean'),median_distance_m=('distance_m','median'),mean_distance_m=('distance_m','mean'),
        mean_candidate_pipe_m=('inspection_pipe_m','mean'),within_100m=('within_100m','mean'),within_500m=('within_500m','mean')).reset_index()
    byboundary.to_csv(OUT/'boundary_interior_run_summary.csv',index=False)
    byrate=rows.groupby(['experiment','method','k','budget','seed','sigma_m','dropout','rate_lps']).agg(
        n_queries=('correct','size'),top1=('correct','mean'),median_distance_m=('distance_m','median'),
        mean_candidate_pipe_m=('inspection_pipe_m','mean')).reset_index()
    byrate.to_csv(OUT/'magnitude_run_summary.csv',index=False)
    checks={}
    for seed,sub in splits.groupby('seed'):
        train=set(sub[sub.role=='train'].node); test=set(sub[sub.role=='test'].node)
        assert train.isdisjoint(test) and len(train|test)==381
        rr=rows[rows.seed==seed]
        assert set(rr.node)==test
        assert set(rr.pred_node.dropna())<=train
    checks['every_prediction_is_train_node_and_truth_is_heldout']=True
    for key,sub in sensors.groupby(['method','budget','seed']):
        assert sub.node.nunique()==key[1]==len(sub)
    checks['sensor_counts_and_uniqueness_match_all_budgets']=True
    total=pipes.length_m.sum()
    for (method,k,seed),sub in geo.groupby(['method','k','seed']):
        st=structure[(structure.method==method)&(structure.k==k)&(structure.seed==seed)].iloc[0]
        assert len(sub)==k and bool(sub.connected.all())
        assert np.isclose(sub.internal_pipe_m.sum()+st.boundary_pipe_m,total)
        assert np.isclose(sub.boundary_pipe_m.sum(),2*st.boundary_pipe_m)
        assert np.isclose(sub.inspection_pipe_m.sum(),total+st.boundary_pipe_m)
    checks['all_90_partitions_have_exact_K_and_connected_regions']=True
    checks['internal_boundary_inspection_length_accounting_consistent']=True
    pp=rows[rows.experiment=='partition']
    assert pp.groupby(['seed','query_id']).pred_node.nunique().max()==1
    checks['node_predictions_identical_across_partition_controls']=True
    assert len(rows)==len(rows.drop_duplicates(['experiment','method','k','budget','seed','sigma_m','dropout','query_id']))
    checks['no_duplicate_configuration_query_rows']=True
    assert np.array_equal(~np.isfinite(rows.distance_m),rows.abstained.to_numpy().astype(bool))
    checks['distance_missing_only_for_declared_abstention']=True
    checks['abstentions_across_all_configuration_query_rows']=int(rows.abstained.sum())
    # Freeze the complete 381-channel observed query fields as well as clean responses.
    z=np.load(OUT/'query_fingerprints_5_10_15_all_nodes.npz'); nodes=list(map(str,z['nodes'])); raw={}
    for seed,sub in splits.groupby('seed'):
        test=sorted([nodes.index(n) for n in sub[sub.role=='test'].node])
        clean=np.concatenate([z['F'][ri,test] for ri in [0,2]])
        noise=np.random.default_rng(seed+17000).normal(size=clean.shape)
        raw[f'seed{seed}_clean']=clean; raw[f'seed{seed}_noise_unit']=noise
        raw[f'seed{seed}_observed_sigma002']=clean+.02*noise
        raw[f'seed{seed}_nodes']=np.array([nodes[i] for rate in [5,15] for i in test])
        raw[f'seed{seed}_rates']=np.repeat([5.,15.],len(test))
    raw['measurement_nodes']=np.array(nodes)
    np.savez_compressed(OUT/'query_observations_all_channels.npz',**raw)
    src=OUT/'frozen_inputs'; src.mkdir(exist_ok=True)
    for source in ['data/reference/sensitivity_matrix.npz','data/Exa7.inp','data/reference/partition_summary.json','data/reference/sensor_summary.json']:
        shutil.copy2(ROOT/source,src/Path(source).name)
    checks['raw_query_fields_and_frozen_inputs_saved']=True
    validity=json.loads((OUT/'hydraulic_validity_summary.json').read_text())
    checks['hydraulic_validity']=validity
    primary_validity=pd.read_csv(OUT/'primary_hydraulic_validity.csv')
    assert len(primary_validity)==1143 and primary_validity.negative_node_times.sum()==0
    checks['primary_hydraulic_scenarios']=1143
    checks['primary_minimum_pressure_m']=float(primary_validity.min_pressure_m.min())
    checks['all_primary_scenarios_have_nonnegative_pressure']=True
    (OUT/'audit.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    manifest['design_scope']=manifest.get('design_scope',manifest.get('limitations',[]))
    manifest.pop('limitations',None)
    manifest['hydraulic_validity']=validity
    manifest['boundary_definition']='True leak node touches at least one physical pipe joining different assigned regions; pumps are excluded from the boundary definition.'
    manifest['primary_response_scenarios_in_final_cache']=1143
    manifest['design_feasibility_screen']=dict(rates_lps=[0,5,20,35],source=validity.get('source','fresh hydraulic simulation'),n_scenarios=validity['new_simulations'])
    manifest['primary_physical_design']='Fixed 10 L/s reference and 5/15 L/s test rates; all 1143 primary hydraulic responses have nonnegative junction pressures.'
    manifest['outputs']={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and 'epanet_work' not in str(p) and p.name!='run_manifest.json'}
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))

if __name__=='__main__': main()
