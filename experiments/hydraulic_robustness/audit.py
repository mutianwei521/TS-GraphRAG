"""Independent output tabulation for the fixed EXA7 hydraulic study."""
from pathlib import Path
import json, hashlib, os
import numpy as np
import pandas as pd
import wntr

OUT=Path(os.environ.get('TS_STUDY_OUTPUT',Path(__file__).resolve().parents[2]/'outputs/hydraulic_robustness')).resolve()

def write_json(p,x):
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def main():
    d=json.loads((OUT/'design.json').read_text(encoding='utf-8'))
    q=pd.read_csv(OUT/'query_metrics.csv')
    hyd=pd.read_csv(OUT/'query_hydraulic_status.csv')
    wn=wntr.network.WaterNetworkModel(str(OUT/'inputs/Exa7.inp'))
    for model_json in (OUT/'models').glob('*_parameters.json'):
        info=json.loads(model_json.read_text())
        if 'original_demand_totals_m3s_by_hour' in info:
            info['pre_spatial_demand_totals_m3s_by_hour']=info.pop('original_demand_totals_m3s_by_hour')
            write_json(model_json,info)
    # Older in-flight workers classified any warning as nonconvergence. Separate
    # negative-pressure warnings from numerical convergence without changing the
    # prespecified pressure admissibility or any retrieval outcome.
    for status_file in (OUT/'cache').glob('*_status.json'):
        status=json.loads(status_file.read_text())
        for st in status:
            assert not any(token in str(st['error_code']) for token in ['Error 106','Error 307','No such file or directory','PermissionError']),status_file
            unstable=[s for s in st['report_warnings'] if 'unbalanced' in s.lower() or 'unstable' in s.lower()]
            st['report_warning_free']=not st['report_warnings']
            st['converged']=bool(st['success'] and st['error_code'] is None and st['pressure_min_m'] is not None and not unstable)
        write_json(status_file,status)
    js=wn.junction_name_list
    services=[i for i,n in enumerate(js) if sum(v.base_value for v in wn.get_node(n).demand_timeseries_list)>0]
    extra=[]
    for s in d['scenarios']:
        for seed in d['seeds']:
            physical=d['seeds'][0] if s['family'] in ['control','uniform_demand'] else seed
            name=f"{s['name']}_seed{physical}"
            z=np.load(OUT/'cache'/f'{name}.npz')
            query_pressures=z['pressures']
            st=json.loads((OUT/'cache'/f'{name}_status.json').read_text())
            bp=z['baseline'];bc=bp[:,services]
            for qi,query in enumerate(d['queries']):
                p=query_pressures[qi];pc=p[:,services]
                finite=bool(np.isfinite(p).all() and np.isfinite(bp).all())
                err=st[qi+1]['error_code'];berr=st[0]['error_code']
                # Pressure warnings are physical diagnostics, not numerical nonconvergence.
                reports=st[qi+1]['report_warnings']+st[0]['report_warnings']
                unstable=[x for x in reports if 'unbalanced' in x.lower() or 'unstable' in x.lower()]
                other_warnings=[x for x in reports if 'negative pressure' not in x.lower()]
                conv=bool(finite and err is None and berr is None and not unstable)
                minp=float(np.min(p)) if finite else None
                minb=float(np.min(bp)) if finite else None
                minservice=float(np.min(pc)) if finite else None
                minbservice=float(np.min(bc)) if finite else None
                pi=np.unravel_index(np.nanargmin(p),p.shape) if finite else (None,None)
                extra.append(dict(seed=seed,scenario=s['name'],query_id=query['query_id'],
                    pressure_min_node=js[pi[1]] if finite else '',pressure_min_hour=pi[0] if finite else None,
                    service_pressure_min_m=minservice,baseline_service_pressure_min_m=minbservice,
                    service_negative_pressure=bool(finite and (minservice<0 or minbservice<0)),
                    service_admissible=bool(conv and not other_warnings and minservice>=0 and minbservice>=0 and pc.max()<=200 and bc.max()<=200),
                    hydraulic_numerical_convergence=conv))
    ex=pd.DataFrame(extra)
    ex.to_csv(OUT/'query_service_pressure_status.csv',index=False,encoding='utf-8-sig')
    q=q.merge(ex,on=['seed','scenario','query_id'],validate='many_to_one')
    q['converged']=q['hydraulic_numerical_convergence']
    # Keep the main CSV schema compact; richer diagnostics are in their own CSV.
    original_columns=pd.read_csv(OUT/'query_metrics.csv',nrows=0).columns
    q[list(original_columns)].to_csv(OUT/'query_metrics.csv',index=False,encoding='utf-8-sig')
    hyd=hyd.merge(ex[['seed','scenario','query_id','hydraulic_numerical_convergence']],on=['seed','scenario','query_id'],validate='one_to_one')
    hyd['converged']=hyd['hydraulic_numerical_convergence']
    hyd.drop(columns='hydraulic_numerical_convergence').to_csv(OUT/'query_hydraulic_status.csv',index=False,encoding='utf-8-sig')
    secondary=[]
    keys=['seed','scenario','family','strength','baseline','library_policy','method']
    for key,g in q.groupby(keys,sort=False):
        service=g[g.service_admissible]
        secondary.append(dict(zip(keys,key))|dict(n_queries=len(g),n_service_admissible=len(service),
            service_failure_pct=100*(~g.service_admissible).mean(),
            service_negative_pressure_pct=100*g.service_negative_pressure.mean(),
            numerical_nonconvergence_pct=100*(~g.hydraulic_numerical_convergence).mean(),
            top1_service_admissible_pct=100*service.top1_raw.mean() if len(service) else np.nan,
            top1_service_all_pct=100*(g.top1_raw*g.service_admissible).mean(),
            minimum_service_pressure_m=g.service_pressure_min_m.min()))
    sdf=pd.DataFrame(secondary);sdf.to_csv(OUT/'service_secondary_seed_summary.csv',index=False,encoding='utf-8-sig')
    summary=sdf.groupby(keys[1:],sort=False).agg(
        top1_service_all_pct_mean=('top1_service_all_pct','mean'),top1_service_all_pct_sd=('top1_service_all_pct','std'),
        top1_service_admissible_pct_mean=('top1_service_admissible_pct','mean'),top1_service_admissible_pct_sd=('top1_service_admissible_pct','std'),
        service_failure_pct_mean=('service_failure_pct','mean'),service_negative_pressure_pct_mean=('service_negative_pressure_pct','mean'),
        numerical_nonconvergence_pct_mean=('numerical_nonconvergence_pct','mean'),minimum_service_pressure_m=('minimum_service_pressure_m','min')).reset_index()
    summary.to_csv(OUT/'service_secondary_summary.csv',index=False,encoding='utf-8-sig')
    expected=len(d['scenarios'])*len(d['seeds'])*len(d['queries'])*2*3*3
    assert len(q)==expected,(len(q),expected)
    assert not q.duplicated(['seed','scenario','query_id','baseline','library_policy','method']).any()
    nominal=q[q.scenario=='nominal']
    piv=nominal.pivot(index=['seed','query_id','library_policy','method'],columns='baseline',values='top1_raw')
    assert (piv.matched==piv.stale_nominal).all()
    assert ((q.top1_all<=q.top1_raw)&(q.top3_all<=q.top3_raw)).all()
    assert q.groupby(keys).size().eq(60).all()
    atoms=json.loads((OUT/'cache/nominal_atoms.json').read_text())
    assert not set(d['excluded_query_nodes'])&{a['node'] for a in atoms}
    cs=np.load(OUT/'cache/nominal_corpus.npz')
    assert cs['F'].shape==(1053,30)
    numerical_nonzero=np.isfinite(cs['F']).all(axis=1)&(np.linalg.norm(cs['F'],axis=1)>1e-12)
    corpus_counts={}
    for period in dict.fromkeys(a['period'] for a in atoms):
        period_mask=np.array([a['period']==period for a in atoms])
        corpus_counts[period]=dict(planned=int(period_mask.sum()),
            hydraulically_eligible=int((period_mask&cs['keep']).sum()),
            nonzero_rank_eligible=int((period_mask&cs['keep']&numerical_nonzero).sum()))
    corpus_counts['all_nominal_periods']=dict(planned=len(atoms),
        hydraulically_eligible=int(cs['keep'].sum()),
        nonzero_rank_eligible=int((cs['keep']&numerical_nonzero).sum()))
    write_json(OUT/'qa/corpus_candidate_counts.json',corpus_counts)
    for period_index,scenario_name in [(0,'uniform_demand_0.7'),(1,'nominal'),(2,'uniform_demand_1.2')]:
        control=np.load(OUT/'cache'/f'{scenario_name}_seed{d["seeds"][0]}.npz')
        assert np.array_equal(cs['baseline'][period_index],control['baseline']),scenario_name
    repeats=list((OUT/'qa').glob('*_repeat_baseline.json'))
    assert len(repeats)==106,len(repeats)
    assert all(json.loads(p.read_text())['identical'] for p in repeats)
    replay=json.loads((OUT/'qa/nominal_corpus_replay.json').read_text())
    assert replay['all_pressure_arrays_identical'] and replay['n_templates']==1053
    active=[]
    expected_cache_names=[]
    for s in d['scenarios']:
        physical_seeds=d['seeds'][:1] if s['family'] in ['control','uniform_demand'] else d['seeds']
        for seed in physical_seeds:
            name=f'{s["name"]}_seed{seed}';expected_cache_names.append(name+'.npz')
            artifacts=[OUT/'cache'/f'{name}.npz',OUT/'cache'/f'{name}_status.json',
                       OUT/'models'/f'{name}.inp',OUT/'models'/f'{name}_parameters.json',
                       OUT/'qa'/f'{name}_repeat_baseline.json']
            active.append(dict(name=name,scenario=s['name'],physics_seed=seed,
                artifacts={str(p.relative_to(OUT)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    bytes=p.stat().st_size,modified_epoch_s=p.stat().st_mtime) for p in artifacts}))
    assert sorted(expected_cache_names)==sorted(p.name for p in (OUT/'cache').glob('*_seed*.npz'))
    write_json(OUT/'qa/active_cache_manifest.json',dict(n_physical_caches=len(active),
        experiment_code_sha256=hashlib.sha256(Path(__file__).with_name('model.py').read_bytes()).hexdigest(),
        all_caches_passed_before_after_baseline_identity=True,
        nominal_period_controls_match_original_arrays=True,caches=active))
    means=pd.read_csv(OUT/'summary.csv')
    seedsum=pd.read_csv(OUT/'seed_summary.csv')
    numerical=ex.groupby(['seed','scenario']).hydraulic_numerical_convergence.apply(lambda x:100*(~x).mean())
    seedsum['nonconvergence_pct']=[numerical.loc[(int(r.seed),r.scenario)] for _,r in seedsum.iterrows()]
    seedsum.to_csv(OUT/'seed_summary.csv',index=False,encoding='utf-8-sig')
    num_mean=seedsum.groupby(keys[1:]).nonconvergence_pct.agg(['mean','std'])
    means['nonconvergence_pct_mean']=[num_mean.loc[tuple(r[k] for k in keys[1:]),'mean'] for _,r in means.iterrows()]
    means['nonconvergence_pct_sd']=[num_mean.loc[tuple(r[k] for k in keys[1:]),'std'] for _,r in means.iterrows()]
    available=seedsum.groupby(keys[1:],sort=False).agg(
        n_realizations_with_admissible_queries=('n_admissible',lambda x:int((x>0).sum())),
        n_admissible_queries_across_realizations=('n_admissible','sum')).reset_index()
    means=means.drop(columns=[c for c in available if c not in keys[1:] and c in means],errors='ignore').merge(available,on=keys[1:],validate='one_to_one')
    means.to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
    detail=means[(means.method=='projection')&(means.library_policy=='nominal_day')]
    readable=[]
    for s in d['scenarios']:
        sm=detail[(detail.scenario==s['name'])&(detail.baseline=='matched')].iloc[0]
        ss=detail[(detail.scenario==s['name'])&(detail.baseline=='stale_nominal')].iloc[0]
        sec=summary[(summary.scenario==s['name'])&(summary.baseline=='matched')&(summary.method=='projection')&(summary.library_policy=='nominal_day')].iloc[0]
        readable.append(dict(scenario=s['name'],matched_top1_pct_mean=sm.top1_all_pct_mean,
            matched_top1_pct_sd=sm.top1_all_pct_sd,stale_top1_pct_mean=ss.top1_all_pct_mean,stale_top1_pct_sd=ss.top1_all_pct_sd,
            strict_failure_pct_mean=sm.failure_pct_mean,strict_negative_pressure_pct_mean=sm.negative_pressure_pct_mean,
            service_failure_pct_mean=sec.service_failure_pct_mean,
            numerical_nonconvergence_pct_mean=sec.numerical_nonconvergence_pct_mean,
            n_realizations_with_admissible_queries=sm.n_realizations_with_admissible_queries,
            matched_admissible_only_top1_pct_mean=sm.top1_admissible_pct_mean,
            service_admissible_top1_pct_mean=sec.top1_service_admissible_pct_mean))
    pd.DataFrame(readable).to_csv(OUT/'report_table.csv',index=False,encoding='utf-8-sig')
    for policy,filename in [('nominal_day','Table_S13_hydraulic_mismatch.csv'),
                            ('all_nominal_periods','Table_S13_union_secondary.csv')]:
        s13=[]
        for s in d['scenarios']:
            g=means[(means.scenario==s['name'])&(means.library_policy==policy)&(means.method=='projection')]
            match=g[g.baseline=='matched'].iloc[0]; stale=g[g.baseline=='stale_nominal'].iloc[0]
            sec=summary[(summary.scenario==s['name'])&(summary.library_policy==policy)&(summary.method=='projection')&(summary.baseline=='matched')].iloc[0]
            s13.append(dict(scenario=s['name'],family=s['family'],strength=s['value'],library_policy=policy,n_realizations=5,n_queries_per_realization=60,
                matched_top1_all_pct_mean=match.top1_all_pct_mean,matched_top1_all_pct_sd=match.top1_all_pct_sd,
                stale_top1_all_pct_mean=stale.top1_all_pct_mean,stale_top1_all_pct_sd=stale.top1_all_pct_sd,
                admissible_pct_mean=100-match.failure_pct_mean,admissible_pct_sd=match.failure_pct_sd,
                matched_top1_admissible_pct_mean=match.top1_admissible_pct_mean,matched_top1_admissible_pct_sd=match.top1_admissible_pct_sd,
                stale_top1_admissible_pct_mean=stale.top1_admissible_pct_mean,stale_top1_admissible_pct_sd=stale.top1_admissible_pct_sd,
                n_realizations_with_admissible_queries=match.n_realizations_with_admissible_queries,
                n_admissible_queries_across_realizations=match.n_admissible_queries_across_realizations,
                negative_pressure_pct_mean=match.negative_pressure_pct_mean,
                numerical_nonconvergence_pct_mean=sec.numerical_nonconvergence_pct_mean,
                service_failure_pct_mean=sec.service_failure_pct_mean))
        pd.DataFrame(s13).to_csv(OUT/filename,index=False,encoding='utf-8-sig')
    stratified=[]
    for key,g in q.groupby(keys+['rate_Lps'],sort=False):
        ok=g[g.admissible]
        stratified.append(dict(zip(keys+['rate_Lps'],key))|dict(n_queries=len(g),n_admissible=len(ok),
            top1_all_pct=100*g.top1_all.mean(),top3_all_pct=100*g.top3_all.mean(),
            top1_admissible_pct=100*ok.top1_raw.mean() if len(ok) else np.nan,
            failure_pct=100*(~g.admissible).mean()))
    pd.DataFrame(stratified).to_csv(OUT/'rate_stratified_seed_summary.csv',index=False,encoding='utf-8-sig')
    write_json(OUT/'qa/independent_audit.json',dict(status='passed',expected_metric_rows=expected,
        actual_metric_rows=len(q),query_hydraulic_rows=len(hyd),n_complete_realizations=5,
        n_queries_per_cell=60,query_nodes_overlap=0,nominal_matched_stale_predictions_identical=True,
        n_service_junctions=len(services),n_all_junctions=len(js),
        n_identical_before_after_baseline_controls=len(repeats),native_solver_worker_cwd_and_tmp_isolated=True,
        exhaustive_nominal_corpus_replay_templates=1053,exhaustive_nominal_corpus_replay_identical=True,
        service_endpoint_note='Post-run diagnostic supplements, but does not replace, the frozen all-junction pressure rule. Pump suction auxiliary nodes can have negative gauge pressure while demand-serving nodes remain positive.',
        all_scenarios_numerical_nonconvergence_pct=float(100*(~ex.hydraulic_numerical_convergence).mean())))
    manifest=json.loads((OUT/'run_manifest.json').read_text(encoding='utf-8'))
    manifest['postprocessing_audit']='qa/independent_audit.json'
    manifest['primary_library_policy']='nominal_day'
    manifest['primary_retrieval_rule']='projection'
    manifest['corpus_candidate_counts']=corpus_counts
    manifest['nominal_day_candidates']=corpus_counts['day_normal']['nonzero_rank_eligible']
    manifest['all_nominal_periods_candidates']=corpus_counts['all_nominal_periods']['nonzero_rank_eligible']
    manifest['baseline_repeat_control_simulations']=106
    manifest['exhaustive_corpus_replay_simulations']=replay['n_simulations']
    manifest['runtime_note']='wall_time_s is measured from the last resumed main invocation; cached corpus and preflight jobs are reused. It is not total experiment cost.'
    manifest['files']={str(p.relative_to(OUT)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
        for p in OUT.rglob('*') if p.is_file() and p.name!='run_manifest.json' and 'hydraulic_logs' not in p.parts and p.suffix not in ['.pyc','.bin','.rpt']}
    write_json(OUT/'run_manifest.json',manifest)
    print(pd.DataFrame(readable).to_string(index=False))

if __name__=='__main__':main()
