"""Frozen EXA7 hydraulic robustness evaluation."""
from pathlib import Path
import os, sys, json, copy, hashlib, time, csv, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('TS_STUDY_OUTPUT', ROOT/'outputs/hydraulic_robustness')).resolve()
sys.path.insert(0, str(ROOT))
import wntr
import pickle

SEEDS = [27101, 27102, 27103, 27104, 27105]
PERIODS = [('night_low', .7), ('day_normal', 1.), ('evening_peak', 1.2)]
RATES = [5., 20.]
NOISE = .02
SCENARIOS = [{'name': 'nominal', 'family': 'control', 'value': 0.}]
for fam, vals in [('roughness', [.05,.10,.20]), ('reservoir_head', [.5,1.,2.]),
                  ('pump_speed', [.01,.03,.05]), ('spatial_demand', [.05,.10,.20]),
                  ('spatiotemporal_demand', [.05,.10,.20]), ('phase_hours', [1.,3.,6.]),
                  ('uniform_demand', [.7,.85,1.10,1.2,1.3])]:
    SCENARIOS += [dict(name=f'{fam}_{v:g}', family=fam, value=v) for v in vals]
SCENARIOS += [dict(name='joint_mild', family='joint', value=.5),
              dict(name='joint_moderate', family='joint', value=1.)]

def dump(path, x):
    path.write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def setup():
    for d in ['inputs','models','cache','hydraulic_logs','qa']:
        (OUT/d).mkdir(parents=True,exist_ok=True)
    if (OUT/'cache/nominal_corpus.npz').exists() and (OUT/'design.json').exists():
        frozen=json.loads((OUT/'design.json').read_text(encoding='utf-8'))
        if frozen['seeds']!=SEEDS or frozen['scenarios']!=SCENARIOS:
            raise RuntimeError('Frozen design differs from requested grid; use a fresh output directory.')
        for relative in ['data/Exa7.inp','data/frozen/historical/exp1b_results.json','data/reference/partitions.pkl']:
            source=ROOT/relative
            if sha(source)!=sha(OUT/'inputs'/source.name):
                raise RuntimeError(f'Input changed: {relative}; use a fresh output directory.')
        return frozen
    held = json.loads((ROOT/'data/frozen/historical/exp1b_results.json').read_text())['config']['held_out_nodes']
    queries = [dict(query_id=f'n{n}_q{q:g}', node=n, partition=int(p), rate_Lps=q)
               for p, ns in held.items() for n in ns for q in RATES]
    sensors = [str(s) for s in np.load(ROOT/'data/frozen/allocation/expH_robust_tensor.npz')['sensors']]
    wn = wntr.network.WaterNetworkModel(str(ROOT/'data/Exa7.inp'))
    design = dict(protocol='Frozen nominal EXA7 library; fixed unseen query locations',
        primary_library_policy='nominal_day', primary_method='projection',
        seeds=SEEDS, scenarios=SCENARIOS, queries=queries, sensors=sensors,
        corpus_reference_Lps=20., corpus_periods=PERIODS,
        corpus_nodes=[n for n in wn.junction_name_list if n not in {q['node'] for q in queries}],
        excluded_query_nodes=sorted({q['node'] for q in queries}, key=int),
        noise_sd_m=NOISE, duration_s=86400, reporting_times='0, 3600, ..., 86400 (25 snapshots)',
        feature='mean of 25 report-time sensor pressures; leak-minus-baseline plus paired Gaussian noise',
        leak_model='additional demand at 5 or 20 L/s multiplied by unchanged original pattern 881',
        admissibility='finite 25x381 pressures, WNTR error_code None, no WARNING in report; every junction pressure >=0 and <=200 m at every report time',
        corpus_admissibility='discard nominal template if pressure check fails, before reading query outcomes; record every discard',
        query_failure_policy='keep every planned query; inadmissible/failed cases score zero in all-query accuracy; admissible-only rate reported separately',
        baseline_modes={'matched':'leak and no-leak pressures under exactly the same perturbed model and demand schedule (ideal contemporaneous baseline)',
                        'stale_nominal':'perturbed leak pressure minus nominal day-normal no-leak pressure (operating-state/baseline drift)'},
        library_policies={'nominal_day':'only day-normal templates (misclassified day condition for true low/peak demand)',
                          'all_nominal_periods':'union of the three nominal periods, no true-period information',
                          'known_nearest_period':'nearest of 0.7/1.0/1.2 to the true uniform multiplier; privileged metadata control'},
        methods=['projection', 'cosine_1NN', 'euclidean_1NN'],
        roughness='independent pipewise lognormal C multiplier, specified coefficient of variation and expectation one',
        reservoir_head='all reservoir heads changed by +/- listed metres; independent seeded sign',
        pump_speed='all 3 existing pump base speeds changed by same +/- fractional amount; independent seeded sign',
        spatial_demand='positive lognormal nodal demand factors from RBF-correlated Gaussian field; target lognormal CV; demand-weighted normalization preserves total nominal demand',
        spatiotemporal_demand='same spatial kernel plus temporal AR(1), rho=0.8, hourly; per-hour demand-weighted normalization preserves nominal total demand',
        spatial_length='0.25 of maximum coordinate span; Euclidean distance in native network coordinate units',
        phase_hours='all nodal demand patterns cyclically shifted by +/- listed hours using first 24 hours, repeat first value at hour 24; leak pattern remains unchanged',
        joint='roughness CV 0.10, reservoir +/-1 m, pumps +/-3%, spatiotemporal demand CV 0.10, phase +/-3 h; mild multiplies all amplitudes by 0.5 (phase rounded to 2 h)',
        inferential_unit='5 complete independent perturbation/noise realizations; fixed 60-query set; noise paired across scenarios, methods and baseline modes',
        design_scope=['one EXA7 network and fixed query split', 'prescribed-demand leakage', 'no valves or PRVs present; no valve/PRV/topology perturbation',
                     'synthetic perturbation magnitudes are stress-test settings, not calibrated operational uncertainty distributions',
                     '24-hour averages do not establish event-time localization or SCADA temporal noise realism',
                     'frozen projection/cosine/Euclidean retrieval only, not optional LLM/fusion engine']),
    design=design[0] if isinstance(design,tuple) else design
    dump(OUT/'design.json', design)
    for f in ['data/Exa7.inp','data/frozen/historical/exp1b_results.json','data/reference/partitions.pkl']:
        p=ROOT/f
        (OUT/'inputs'/p.name).write_bytes(p.read_bytes())
    dump(OUT/'inputs/source_hashes.json',{f:sha(ROOT/f) for f in [
        'data/Exa7.inp','data/frozen/historical/exp1b_results.json','data/reference/partitions.pkl',
        'data/frozen/allocation/expH_robust_tensor.npz','data/reference/sensor_placement.csv']})
    return design

def sim(wn, label):
    prefix=OUT/'hydraulic_logs'/label
    start=time.time()
    try:
        with warnings.catch_warnings(record=True) as wmsg:
            warnings.simplefilter('always')
            r=wntr.sim.EpanetSimulator(copy.deepcopy(wn)).run_sim(file_prefix=str(prefix))
        p=r.node['pressure'][wn.junction_name_list].values.astype('float32')
        report=Path(str(prefix)+'.rpt').read_text(errors='replace')
        warn_lines=[s.strip() for s in report.splitlines() if 'WARNING' in s.upper()]
        finite=bool(np.isfinite(p).all())
        complete=p.shape==(25,len(wn.junction_name_list))
        convergence_warnings=[s for s in warn_lines if 'unbalanced' in s.lower() or 'unstable' in s.lower()]
        converged=r.error_code is None and complete and finite and not convergence_warnings
        mn=float(np.min(p)) if finite else None
        mx=float(np.max(p)) if finite else None
        status=dict(label=label,success=True,converged=converged,error_code=r.error_code,
            pressure_min_m=mn,pressure_max_m=mx,negative_pressure=bool(mn is not None and mn<0),
            admissible=bool(converged and not warn_lines and mn>=0 and mx<=200),report_warnings=warn_lines,
            python_warnings=[str(z.message) for z in wmsg],elapsed_s=time.time()-start)
    except Exception as exc:
        p=np.full((25,len(wn.junction_name_list)),np.nan,dtype='float32')
        status=dict(label=label,success=False,converged=False,error_code=str(exc),pressure_min_m=None,
                    pressure_max_m=None,negative_pressure=False,admissible=False,
                    report_warnings=[],python_warnings=[],elapsed_s=time.time()-start)
    # INP and RPT persist for every solve; binary repeats are not needed after arrays are saved.
    for suffix in ['.bin','.hyd']:
        f=Path(str(prefix)+suffix)
        if f.exists(): f.unlink()
    return p,status

def nominal(wn, mult):
    w=copy.deepcopy(wn)
    for _,j in w.junctions():
        for d in j.demand_timeseries_list: d.base_value*=mult
    return w

def build_corpus(design):
    path=OUT/'cache/nominal_corpus.npz'
    signature=dict(inp_sha256=sha(OUT/'inputs/Exa7.inp'),corpus_nodes=design['corpus_nodes'],
                   sensors=design['sensors'],periods=design['corpus_periods'],reference_Lps=20.)
    provenance=OUT/'cache/nominal_corpus_provenance.json'
    if path.exists():
        if provenance.exists() and json.loads(provenance.read_text())!=signature:
            raise RuntimeError('Existing corpus input signature differs; use a fresh output directory.')
        if not provenance.exists():dump(provenance,signature)
        return
    w0=wntr.network.WaterNetworkModel(str(OUT/'inputs/Exa7.inp'))
    junctions=w0.junction_name_list
    si=[junctions.index(s) for s in design['sensors']]
    allp=[]; baseline=[]; atoms=[]; status=[]; F=[]; keep=[]
    for pname,mult in PERIODS:
        w=nominal(w0,mult)
        bp, bs=sim(w,'corpus_'+pname+'_baseline')
        baseline.append(bp)
        status.append(bs)
        for ni,n in enumerate(design['corpus_nodes']):
            z=copy.deepcopy(w); z.get_node(n).add_demand(.02,'881')
            p,st=sim(z,f'corpus_{pname}_n{n}')
            allp.append(p); status.append(st)
            atoms.append(dict(node=n,period=pname,multiplier=mult,rate_Lps=20.,hydraulic_label=st['label']))
            F.append(p[:,si].mean(0)-bp[:,si].mean(0))
            keep.append(bs['admissible'] and st['admissible'])
            if (ni+1)%100==0: print('corpus',pname,ni+1,flush=True)
    np.savez_compressed(path,pressures=np.array(allp),baseline=np.array(baseline),F=np.array(F),
                       keep=np.array(keep),junctions=np.array(junctions),sensors=np.array(design['sensors']))
    dump(OUT/'cache/nominal_atoms.json',atoms); dump(OUT/'cache/nominal_status.json',status)
    dump(provenance,signature)

def correlated_fields(wn,seed):
    rng=np.random.default_rng(seed)
    xy=np.array([wn.get_node(n).coordinates for n in wn.junction_name_list])
    length=.25*np.ptp(xy,axis=0).max()
    d=((xy[:,None,:]-xy[None,:,:])**2).sum(2)
    chol=np.linalg.cholesky(np.exp(-d/(2*length**2))+np.eye(len(xy))*1e-8)
    spatial=chol@rng.normal(size=len(xy))
    fields=[]; prev=chol@rng.normal(size=len(xy))
    for t in range(25):
        if t: prev=.8*prev+np.sqrt(1-.8**2)*(chol@rng.normal(size=len(xy)))
        fields.append(prev.copy())
    return spatial,np.array(fields).T,length

def perturbed(design,scenario,seed):
    w=wntr.network.WaterNetworkModel(str(OUT/'inputs/Exa7.inp'))
    fam,v=scenario['family'],scenario['value']
    rng=np.random.default_rng(seed)
    pipe_z=rng.normal(size=len(w.pipe_name_list))
    head_sign=int(rng.choice([-1,1])); pump_sign=int(rng.choice([-1,1])); phase_sign=int(rng.choice([-1,1]))
    spatial,temporal,length=correlated_fields(w,seed+100000)
    c=0.; head=0.; speed=0.; ds=0.; dt=0.; shift=0; uniform=1.
    if fam=='roughness':c=v
    if fam=='reservoir_head':head=v
    if fam=='pump_speed':speed=v
    if fam=='spatial_demand':ds=v
    if fam=='spatiotemporal_demand':dt=v
    if fam=='phase_hours':shift=int(v)
    if fam=='uniform_demand':uniform=v
    if fam=='joint':c=.1*v;head=v;speed=.03*v;dt=.1*v;shift=int(np.floor(3*v+.5))
    rough_mult=np.exp(np.sqrt(np.log1p(c*c))*pipe_z-.5*np.log1p(c*c))
    for n,m in zip(w.pipe_name_list,rough_mult):w.get_link(n).roughness*=float(m)
    for _,r in w.reservoirs():r.base_head+=head_sign*head
    for _,p in w.pumps():p.base_speed*=1+pump_sign*speed
    base_pattern=np.array(w.get_pattern('881').multipliers)
    leak_pattern='audit_leak_nominal'
    w.add_pattern(leak_pattern,base_pattern)
    # Preserve every original demand category. In this INP, a blank pattern
    # means constant demand; it must not acquire pattern 881 in the control.
    category_patterns={}
    demand_time=np.zeros((len(w.junction_name_list),25))
    for i,n in enumerate(w.junction_name_list):
        category_patterns[n]=[]
        for d in w.get_node(n).demand_timeseries_list:
            orig=(np.array(w.get_pattern(d.pattern_name).multipliers)
                  if d.pattern_name else np.ones(25))
            if shift:
                shifted=np.roll(orig[:24],phase_sign*shift)
                pat=np.append(shifted,shifted[0])
            else:pat=orig.copy()
            category_patterns[n].append(pat)
            demand_time[i]+=d.base_value*pat
    coeff=np.exp(np.sqrt(np.log1p(ds*ds))*spatial[:,None]+np.sqrt(np.log1p(dt*dt))*temporal)
    if ds or dt: coeff/=np.sum(coeff*demand_time,axis=0)[None,:]/demand_time.sum(axis=0)[None,:]
    else:coeff=np.ones_like(demand_time)
    for i,n in enumerate(w.junction_name_list):
        for category,d in enumerate(w.get_node(n).demand_timeseries_list):
            d.base_value*=uniform
            if ds or dt or shift:
                name=f'audit_demand_{n}_category{category}'
                w.add_pattern(name,category_patterns[n][category]*coeff[i])
                d.pattern_name=name
    info=dict(scenario=scenario,seed=seed,roughness_multipliers=dict(zip(w.pipe_name_list,rough_mult.tolist())),
              reservoir_head_delta_m=head_sign*head,pump_speed_multiplier=1+pump_sign*speed,
              phase_shift_hours=phase_sign*shift,uniform_demand_multiplier=uniform,spatial_length=length,
              demand_factors=coeff.tolist(),realized_demand_factor_cv_by_hour=(coeff.std(0)/coeff.mean(0)).tolist(),
              nominal_leak_pattern=base_pattern.tolist(),
              pre_spatial_demand_totals_m3s_by_hour=demand_time.sum(axis=0).tolist(),
              perturbed_demand_totals_m3s_by_hour=(uniform*np.sum(coeff*demand_time,axis=0)).tolist(),
              demand_pattern_policy='Preserve original pattern of each demand category; blank pattern is constant; alter patterns only for explicitly spatial/temporal scenarios')
    return w,info,leak_pattern

def query_job(payload):
    # EPANET's native temporary hydraulics files can collide across processes
    # sharing one cwd on Windows, despite distinct public file_prefix values.
    # Isolate each process's working directory and native temporary directory.
    scratch=OUT/'worker_scratch'/str(os.getpid())
    scratch.mkdir(parents=True,exist_ok=True)
    os.chdir(scratch)
    os.environ['TMP']=str(scratch)
    os.environ['TEMP']=str(scratch)
    scenario,seed=payload
    design=json.loads((OUT/'design.json').read_text(encoding='utf-8'))
    name=f"{scenario['name']}_seed{seed}"
    cache=OUT/'cache'/f'{name}.npz'
    if cache.exists() and (OUT/'cache'/f'{name}_status.json').exists():return name
    w,info,leak_pattern=perturbed(design,scenario,seed)
    dump(OUT/'models'/f'{name}_parameters.json',info)
    wntr.network.write_inpfile(w,str(OUT/'models'/f'{name}.inp'))
    baseline,bs=sim(w,name+'_baseline')
    if scenario['family']=='control' or (scenario['family']=='uniform_demand' and scenario['value'] in [.7,1.2]):
        ci=1 if scenario['family']=='control' else (0 if scenario['value']==.7 else 2)
        control=np.load(OUT/'cache/nominal_corpus.npz')['baseline'][ci]
        if not np.array_equal(baseline,control):
            raise RuntimeError(f'Control identity failed for {name}: {np.nanmax(np.abs(baseline-control))}')
    pressures=[]; status=[bs]
    for q in design['queries']:
        z=copy.deepcopy(w); z.get_node(q['node']).add_demand(q['rate_Lps']/1000,leak_pattern)
        p,st=sim(z,name+'_'+q['query_id'])
        pressures.append(p); status.append(st)
    repeat,repeat_status=sim(w,name+'_baseline_repeat')
    if not np.array_equal(baseline,repeat,equal_nan=True):
        raise RuntimeError(f'Baseline changed during batch: {name}')
    dump(OUT/'qa'/f'{name}_repeat_baseline.json',dict(identical=True,status=repeat_status))
    np.savez_compressed(cache,baseline=baseline,pressures=np.array(pressures))
    dump(OUT/'cache'/f'{name}_status.json',status)
    return name

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def evaluate(design):
    z=np.load(OUT/'cache/nominal_corpus.npz')
    atoms=json.loads((OUT/'cache/nominal_atoms.json').read_text())
    partitions=pickle.loads((OUT/'inputs/partitions.pkl').read_bytes())[15]['node_to_community']
    sensors=design['sensors'];junctions=z['junctions'].tolist();si=[junctions.index(s) for s in sensors]
    frozenF=z['F'].astype(float);nonzero=np.linalg.norm(frozenF,axis=1)>1e-12
    eligible=z['keep']&nonzero
    wn=wntr.network.WaterNetworkModel(str(OUT/'inputs/Exa7.inp'))
    xy={n:np.array(wn.get_node(n).coordinates) for n in junctions}
    n2p={n:int(partitions[n]) for n in junctions}
    assert set(design['excluded_query_nodes']).isdisjoint({a['node'] for a in atoms})
    summary=[];rows=[]; hyd=[]
    for seed in SEEDS:
        noise=np.random.default_rng(seed+200000).normal(0,NOISE,(len(design['queries']),len(sensors)))
        np.save(OUT/'inputs'/f'noise_seed{seed}.npy',noise)
        for scenario in SCENARIOS:
            physics_seed=SEEDS[0] if scenario['family'] in ['control','uniform_demand'] else seed
            name=f"{scenario['name']}_seed{physics_seed}"
            d=np.load(OUT/'cache'/f'{name}.npz')
            sts=json.loads((OUT/'cache'/f'{name}_status.json').read_text())
            bp=d['baseline'][:,si].mean(0)
            lp=d['pressures'][:,:,si].mean(1)
            stale=z['baseline'][1][:,si].mean(0)
            closest=min(PERIODS,key=lambda item:abs(item[1]-(scenario['value'] if scenario['family']=='uniform_demand' else 1.)))[0]
            for qi,q in enumerate(design['queries']):
                st=sts[qi+1];bs=sts[0]
                admissible=st['admissible'] and bs['admissible']
                hyd.append(dict(seed=seed,physics_seed=physics_seed,scenario=scenario['name'],query_id=q['query_id'],
                    query_admissible=admissible,leak_admissible=st['admissible'],baseline_admissible=bs['admissible'],
                    negative_pressure=st['negative_pressure'] or bs['negative_pressure'],
                    converged=st['converged'] and bs['converged'],pressure_min_m=st['pressure_min_m'],
                    baseline_pressure_min_m=bs['pressure_min_m'],pressure_max_m=st['pressure_max_m'],
                    report_warnings=' | '.join(st['report_warnings']+bs['report_warnings'])))
                for baseline,b in [('matched',bp),('stale_nominal',stale)]:
                    y=lp[qi]-b+noise[qi]
                    for policy in ['nominal_day','all_nominal_periods','known_nearest_period']:
                        ids=np.flatnonzero(eligible & np.array([policy=='all_nominal_periods' or a['period']==('day_normal' if policy=='nominal_day' else closest) for a in atoms]))
                        F=frozenF[ids];norm=np.linalg.norm(F,axis=1)
                        scores={}
                        if np.isfinite(y).all() and len(ids):
                            gain=(F@y)/(norm**2)
                            scores['projection']=np.linalg.norm(y[None,:]-gain[:,None]*F,axis=1)
                            scores['cosine_1NN']=-(F@y)/norm/max(np.linalg.norm(y),1e-12)
                            scores['euclidean_1NN']=np.linalg.norm(y[None,:]-F,axis=1)
                        for method in design['methods']:
                            pred_node='';pred_part=-1;top1=top3=mrr=0.;dist=None;rank=[]
                            if method in scores:
                                order=np.argsort(scores[method],kind='stable')
                                pred_node=atoms[ids[order[0]]]['node'];pred_part=n2p[pred_node]
                                for oi in order:
                                    pp=n2p[atoms[ids[oi]]['node']]
                                    if pp not in rank:rank.append(pp)
                                top1=int(pred_part==q['partition']);top3=int(q['partition'] in rank[:3])
                                mrr=1./(rank.index(q['partition'])+1) if q['partition'] in rank else 0.
                                dist=float(np.linalg.norm(xy[pred_node]-xy[q['node']]))
                            rows.append(dict(seed=seed,physics_seed=physics_seed,scenario=scenario['name'],family=scenario['family'],
                                strength=scenario['value'],baseline=baseline,library_policy=policy,method=method,
                                **q,n_candidates=len(ids),admissible=admissible,converged=st['converged'] and bs['converged'],
                                negative_pressure=st['negative_pressure'] or bs['negative_pressure'],pred_node=pred_node,pred_partition=pred_part,
                                top1_all=int(top1 and admissible),top3_all=int(top3 and admissible),mrr_all=mrr if admissible else 0.,
                                top1_raw=top1,top3_raw=top3,mrr_raw=mrr,coordinate_distance=dist,
                                sensor_signal_norm_m=float(np.linalg.norm(lp[qi]-bp)),
                                baseline_drift_norm_m=float(np.linalg.norm(bp-stale))))
            print('evaluate',seed,scenario['name'],flush=True)
    write_csv(OUT/'query_metrics.csv',rows);write_csv(OUT/'query_hydraulic_status.csv',hyd)
    import pandas as pd
    df=pd.DataFrame(rows)
    keys=['seed','scenario','family','strength','baseline','library_policy','method']
    for key,g in df.groupby(keys,sort=False):
        ok=g[g.admissible]
        summary.append(dict(zip(keys,key))|dict(n_queries=len(g),n_admissible=len(ok),
            top1_all_pct=100*g.top1_all.mean(),top3_all_pct=100*g.top3_all.mean(),mrr_all=g.mrr_all.mean(),
            top1_admissible_pct=100*ok.top1_raw.mean() if len(ok) else None,
            top3_admissible_pct=100*ok.top3_raw.mean() if len(ok) else None,
            median_coordinate_distance=ok.coordinate_distance.median() if len(ok) else None,
            failure_pct=100*(~g.admissible).mean(),negative_pressure_pct=100*g.negative_pressure.mean(),
            nonconvergence_pct=100*(~g.converged).mean(),baseline_drift_norm_m=g.baseline_drift_norm_m.mean()))
    write_csv(OUT/'seed_summary.csv',summary)
    sdf=pd.DataFrame(summary); final=[]
    for key,g in sdf.groupby(keys[1:],sort=False):
        item=dict(zip(keys[1:],key));item['n_realizations']=len(g)
        for metric in ['top1_all_pct','top3_all_pct','mrr_all','top1_admissible_pct','top3_admissible_pct',
                       'median_coordinate_distance','failure_pct','negative_pressure_pct','nonconvergence_pct','baseline_drift_norm_m']:
            vals=g[metric].dropna();item[metric+'_mean']=vals.mean() if len(vals) else None
            item[metric+'_sd']=vals.std(ddof=1) if len(vals)>1 else None
        base=sdf[(sdf.scenario=='nominal')&(sdf.baseline==key[3])&(sdf.library_policy==key[4])&(sdf.method==key[5])]
        pairs=g.merge(base,on='seed',suffixes=('','_control'))
        delta=pairs.top1_all_pct-pairs.top1_all_pct_control
        item['paired_top1_delta_pp_mean']=delta.mean();item['paired_top1_delta_pp_sd']=delta.std(ddof=1)
        final.append(item)
    write_csv(OUT/'summary.csv',final)
    allstatus=json.loads((OUT/'cache/nominal_status.json').read_text())
    dump(OUT/'qa/corpus_audit.json',dict(planned_templates=len(atoms),eligible_templates=int(eligible.sum()),
        inadmissible_templates=int((~z['keep']).sum()),zero_response_eligible_templates=int((z['keep']&~nonzero).sum()),
        query_node_overlap=0,partitions_retained=sorted({n2p[atoms[i]['node']] for i in np.flatnonzero(eligible)}),
        baseline_status=[s for s in allstatus if s['label'].endswith('_baseline')]))

def main():
    start=time.time();design=setup();build_corpus(design)
    jobs=[(s,SEEDS[0]) for s in SCENARIOS if s['family'] in ['control','uniform_demand']]
    jobs += [(s,seed) for s in SCENARIOS if s['family'] not in ['control','uniform_demand'] for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(query_job,j) for j in jobs]
        for i,f in enumerate(as_completed(futures),1):print('simulations',i,'/',len(jobs),f.result(),f'{time.time()-start:.0f}s',flush=True)
    evaluate(design)
    files={str(p.relative_to(OUT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in OUT.rglob('*') if p.is_file() and p.name!='run_manifest.json' and p.suffix not in ['.inp','.rpt']}
    dump(OUT/'run_manifest.json',dict(status='completed',wall_time_s=time.time()-start,wntr_version=wntr.__version__,
        python=sys.version,root_input_sha256=sha(ROOT/'data/Exa7.inp'),n_physics_jobs=len(jobs),
        planned_epanet_simulations=len(jobs)*61+len(design['corpus_nodes'])*3+3,
        n_independent_realizations=5,n_queries_per_realization=60,files=files))
    print('DONE',time.time()-start,flush=True)

if __name__=='__main__':main()
