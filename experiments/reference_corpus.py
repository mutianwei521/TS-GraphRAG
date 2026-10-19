"""Rebuild the 540-scenario reference fingerprints directly from EXA7 hydraulics."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def checked_pressures(result, model, label):
    code=getattr(result,'error_code','missing')
    if code is not None:
        raise RuntimeError(f'Failed hydraulic solve for {label}: error_code={code}')
    pressure=result.node.get('pressure')
    times=np.arange(model.options.time.report_start,model.options.time.duration+1,
                    model.options.time.report_timestep)
    if pressure is None or set(pressure.columns)!=set(model.node_name_list) or not np.array_equal(pressure.index.to_numpy(),times):
        raise RuntimeError(f'Incomplete hydraulic pressure array for {label}')
    if not np.isfinite(pressure.to_numpy()).all():
        raise RuntimeError(f'Nonfinite hydraulic pressure array for {label}')
    return pressure


def rebuild_reference(inp, reference_dir, output_dir, limit=0, fingerprint_file=None):
    import wntr
    from experiments.allocation.build_tensor import hydraulic_workspace

    inp, reference_dir, output_dir = map(lambda p: Path(p).resolve(), (inp, reference_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir/'sensor_fingerprints.json'
    if output.exists():
        raise FileExistsError(f'Use a fresh output directory: {output}')
    fingerprint_file=Path(fingerprint_file).resolve() if fingerprint_file else reference_dir/'sensor_fingerprints.json'
    design = json.loads(fingerprint_file.read_text(encoding='utf-8'))
    partition = json.loads((reference_dir/'partition_summary.json').read_text(encoding='utf-8'))['15']['node_assignments']
    sensors = design['sensor_nodes']
    scenarios = design['fingerprints'][:limit] if limit else design['fingerprints']
    wn = wntr.network.WaterNetworkModel(str(inp))
    counts = {'junctions':len(wn.junction_name_list), 'pipes':len(wn.pipe_name_list),
              'pumps':len(wn.pump_name_list), 'reservoirs':len(wn.reservoir_name_list),
              'tanks':len(wn.tank_name_list)}
    started = time.perf_counter()
    baselines = {}
    rebuilt, comparisons = [], []
    raw_dir = output_dir/'simulation_results'
    raw_dir.mkdir(exist_ok=True)
    with hydraulic_workspace(output_dir) as scratch:
        if scratch:
            for i, scenario in enumerate(scenarios):
                multiplier = scenario['demand_multiplier']
                if multiplier not in baselines:
                    state = copy.deepcopy(wn)
                    for node in state.junction_name_list:
                        for demand in state.get_node(node).demand_timeseries_list:
                            demand.base_value *= multiplier
                    base_result = wntr.sim.EpanetSimulator(copy.deepcopy(state)).run_sim(file_prefix='baseline')
                    base_pressure=checked_pressures(base_result,state,f'baseline x{multiplier}')
                    baselines[multiplier] = (state, base_pressure.mean(axis=0).to_dict())
                state, baseline = baselines[multiplier]
                leak = copy.deepcopy(state)
                leak.get_node(scenario['leak_node']).add_demand(scenario['leak_rate_Ls']/1000., '881')
                result = wntr.sim.EpanetSimulator(leak).run_sim(file_prefix='leak')
                pressure = checked_pressures(result,leak,scenario['scenario_id']).mean(axis=0).to_dict()
                delta = {node:round(pressure[node]-baseline[node],4) for node in baseline
                         if abs(pressure[node]-baseline[node]) > .01}
                affected = [node for node, value in delta.items() if value < -.5]
                vector = [delta.get(node,0.) for node in sensors]
                if not np.isfinite(vector).all():
                    raise ValueError(f'Nonfinite hydraulic fingerprint: {scenario["scenario_id"]}')
                record = dict(scenario)
                record.update(sensor_fingerprint=vector,
                              max_pressure_drop=round(abs(min(delta.values())),4) if delta else 0.,
                              num_affected_nodes=len(affected),
                              affected_partitions=sorted({partition.get(node,-1) for node in affected}))
                rebuilt.append(record)
                (raw_dir/(scenario['scenario_id']+'.json')).write_text(
                    json.dumps({**record,'delta_pressure':delta},indent=2),encoding='utf-8')
                comparisons.append(float(np.max(np.abs(np.asarray(vector)-scenario['sensor_fingerprint']))))
                if (i+1)%90==0:
                    print(f'Rebuilt {i+1}/{len(scenarios)} hydraulic scenarios',flush=True)
    payload = {k:v for k,v in design.items() if k!='fingerprints'}
    payload.update(fingerprints=rebuilt,num_scenarios=len(rebuilt))
    output.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    report = {'source':'EPANET hydraulic simulation','network_counts':counts,
              'simulated_scenarios':len(rebuilt),'baseline_states':len(baselines),
              'sensor_count':len(sensors),'sensor_nodes':sensors,
              'input_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),
              'reference_scenarios_sha256':hashlib.sha256(fingerprint_file.read_bytes()).hexdigest(),
              'max_abs_difference_m':max(comparisons,default=0.),
              'exact_fingerprint_rows':sum(value==0 for value in comparisons),
              'wntr_version':wntr.__version__,'elapsed_seconds':time.perf_counter()-started,
              'solver_checks':{'success_error_code':None,'wntr_error_code_zero_means':'nonconvergence',
                               'all_pressure_arrays_complete_and_finite':True,
                               'checked_leak_solves':len(rebuilt),'checked_baseline_solves':len(baselines)},
              'feature_definition':'24-hour mean leak pressure minus baseline; absolute changes >0.01 m retained and rounded to four decimals',
              'physical_setup':'Frozen 15-partition map, 30 ordered sensors and 540 scenario IDs'}
    (output_dir/'rebuild_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    if report['max_abs_difference_m'] > .00011:
        raise AssertionError('Hydraulic fingerprints differ from the archived reference by more than 0.00011 m; inspect rebuild_report.json')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inp',type=Path,default=ROOT/'data/Exa7.inp')
    parser.add_argument('--reference-dir',type=Path,default=ROOT/'data/reference')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/reference_corpus')
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    if args.limit<0: parser.error('--limit must be nonnegative')
    rebuild_reference(args.inp,args.reference_dir,args.output_dir,args.limit)


if __name__=='__main__':
    main()
