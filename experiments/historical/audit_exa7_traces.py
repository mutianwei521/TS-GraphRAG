"""Audit saved EXA7 execution traces against the final WNTR success and grid policy."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
EXPECTED_SIMULATIONS={'pressure_dependent':1282,'sensor_degradation':175,'resolution':0,
                      'multileak_exhaustive':81,'multileak_sparse':721}


def audit(input_dir,output,studies=None):
    import wntr
    input_dir=Path(input_dir).resolve();output=Path(output).resolve()
    model=wntr.network.WaterNetworkModel(str(ROOT/'data/Exa7.inp'))
    nodes=model.node_name_list
    times=np.arange(model.options.time.report_start,model.options.time.duration+1,
                    model.options.time.report_timestep).tolist()
    records=[];errors=[];count=0
    for study in studies or EXPECTED_SIMULATIONS:
        directory=input_dir/study
        report_path=directory/'execution_report.json';trace_path=directory/'simulation_trace.jsonl.gz'
        report=json.loads(report_path.read_text(encoding='utf-8'))
        n=0
        with gzip.open(trace_path,'rt',encoding='utf-8') as handle:
            for line in handle:
                record=json.loads(line);n+=1
                if 'solver_error_code' not in record or record['solver_error_code'] is not None:
                    errors.append(f'Invalid solver status in {study} simulation {n}: {record.get("solver_error_code","missing")}')
                if record.get('simulation_index')!=n: errors.append(f'Invalid simulation index in {study}: {n}')
                if record.get('report_times')!=times: errors.append(f'Incomplete report-time grid in {study}: {n}')
                if record.get('pressure_grid_shape')!=[len(times),len(nodes)]: errors.append(f'Incomplete pressure-grid shape in {study}: {n}')
                pressure_nodes=record.get('pressure_nodes',[])
                if len(pressure_nodes)!=len(nodes) or set(pressure_nodes)!=set(nodes): errors.append(f'Incomplete node grid in {study}: {n}')
                mean=np.asarray(record.get('mean_pressure',[]),dtype=float)
                if mean.shape!=(len(nodes),) or not np.isfinite(mean).all(): errors.append(f'Invalid mean-pressure vector in {study}: {n}')
                if re.fullmatch(r'[0-9a-f]{64}',record.get('full_pressure_grid_sha256','')) is None:
                    errors.append(f'Missing full-grid SHA-256 in {study}: {n}')
        if report.get('status')!='complete' or not report.get('completed_full_study'):
            errors.append(f'Original run was not complete: {study}')
        if n!=report.get('hydraulic_simulations') or n!=EXPECTED_SIMULATIONS[study]:
            errors.append(f'Incomplete simulation count in {study}: {n}')
        if report.get('solver_checks',{}).get('validated_simulations')!=n:
            errors.append(f'Original validation count differs in {study}: {n}')
        records.append(dict(study=study,hydraulic_simulations=n,
                            execution_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                            trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                            original_run_input_sha256=report.get('input_sha256',{}),
                            original_solver_policy=report.get('solver_checks',{}),
                            original_numeric_comparison=report.get('archived_metrics_comparison')))
        count+=n
    sources=[Path(__file__),ROOT/'experiments/reference_corpus.py',ROOT/'experiments/allocation/build_tensor.py',ROOT/'data/Exa7.inp']
    result=dict(status='failed' if errors else 'complete',success_error_code=None,hydraulic_simulations=count,
                studies=records,errors=errors,
                final_policy_sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                trace_checks='Every stored solver status is None; report-time/node grids, shapes, finite means, simulation counts and grid-hash format are checked independently',
                full_grid_hash_scope='The original execution validated unaveraged matrices before hashing; this audit verifies hash presence/format, not a recomputation from raw grids, which are not serialized',
                isolation='Per-run scratch cwd; process-lifetime native TMP/TEMP/TMPDIR synchronized with Windows CRTs and restored on context exit',
                provenance='Original execution reports and their source bindings are preserved unchanged; this is a separate audit under the final policy')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    if errors: raise AssertionError(errors[0])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.input_dir,args.output)
    print(json.dumps(dict(status=result['status'],hydraulic_simulations=result['hydraulic_simulations'],studies=len(result['studies']))))


if __name__=='__main__':main()
