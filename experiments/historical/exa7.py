"""Execute EXA7 emitter, sensor-degradation, resolution and simultaneous-leak studies."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
import gzip
import hashlib
import json
import math
from pathlib import Path
import runpy
import time
import traceback
import warnings

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.allocation.build_tensor import hydraulic_workspace
from experiments.reference_corpus import checked_pressures
from . import exa7_support

ROOT=Path(__file__).resolve().parents[2]
STUDIES={
    'pressure_dependent':('exp3_results.json','exp3_pressure_dependent_leak.py'),
    'sensor_degradation':('expA2_results.json','expA2_sensor_robustness.py'),
    'resolution':('expA1a_results.json','expA1a_fix.py'),
    'multileak_exhaustive':('expA1b_results.json','expA1b_multileak_limit.py'),
    'multileak_sparse':('expA1_v2_results.json','expA1_multileak_v2.py')}
FIELD_ALIASES={'median_coordinate_distance':'med_m','class_span_coordinate_units':'class_span_m',
               'distance_median_coordinate_units':'dist_median'}


def public_result(value):
    inverse={v:k for k,v in FIELD_ALIASES.items()}
    if isinstance(value,dict): return {inverse.get(k,k):public_result(v) for k,v in value.items()}
    if isinstance(value,list): return [public_result(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value): return None
    return value


def compare_results(actual,expected):
    """Compare the original result structure before applying public field aliases."""
    differences=[];numeric_count=0;maximum=0.
    def visit(a,b,path):
        nonlocal numeric_count,maximum
        if isinstance(b,dict):
            if not isinstance(a,dict) or set(a)!=set(b):
                differences.append(dict(path=path,reason='different object fields'));return
            for k in b:visit(a[k],b[k],path+'.'+k)
        elif isinstance(b,list):
            if not isinstance(a,list) or len(a)!=len(b):
                differences.append(dict(path=path,reason='different list length'));return
            for i,(x,y) in enumerate(zip(a,b)):visit(x,y,f'{path}[{i}]')
        elif isinstance(b,(int,float)) and not isinstance(b,bool):
            numeric_count+=1
            if isinstance(a,(int,float)) and math.isnan(a) and math.isnan(b):return
            delta=abs(float(a)-float(b));maximum=max(maximum,delta)
            if not np.isclose(a,b,rtol=1e-8,atol=1e-8,equal_nan=True):
                differences.append(dict(path=path,actual=a,archived=b))
        elif a!=b:differences.append(dict(path=path,actual=a,archived=b))
    visit(actual,expected,'result')
    return dict(match=not differences,numeric_values_compared=numeric_count,
                maximum_absolute_numeric_difference=maximum,differences=differences[:30],difference_count=len(differences))


@threadpool_limits.wrap(limits=1)
def run_study(study,output_dir,*,inp=None,reference_dir=None,frozen_dir=None,sensitivity=None,dense_corpus=None,smoke=False):
    import wntr
    if study not in STUDIES: raise ValueError(f'Unknown EXA7 study: {study}')
    output_dir=Path(output_dir).resolve();output_dir.mkdir(parents=True,exist_ok=True)
    filename,original_script=STUDIES[study]
    report_path=output_dir/'execution_report.json'
    if report_path.exists() or (output_dir/filename).exists(): raise FileExistsError(f'Use a fresh output directory: {output_dir}')
    reference_dir=Path(reference_dir or ROOT/'data/reference').resolve()
    frozen_dir=Path(frozen_dir or ROOT/'data/frozen/historical').resolve()
    source=Path(__file__).with_name('exa7_sources')/(study+'.py')
    inputs=dict(inp=str(Path(inp or ROOT/'data/Exa7.inp').resolve()),reference_dir=str(reference_dir),
                sensitivity=str(Path(sensitivity or reference_dir/'sensitivity_matrix.npz').resolve()),
                dense=str(Path(dense_corpus or frozen_dir/'exp2a_dense_corpus.npz').resolve()),
                held=str(frozen_dir/'exp1b_results.json'))
    paths=[source,Path(__file__),Path(exa7_support.__file__),ROOT/'experiments/reference_corpus.py',
           ROOT/'experiments/allocation/build_tensor.py',Path(inputs['inp']),reference_dir/'sensor_fingerprints.json',
           reference_dir/'partition_summary.json',Path(inputs['sensitivity']),frozen_dir/filename]
    if study=='pressure_dependent':paths.extend([Path(inputs['dense']),Path(inputs['held'])])
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if smoke: inputs.update(N_SCEN=1,N_QUERY=2,COUNTS=[5,30],probe_count=2,dense_count=12,query_count=4)
    report=dict(study=study,profile='smoke' if smoke else 'full',completed_full_study=False,status='running',
                original_script=original_script,input_sha256=hashes,distance_units='native_network_coordinate_units',
                archive_field_aliases=FIELD_ALIASES,hydraulic_simulations=0,
                startpoint=('fixed sensitivity matrix and network coordinates; linear algebra with no fresh hydraulic simulations'
                            if study=='resolution' else 'fixed reference sensor/partition design and sensitivity matrix; fresh EPANET event simulations'),
                isolation='per-run scratch cwd; process-lifetime native TMP/TEMP/TMPDIR synchronized with Windows CRTs and restored on context exit')
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    original_run_sim=wntr.sim.EpanetSimulator.run_sim
    saved_inputs=dict(exa7_support.INPUTS);started=time.perf_counter();count=0
    error=None;raw=None
    try:
        with hydraulic_workspace(output_dir) as scratch, gzip.open(output_dir/'simulation_trace.jsonl.gz','wt',encoding='utf-8') as trace, (output_dir/'execution.log').open('w',encoding='utf-8') as log:
            inputs['epanet_prefix']=str(scratch/'ep')
            exa7_support.INPUTS.clear();exa7_support.INPUTS.update(inputs)
            def traced_simulator(simulator,*args,**kwargs):
                nonlocal count
                result=original_run_sim(simulator,*args,**kwargs)
                grid=checked_pressures(result,simulator._wn,f'{study} simulation {count+1}')
                if grid.shape[1]!=len(simulator._wn.node_name_list):
                    raise RuntimeError(f'Incomplete hydraulic node grid for {study} simulation {count+1}')
                pressure=grid.mean(axis=0)
                count+=1
                record=dict(simulation_index=count,solver_error_code=getattr(result,'error_code',None),
                            report_times=grid.index.tolist(),pressure_grid_shape=list(grid.shape),
                            pressure_nodes=pressure.index.tolist(),mean_pressure=pressure.to_numpy().tolist(),
                            full_pressure_grid_sha256=hashlib.sha256(grid.to_numpy().tobytes()).hexdigest())
                trace.write(json.dumps(record)+'\n')
                if count%100==0: print(f'{study}: {count} hydraulic simulations',flush=True)
                return result
            wntr.sim.EpanetSimulator.run_sim=traced_simulator
            with warnings.catch_warnings(),redirect_stdout(log):
                runpy.run_path(str(source),run_name='__main__')
            raw=json.loads((scratch/filename).read_text(encoding='utf-8'))
    except Exception as exc:
        error=exc;report['error']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    finally:
        wntr.sim.EpanetSimulator.run_sim=original_run_sim
        exa7_support.INPUTS.clear();exa7_support.INPUTS.update(saved_inputs)
    report.update(hydraulic_simulations=count,elapsed_seconds=time.perf_counter()-started)
    report['solver_checks']=dict(success_error_codes=[None],require_finite_full_pressure_grid=True,
                                 require_exact_report_times_and_node_ids=True,validated_before_averaging=True,
                                 validated_simulations=count)
    if raw is not None:
        public=public_result(raw)
        public.update(distance_units='native_network_coordinate_units',archive_field_aliases=FIELD_ALIASES)
        (output_dir/filename).write_text(json.dumps(public,indent=2,allow_nan=False),encoding='utf-8')
        if smoke: report['archived_metrics_comparison']='not applicable to smoke scope'
        else:
            comparison=compare_results(raw,json.loads((frozen_dir/filename).read_text(encoding='utf-8')))
            report['archived_metrics_comparison']=comparison
            if not comparison['match']:
                error=AssertionError(f'{study} differs from the frozen historical results; inspect execution_report.json')
    report.update(status='failed' if error else 'complete',completed_full_study=not smoke and error is None)
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    if error:raise error
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',choices=STUDIES,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--inp',type=Path)
    parser.add_argument('--reference-dir',type=Path)
    parser.add_argument('--frozen-dir',type=Path)
    parser.add_argument('--sensitivity',type=Path)
    parser.add_argument('--dense-corpus',type=Path)
    parser.add_argument('--smoke',action='store_true',help='Explicit reduced workload; never reported as complete full study')
    args=parser.parse_args()
    report=run_study(args.study,args.output_dir,inp=args.inp,reference_dir=args.reference_dir,frozen_dir=args.frozen_dir,
                     sensitivity=args.sensitivity,dense_corpus=args.dense_corpus,smoke=args.smoke)
    print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},indent=2))


if __name__=='__main__':main()
