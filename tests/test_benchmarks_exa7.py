"""Executable EXA7 historical-study boundaries and known physical-design checks."""
import importlib
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_resolution_port_keeps_complete_link_classes_and_coordinate_units(tmp_path):
    module=importlib.import_module('experiments.historical.exa7')
    report=module.run_study('resolution',tmp_path,smoke=True)
    assert report['profile']=='smoke' and report['completed_full_study'] is False
    output=json.loads((tmp_path/'expA1a_results.json').read_text(encoding='utf-8'))
    assert {k:v['n'] for k,v in output['classes_deployed'].items()}=={'0.99':39,'0.999':74,'0.9999':127}
    assert output['distance_units']=='native_network_coordinate_units'
    assert output['archive_field_aliases']['class_span_coordinate_units']=='class_span_m'
    assert output['n_classes_0999']==74


def test_sensor_port_runs_actual_hydraulics_with_explicit_smoke_scope(tmp_path):
    module=importlib.import_module('experiments.historical.exa7')
    report=module.run_study('sensor_degradation',tmp_path,smoke=True)
    assert report['profile']=='smoke' and report['completed_full_study'] is False
    assert report['hydraulic_simulations']>0
    assert report['archived_metrics_comparison']=='not applicable to smoke scope'
    result=json.loads((tmp_path/'expA2_results.json').read_text(encoding='utf-8'))
    assert result['n_queries']==4
    assert result['seed']==31
    assert 'median_coordinate_distance' in result['results']['baseline, 30 deployed sensors']['L2']
    assert (tmp_path/'simulation_trace.jsonl.gz').is_file()


def test_exa7_full_refuses_existing_outputs(tmp_path):
    module=importlib.import_module('experiments.historical.exa7')
    (tmp_path/'execution_report.json').write_text('{}',encoding='utf-8')
    import pytest
    with pytest.raises(FileExistsError): module.run_study('resolution',tmp_path)


@pytest.mark.parametrize('invalid',['error_status','nonconverged_zero','internal_nan','missing_time','missing_node'])
def test_exa7_rejects_invalid_hydraulic_grid_before_mean(tmp_path,monkeypatch,invalid):
    module=importlib.import_module('experiments.historical.exa7')
    import wntr
    def invalid_simulation(simulator,*args,**kwargs):
        model=simulator._wn
        times=np.arange(model.options.time.report_start,model.options.time.duration+1,model.options.time.report_timestep)
        pressure=pd.DataFrame(20.,index=times,columns=model.node_name_list)
        code=None
        if invalid=='error_status': code=999
        elif invalid=='nonconverged_zero': code=0
        elif invalid=='internal_nan': pressure.iloc[3,3]=np.nan
        elif invalid=='missing_time': pressure=pressure.iloc[:-1]
        elif invalid=='missing_node': pressure=pressure.iloc[:,:-1]
        return SimpleNamespace(error_code=code,node={'pressure':pressure})
    monkeypatch.setattr(wntr.sim.EpanetSimulator,'run_sim',invalid_simulation)
    with pytest.raises(RuntimeError,match='hydraulic'):
        module.run_study('sensor_degradation',tmp_path,smoke=True)
    report=json.loads((tmp_path/'execution_report.json').read_text(encoding='utf-8'))
    assert report['status']=='failed'
    assert report['hydraulic_simulations']==0
    assert report['completed_full_study'] is False


def test_final_policy_trace_audit_rejects_complete_grid_with_zero_error_code(tmp_path):
    module=importlib.import_module('experiments.historical.audit_exa7_traces')
    import wntr
    model=wntr.network.WaterNetworkModel(str(module.ROOT/'data/Exa7.inp'))
    study_dir=tmp_path/'sensor_degradation';study_dir.mkdir()
    times=np.arange(model.options.time.report_start,model.options.time.duration+1,model.options.time.report_timestep).tolist()
    record=dict(simulation_index=1,solver_error_code=0,report_times=times,
                pressure_grid_shape=[len(times),len(model.node_name_list)],pressure_nodes=model.node_name_list,
                mean_pressure=[20.]*len(model.node_name_list),full_pressure_grid_sha256='a'*64)
    with gzip.open(study_dir/'simulation_trace.jsonl.gz','wt',encoding='utf-8') as handle:
        handle.write(json.dumps(record)+'\n')
    (study_dir/'execution_report.json').write_text(json.dumps(dict(status='complete',completed_full_study=True,
        hydraulic_simulations=1,solver_checks={'validated_simulations':1})),encoding='utf-8')
    with pytest.raises(AssertionError,match='solver status'):
        module.audit(tmp_path,tmp_path/'policy_audit.json',studies=['sensor_degradation'])
