"""Known-answer checks for the portable benchmark calculations."""
import importlib
import csv
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_e4_replay_recomputes_predictions_and_projection(tmp_path):
    module = importlib.import_module('experiments.repeated_benchmarks.run')
    report = module.replay('main90', tmp_path)
    assert report['prediction_rows'] == 18000
    assert report['summary_cells'] == 40
    assert report['projection_queries_verified'] == 2250
    assert report['archive_statistics_match'] is True
    assert report['projection_prediction_mismatches'] == 0


def test_e5_replay_reconstructs_the_fifteen_draws(tmp_path):
    module = importlib.import_module('experiments.allocation.run')
    report = module.replay(tmp_path)
    assert report['run_count'] == 1500
    assert report['actual_seeds'] == list(range(41, 56))
    assert report['tensor_shape'] == [350, 12, 3, 30]
    assert report['sampling_pool_size'] == 326
    assert report['archive_tables_match'] is True
    assert report['G2_H0_equals_H2_all_metrics'] is True


def test_e5_full_one_draw_matches_original_queries(tmp_path):
    module = importlib.import_module('experiments.allocation.run')
    report = module.full(tmp_path, draws=[0], protocols=['H0'], groups=['G2'])
    assert report['run_count'] == 5
    assert report['archive_metrics_match'] is True
    assert report['max_absolute_metric_difference'] < 1e-10


def test_e5_fresh_distance_outputs_use_native_coordinate_units(tmp_path):
    module=importlib.import_module('experiments.allocation.run')
    replay_dir=tmp_path/'replay'
    report=module.replay(replay_dir)
    assert report['distance_units']=='native_network_coordinate_units'
    assert report['archive_metric_aliases']['median_coordinate_distance']=='med_m'
    with (replay_dir/'e5_h0_summary.csv').open(encoding='utf-8',newline='') as handle:
        distances=[r for r in csv.DictReader(handle) if r['metric']=='median_coordinate_distance']
    assert len(distances)==10
    assert {r['units'] for r in distances}=={'native_network_coordinate_units'}
    assert {r['archived_metric'] for r in distances}=={'med_m'}
    with (replay_dir/'e5_h0_draws.csv').open(encoding='utf-8',newline='') as handle:
        draw=next(csv.DictReader(handle))
    assert 'median_error_coordinate_units' in draw and 'median_error_m' not in draw
    stats=json.loads((replay_dir/'all_statistics.json').read_text(encoding='utf-8'))
    assert 'median_coordinate_distance' in stats['summaries']['G2']['H0']['A']
    assert 'med_m' not in stats['summaries']['G2']['H0']['A']
    full_dir=tmp_path/'full'
    full_report=module.full(full_dir,draws=[0],protocols=['H0'],groups=['G2'])
    assert full_report['distance_units']=='native_network_coordinate_units'
    with gzip.open(full_dir/'query_predictions.csv.gz','rt',encoding='utf-8',newline='') as handle:
        query=next(csv.DictReader(handle))
    assert 'distance_coordinate_units' in query and 'distance_m' not in query
    payload=json.loads((full_dir/'expH_robust_results.json').read_text(encoding='utf-8'))
    assert 'median_coordinate_distance' in payload['runs'][0]
    assert 'med_m' not in payload['runs'][0]
    original=json.loads((ROOT/'data/frozen/allocation/expH_robust_results.json').read_text(encoding='utf-8'))
    for fresh in payload['runs']:
        old=next(r for r in original['runs'] if all(r[k]==fresh[k] for k in ['seed','design','group','protocol']))
        assert fresh['median_coordinate_distance']==old['med_m']


def test_historical_replay_recalculates_physics_accuracy(tmp_path):
    module = importlib.import_module('experiments.historical.run')
    report = module.replay(tmp_path)
    assert report['physics_cells'] == 80
    assert report['physics_raw_accuracy_match'] is True
    assert report['llm_mode'] == 'archived statistics only'


def test_historical_projection_uses_original_observations(tmp_path):
    module = importlib.import_module('experiments.historical.physics')
    report = module.run(tmp_path, methods=['projection', 'cosine 1-NN'], protocols=['H0', 'H2', 'H8'])
    assert report['compared_query_predictions'] == 3240
    assert report['archived_prediction_mismatches'] == 0


@pytest.mark.parametrize('entrypoint',['standalone','historical_full'])
def test_nnls_fit_errors_fail_both_clis_after_writing_failed_artifacts(tmp_path,entrypoint):
    output_dir=tmp_path/entrypoint
    program='''
import sys
from experiments.historical import physics, run as historical
def failed_nnls(*args,**kwargs):
    raise RuntimeError('injected NNLS failure')
physics.nnls=failed_nnls
output=sys.argv[1]
if sys.argv[2]=='standalone':
    sys.argv=['physics','--methods','WLS/NNLS','--protocols','H9','--output-dir',output]
    physics.main()
else:
    complete_physics_run=physics.run
    def selected_run(path):
        return complete_physics_run(path,methods=['WLS/NNLS'],protocols=['H9'])
    physics.run=selected_run
    sys.argv=['historical','--profile','full','--output-dir',output]
    historical.main()
'''
    result=subprocess.run([sys.executable,'-c',program,str(output_dir),entrypoint],cwd=ROOT,
                          capture_output=True,text=True)
    assert result.returncode!=0,result.stdout
    physics_dir=output_dir if entrypoint=='standalone' else output_dir/'physics'
    report=json.loads((physics_dir/'full_report.json').read_text(encoding='utf-8'))
    assert report['fit_errors']==540
    assert report['archived_prediction_mismatches']==0
    assert report['status']=='failed'
    assert report['completed'] is False
    assert (physics_dir/'predictions.jsonl.gz').is_file()
    payload=json.loads((physics_dir/'expH_physics_baselines_results.json').read_text(encoding='utf-8'))
    assert payload['fit_errors'][0]['query_index']==0
    assert payload['fit_errors'][0]['protocol']=='H9'


def test_ordinary_nnls_h9_still_completes_without_fit_errors(tmp_path):
    module=importlib.import_module('experiments.historical.physics')
    report=module.run(tmp_path,methods=['WLS/NNLS'],protocols=['H9'])
    assert report['fit_errors']==0
    assert report['archived_prediction_mismatches']==0
    assert report['status']=='complete'
    assert report['completed'] is True


def test_e4_pool_cache_rejects_changed_inputs_or_missing_predictions():
    module=importlib.import_module('experiments.repeated_benchmarks.classical')
    task={'key':'pool','pairs':[['H0',3],['H0',8]]}
    cached={'key':'pool','binding':'old-inputs','query_indices':[3,8]}
    assert module.cache_valid(cached,task,'new-inputs') is False
    cached['binding']='new-inputs'
    assert module.cache_valid(cached,task,'new-inputs') is True
    cached['query_indices']=[3]
    assert module.cache_valid(cached,task,'new-inputs') is False


def test_tensor_workspace_restores_cwd_and_temp_after_failure(tmp_path,monkeypatch):
    module=importlib.import_module('experiments.allocation.build_tensor')
    original=Path.cwd()
    monkeypatch.setenv('TMP','saved-temp')
    monkeypatch.delenv('TEMP',raising=False)
    with pytest.raises(RuntimeError,match='simulation interrupted'):
        with module.hydraulic_workspace(tmp_path) as scratch:
            assert Path.cwd()==scratch
            native_temp=Path(os.environ['TMP'])
            assert native_temp.is_absolute() and native_temp.is_dir()
            assert os.environ['TEMP']==str(native_temp)
            assert os.environ['TMP']!='saved-temp'
            raise RuntimeError('simulation interrupted')
    assert Path.cwd()==original
    assert os.environ['TMP']=='saved-temp'
    assert 'TEMP' not in os.environ
    assert not scratch.exists()
    assert native_temp.is_dir()  # EPANET's native CRT may reuse this process-lifetime path.


def test_tensor_builder_refuses_overwrite_before_simulation(tmp_path,monkeypatch):
    module=importlib.import_module('experiments.allocation.build_tensor')
    target=tmp_path/'expH_robust_tensor.npz'
    target.write_bytes(b'existing evidence')
    import wntr
    def unexpected_simulation(*args,**kwargs):
        pytest.fail('Fresh-output guard must run before loading the hydraulic network')
    monkeypatch.setattr(wntr.network,'WaterNetworkModel',unexpected_simulation)
    with pytest.raises(FileExistsError):
        module.build_tensor(ROOT/'data/Exa7.inp',tmp_path)
    assert target.read_bytes()==b'existing evidence'


def test_tensor_builder_rejects_changed_values_with_matching_axes(tmp_path,monkeypatch):
    import numpy as np
    module=importlib.import_module('experiments.allocation.build_tensor')
    with np.load(ROOT/'data/frozen/allocation/expH_robust_tensor.npz') as archive:
        tensor=archive['T'].copy();tensor.flat[0]+=.1
        nodes=archive['nodes'].tolist();sensors=archive['sensors'].tolist()
    monkeypatch.setattr(module,'_simulate_tensor',lambda *args:(tensor,nodes,sensors))
    with pytest.raises(AssertionError,match='tensor values'):
        module.build_tensor(ROOT/'data/Exa7.inp',tmp_path)
    report=json.loads((tmp_path/'rebuild_report.json').read_text())
    assert all(report['comparison'].values())
    assert report['max_absolute_tensor_difference_m']>.09


def test_classical_limited_fit_is_explicitly_labeled_smoke(tmp_path):
    module=importlib.import_module('experiments.repeated_benchmarks.classical')
    report=module.run('main90',1,tmp_path,max_pools=1)
    assert report['profile']=='smoke'
    assert report['completed'] is False
    assert report['evaluated_pools']==1
    assert report['expected_full_pools']==650
    assert not (tmp_path/'classical_main90_results.json').exists()
