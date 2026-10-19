from pathlib import Path
import shutil
import numpy as np
import wntr

from experiments.hydraulic_robustness import model as h
from experiments.partition_sensor import model as p


ROOT = Path(__file__).resolve().parents[1]


def test_nominal_perturbator_preserves_all_original_demand_categories(tmp_path, monkeypatch):
    inputs=tmp_path/'inputs'; inputs.mkdir()
    shutil.copyfile(ROOT/'data/Exa7.inp',inputs/'Exa7.inp')
    monkeypatch.setattr(h,'OUT',tmp_path)
    original=wntr.network.WaterNetworkModel(str(inputs/'Exa7.inp'))
    rebuilt, info, pattern=h.perturbed({}, {'name':'nominal','family':'control','value':0.},27101)
    for name in original.junction_name_list:
        before=original.get_node(name).demand_timeseries_list
        after=rebuilt.get_node(name).demand_timeseries_list
        assert [(d.base_value,d.pattern_name) for d in before] == [(d.base_value,d.pattern_name) for d in after]
    np.testing.assert_array_equal(info['pre_spatial_demand_totals_m3s_by_hour'],info['perturbed_demand_totals_m3s_by_hour'])


def test_spatial_perturbations_preserve_hourly_network_demand(tmp_path, monkeypatch):
    inputs=tmp_path/'inputs'; inputs.mkdir()
    shutil.copyfile(ROOT/'data/Exa7.inp',inputs/'Exa7.inp')
    monkeypatch.setattr(h,'OUT',tmp_path)
    _, info, _=h.perturbed({}, {'name':'spatial_demand_0.1','family':'spatial_demand','value':.1},27101)
    np.testing.assert_allclose(info['pre_spatial_demand_totals_m3s_by_hour'],info['perturbed_demand_totals_m3s_by_hour'],rtol=1e-12)


def test_design_split_matches_published_node_counts():
    import json
    mapping=json.loads((ROOT/'data/reference/partition_summary.json').read_text())['15']['node_assignments']
    wn=wntr.network.WaterNetworkModel(str(ROOT/'data/Exa7.inp'))
    train,test=p.stratified_split(wn.junction_name_list,mapping,101)
    assert len(train)==303 and len(test)==78
    assert set(train).isdisjoint(test)
