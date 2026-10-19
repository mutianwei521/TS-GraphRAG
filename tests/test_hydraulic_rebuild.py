import json
from pathlib import Path

import numpy as np
import pytest

from experiments.reference_corpus import rebuild_reference


ROOT = Path(__file__).resolve().parents[1]


def test_wntr_error_status_zero_rejected_even_with_complete_finite_grid():
    import pandas as pd
    import wntr
    from types import SimpleNamespace
    from experiments.reference_corpus import checked_pressures
    wn=wntr.network.WaterNetworkModel(str(ROOT/'data/Exa7.inp'))
    result=SimpleNamespace(error_code=0,node={'pressure':pd.DataFrame(20.,
        index=np.arange(0,86401,3600),columns=wn.node_name_list)})
    with pytest.raises(RuntimeError,match='error_code=0'):
        checked_pressures(result,wn,'WNTR nonconvergence')


def test_reference_rebuild_from_network_preserves_sensor_response(tmp_path):
    report = rebuild_reference(ROOT/'data/Exa7.inp', ROOT/'data/reference', tmp_path, limit=1)
    original = json.loads((ROOT/'data/reference/sensor_fingerprints.json').read_text())
    rebuilt = json.loads((tmp_path/'sensor_fingerprints.json').read_text())
    assert report['source'] == 'EPANET hydraulic simulation'
    assert report['simulated_scenarios'] == 1
    assert report['network_counts'] == {'junctions':381,'pipes':469,'pumps':3,'reservoirs':1,'tanks':1}
    assert rebuilt['sensor_nodes'] == original['sensor_nodes']
    assert rebuilt['fingerprints'][0]['scenario_id'] == original['fingerprints'][0]['scenario_id']
    np.testing.assert_allclose(rebuilt['fingerprints'][0]['sensor_fingerprint'],
                               original['fingerprints'][0]['sensor_fingerprint'], atol=0.0001, rtol=0)


def test_rebuild_rejects_existing_outputs(tmp_path):
    (tmp_path/'sensor_fingerprints.json').write_text('{}')
    import pytest
    with pytest.raises(FileExistsError):
        rebuild_reference(ROOT/'data/Exa7.inp', ROOT/'data/reference', tmp_path, limit=1)


def test_reference_hydraulics_isolate_native_temporary_directory(tmp_path,monkeypatch):
    import os
    import wntr
    original=wntr.sim.EpanetSimulator.run_sim
    before={name:os.environ.get(name) for name in ['TMP','TEMP','TMPDIR']}
    starting=Path.cwd()
    seen=[]
    def isolated(simulator,*args,**kwargs):
        from experiments.allocation.build_tensor import native_temporary_directory
        scratch=Path.cwd()
        assert scratch.is_relative_to(tmp_path/'epanet_work')
        for name in before:assert Path(os.environ[name])==native_temporary_directory()
        seen.append(scratch)
        return original(simulator,*args,**kwargs)
    monkeypatch.setattr(wntr.sim.EpanetSimulator,'run_sim',isolated)
    rebuild_reference(ROOT/'data/Exa7.inp',ROOT/'data/reference',tmp_path,limit=1)
    assert len(seen)==2
    assert Path.cwd()==starting
    assert before=={name:os.environ.get(name) for name in before}


@pytest.mark.parametrize('error_code,incomplete',[(110,False),(None,False),(0,True)])
def test_allocation_rejects_invalid_hydraulic_arrays(tmp_path,monkeypatch,error_code,incomplete):
    import pandas as pd
    import wntr
    from types import SimpleNamespace
    from experiments.allocation.build_tensor import _simulate_tensor
    wn=wntr.network.WaterNetworkModel(str(ROOT/'data/Exa7.inp'))
    bad=SimpleNamespace(error_code=error_code,node={'pressure':pd.DataFrame(1. if incomplete else np.nan,
        index=np.arange(0,82801 if incomplete else 86401,3600),columns=wn.node_name_list)})
    monkeypatch.setattr(wntr.sim.EpanetSimulator,'run_sim',lambda *args,**kwargs:bad)
    with pytest.raises(RuntimeError,match='hydraulic'):
        _simulate_tensor(ROOT/'data/Exa7.inp',['34'],tmp_path/'ep')


@pytest.mark.parametrize('error_code,incomplete',[(110,False),(None,False),(0,True)])
def test_failed_solver_cannot_become_an_exact_zero_fingerprint(tmp_path,monkeypatch,error_code,incomplete):
    import pandas as pd
    import wntr
    from types import SimpleNamespace
    import pytest
    design=json.loads((ROOT/'data/reference/sensor_fingerprints.json').read_text())
    design['fingerprints']=[next(row for row in design['fingerprints'] if np.linalg.norm(row['sensor_fingerprint'])==0)]
    selected=tmp_path/'selected.json';selected.write_text(json.dumps(design))
    wn=wntr.network.WaterNetworkModel(str(ROOT/'data/Exa7.inp'))
    bad=SimpleNamespace(error_code=error_code,node={'pressure':pd.DataFrame(1. if incomplete else np.nan,
        index=np.arange(0,82801 if incomplete else 86401,3600),columns=wn.node_name_list)})
    monkeypatch.setattr(wntr.sim.EpanetSimulator,'run_sim',lambda *args,**kwargs:bad)
    with pytest.raises(RuntimeError,match='hydraulic'):
        rebuild_reference(ROOT/'data/Exa7.inp',ROOT/'data/reference',tmp_path/'output',limit=1,fingerprint_file=selected)
