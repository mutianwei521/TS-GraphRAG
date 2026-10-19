import json
from pathlib import Path
import subprocess
import sys
import types
import yaml
import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_build_help_runs_without_semantic_dependencies():
    result=subprocess.run([sys.executable,str(ROOT/'build_knowledge_base.py'),'--help'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '--reference-dir' in result.stdout


def test_partition_cli_exports_frozen_physical_design_from_external_directory(tmp_path):
    output=tmp_path/'partition'
    result=subprocess.run([sys.executable,str(ROOT/'wds_partition_leiden_main.py'),'--output-dir',str(output)],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    data=json.loads((output/'partition_summary.json').read_text())
    assert len(set(data['15']['node_assignments'].values()))==15
    report=json.loads((output/'run_manifest.json').read_text())
    assert report['design']=='frozen'
    assert report['network_counts']['junctions']==381


def test_explicit_fusion_receives_effective_propagation_beta(tmp_path, monkeypatch):
    import main
    captured={}
    class Locator:
        def __init__(self,**kwargs):pass
        def localize_leak(self,*args,**kwargs):captured.update(kwargs);return {'ranking':[0]}
    stub=types.ModuleType('src.phase3_query.leak_locator');stub.LeakLocator=Locator
    monkeypatch.setitem(sys.modules,'src.phase3_query.leak_locator',stub)
    config={'inference':{'method':'fusion','beta':.3},'paths':{'fingerprints':str(ROOT/'data/reference/sensor_fingerprints.json')}}
    path=tmp_path/'fusion.yaml';path.write_text(yaml.safe_dump(config))
    main.main(['--mode','inference','--config',str(path),'--observations',str(ROOT/'data/example_observation.csv')])
    assert captured['propagation_beta']==.3


@pytest.mark.parametrize('override',[{'partitioning':{'algorithm':'nonexistent'}},
                                     {'sensor_placement':{'sensors_per_partition':999}}])
def test_configuration_rejects_unimplemented_physical_design(tmp_path, override):
    from src.configuration import load_config
    path=tmp_path/'invalid.yaml';path.write_text(yaml.safe_dump(override))
    with pytest.raises(ValueError):load_config(path)
