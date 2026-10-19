"""Windows EPANET uses a CRT environment distinct from Python's environment."""
import ctypes
import os
from pathlib import Path

import pytest

from experiments.allocation.build_tensor import hydraulic_workspace


@pytest.mark.skipif(os.name != 'nt', reason='Windows CRT environment boundary')
@pytest.mark.parametrize('library', ['msvcrt', 'ucrtbase'])
def test_workspace_replaces_stale_native_crt_temp_and_restores_it(tmp_path, library):
    crt=ctypes.CDLL(library)
    crt.getenv.argtypes=[ctypes.c_char_p]
    crt.getenv.restype=ctypes.c_char_p
    crt._putenv_s.argtypes=[ctypes.c_char_p,ctypes.c_char_p]
    crt._putenv_s.restype=ctypes.c_int
    names=[b'TMP',b'TEMP',b'TMPDIR']
    original={name:crt.getenv(name) for name in names}
    python_original={name.decode():os.environ.get(name.decode()) for name in names}
    stale=tmp_path/'deleted_native_temporary_directory'
    stale_bytes=str(stale).encode('mbcs')
    try:
        for name in names: assert crt._putenv_s(name,stale_bytes)==0
        assert not stale.exists()
        with pytest.raises(RuntimeError,match='deliberate interruption'):
            with hydraulic_workspace(tmp_path/'run') as scratch:
                native=Path(os.environ['TMP'])
                assert native.is_dir() and native!=scratch
                for name in names:
                    assert crt.getenv(name)==str(native).encode('mbcs')
                raise RuntimeError('deliberate interruption')
        for name in names: assert crt.getenv(name)==stale_bytes
        assert python_original=={name:os.environ.get(name) for name in python_original}
        assert not scratch.exists()
    finally:
        for name,value in original.items(): assert crt._putenv_s(name,value or b'')==0
        for name,value in python_original.items():
            if value is None: os.environ.pop(name,None)
            else: os.environ[name]=value


def test_consecutive_workspaces_support_real_epanet_solves(tmp_path):
    import wntr
    from experiments.reference_corpus import checked_pressures
    root=Path(__file__).resolve().parents[1]
    original=Path.cwd()
    native_paths=[]
    scratch_paths=[]
    for index in range(2):
        model=wntr.network.WaterNetworkModel(str(root/'data/Exa7.inp'))
        with hydraulic_workspace(tmp_path/str(index)) as scratch:
            native_paths.append(Path(os.environ['TMP']))
            scratch_paths.append(scratch)
            result=wntr.sim.EpanetSimulator(model).run_sim(file_prefix='actual')
            pressures=checked_pressures(result,model,f'consecutive workspace {index}')
            assert pressures.shape==(25,383)
        assert Path.cwd()==original and not scratch.exists()
    assert native_paths[0]==native_paths[1] and native_paths[0].is_dir()
    assert scratch_paths[0]!=scratch_paths[1]
