"""Rebuild the original E5 hydraulic tensor with fixed reference sensors and EXA7."""
import argparse
import atexit
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from .run import ROOT, DESIGNS
PERIOD = {'night_low':.7,'day_normal':1.,'evening_peak':1.2}
QMAX = 50.
ALL_MAGS = sorted({m for _,ms in DESIGNS.values() for m in ms}|{5.,20.,50.})
_NATIVE_TEMP=None


def native_temporary_directory():
    """Keep EPANET's native CRT temp path valid across consecutive runs in one process."""
    global _NATIVE_TEMP
    if _NATIVE_TEMP is None:
        _NATIVE_TEMP=tempfile.TemporaryDirectory(prefix='tsgraphrag_epanet_')
        atexit.register(_NATIVE_TEMP.cleanup)
    return Path(_NATIVE_TEMP.name).resolve()


@contextmanager
def native_temporary_environment(directory):
    """Synchronize Python and Windows CRT temp paths, restoring each original value."""
    names=['TMP','TEMP','TMPDIR']
    saved={name:os.environ.get(name) for name in names}
    crts=[]
    if os.name=='nt':
        import ctypes
        # EPANET's MSVCRT may retain an environment separate from Python's UCRT.
        for library in ['msvcrt','ucrtbase']:
            crt=ctypes.CDLL(library)
            crt.getenv.argtypes=[ctypes.c_char_p]
            crt.getenv.restype=ctypes.c_char_p
            crt._putenv_s.argtypes=[ctypes.c_char_p,ctypes.c_char_p]
            crt._putenv_s.restype=ctypes.c_int
            crts.append((crt,{name:crt.getenv(name.encode('ascii')) for name in names}))
    try:
        for name in names: os.environ[name]=str(directory)
        for crt,_ in crts:
            for name in names:
                if crt._putenv_s(name.encode('ascii'),str(directory).encode('mbcs')):
                    raise RuntimeError(f'Could not set native hydraulic temporary variable {name}')
        yield
    finally:
        for name,value in saved.items():
            if value is None: os.environ.pop(name,None)
            else: os.environ[name]=value
        for crt,values in crts:
            for name,value in values.items():
                if crt._putenv_s(name.encode('ascii'),value or b''):
                    raise RuntimeError(f'Could not restore native hydraulic temporary variable {name}')


@contextmanager
def hydraulic_workspace(output_dir):
    """Use a per-run working directory and a process-lifetime native temporary directory."""
    work_root=Path(output_dir).resolve()/'epanet_work'
    work_root.mkdir(parents=True,exist_ok=True)
    temporary=tempfile.TemporaryDirectory(prefix='e5_',dir=work_root)
    scratch=Path(temporary.name).resolve()
    original=Path.cwd()
    native_temp=native_temporary_directory()
    try:
        os.chdir(scratch)
        with native_temporary_environment(native_temp):
            yield scratch
    finally:
        os.chdir(original)
        temporary.cleanup()


def _simulate_tensor(inp,sensors,prefix):
    """Simulate each admissible location, magnitude and demand period using the original formulas."""
    import wntr
    from experiments.reference_corpus import checked_pressures
    def pressures(model,label):
        result=wntr.sim.EpanetSimulator(model).run_sim(file_prefix=str(prefix))
        return checked_pressures(result,model,label)
    INP=str(inp)
    wn0 = wntr.network.WaterNetworkModel(INP)
    juncs = list(wn0.junction_name_list)
    models, base = ({}, {})
    for per, mult in PERIOD.items():
        wn = copy.deepcopy(wn0)
        for j in juncs:
            for d in wn.get_node(j).demand_timeseries_list:
                d.base_value *= mult
        models[per] = wn
        P = pressures(copy.deepcopy(wn),f'{per} baseline')[juncs]
        if not np.isfinite(P.values).all() or P.values.min() < -5 or P.values.max() > 200:
            raise ValueError(f'Invalid EXA7 {per} baseline pressure')
        base[per] = P.mean(axis=0)
    bp = base['day_normal']
    keepn = []
    print(f'  admissibility screen at {QMAX:g} L/s over {len(juncs)} junctions', flush=True)
    for i, nd in enumerate(juncs):
        w = copy.deepcopy(models['day_normal'])
        w.get_node(nd).add_demand(QMAX / 1000.0, '881')
        lp = pressures(w,f'admissibility {nd}').mean(axis=0)
        d = np.array([float(lp[j] - bp[j]) for j in juncs])
        if float(lp[nd]) >= 0.0 and np.abs(d).max() <= 5.0 * QMAX:
            keepn.append(nd)
        if (i + 1) % 150 == 0:
            print(f'    {i + 1}/{len(juncs)}', flush=True)
    print(f'  {len(juncs) - len(keepn)} of {len(juncs)} junctions excluded ({100 * (len(juncs) - len(keepn)) / len(juncs):.1f}%)', flush=True)
    T = np.zeros((len(keepn), len(ALL_MAGS), 3, len(sensors)))
    t0, n = (time.time(), 0)
    for pi, per in enumerate(PERIOD):
        for mi, q in enumerate(ALL_MAGS):
            for ni, nd in enumerate(keepn):
                w = copy.deepcopy(models[per])
                w.get_node(nd).add_demand(q / 1000.0, '881')
                lp = pressures(w,f'{per} {q:g} L/s {nd}').mean(axis=0)
                T[ni, mi, pi] = [float(lp[s] - base[per][s]) for s in sensors]
                n += 1
            print(f'    {per} {q:g} L/s done ({n}/{len(keepn) * len(ALL_MAGS) * 3}, {time.time() - t0:.0f}s)', flush=True)
    if not np.isfinite(T).all(): raise ValueError('Nonfinite EXA7 tensor')
    return (T, keepn, sensors)


def build_tensor(inp,output_dir):
    """Rebuild the EXA7 tensor and record its source binding and comparison with the archive."""
    inp=Path(inp).resolve();output_dir=Path(output_dir).resolve()
    tensor_path=output_dir/'expH_robust_tensor.npz';report_path=output_dir/'rebuild_report.json'
    if tensor_path.exists() or report_path.exists():
        raise FileExistsError(f'Use a fresh output directory; tensor/report already exists in {output_dir}')
    output_dir.mkdir(parents=True,exist_ok=True)
    sensor_path=ROOT/'data/reference/sensor_fingerprints.json'
    frozen_path=ROOT/'data/frozen/allocation/expH_robust_tensor.npz'
    inputs=[inp,sensor_path,Path(__file__),frozen_path]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    sensors=sorted(json.loads(sensor_path.read_text(encoding='utf-8'))['sensor_nodes'],key=lambda x:int(x))
    started=time.perf_counter()
    with hydraulic_workspace(output_dir) as scratch:
        T,nodes,sensors=_simulate_tensor(inp,sensors,scratch/'ep')
    np.savez_compressed(tensor_path,T=T,nodes=np.array(nodes),mags=np.array(ALL_MAGS),
                        periods=np.array(list(PERIOD)),sensors=np.array(sensors))
    archive=np.load(frozen_path,allow_pickle=False)
    axes=dict(shape_match=list(T.shape)==list(archive['T'].shape),
              nodes_match=nodes==archive['nodes'].tolist(),magnitudes_match=ALL_MAGS==archive['mags'].tolist(),
              periods_match=list(PERIOD)==archive['periods'].tolist(),sensors_match=sensors==archive['sensors'].tolist())
    maximum=float(np.max(np.abs(T-archive['T']))) if all(axes.values()) else None
    report=dict(study='allocation_tensor',profile='raw_hydraulics',input_sha256=hashes,
                binding_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
                tensor_shape=list(T.shape),reference_tensor_shape=list(archive['T'].shape),sensor_nodes=sensors,
                comparison=axes,max_absolute_tensor_difference_m=maximum,exact_tensor_match=maximum==0.,
                simulations=dict(baselines=3,screen=381,tensor=len(nodes)*len(ALL_MAGS)*3),
                output_sha256=hashlib.sha256(tensor_path.read_bytes()).hexdigest(),elapsed_seconds=time.perf_counter()-started,
                isolation='per-run scratch cwd; process-lifetime native TMP/TEMP/TMPDIR; environment restored after each run')
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    if not all(axes.values()): raise AssertionError('Rebuilt E5 tensor axes differ from the frozen experiment; inspect rebuild_report.json')
    if maximum is None or not np.isfinite(maximum) or maximum>1e-8:
        raise AssertionError('Rebuilt E5 tensor values differ from the frozen experiment; inspect rebuild_report.json')
    return T,nodes,sensors

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inp',type=Path,default=ROOT/'data/Exa7.inp')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/allocation/tensor')
    args=parser.parse_args()
    tensor,nodes,sensors=build_tensor(args.inp,args.output_dir)
    print(json.dumps({'tensor_shape':list(tensor.shape),'nodes':len(nodes),'sensors':sensors},indent=2))

if __name__=='__main__': main()
