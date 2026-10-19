"""Execute the fixed hydraulic/demand study from a fresh EXA7 input."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def verify_nominal_corpus(h, design):
    """Independently solve every nominal atom in one isolated native workspace."""
    started=time.perf_counter()
    scratch=h.OUT/'worker_scratch'/'independent_nominal_replay';scratch.mkdir(parents=True,exist_ok=True)
    previous=Path.cwd();oldtemp={name:os.environ.get(name) for name in ['TMP','TEMP']}
    os.chdir(scratch)
    for name in oldtemp:os.environ[name]=str(scratch)
    try:
        z=np.load(h.OUT/'cache/nominal_corpus.npz')
        wn=h.wntr.network.WaterNetworkModel(str(h.OUT/'inputs/Exa7.inp'))
        ix=0
        for pi,(period,multiplier) in enumerate(h.PERIODS):
            state=h.nominal(wn,multiplier)
            baseline, status=h.sim(state,'verify_'+period+'_baseline')
            if not status['success'] or not np.array_equal(baseline,z['baseline'][pi]):
                raise AssertionError(f'Nominal baseline identity: {period}')
            for node in design['corpus_nodes']:
                leak=copy.deepcopy(state);leak.get_node(node).add_demand(.02,'881')
                pressures,status=h.sim(leak,'verify_'+period+'_n'+node)
                if not status['success'] or not np.array_equal(pressures,z['pressures'][ix],equal_nan=True):
                    raise AssertionError(f'Nominal template identity: {period}/{node}')
                ix+=1
        counts={period:int(z['keep'][pi*len(design['corpus_nodes']):(pi+1)*len(design['corpus_nodes'])].sum())
                for pi,(period,_) in enumerate(h.PERIODS)}
        if counts != {'night_low':351,'day_normal':350,'evening_peak':340}:
            raise AssertionError(f'Nominal retained template counts: {counts}')
        h.dump(h.OUT/'qa/nominal_corpus_replay.json',
               {'n_templates':ix,'n_baselines':3,'n_simulations':ix+3,
                'all_pressure_arrays_identical':True,'mismatches':[],
                'candidate_counts':counts,'elapsed_s':time.perf_counter()-started})
        z.close()
    finally:
        os.chdir(previous)
        for name,value in oldtemp.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/hydraulic_robustness')
    parser.add_argument('--stage',choices=['all','build','evaluate','audit'],default='all')
    args=parser.parse_args()
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    os.environ['TS_STUDY_OUTPUT']=str(output)
    from . import model as h
    h.OUT=output
    if args.stage in ['all','build']:
        native=output/'worker_scratch'/'initial_nominal';native.mkdir(parents=True,exist_ok=True)
        previous=Path.cwd();os.chdir(native)
        try:
            design=h.setup();h.build_corpus(design)
        finally:os.chdir(previous)
        verify_nominal_corpus(h,design)
    if args.stage=='all':
        h.main()
    if args.stage=='evaluate':
        h.evaluate(h.setup())
    if args.stage in ['all','audit']:
        from . import audit
        audit.OUT=output;audit.main()
    print(json.dumps({'study':'hydraulic_robustness','stage':args.stage,'output_dir':str(output)}))


if __name__=='__main__':main()
