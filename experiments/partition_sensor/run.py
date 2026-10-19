"""Run paired partition and sensor controls with training-only response designs."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/partition_sensor')
    parser.add_argument('--profile',choices=['full','replay'],default='full')
    args=parser.parse_args()
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    os.environ['TS_STUDY_OUTPUT']=str(output)
    from . import model
    model.OUT=output
    if args.profile=='replay':
        frozen=ROOT/'data/frozen/partition_sensor'
        for name in ['query_fingerprints_5_10_15_all_nodes.npz','hydraulic_validity.csv','hydraulic_validity_summary.json','primary_hydraulic_validity.csv']:
            shutil.copyfile(frozen/name,output/name)
    previous=Path.cwd()
    oldtemp={name:os.environ.get(name) for name in ['TMP','TEMP']}
    try:
        if args.profile=='full' and not (output/'hydraulic_validity_summary.json').exists():
            work=output/'epanet_work';work.mkdir(parents=True,exist_ok=True)
            os.chdir(work)
            os.environ['TMP']=str(work);os.environ['TEMP']=str(work)
            from . import validity
            validity.OUT=output;validity.main()
        model.main()
        from . import audit
        audit.OUT=output;audit.main()
    finally:
        os.chdir(previous)
        for name,value in oldtemp.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value


if __name__=='__main__':main()
