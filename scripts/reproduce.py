"""Run the paper's frozen evaluation and data-to-figure workflows."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.verify_outputs import verify_assets

STUDIES=['all','exa7','strict_holdout','sensitivity','node_selection','repeated_benchmarks',
         'allocation','historical','hydraulic_robustness','partition_sensor','source_data','figures']


def main():
    if not __debug__:
        raise RuntimeError('Reproduction workflows require standard Python execution with validation enabled')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['smoke','replay','full'],default='smoke')
    parser.add_argument('--study',choices=STUDIES,default='all')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/reproduction')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--include-gnn',action='store_true')
    parser.add_argument('--model',help='Fixed MiniLM checkpoint path or model ID for full semantic rebuild')
    parser.add_argument('--allow-download',action='store_true')
    parser.add_argument('--benchmark-scope',choices=['main90','all540','both'],default='both')
    args=parser.parse_args()
    if args.profile=='smoke' and args.study!='all':
        parser.error('smoke runs the fixed minimal suite; --study selects replay/full workflows')
    out=args.output_dir.resolve()
    if (out/'run_report.json').exists():parser.error('Choose a fresh --output-dir for each recorded run')
    out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[name]='1'
    report={'profile':args.profile,'study':args.study,'started_utc':datetime.now(timezone.utc).isoformat(),
            'python':sys.version,'platform':platform.platform(),'status':'running','steps':[],
            'input_integrity':verify_assets()}
    report['asset_manifest_sha256']=hashlib.sha256((ROOT/'reproducibility/assets.json').read_bytes()).hexdigest()
    packages=['numpy','scipy','pandas','scikit-learn','wntr','networkx','leidenalg','igraph',
              'matplotlib','PyYAML','openpyxl','pypdf','torch','torch-geometric','sentence-transformers']
    report['packages']={}
    for package in packages:
        try:report['packages'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:report['packages'][package]=None
    report['implementation_sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ['src','scripts','experiments'] for p in (ROOT/folder).rglob('*.py')}
    report['implementation_sha256'].update({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')})
    report['environment_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('requirements*.txt')}
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    report['git_head']=git.stdout.strip() if git.returncode==0 else None
    started=time.perf_counter()
    def save():
        report['elapsed_seconds']=time.perf_counter()-started
        (out/'run_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def step(name,arguments):
        command=[sys.executable,*map(str,arguments)]
        print(f'Running {name}',flush=True)
        log=out/(name+'.log');t=time.perf_counter()
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        report['steps'].append({'name':name,'command':command,'exit_code':result.returncode,
                                'elapsed_seconds':time.perf_counter()-t,'log':log.name})
        save()
        if result.returncode:
            print(log.read_text(encoding='utf-8')[-6000:])
            raise RuntimeError(f'{name} failed; inspect {log}')
    def module(name,module_name,*arguments):step(name,['-m',module_name,*arguments])
    try:
        if args.profile=='smoke':
            step('offline_inference',['main.py','--mode','inference','--observations',ROOT/'data/example_observation.csv'])
            module('reference_hydraulics','experiments.reference_corpus','--limit',1,'--output-dir',out/'reference')
            module('strict_holdout','experiments.strict_engine.run','--study','all','--limit',2,'--output-dir',out/'strict')
            module('frozen_node_selection','experiments.node_selection.run','--stage','replay','--limit',1,'--output-dir',out/'node_selection')
            step('source_data',['scripts/build_source_data.py','--output',out/'source_data'])
        else:
            selected=STUDIES[1:] if args.study=='all' else [args.study]
            for study in selected:
                target=out/study
                if study=='exa7':
                    if args.profile=='full':module(study,'experiments.reference_corpus','--output-dir',target/'reference')
                    for protocol in ['H0','H2','H8','H9']:
                        arguments=['main.py','--mode','evaluate','--protocol',protocol,'--output-dir',target/protocol]
                        if args.profile=='full':arguments+=['--fingerprints',target/'reference/sensor_fingerprints.json']
                        step('numerical_'+protocol,arguments)
                elif study in ['strict_holdout','sensitivity']:
                    extra=[]
                    if args.profile=='full':
                        reference=target/'reference';reference.mkdir(parents=True,exist_ok=True)
                        for asset in (ROOT/'data/reference').iterdir():
                            if asset.is_file() and asset.name!='sensor_fingerprints.json':shutil.copyfile(asset,reference/asset.name)
                        module(study+'_hydraulics','experiments.reference_corpus','--output-dir',reference)
                        extra=['--data-dir',reference,'--embedding-mode','rebuild']
                        if args.model:extra+=['--model',args.model]
                        if args.allow_download:extra+=['--allow-download']
                    if study=='strict_holdout':extra+=['--legacy-attribution']
                    module(study,'experiments.strict_engine.run','--study','holdout' if study=='strict_holdout' else 'sensitivity','--output-dir',target,*extra)
                    module(study+'_verification','experiments.strict_engine.verify','--output-dir',target,'--compare-frozen')
                elif study=='node_selection':
                    module(study,'experiments.node_selection.run','--stage','all' if args.profile=='full' else 'replay','--output-dir',target)
                elif study=='repeated_benchmarks':
                    scopes=['main90','all540'] if args.benchmark_scope=='both' else [args.benchmark_scope]
                    for scope in scopes:
                        extra=['--include-gnn'] if args.include_gnn and scope=='main90' else []
                        module(study+'_'+scope,'experiments.repeated_benchmarks.run','--profile',args.profile,'--scope',scope,'--output-dir',target/scope,'--workers',args.workers,*extra)
                        if args.profile=='full':step(study+'_'+scope+'_verification',['scripts/verify_outputs.py','--run-dir',target/scope,'--classical-refit',scope])
                elif study=='allocation':
                    extra=[]
                    if args.profile=='full':
                        module(study+'_hydraulics','experiments.allocation.build_tensor','--output-dir',target/'tensor')
                        extra=['--tensor',target/'tensor/expH_robust_tensor.npz']
                    module(study,'experiments.allocation.run','--profile',args.profile,'--output-dir',target,*extra)
                elif study=='historical':
                    module(study,'experiments.historical.run','--profile',args.profile,'--output-dir',target)
                    if args.profile=='full':
                        for component in ['pressure_dependent','sensor_degradation','resolution','multileak_exhaustive','multileak_sparse']:
                            module(study+'_'+component,'experiments.historical.exa7','--study',component,'--output-dir',target/component)
                elif study=='partition_sensor':
                    module(study,'experiments.partition_sensor.run','--profile',args.profile,'--output-dir',target)
                elif study=='hydraulic_robustness':
                    if args.profile=='full':module(study,'experiments.hydraulic_robustness.run','--output-dir',target)
                    else:module(study,'experiments.hydraulic_robustness.replay','--output-dir',target)
                elif study=='source_data':step(study,['scripts/build_source_data.py','--output',out/'source_data'])
                elif study=='figures':step(study,['scripts/figures/replay.py','all','--output',target])
        report['status']='passed'
    except Exception as error:
        report['status']='failed';report['error']=str(error);save();raise
    for item in report['steps']:
        p=out/item['log']
        item['outputs']=[{'path':p.relative_to(out).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}]
    report['result_files']=[{'path':p.relative_to(out).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in out.rglob('*') if p.is_file() and p.name!='run_report.json' and not any(part in ['epanet_work','hydraulic_logs','worker_scratch'] for part in p.parts)]
    save();print(json.dumps({'status':report['status'],'steps':len(report['steps']),'elapsed_seconds':report['elapsed_seconds'],'report':str(out/'run_report.json')},indent=2))


if __name__=='__main__':main()
