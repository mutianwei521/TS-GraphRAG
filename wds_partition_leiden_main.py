"""Export the frozen hydraulic partition or generate a seeded Leiden design."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle

ROOT=Path(__file__).resolve().parent


def main(inp_file=None, output_dir=None, num_partitions=15, seed=41, design='frozen'):
    import wntr
    inp=Path(inp_file or ROOT/'data/Exa7.inp').resolve()
    output=Path(output_dir or ROOT/'outputs/partition').resolve();output.mkdir(parents=True,exist_ok=True)
    wn=wntr.network.WaterNetworkModel(str(inp))
    if design=='frozen':
        if hashlib.sha256(inp.read_bytes()).digest()!=hashlib.sha256((ROOT/'data/Exa7.inp').read_bytes()).digest():
            raise ValueError('Frozen physical partition requires the released EXA7 input; use --design leiden for a new network')
        summary=json.loads((ROOT/'data/reference/partition_summary.json').read_text())
        if str(num_partitions) not in summary:raise ValueError(f'No frozen partition at K={num_partitions}')
        row=summary[str(num_partitions)]
        partitions={num_partitions:{'node_to_community':row['node_assignments'],'resolution':row['resolution']}}
    else:
        from src.phase1_partition.hydraulic import calculate_average_pressure
        from src.phase1_partition.similarity import create_network_graph
        from src.phase1_partition.partitioning_leiden import run_leiden_partitioning,extract_partitions_with_merge
        import os
        work=output/'epanet_work';work.mkdir(exist_ok=True)
        previous=Path.cwd();os.chdir(work)
        try:result=wntr.sim.EpanetSimulator(wn).run_sim(file_prefix='partition_baseline')
        finally:os.chdir(previous)
        graph,_=create_network_graph(wn,calculate_average_pressure(result,wn))
        candidates=run_leiden_partitioning(graph,(.001,3.),300,seed=seed)
        all_parts=extract_partitions_with_merge(graph,candidates,merge_range=(2,max(15,num_partitions)),target_k=num_partitions)
        if num_partitions not in all_parts:raise ValueError(f'Seeded search did not produce K={num_partitions}')
        partitions={num_partitions:all_parts[num_partitions]}
    summary={str(k):{'num_communities':k,'resolution':v['resolution'],'node_assignments':v['node_to_community']} for k,v in partitions.items()}
    (output/'partition_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (output/'partitions.pkl').write_bytes(pickle.dumps(partitions))
    manifest={'design':design,'seed':seed if design=='leiden' else None,'num_partitions':num_partitions,
              'network_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),
              'network_counts':{'junctions':len(wn.junction_name_list),'pipes':len(wn.pipe_name_list),'pumps':len(wn.pump_name_list),'reservoirs':len(wn.reservoir_name_list),'tanks':len(wn.tank_name_list)}}
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2));return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inp',type=Path,default=ROOT/'data/Exa7.inp')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/partition')
    parser.add_argument('--num-partitions','--partition-k',type=int,default=15)
    parser.add_argument('--seed',type=int,default=41)
    parser.add_argument('--design',choices=['frozen','leiden'],default='frozen')
    args=parser.parse_args()
    main(args.inp,args.output_dir,args.num_partitions,args.seed,args.design)
