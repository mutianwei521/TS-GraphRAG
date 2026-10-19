"""Check the prescribed-demand design screen at 0, 5, 20 and 35 L/s."""
from pathlib import Path
import copy
import json
import time
import wntr
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/partition_sensor'


def main():
    started=time.perf_counter()
    wn=wntr.network.WaterNetworkModel(str(ROOT/'data/Exa7.inp'))
    nodes=wn.junction_name_list;rows=[]
    prefix=str(OUT/'epanet_work'/'validity')
    baseline=wntr.sim.EpanetSimulator(copy.deepcopy(wn)).run_sim(file_prefix=prefix)
    for rate in [0,5,20,35]:
        for node in ([None] if rate==0 else nodes):
            if node is None:result=baseline
            else:
                leak=copy.deepcopy(wn);leak.get_node(node).add_demand(rate/1000.,'881')
                result=wntr.sim.EpanetSimulator(leak).run_sim(file_prefix=prefix)
            pressure=result.node['pressure'][nodes].values
            rows.append(dict(node=node or 'baseline',rate_lps=rate,min_pressure_m=float(pressure.min()),
                             negative_node_times=int((pressure<0).sum()),nonfinite=int((~np.isfinite(pressure)).sum()),node_times=pressure.size))
        print('Design feasibility rate',rate,'elapsed_s',round(time.perf_counter()-started),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'hydraulic_validity.csv',index=False)
    report={'source':'fresh hydraulic simulation','model':'DDA','new_simulations':len(rows),
            'rates_lps':[0,5,20,35],'rows_with_negative_pressure':sum(row['negative_node_times']>0 for row in rows),
            'minimum_pressure_m':min(row['min_pressure_m'] for row in rows),
            'nonfinite_node_times':sum(row['nonfinite'] for row in rows),'elapsed_s':time.perf_counter()-started}
    (OUT/'hydraulic_validity_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
