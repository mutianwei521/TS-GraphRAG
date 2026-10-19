"""Direct physical inputs and shared primitives for the EXA7 historical studies."""
import json
from pathlib import Path
import numpy as np

INPUTS={}


class FrozenReference:
    """Read the original ordered sensor configuration without embedding initialization."""
    def __init__(self):
        data=json.loads((Path(INPUTS['reference_dir'])/'sensor_fingerprints.json').read_text(encoding='utf-8'))
        self.sensor_nodes=sorted(set(map(str,data['sensor_nodes'])),key=lambda x:int(x))
        self.fp_sensor_nodes=list(data['sensor_nodes'])
        self.fingerprints=list(data['fingerprints'])


def load_partition_map():
    data=json.loads((Path(INPUTS['reference_dir'])/'partition_summary.json').read_text(encoding='utf-8'))
    return {str(n):int(p) for n,p in data['15']['node_assignments'].items()}


class Guards:
    strict=True
    def prob(self,p,what='probability'):
        if not np.isfinite(p) or not 0<=p<=1: raise ValueError(f'{what}: invalid probability {p}')
        return p
    def comparable(self,counts,what='comparison'):
        if self.strict and len(set(counts))>1: raise ValueError(f'{what}: different class counts')
        return len(set(counts))==1


G=Guards()


def complete_link_classes(F,thr):
    """Complete-linkage cosine classes using the original threshold and zero-vector rule."""
    from scipy.cluster.hierarchy import linkage,fcluster
    from scipy.spatial.distance import squareform
    nr=np.linalg.norm(F,axis=1);ok=nr>1e-12
    U=np.zeros_like(F);U[ok]=F[ok]/nr[ok,None]
    C=np.clip(U@U.T,-1,1);D=1-C
    np.fill_diagonal(D,0.);D=(D+D.T)/2
    return fcluster(linkage(squareform(D,checks=False),method='complete'),t=1-thr,criterion='distance')
