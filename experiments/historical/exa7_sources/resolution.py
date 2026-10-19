"""Portable original resolution computation with explicit frozen inputs."""
from experiments.historical.exa7_support import INPUTS, FrozenReference, load_partition_map
import os, sys, json, random, warnings, pickle
import numpy as np
warnings.filterwarnings('ignore')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
pass
pass
import wntr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
INP = INPUTS['inp']
NOISE = 0.02
Q_REF = 20.0
SEED = 5
THRESH = [0.99, 0.999, 0.9999]
COUNTS = INPUTS.get('COUNTS', [5, 10, 15, 20, 30, 45, 60, 90, 150, 381])
z = np.load(INPUTS['sensitivity'], allow_pickle=True)
S = z['S']
nodes = [str(x) for x in z['nodes']]
meas = [str(x) for x in z['meas']]
wn0 = wntr.network.WaterNetworkModel(INP)
n2c = load_partition_map()
xy = {n: np.array(wn0.get_node(n).coordinates) for n in wn0.junction_name_list}
midx = {n: i for i, n in enumerate(meas)}
loc = FrozenReference()
deployed = [midx[str(s)] for s in loc.sensor_nodes if str(s) in midx]

def complete_link_classes(Fs, thr):
    nrm = np.linalg.norm(Fs, axis=1)
    ok = nrm > 1e-12
    U = np.zeros_like(Fs)
    U[ok] = Fs[ok] / nrm[ok, None]
    C = np.clip(U @ U.T, -1, 1)
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method='complete')
    lab = fcluster(Z, t=1.0 - thr, criterion='distance')
    return lab

def eff_rank(Fs, noise=NOISE):
    s = np.linalg.svd(Fs, compute_uv=False)
    return (int((s > noise * np.sqrt(Fs.shape[1])).sum()), s)
pass
pass
pass
pass
pass
pass
out1 = {}
for thr in THRESH:
    lab = complete_link_classes(S[:, deployed], thr)
    sizes = np.bincount(lab)[1:]
    out1[str(thr)] = dict(n=int(len(sizes)), largest=int(sizes.max()), median=float(np.median(sizes)), singletons=int((sizes == 1).sum()))
    pass
lab = complete_link_classes(S[:, deployed], 0.999)
sizes = np.bincount(lab)[1:]
spans = []
for c in range(1, lab.max() + 1):
    ns = [nodes[i] for i in np.where(lab == c)[0]]
    if len(ns) < 2:
        spans.append(0.0)
        continue
    P = np.array([xy[n] for n in ns])
    spans.append(float(np.max(np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1))))
spans = np.array(spans)
pass
pure = sum((1 for c in range(1, lab.max() + 1) if len({int(n2c[nodes[i]]) for i in np.where(lab == c)[0]}) == 1))
pass
pass
pass
pass
rng = random.Random(SEED)
out2 = {}

def greedy(budget, seed=0):
    r = np.random.default_rng(seed)
    chosen = [int(r.integers(S.shape[1]))]
    for _ in range(budget - 1):
        best, bj = (np.inf, None)
        for j in r.choice(S.shape[1], size=min(60, S.shape[1]), replace=False):
            if j in chosen:
                continue
            Fs = S[:, chosen + [int(j)]]
            nrm = np.linalg.norm(Fs, axis=1)
            ok = nrm > 1e-12
            if ok.sum() < 2:
                continue
            U = Fs[ok] / nrm[ok, None]
            Gm = U @ U.T
            np.fill_diagonal(Gm, 0.0)
            sc = float(Gm.max(axis=1).mean())
            if sc < best:
                best, bj = (sc, int(j))
        if bj is None:
            break
        chosen.append(bj)
    return chosen
for b in COUNTS:
    row = {}
    if b <= len(deployed):
        row['deployed'] = int(complete_link_classes(S[:, deployed[:b]], 0.999).max())
        row['rank_dep'] = eff_rank(S[:, deployed[:b]])[0]
    vals = [int(complete_link_classes(S[:, rng.sample(range(len(meas)), b)], 0.999).max()) for _ in range(5)]
    row['random_mean'] = float(np.mean(vals))
    g = greedy(b, SEED) if b < len(meas) else list(range(len(meas)))
    row['greedy'] = int(complete_link_classes(S[:, g], 0.999).max())
    row['rank_greedy'] = eff_rank(S[:, g])[0]
    out2[str(b)] = row
    pass
json.dump(dict(classes_deployed=out1, class_span_m=dict(median=float(np.median(spans)), p75=float(np.percentile(spans, 75)), max=float(spans.max())), pure_classes=pure, n_classes_0999=int(len(sizes)), resolution_vs_sensors=out2), open('expA1a_results.json', 'w'), indent=1)
pass
