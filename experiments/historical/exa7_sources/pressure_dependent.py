"""Portable original pressure dependent computation with explicit frozen inputs."""
from experiments.historical.exa7_support import INPUTS, FrozenReference, load_partition_map
import os, sys, json, time, copy, pickle, random, warnings, tempfile, uuid
import numpy as np
warnings.filterwarnings('ignore')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
pass
import wntr
from scipy.stats import binomtest
_P = INPUTS['epanet_prefix']
KB, INP = ('knowledge_base', INPUTS['inp'])
PART_PKL, K = ('partition_results_leiden/partitions.pkl', 15)
NOISE, SEED, PER = (0.02, 7, 'day_normal')
Q_REF = 20.0
ALPHAS = [0.5, 1.0, 1.5]
TARGETS = [5.0, 20.0]
INVAR_TARGETS = [2.0, 5.0, 10.0, 20.0, 35.0, 50.0]
DENSE_CACHE, OUT = (INPUTS['dense'], 'exp3_results.json')

def proj_res_mat(y, F, Fn):
    b = F @ y / np.maximum(Fn ** 2, 1e-18)
    r = np.linalg.norm(y[None, :] - b[:, None] * F, axis=1)
    r[Fn == 0] = np.inf
    return r

def main():
    pass
    pass
    pass
    loc = FrozenReference()
    sensors = sorted(loc.sensor_nodes, key=lambda x: int(x))
    wn0 = wntr.network.WaterNetworkModel(INP)
    juncs = wn0.junction_name_list
    n2c = load_partition_map()
    xy = {n: np.array(wn0.get_node(n).coordinates) for n in juncs}
    r0 = wntr.sim.EpanetSimulator(copy.deepcopy(wn0)).run_sim(file_prefix=_P)
    p0 = r0.node['pressure'].mean(axis=0)
    d0 = r0.node['demand'].mean(axis=0)
    n_sims = 1

    def emitter_dp(node, target_Lps, alpha):
        """Emitter leak sized so the nominal discharge at baseline pressure is target_Lps."""
        nonlocal n_sims
        w = copy.deepcopy(wn0)
        w.options.hydraulic.emitter_exponent = alpha
        C = target_Lps / 1000.0 / max(float(p0[node]), 0.1) ** alpha
        w.get_node(node).emitter_coefficient = C
        r = wntr.sim.EpanetSimulator(w).run_sim(file_prefix=_P)
        n_sims += 1
        pp = r.node['pressure'].mean(axis=0)
        realized = float((r.node['demand'].mean(axis=0)[node] - d0[node]) * 1000)
        return (pp - p0, realized)

    def demand_dp(node, q_Lps):
        nonlocal n_sims
        w = copy.deepcopy(wn0)
        w.get_node(node).add_demand(q_Lps / 1000.0, '881')
        r = wntr.sim.EpanetSimulator(w).run_sim(file_prefix=_P)
        n_sims += 1
        return r.node['pressure'].mean(axis=0) - p0

    def vec(series):
        return np.array([float(series[s]) for s in sensors])
    rng = random.Random(SEED)
    probe = rng.sample(list(juncs), INPUTS.get('probe_count', 20))
    pass
    pass
    pass
    q1 = {}
    for label, fn in [('fixed demand (as published)', lambda nd, t: (demand_dp(nd, t), t)), ('emitter alpha=0.5', lambda nd, t: emitter_dp(nd, t, 0.5)), ('emitter alpha=1.0', lambda nd, t: emitter_dp(nd, t, 1.0)), ('emitter alpha=1.5', lambda nd, t: emitter_dp(nd, t, 1.5))]:
        cosines, worst = ([], (2.0, None))
        for nd in probe:
            V = []
            for t in INVAR_TARGETS:
                dp, _ = fn(nd, t)
                V.append(vec(dp))
            for i in range(len(V)):
                for j in range(i + 1, len(V)):
                    a, b = (V[i], V[j])
                    na, nb = (np.linalg.norm(a), np.linalg.norm(b))
                    if na == 0 or nb == 0:
                        continue
                    c = float(np.dot(a, b) / (na * nb))
                    cosines.append(c)
                    if c < worst[0]:
                        worst = (c, f'{nd}: {INVAR_TARGETS[i]:.0f} vs {INVAR_TARGETS[j]:.0f} L/s')
        cosines = np.array(cosines)
        q1[label] = dict(min=float(cosines.min()), median=float(np.median(cosines)), mean=float(cosines.mean()))
        pass
    pass
    z = np.load(DENSE_CACHE, allow_pickle=True)
    dn = [str(x) for x in z['nodes']]
    if INPUTS.get('dense_count'):
        dn = dn[:INPUTS['dense_count']]
    Fd = z['F_day_normal'][:len(dn)]
    Fdn = np.linalg.norm(Fd, axis=1)
    pass
    t0 = time.time()
    Fe = np.zeros_like(Fd)
    for i, nd in enumerate(dn):
        dp, _ = emitter_dp(nd, Q_REF, 0.5)
        Fe[i] = vec(dp)
        if (i + 1) % 100 == 0:
            pass
    Fen = np.linalg.norm(Fe, axis=1)
    pass
    held = {int(k): v for k, v in json.load(open(INPUTS['held']))['config']['held_out_nodes'].items()}
    qs = [(c, n, t) for c, ns in held.items() for n in ns for t in TARGETS]
    random.Random(SEED).shuffle(qs)
    if INPUTS.get('query_count'):
        qs = qs[:INPUTS['query_count']]
    pass
    pass
    res = {}
    for alpha in ALPHAS:
        for cname, F, Fn in (('Cd fixed-demand', Fd, Fdn), ('Ce emitter a=0.5', Fe, Fen)):
            np.random.seed(SEED)
            ok, nok, dist, real = ([], [], [], [])
            for gp, gn, t in qs:
                dp, rq = emitter_dp(gn, t, alpha)
                real.append(rq)
                y = vec(dp) + np.random.normal(0, NOISE, len(sensors))
                r = proj_res_mat(y, F, Fn)
                i = int(np.argmin(r))
                pn = dn[i]
                ok.append(int(int(n2c[pn]) == gp))
                nok.append(int(pn == gn))
                dist.append(float(np.linalg.norm(xy[pn] - xy[gn])))
            res[f'a{alpha}_{cname[:2]}'] = dict(alpha=alpha, corpus=cname, part=float(np.mean(ok)), node=float(np.mean(nok)), med_m=float(np.median(dist)), ok=ok)
            pass
    np.random.seed(SEED)
    ok = []
    for gp, gn, t in qs:
        y = vec(demand_dp(gn, t)) + np.random.normal(0, NOISE, len(sensors))
        i = int(np.argmin(proj_res_mat(y, Fd, Fdn)))
        ok.append(int(int(n2c[dn[i]]) == gp))
    pass
    res['control'] = dict(part=float(np.mean(ok)), ok=ok)
    pass
    for alpha in ALPHAS:
        a = res[f'a{alpha}_Cd']['ok']
        b = res[f'a{alpha}_Ce']['ok']
        n01 = sum((1 for x, y_ in zip(a, b) if x == 0 and y_ == 1))
        n10 = sum((1 for x, y_ in zip(a, b) if x == 1 and y_ == 0))
        p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        pass
    json.dump(dict(q1_direction_stability=q1, q2=res, n_simulations=n_sims, probe_nodes=probe, targets=TARGETS, alphas=ALPHAS), open(OUT, 'w'), indent=1)
    pass
if __name__ == '__main__':
    main()
