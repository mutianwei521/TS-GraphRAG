"""Portable original sensor degradation computation with explicit frozen inputs."""
from experiments.historical.exa7_support import INPUTS, FrozenReference, load_partition_map
import os, sys, json, time, copy, pickle, random, warnings, tempfile, uuid
import numpy as np
warnings.filterwarnings('ignore')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
pass
import wntr
from experiments.historical.exa7_support import G, complete_link_classes
_P = INPUTS['epanet_prefix']
INP = INPUTS['inp']
NOISE, SEED, COS = (0.02, 31, 0.999)
MAGS = [5.0, 20.0]
N_QUERY = INPUTS.get('N_QUERY', 100)
TRIM = 0.2
OUT = 'expA2_results.json'

def fit_residual(y, D, mask, trim=0.0):
    """Projection residual restricted to `mask` sensors; optionally trimming the worst terms."""
    Dm = D[:, mask]
    ym = y[mask]
    dn2 = np.maximum((Dm ** 2).sum(axis=1), 1e-18)
    b = Dm @ ym / dn2
    R = ym[None, :] - b[:, None] * Dm
    if trim > 0 and Dm.shape[1] > 4:
        keep = int(round(Dm.shape[1] * (1 - trim)))
        R2 = np.sort(R ** 2, axis=1)[:, :keep]
        return np.sqrt(R2.sum(axis=1))
    return np.linalg.norm(R, axis=1)

def main():
    pass
    pass
    pass
    loc = FrozenReference()
    deployed = sorted(loc.sensor_nodes, key=lambda x: int(x))
    wn0 = wntr.network.WaterNetworkModel(INP)
    juncs = list(wn0.junction_name_list)
    xy = {n: np.array(wn0.get_node(n).coordinates) for n in juncs}
    z = np.load(INPUTS['sensitivity'], allow_pickle=True)
    S, nodes, meas = (z['S'], [str(x) for x in z['nodes']], [str(x) for x in z['meas']])
    midx = {n: i for i, n in enumerate(meas)}

    def make_dict(sensor_nodes):
        cols = [midx[str(s)] for s in sensor_nodes]
        F = S[:, cols]
        lab = complete_link_classes(F, COS)
        nc = int(lab.max())
        D = np.zeros((nc, len(cols)))
        cxy = {}
        for c in range(nc):
            ii = np.where(lab == c + 1)[0]
            D[c] = F[ii].mean(axis=0)
            cxy[c] = np.mean([xy[nodes[i]] for i in ii], axis=0)
        return (D, {nodes[i]: int(lab[i]) - 1 for i in range(len(nodes))}, nc, cxy)
    wn = copy.deepcopy(wn0)
    n_sims = 0
    bp = wntr.sim.EpanetSimulator(copy.deepcopy(wn)).run_sim(file_prefix=_P).node['pressure'].mean(axis=0)
    n_sims += 1
    sim_cache = {}

    def dp(nd, q, sensor_nodes):
        nonlocal n_sims
        key = (nd, q)
        if key not in sim_cache:
            w = copy.deepcopy(wn)
            w.get_node(nd).add_demand(q / 1000.0, '881')
            lp = wntr.sim.EpanetSimulator(w).run_sim(file_prefix=_P).node['pressure'].mean(axis=0)
            n_sims += 1
            sim_cache[key] = {s: float(lp[s] - bp[s]) for s in meas}
        d = sim_cache[key]
        return np.array([d[str(s)] for s in sensor_nodes])
    rng = random.Random(SEED)
    pool = [n for n in juncs if n not in set(map(str, deployed))]
    queries = [(rng.choice(pool), q) for _ in range(N_QUERY) for q in MAGS]
    pass

    def run(sensor_nodes, corrupt, label):
        D, ncl, nc, cxy = make_dict(sensor_nodes)
        m = len(sensor_nodes)
        out = {}
        for tag, trim in (('L2', 0.0), ('trimmed', TRIM)):
            npr = np.random.default_rng(SEED)
            hit, dist = (0, [])
            for nd, q in queries:
                y = dp(nd, q, sensor_nodes) + npr.normal(0, NOISE, m)
                mask = np.ones(m, bool)
                y, mask = corrupt(y, mask, npr, m)
                if mask.sum() < 3:
                    continue
                r = fit_residual(y, D, mask, trim)
                c = int(np.argmin(r))
                hit += int(c == ncl[nd])
                dist.append(float(np.linalg.norm(cxy[c] - xy[nd])))
            n = len(dist)
            G.prob(hit / max(n, 1), f'{label}/{tag} recall')
            out[tag] = dict(recall=hit / max(n, 1), med_m=float(np.median(dist)), n=n)
        return (out, nc)

    def none(y, mask, npr, m):
        return (y, mask)

    def drift(sd):

        def f(y, mask, npr, m):
            return (y + npr.normal(0, sd, m), mask)
        return f

    def missing(p):

        def f(y, mask, npr, m):
            mask = mask & (npr.random(m) > p)
            return (y, mask)
        return f

    def stuck(k):

        def f(y, mask, npr, m):
            idx = npr.choice(m, size=min(k, m), replace=False)
            y = y.copy()
            y[idx] = 0.0
            return (y, mask)
        return f

    def spikes(p, amp=1.0):

        def f(y, mask, npr, m):
            y = y.copy()
            hit = npr.random(m) < p
            y[hit] += npr.choice([-1, 1], size=hit.sum()) * amp
            return (y, mask)
        return f

    def bias_control(sd):
        """A constant per-sensor bias present in BOTH baseline and event: must cancel."""

        def f(y, mask, npr, m):
            return (y, mask)
        return f
    results = {}
    pass
    pass
    scenarios = [('baseline, 30 deployed sensors', deployed, none), ('drift sd=0.05 m', deployed, drift(0.05)), ('drift sd=0.15 m', deployed, drift(0.15)), ('drift sd=0.30 m', deployed, drift(0.3)), ('constant bias (control: cancels)', deployed, bias_control(0.3)), ('missing 10%', deployed, missing(0.1)), ('missing 30%', deployed, missing(0.3)), ('1 sensor stuck', deployed, stuck(1)), ('3 sensors stuck', deployed, stuck(3)), ('5% impulsive spikes (1 m)', deployed, spikes(0.05)), ('bad day: drift .15 + 10% miss + 2 stuck + spikes', deployed, lambda y, mk, npr, m: spikes(0.05)(*stuck(2)(*missing(0.1)(*drift(0.15)(y, mk, npr, m), npr, m), npr, m), npr, m))]
    for label, sn, corrupt in scenarios:
        r, nc = run(sn, corrupt, label)
        results[label] = r
        d = 100 * (r['trimmed']['recall'] - r['L2']['recall'])
        pass
    pass
    pass
    pass
    pass
    pass
    pass
    counts = []
    r0, nc0 = run(deployed, none, '30 deployed')
    results['30 deployed sensors'] = r0
    counts.append(nc0)
    pass
    for k in [20, 15, 10, 5]:
        r, nc = run(deployed[:k], none, f'{k} deployed')
        results[f'{k} deployed sensors'] = r
        rnd = [meas[i] for i in random.Random(100 + k).sample(range(len(meas)), k)]
        r2, nc2 = run(rnd, none, f'{k} random')
        results[f'{k} random sensors'] = r2
        counts += [nc, nc2]
        pass
        pass
    G.strict = False
    G.comparable(counts, 'class recall across sensor budgets')
    G.strict = True
    pass
    json.dump(dict(results=results, trim=TRIM, n_queries=len(queries), mags=MAGS, noise=NOISE, seed=SEED, n_simulations=n_sims), open(OUT, 'w'), indent=1)
    pass
if __name__ == '__main__':
    main()
