"""Train the original E4 GNN architectures with fixed retained pools and five seeds."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SEEDS = [41, 581, 1121, 1661, 2201]
PROTOCOLS = ["H0", "H2", "H3", "H4", "H8"]
METHODS = ["GCN+", "GAT+", "GraphSAGE+"]
CONFIG = dict(seeds=SEEDS, protocols=PROTOCOLS, methods=METHODS, noise_m=0.02,
              epochs=200, batch_size=64, validation_fraction=0.15,
              validation_minimum=15, validation_noise=False, check_every=10,
              learning_rate=0.001, weight_decay=0.0001, scheduler_step=60,
              scheduler_gamma=0.5, torch_threads=1,
              query_noise="np.random.RandomState(run_seed + original_corpus_index)",
              run_seed_spacing=540,
              query_sampling="fixed existing qidx90; no redraw between seeds",
              architecture="unchanged build() from expH_gnn2.py",
              optimizer="Adam", input_preprocessing="none",
              uncertainty="sample SD (ddof=1) across five run seeds",
              deduplicate_retained_pools=True, batched_query_prediction=True)
SOURCE_FILES = ["experiments/repeated_benchmarks/architecture.py", "data/reference/sensor_fingerprints.json",
                "data/reference/partition_adjacency.json", "data/frozen/historical/expH_gnn2_results.json.partial",
                "experiments/repeated_benchmarks/primitives.py"]


def digest_bytes(b):
    return hashlib.sha256(b).hexdigest()


def digest_json(x):
    return digest_bytes(json.dumps(x, sort_keys=True, separators=(",", ":")).encode())


def read_json(p):
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, p)


def projection(vec, dictionary):
    norm = np.linalg.norm(dictionary, axis=1)
    beta = (dictionary @ vec) / np.maximum(norm**2, 1e-18)
    res = np.linalg.norm(vec[None, :] - beta[:, None]*dictionary, axis=1)
    res[norm == 0] = np.inf
    return res


class Inputs:
    def __init__(self):
        self.hashes = {n: digest_bytes((ROOT / n).read_bytes()) for n in SOURCE_FILES}
        self.script_hash = digest_bytes(Path(__file__).read_bytes())
        fpd = read_json(ROOT / "data/reference/sensor_fingerprints.json")
        self.all = fpd["fingerprints"]
        self.sensors = fpd["sensor_nodes"]
        self.ids = [f["scenario_id"] for f in self.all]
        self.X = np.asarray([f["sensor_fingerprint"] for f in self.all], dtype=float)
        self.y = np.asarray([f["leak_partition"] for f in self.all], dtype=int)
        ck = read_json(ROOT / "data/frozen/historical/expH_gnn2_results.json.partial")
        old = ck
        self.qidx = ck["qidx"]
        assert self.qidx == ck["done_idx"]
        assert len(self.qidx) == len(set(self.qidx)) == 90
        assert self.qidx == sorted(self.qidx)
        assert self.X.shape == (540, 30) and np.isfinite(self.X).all()
        assert sorted(set(self.y)) == list(range(15))
        assert len(self.ids) == len(set(self.ids)) == 540
        assert all(0 <= qi < 540 for qi in self.qidx)
        noise_seeds = [s + qi for s in SEEDS for qi in range(540)]
        assert len(noise_seeds) == len(set(noise_seeds)) == 2700
        assert np.bincount(self.y[self.qidx]).tolist() == [6]*15
        for proto in PROTOCOLS:
            for name in ["projection"] + METHODS:
                assert old["raw"][proto][name] == ck["raw"][proto][name]
        n2p = {str(s): int(p) for p, lst in fpd["partition_sensors"].items() for s in lst}
        pag = read_json(ROOT / "data/reference/partition_adjacency.json")
        adj = {(min(int(a), int(b)), max(int(a), int(b)))
               for a, ns in pag["adjacency"].items() for b in ns}
        src, dst = [], []
        for i, si in enumerate(self.sensors):
            for j, sj in enumerate(self.sensors):
                if i == j:
                    continue
                pi, pj = n2p.get(str(si), -1), n2p.get(str(sj), -1)
                if pi == pj or (min(pi, pj), max(pi, pj)) in adj:
                    src.append(i)
                    dst.append(j)
        self.edge = [src, dst]
        assert len(src) == 198
        self.pools = {}
        self.cell_pool = {}
        for proto in PROTOCOLS:
            for qi in self.qidx:
                idx = self.keep(qi, proto)
                key = digest_bytes(idx.astype("<i8").tobytes())[:20]
                rec = self.pools.setdefault(key, dict(indices=idx.tolist(), cells=[]))
                assert rec["indices"] == idx.tolist()
                rec["cells"].append([proto, qi])
                self.cell_pool[(proto, qi)] = key
        self.binding = digest_json(dict(config=CONFIG, sources=self.hashes,
                                        runner_sha256=self.script_hash, qidx=self.qidx))
        verification = {}
        for proto in PROTOCOLS:
            hits = []
            for qi in self.qidx:
                idx = self.keep(qi, proto)
                hit = int(self.y[idx[np.argmin(projection(self.observation(41, qi), self.X[idx]))]]
                          == self.y[qi])
                hits.append(hit)
            assert hits == old["raw"][proto]["projection"], f"seed41 anchor failed: {proto}"
            verification[proto] = dict(n=90, hits=sum(hits), matches_prior_raw=True)
        self.verification = verification

    def keep(self, qi, proto):
        q = self.all[qi]
        sn = np.asarray([f["leak_node"] == q["leak_node"] for f in self.all])
        sq = np.asarray([f["leak_rate_Ls"] == q["leak_rate_Ls"] for f in self.all])
        st = np.asarray([f["demand_period"] == q["demand_period"] for f in self.all])
        return np.flatnonzero({"H0": np.ones(540, bool), "H2": ~sn, "H3": ~sq,
                               "H4": ~st, "H8": ~(sn | sq | st)}[proto])

    def observation(self, seed, qi):
        return self.X[qi] + np.random.RandomState(seed + qi).normal(0, 0.02, 30)

    def manifest(self):
        queries = [{"corpus_index": qi, "scenario_id": self.ids[qi],
                    "true_label": int(self.y[qi]), "leak_node": self.all[qi]["leak_node"],
                    "magnitude_Ls": self.all[qi]["leak_rate_Ls"],
                    "demand_period": self.all[qi]["demand_period"]} for qi in self.qidx]
        return dict(binding=self.binding, configuration=CONFIG, sources_sha256=self.hashes,
                    runner_sha256=self.script_hash, query_index=queries,
                    sensors=self.sensors, edges=self.edge, pools=self.pools,
                    label_order=list(range(15)), seed41_projection_check=self.verification,
                    unique_pools=len(self.pools), fits_required=len(self.pools)*3*5)


class Runner:
    def __init__(self, data):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch_geometric
        from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
        self.t, self.F, self.data = torch, F, data
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert self.device.type == "cuda", "The exact E4 GNN training profile requires CUDA and the optional pinned GNN environment."
        from .architecture import build
        self.models = build(torch, nn, F, (GCNConv, GATConv, SAGEConv),
                            (global_mean_pool, global_max_pool))
        self.X = torch.tensor(data.X, dtype=torch.float, device=self.device)
        self.y = torch.tensor(data.y, dtype=torch.long, device=self.device)
        self.edge = torch.tensor(data.edge, dtype=torch.long, device=self.device)
        self._blocks = {}
        self.env = dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                        torch_geometric=torch_geometric.__version__, cuda=torch.version.cuda,
                        device=torch.cuda.get_device_name(), torch_threads=torch.get_num_threads(),
                        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())

    def block(self, size):
        if size not in self._blocks:
            self._blocks[size] = (self.t.cat([self.edge + k*30 for k in range(size)], dim=1),
                                 self.t.arange(size, device=self.device).repeat_interleave(30))
        return self._blocks[size]

    class Batch:
        __slots__ = ("x", "edge_index", "batch", "y")
        def __init__(self, x, edge, batch, y):
            self.x, self.edge_index, self.batch, self.y = x, edge, batch, y

    def batches(self, indices, size, generator=None, noise=None):
        t = self.t
        order = indices[t.randperm(indices.numel(), generator=generator, device=self.device)] \
            if generator is not None else indices
        result = []
        for i in range(0, order.numel(), size):
            sel = order[i:i+size]
            n = sel.numel()
            edge, batch = self.block(n)
            xb = self.X[sel]
            if noise is not None:
                xb = xb + noise*t.randn(xb.shape, device=self.device, generator=generator)
            result.append(self.Batch(xb.reshape(n*30, 1), edge, batch, self.y[sel]))
        return result

    def fit(self, model_name, indices, seed):
        t, F = self.t, self.F
        t.manual_seed(seed)
        perm = np.random.RandomState(seed).permutation(len(indices))
        nv = max(15, int(0.15*len(indices)))
        vi_np, ti_np = indices[perm[:nv]], indices[perm[nv:]]
        vi = t.tensor(vi_np, dtype=t.long, device=self.device)
        ti = t.tensor(ti_np, dtype=t.long, device=self.device)
        gen = t.Generator(device=self.device)
        gen.manual_seed(seed)
        va = self.batches(vi, 256)
        model = self.models[model_name](30).to(self.device)
        opt = t.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        sch = t.optim.lr_scheduler.StepLR(opt, step_size=60, gamma=0.5)
        best, state, best_ep, trace = -1.0, None, None, []
        t.cuda.synchronize()
        begin = time.perf_counter()
        for ep in range(1, 201):
            model.train()
            for b in self.batches(ti, 64, gen, noise=0.02):
                opt.zero_grad()
                F.cross_entropy(model(b), b.y).backward()
                opt.step()
            sch.step()
            if ep % 10 == 0:
                model.eval()
                ok = tot = 0
                with t.no_grad():
                    for b in va:
                        ok += int((model(b).argmax(1) == b.y).sum())
                        tot += int(b.y.numel())
                trace.append(dict(epoch=ep, hits=ok, n=tot))
                if tot and ok/tot > best:
                    best, best_ep = ok/tot, ep
                    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        assert state is not None
        model.load_state_dict(state)
        model.eval()
        t.cuda.synchronize()
        elapsed = time.perf_counter()-begin
        h = hashlib.sha256()
        for key, val in state.items():
            a = val.detach().cpu().numpy()
            h.update(key.encode())
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        meta = dict(train_indices=ti_np.tolist(), validation_indices=vi_np.tolist(),
                    train_scenario_ids=[self.data.ids[i] for i in ti_np],
                    validation_scenario_ids=[self.data.ids[i] for i in vi_np],
                    best_epoch=best_ep, best_validation_accuracy=best,
                    validation_trace=trace, model_state_sha256=h.hexdigest(), fit_seconds=elapsed)
        return model, meta

    def predict(self, model, vectors):
        t = self.t
        n = len(vectors)
        edge, batch = self.block(n)
        with t.no_grad():
            logits = model(self.Batch(t.tensor(np.asarray(vectors), dtype=t.float,
                                              device=self.device).reshape(n*30, 1),
                                      edge, batch, None))
            return logits.cpu().numpy(), t.softmax(logits, dim=1).cpu().numpy()


def ensure_manifest(data):
    path = OUT / "gnn_manifest.json"
    manifest = data.manifest()
    if path.exists():
        assert read_json(path)["binding"] == data.binding, "Refuse incompatible checkpoint manifest"
    else:
        save_json(path, manifest)


def result_path(name, seed, key):
    return OUT / "gnn_fits" / name.replace("+", "plus") / str(seed) / f"{key}.json"


def worker(name):
    data = Inputs()
    ensure_manifest(data)
    runner = Runner(data)
    print(json.dumps(dict(worker=name, environment=runner.env)), flush=True)
    total = len(data.pools)*len(SEEDS)
    completed, started = 0, time.perf_counter()
    for seed in SEEDS:
        for key, pool in data.pools.items():
            path = result_path(name, seed, key)
            if path.exists():
                prior = read_json(path)
                assert prior["binding"] == data.binding
                assert prior["method"] == name and prior["run_seed"] == seed
                assert prior["pool_key"] == key
                completed += 1
                continue
            idx = np.asarray(pool["indices"], dtype=int)
            model, meta = runner.fit(name, idx, seed)
            qis = sorted({qi for proto, qi in pool["cells"]})
            observations = [data.observation(seed, qi) for qi in qis]
            logits, probabilities = runner.predict(model, observations)
            pred = logits.argmax(axis=1)
            assert np.allclose(probabilities.sum(1), 1, atol=1e-6)
            lookup = {qi: k for k, qi in enumerate(qis)}
            records = []
            for proto, qi in pool["cells"]:
                k = lookup[qi]
                records.append(dict(protocol=proto, corpus_index=qi, scenario_id=data.ids[qi],
                                    true_label=int(data.y[qi]), predicted_label=int(pred[k]),
                                    correct=bool(pred[k] == data.y[qi]), logits=logits[k].tolist(),
                                    probabilities=probabilities[k].tolist(),
                                    observation_sha256=digest_bytes(observations[k].astype("<f8").tobytes())))
            save_json(path, dict(binding=data.binding, run_seed=seed, method=name, pool_key=key,
                                 retained_indices=idx.tolist(),
                                 retained_scenario_ids=[data.ids[i] for i in idx],
                                 environment=runner.env, training=meta, predictions=records))
            del model
            completed += 1
            print(f"{name} seed={seed} completed={completed}/{total} "
                  f"fit={meta['fit_seconds']:.2f}s best_epoch={meta['best_epoch']} "
                  f"elapsed={time.perf_counter()-started:.1f}s", flush=True)


def benchmark():
    data = Inputs()
    runner = Runner(data)
    print(json.dumps(data.verification), flush=True)
    print(json.dumps(runner.env), flush=True)
    print(f"unique pools={len(data.pools)}; five-seed fits={len(data.pools)*15}", flush=True)
    result = dict(binding=data.binding, environment=runner.env, benchmarks=[])
    for name in METHODS:
        m, meta = runner.fit(name, np.arange(540), 41)
        vectors = [data.observation(41, qi) for qi in data.qidx]
        batch_logits, _ = runner.predict(m, vectors)
        single_logits = np.vstack([runner.predict(m, [v])[0] for v in vectors])
        maximum_delta = float(np.abs(batch_logits-single_logits).max())
        same_labels = bool(np.array_equal(batch_logits.argmax(1), single_logits.argmax(1)))
        assert same_labels, f"Batched predictions changed labels: {name}"
        assert np.allclose(batch_logits, single_logits, atol=2e-5, rtol=2e-5), maximum_delta
        entry = dict(method=name, fit_seconds=meta["fit_seconds"], best_epoch=meta["best_epoch"],
                     batch_vs_single_max_logit_delta=maximum_delta, identical_labels=same_labels,
                     h0_hits=int((batch_logits.argmax(1) == data.y[data.qidx]).sum()))
        result["benchmarks"].append(entry)
        print(json.dumps(entry), flush=True)
        del m
    save_json(OUT / "gnn_benchmark.json", result)


def run_all():
    data = Inputs()
    ensure_manifest(data)
    children = []
    for name in METHODS:
        handle = (OUT / ("gnn_" + name.replace("+", "plus") + ".log")).open("a", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.Popen([sys.executable, "-B", "-m", "experiments.repeated_benchmarks.gnn", "worker", "--model", name, "--output-dir", str(OUT)],
                                cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        children.append((name, proc, handle))
        print(f"Started {name}: pid={proc.pid}", flush=True)
    while any(proc.poll() is None for _, proc, _ in children):
        done = len(list((OUT / "gnn_fits").glob("*/*/*.json")))
        print(f"Checkpointed fits {done}/{len(data.pools)*15}", flush=True)
        time.sleep(30)
    errors = []
    for name, proc, handle in children:
        handle.close()
        if proc.returncode:
            errors.append((name, proc.returncode))
    if errors:
        raise RuntimeError(errors)
    summarize()


def summarize():
    data = Inputs()
    ensure_manifest(data)
    records = []
    for name in METHODS:
        for seed in SEEDS:
            for key in data.pools:
                fit = read_json(result_path(name, seed, key))
                assert fit["binding"] == data.binding
                for rec in fit["predictions"]:
                    records.append(dict(run_seed=seed, method=name, pool_key=key, **rec))
    assert len(records) == 5*3*5*90
    lookup = {(r["run_seed"], r["method"], r["protocol"], r["corpus_index"]): r for r in records}
    assert len(lookup) == len(records)
    summary = {}
    for proto in PROTOCOLS:
        summary[proto] = {}
        for name in METHODS:
            counts = [sum(lookup[(seed, name, proto, qi)]["correct"] for qi in data.qidx) for seed in SEEDS]
            rates = 100*np.asarray(counts)/90
            summary[proto][name] = dict(n_queries=90, seeds=SEEDS, hits_by_seed=counts,
                                       top1_by_seed=rates.tolist(), mean_percent=float(rates.mean()),
                                       sample_sd_percent=float(rates.std(ddof=1)))
    for filename, expected in data.hashes.items():
        assert digest_bytes((ROOT / filename).read_bytes()) == expected
    save_json(OUT / "gnn_repeats_results.json", dict(binding=data.binding, configuration=CONFIG,
                                                     summary=summary, predictions=records,
                                                     original_sources_unchanged=True))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["verify", "benchmark", "worker", "run", "summarize"])
    parser.add_argument("--model", choices=METHODS)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/repeated_benchmarks")
    a = parser.parse_args()
    OUT = a.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "verify":
        data = Inputs()
        ensure_manifest(data)
        print(json.dumps(dict(binding=data.binding, unique_pools=len(data.pools),
                              fits_required=len(data.pools)*15,
                              seed41_projection_check=data.verification), indent=2), flush=True)
    elif a.mode == "benchmark":
        benchmark()
    elif a.mode == "worker":
        assert a.model
        worker(a.model)
    elif a.mode == "run":
        run_all()
    else:
        summarize()


def run(output_dir):
    global OUT
    OUT = Path(output_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    run_all()


if __name__ == "__main__":
    main()
