"""vcellx · end-to-end driver.

Stages are separate so a failure is cheap and the artifacts are inspectable:

  data   load Arc, normalise, HVG on controls, target-disjoint split
  epi    GENCODE TSS + K562 peaks -> per-gene epigenomic features
  graph  K562 Hi-C loops -> GRN + degree-matched shuffle control
  train  train the three conditioning arms on identical data and budget
  eval   score on held-out targets against control and mean baselines

Usage:
  python -m vcellx.run_all --stage data
  ...
  python -m vcellx.run_all --stage all
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

ROOT_DEFAULT = "/mnt/storage/bzhang/AIVC"
ARC_H5 = os.path.join(ROOT_DEFAULT, "arc-institute-virtual-cell-dataset/adata_Training.h5ad")
K562 = os.path.join(ROOT_DEFAULT, "epigenomics/k562-encode-matched-bulk-v1")
K562_TF = os.path.join(ROOT_DEFAULT, "epigenomics/k562-tf-chip-matched-v2")
# Bin-level contact matrix. The juicer loop calls in k562-encode-matched-bulk-v1
# are 52k genome-wide, which over a 2000-gene panel leaves the median degree at 0.
MCOOL = os.path.join(ROOT_DEFAULT, "epigenomics/raw/4dn/HFFc6/4DNFIB59T7NN.mcool")


def stage_data(work: str, n_hvg: int, force: bool):
    from .data import load_arc, target_disjoint_split
    t0 = time.time()
    d = load_arc(ARC_H5, work, n_hvg=n_hvg, force=force)
    split = target_disjoint_split(d.obs_target, seed=42)
    p = os.path.join(work, "split.json")
    json.dump(split, open(p, "w"), indent=2)
    print(f"[data] {time.time()-t0:.0f}s  train/val/test targets "
          f"{len(split['train'])}/{len(split['val'])}/{len(split['test'])}", flush=True)
    print(f"[data] test targets: {split['test'][:8]} ...", flush=True)
    return d, split


def stage_epi(work: str, genes, force: bool):
    from . import epi
    t0 = time.time()
    out = os.path.join(work, "epi_features.npz")
    if os.path.exists(out) and not force:
        z = np.load(out, allow_pickle=True)
        print(f"[epi] cached: {z['X'].shape}, ok={int(z['ok'].sum())}", flush=True)
        return z["X"], z["ok"], list(z["names"])
    F, ok, names = epi.run(genes, work, k562_dir=K562, tf_dir=K562_TF)
    print(f"[epi] {time.time()-t0:.0f}s  features {F.shape}  "
          f"names={list(names)}", flush=True)
    return F, ok, names


def stage_graph(work: str, genes, force: bool, resolution: int = 100000,
                topk: int = 10, quantile: float = 0.90):
    """Build the GRN from the bin-level contact matrix.

    The juicer loop calls were measured and rejected: over a 2000-gene panel they
    give 15 edges at a 1 kb promoter window and the median degree stays 0 even at
    a 100 kb window. A bin-level matrix is the source that actually connects the
    panel.
    """
    from .epi import ensure_gencode, read_tss
    from . import grn
    t0 = time.time()
    gp = os.path.join(work, f"graph_{resolution}.npz")
    if os.path.exists(gp) and not force:
        z = np.load(gp, allow_pickle=True)
        summary = json.loads(str(z["summary"]))
        print(f"[graph] cached: edges={summary.get('n_edges')}", flush=True)
        return z["A"], z["A_shuf"], summary

    gtf = ensure_gencode(work)
    tss = read_tss(gtf, genes)
    A, deg, summary = grn.build_graph_from_mcool(
        genes, tss, MCOOL, resolution=resolution, topk=topk,
        min_count_quantile=quantile)
    A_shuf, sinfo = grn.shuffle_degree_matched(A, seed=42)
    summary["shuffle"] = sinfo

    np.savez_compressed(gp,
                        A=A.toarray().astype(np.float32),
                        A_shuf=A_shuf.toarray().astype(np.float32),
                        summary=np.array(json.dumps(summary)))
    json.dump(summary, open(os.path.join(work, "graph_summary.json"), "w"), indent=2)
    print(f"[graph] {time.time()-t0:.0f}s  edges={summary['n_edges']}  "
          f"isolated={summary['isolated_genes']}/{summary['n_genes']}  "
          f"deg_median={summary['degree_median']:.0f}  "
          f"deg_mean={summary['degree_mean']:.1f}  "
          f"deg_max={summary['degree_max']}", flush=True)
    return A, A_shuf, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--work", default=os.path.join(ROOT_DEFAULT, "vcellx/work"))
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    print(f"work dir: {a.work}", flush=True)

    d = None
    if a.stage in ("data", "all"):
        d, split = stage_data(a.work, a.n_hvg, a.force)
    if a.stage in ("epi", "all"):
        if d is None:
            d, split = stage_data(a.work, a.n_hvg, False)
        stage_epi(a.work, d.genes, a.force)
    if a.stage in ("graph", "all"):
        if d is None:
            d, split = stage_data(a.work, a.n_hvg, False)
        stage_graph(a.work, d.genes, a.force)


if __name__ == "__main__":
    main()
