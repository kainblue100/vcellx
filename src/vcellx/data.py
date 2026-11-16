"""vcellx · data loading, splitting and set batching.

Design choices that matter for validity:

* Feature selection (HVGs) and normalisation statistics are computed on
  **control cells only**. Using perturbed cells, and especially held-out
  targets, to pick genes would leak the thing we are trying to predict.
* The split is **target-disjoint**: a target gene appears in exactly one of
  train/val/test. Same-target splits are how these models report numbers that
  do not survive an honest held-out evaluation.
* Training examples are **sets** of cells, not single cells, because the loss
  compares distributions (see train.py). Sets are built within a
  (target, batch) group and paired with control sets from the same batch.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import h5py
import numpy as np
import scipy.sparse as sp

CONTROL_LABEL = "non-targeting"


@dataclass
class ArcData:
    obs_target: np.ndarray            # (n_cells,) str
    obs_batch: np.ndarray             # (n_cells,) str
    X: sp.csr_matrix                  # (n_cells, n_genes) log1p-normalised
    genes: np.ndarray                 # (n_genes,) str
    hvg_idx: np.ndarray               # indices kept
    control_mask: np.ndarray          # bool


def _read_cats(f, path):
    g = f[path]
    cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in g["categories"][:]])
    codes = g["codes"][:]
    return cats, codes


def load_arc(h5_path: str, out_dir: str, n_hvg: int = 2000, force: bool = False) -> ArcData:
    """Load the Arc VCC training set, normalise, and cache a compact npz."""
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, f"arc_hvg{n_hvg}.npz")
    meta_p = os.path.join(out_dir, f"arc_hvg{n_hvg}.json")

    if os.path.exists(cache) and os.path.exists(meta_p) and not force:
        z = np.load(cache, allow_pickle=True)
        meta = json.load(open(meta_p))
        return ArcData(
            obs_target=z["obs_target"], obs_batch=z["obs_batch"],
            X=sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(meta["shape"])),
            genes=z["genes"], hvg_idx=np.arange(len(z["genes"])),
            control_mask=z["control_mask"],
        )

    with h5py.File(h5_path, "r") as f:
        genes = np.array([g.decode() if isinstance(g, bytes) else str(g)
                          for g in f["var"]["_index"][:]])
        tcats, tcodes = _read_cats(f, "obs/target_gene")
        bcats, bcodes = _read_cats(f, "obs/batch")
        Xg = f["X"]
        shape = tuple(int(v) for v in Xg.attrs["shape"])
        X = sp.csr_matrix((Xg["data"][:], Xg["indices"][:], Xg["indptr"][:]),
                          shape=shape, dtype=np.float32)
    obs_target = tcats[tcodes]
    obs_batch = bcats[bcodes]
    control_mask = obs_target == CONTROL_LABEL
    print(f"cells {X.shape[0]:,}  genes {X.shape[1]:,}  "
          f"control {int(control_mask.sum()):,}  "
          f"targets {len(set(obs_target)) - 1}", flush=True)

    # ---- normalise on control cells only ----
    lib = np.asarray(X.sum(axis=1)).ravel()
    lib_ctrl = lib[control_mask]
    target_depth = float(np.median(lib_ctrl))
    print(f"control median depth {target_depth:,.0f}", flush=True)

    Xn = X.copy()
    scale = np.ones_like(lib, dtype=np.float32)
    nz = lib > 0
    scale[nz] = (target_depth / lib[nz]).astype(np.float32)
    Xn = sp.diags(scale) @ Xn
    Xn = Xn.tocsr()
    Xn.data = np.log1p(Xn.data)

    # ---- HVG by variance on control cells only ----
    C = Xn[control_mask]
    mean = np.asarray(C.mean(axis=0)).ravel()
    sq = np.asarray(C.multiply(C).mean(axis=0)).ravel()
    var = np.maximum(sq - mean ** 2, 0.0)
    hvg_idx = np.argsort(-var)[:n_hvg]
    hvg_idx.sort()
    print(f"HVG selected on controls: {len(hvg_idx)} genes", flush=True)

    Xh = Xn[:, hvg_idx].tocsr().astype(np.float32)
    genes_h = genes[hvg_idx]

    np.savez_compressed(
        cache, data=Xh.data, indices=Xh.indices, indptr=Xh.indptr,
        obs_target=obs_target, obs_batch=obs_batch,
        genes=genes_h, control_mask=control_mask,
    )
    json.dump({"shape": list(Xh.shape), "target_depth": target_depth,
               "source": h5_path, "hvg_source": "control cells only"},
              open(meta_p, "w"), indent=2)

    return ArcData(obs_target=obs_target, obs_batch=obs_batch, X=Xh,
                   genes=genes_h, hvg_idx=np.arange(len(genes_h)),
                   control_mask=control_mask)


# --------------------------------------------------------------------------
# target-disjoint split
# --------------------------------------------------------------------------
def target_disjoint_split(targets, seed: int = 42, frac_val=0.15, frac_test=0.20):
    """Return dict with lists of target genes for train / val / test.

    Deterministic given the seed. Controls are not a target and never appear
    here; they are shared by all splits as the input distribution.
    """
    t = np.array(sorted(set(targets) - {CONTROL_LABEL}))
    rng = np.random.default_rng(seed)
    rng.shuffle(t)
    n = len(t)
    n_test = max(1, int(round(n * frac_test)))
    n_val = max(1, int(round(n * frac_val)))
    test = sorted(t[:n_test].tolist())
    val = sorted(t[n_test:n_test + n_val].tolist())
    train = sorted(t[n_test + n_val:].tolist())
    return {"train": train, "val": val, "test": test}


# --------------------------------------------------------------------------
# set batching
# --------------------------------------------------------------------------
@dataclass
class SetBatch:
    ctrl: np.ndarray        # (B, S, G) float32
    pert: np.ndarray        # (B, S, G) float32
    tgt_label: list         # list[str] length B
    batch_label: list       # list[str] length B
    sizes: np.ndarray = field(default_factory=lambda: np.zeros(0))


def make_set_batches(d: ArcData, split: dict, which: str, set_size: int = 64,
                     n_per_target: int = 40, seed: int = 0, control_pool=None):
    """Build set pairs. Each item is (perturbed set, matched control set).

    Matching is on batch, which is the only nuisance covariate the training
    file carries. Controls are drawn without replacement per group where
    possible so a single control cell is not reused across every set.
    """
    rng = np.random.default_rng(seed)
    targets = set(split[which])
    X = d.X
    obs_t, obs_b = d.obs_target, d.obs_batch
    ctrl_mask = d.control_mask

    by_batch_ctrl = {}
    for b in np.unique(obs_b):
        idx = np.where(ctrl_mask & (obs_b == b))[0]
        if len(idx) >= set_size:
            by_batch_ctrl[b] = idx

    samples = []
    for t in sorted(targets):
        for b in by_batch_ctrl:
            idx = np.where((obs_t == t) & (obs_b == b))[0]
            if len(idx) < set_size:
                continue
            for _ in range(n_per_target):
                pi = rng.choice(idx, size=set_size, replace=False)
                ci = rng.choice(by_batch_ctrl[b], size=set_size, replace=False)
                samples.append((ci, pi, t, b))

    rng.shuffle(samples)
    B = len(samples)
    G = X.shape[1]
    ctrl = np.zeros((B, set_size, G), dtype=np.float32)
    pert = np.zeros((B, set_size, G), dtype=np.float32)
    tl, bl = [], []
    for k, (ci, pi, t, b) in enumerate(samples):
        ctrl[k] = X[ci].toarray()
        pert[k] = X[pi].toarray()
        tl.append(t); bl.append(b)
    return SetBatch(ctrl=ctrl, pert=pert, tgt_label=tl, batch_label=bl,
                    sizes=np.full(B, set_size))
