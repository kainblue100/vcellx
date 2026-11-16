"""vcellx · stage P: pretrain a cell-embedding foundation model.

Not a copy of anyone's architecture. The idea is the simplest thing that
actually forces a useful cell representation:

  mask ~30% of the genes in each cell
  encode the remaining 70% to one latent vector
  reconstruct the *masked* genes only from that vector

Reconstructing only the masked positions is what makes the latent carry cell
state rather than a copy of the input. Training across several datasets with a
shared gene panel is what makes it a pretrained representation rather than a
per-dataset autoencoder.

Panel choice matters for honesty: it is selected ONLY from genes present in
both the external corpora and the Arc panel, by variance in the external data.
No Arc perturbed cells are used to choose features.
"""
from __future__ import annotations

import json
import os

import h5py
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

B = "/mnt/storage/bzhang/AIVC"


# --------------------------------------------------------------------------
# gene panel
# --------------------------------------------------------------------------
def read_names(group, name):
    if name not in group:
        return None
    node = group[name]
    if isinstance(node, h5py.Group):
        cats = node["categories"][:]
        cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
        codes = node["codes"][:]
        out = np.empty(len(codes), dtype=object)
        ok = codes >= 0
        out[ok] = cats[codes[ok]]
        out[~ok] = None
        return out
    return np.array([x.decode() if isinstance(x, bytes) else x for x in node[:]])


def tabula_panel(n_genes: int, arc_genes: list, out_npz: str, max_cells: int | None = None):
    """Pick the panel by variance in Tabula among genes that Arc also has."""
    if os.path.exists(out_npz):
        z = np.load(out_npz, allow_pickle=True)
        print(f"[panel] cached {z['genes'].shape}", flush=True)
        return z["genes"], z["mean"], z["std"]

    p = os.path.join(B, "Tabula-Sapiens-v2-CELLxGENE/Tabula_Sapiens_v2_All_Cells.h5ad")
    print("[panel] scanning Tabula var ...", flush=True)
    with h5py.File(p, "r") as f:
        sym = read_names(f["var"], "feature_name")
        keep = np.array([s in set(arc_genes) for s in sym])
        col_idx = np.where(keep)[0]
        genes_kept = sym[col_idx]
        n_cells = int(f["X"].attrs["shape"][0])
    print(f"[panel] Tabula genes also in Arc: {keep.sum():,} / {len(sym):,}", flush=True)

    # two-pass variance over a bounded subsample, then keep the top n_genes
    n_use = min(n_cells, max_cells or 120_000)
    rows = np.sort(np.random.default_rng(0).choice(n_cells, n_use, replace=False))
    Xs = _read_submatrix(p, rows, col_idx)
    Xs = _normalise(Xs)
    mean = np.asarray(Xs.mean(axis=0)).ravel()
    sq = np.asarray(Xs.multiply(Xs).mean(axis=0)).ravel()
    var = np.maximum(sq - mean ** 2, 0)
    order = np.argsort(-var)[:n_genes]
    order.sort()
    genes = genes_kept[order]
    mean = mean[order]
    std = np.sqrt(var[order]).astype(np.float32)
    np.savez(out_npz, genes=genes, mean=mean, std=std)
    print(f"[panel] selected {len(genes)} genes by Tabula variance", flush=True)
    return genes, mean, std


def _read_submatrix(h5_path, rows, cols):
    """Read a row/col subset of the h5ad X (CSR, possibly gzip) as CSR float32."""
    with h5py.File(h5_path, "r") as f:
        Xg = f["X"]
        shape = tuple(int(v) for v in Xg.attrs["shape"])
        indptr = Xg["indptr"][:]
        data = Xg["data"]
        indices = Xg["indices"]
        out_r, out_c, out_v = [], [], []
        cursor = 0
        for i in rows:
            s, e = int(indptr[i]), int(indptr[i + 1])
            if e > s:
                idx = indices[s:e]
                v = data[s:e]
                m = np.isin(idx, cols)
                if m.any():
                    pos = np.searchsorted(cols, idx[m])
                    out_r.append(np.full(m.sum(), cursor, dtype=np.int64))
                    out_c.append(pos.astype(np.int64))
                    out_v.append(v[m].astype(np.float32))
            cursor += 1
    if not out_r:
        return sp.csr_matrix((len(rows), len(cols)), dtype=np.float32)
    r = np.concatenate(out_r); c = np.concatenate(out_c); v = np.concatenate(out_v)
    return sp.csr_matrix((v, (r, c)), shape=(len(rows), len(cols)), dtype=np.float32)


def _normalise(X: sp.csr_matrix, target: float = 1e4) -> sp.csr_matrix:
    lib = np.asarray(X.sum(axis=1)).ravel()
    scale = np.ones_like(lib, dtype=np.float32)
    nz = lib > 0
    scale[nz] = (target / lib[nz]).astype(np.float32)
    Y = (sp.diags(scale) @ X).tocsr()
    Y.data = np.log1p(Y.data)
    return Y


# --------------------------------------------------------------------------
# masked autoencoder
# --------------------------------------------------------------------------
class CellMAE(nn.Module):
    """Masked-expression denoising autoencoder.

    The first projection (n_genes -> hidden) dominates the FLOPs because
    n_genes is 4096. On CPU-only torch a hidden of 1024 costs about 20 minutes
    per epoch over 400k cells, so the default here is deliberately smaller:
    the representation is what matters, not the width of the first layer.
    """

    def __init__(self, n_genes: int, d: int = 256, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_genes, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, d),
        )
        self.dec = nn.Sequential(
            nn.Linear(d, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, n_genes),
        )
        self.n_genes = n_genes

    def forward(self, x, keep_mask):
        """x: (B, G) with masked entries already zeroed. keep_mask: (B, G) bool."""
        z = self.enc(x)
        rec = self.dec(z)
        return z, rec


def pretrain(npz_path: str, genes: np.ndarray, out_dir: str, max_cells: int = 400_000,
             epochs: int = 12, bs: int = 512, lr: float = 1e-3, mask_frac: float = 0.30,
             d: int = 256, hidden: int = 512, threads: int = 16,
             seed: int = 0, device: str = "cpu", sources=None):
    """Train the MAE on cached float16 expression matrices from one or more sources.

    sources: list of (npz_path, n_cells) describing cached panels with arrays
             'X' (n_cells, n_genes) float16 and 'genes'.
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cpu" and threads:
        torch.set_num_threads(threads)
        print(f"[pretrain] torch threads: {torch.get_num_threads()}", flush=True)

    mats = []
    for path, desc in (sources or []):
        if not os.path.exists(path):
            print(f"[pretrain] skip missing {path}", flush=True)
            continue
        z = np.load(path, allow_pickle=True)
        src_genes = list(z["genes"])
        g2i = {g: i for i, g in enumerate(src_genes)}
        take = np.array([g2i.get(g, -1) for g in genes])
        have = take >= 0
        if have.sum() < 0.9 * len(genes):
            print(f"[pretrain] {desc}: only {have.sum()}/{len(genes)} panel genes, skip", flush=True)
            continue
        X = np.zeros((z["X"].shape[0], len(genes)), dtype=np.float16)
        idx = np.where(have)[0]
        X[:, idx] = z["X"][:, take[have]]
        mats.append(X)
        print(f"[pretrain] {desc}: {X.shape[0]:,} cells, {have.sum()} genes filled", flush=True)

    if not mats:
        raise SystemExit("no pretraining sources available")
    X = np.concatenate(mats, 0)
    if X.shape[0] > max_cells:
        sel = np.random.default_rng(seed).choice(X.shape[0], max_cells, replace=False)
        X = X[sel]
    print(f"[pretrain] training matrix {X.shape}", flush=True)

    model = CellMAE(len(genes), d=d, hidden=hidden).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[pretrain] model params {n_par/1e6:.2f} M  (d={d}, hidden={hidden})", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = X.shape[0]
    hist = []
    for ep in range(epochs):
        perm = np.random.permutation(n)
        run, nb = 0.0, 0
        for i in range(0, n, bs):
            b = torch.as_tensor(X[perm[i:i + bs]].astype(np.float32))
            mask = torch.rand(b.shape) < mask_frac
            xin = b.masked_fill(mask, 0.0)
            z, rec = model(xin, ~mask)
            loss = ((rec - b) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += float(loss.detach()); nb += 1
        sched.step()
        hist.append(run / max(nb, 1))
        print(f"  [pretrain] epoch {ep+1}/{epochs}  masked MSE {hist[-1]:.5f}", flush=True)

    torch.save({"enc": model.enc.state_dict(), "d": d,
                "n_genes": len(genes), "genes": np.array(genes)},
               os.path.join(out_dir, "encoder.pt"))
    json.dump({"epochs": epochs, "mask_frac": mask_frac, "d": d,
               "cells": int(n), "genes": int(len(genes)),
               "loss": hist,
               "sources": [s[1] for s in (sources or [])]},
              open(os.path.join(out_dir, "pretrain.json"), "w"), indent=2)
    print(f"[pretrain] saved encoder.pt", flush=True)
    return hist


def cache_tabula(npz_path: str, genes: np.ndarray, max_cells: int = 400_000,
                 chunk: int = 40_000, seed: int = 0):
    """Slice Tabula onto the panel and cache as float16 for pretraining."""
    if os.path.exists(npz_path):
        print(f"[cache] exists {npz_path}", flush=True)
        return
    p = os.path.join(B, "Tabula-Sapiens-v2-CELLxGENE/Tabula_Sapiens_v2_All_Cells.h5ad")
    with h5py.File(p, "r") as f:
        sym = read_names(f["var"], "feature_name")
        n_cells = int(f["X"].attrs["shape"][0])
    g2i = {g: i for i, g in enumerate(sym)}
    cols = np.array(sorted(g2i[g] for g in genes if g in g2i))
    gene_ok = np.array([g in g2i for g in genes])
    print(f"[cache] panel genes present in Tabula: {gene_ok.sum()}/{len(genes)}", flush=True)

    rng = np.random.default_rng(seed)
    n_use = min(n_cells, max_cells)
    rows = np.sort(rng.choice(n_cells, n_use, replace=False))

    out = np.zeros((n_use, len(genes)), dtype=np.float16)
    pos_in_cols = {c: k for k, c in enumerate(cols)}
    gene_to_col = {}
    for gi, g in enumerate(genes):
        if g in g2i and g2i[g] in pos_in_cols:
            gene_to_col[gi] = pos_in_cols[g2i[g]]

    with h5py.File(p, "r") as f:
        Xg = f["X"]
        indptr = Xg["indptr"][:]
        data_all = Xg["data"]; ind_all = Xg["indices"]
        done = 0
        for start in range(0, n_use, chunk):
            rs = rows[start:start + chunk]
            sub_r, sub_c, sub_v = [], [], []
            for k, r in enumerate(rs):
                s, e = int(indptr[r]), int(indptr[r + 1])
                if e <= s:
                    continue
                idx = ind_all[s:e]; v = data_all[s:e]
                m = np.isin(idx, cols)
                if not m.any():
                    continue
                sub_r.append(np.full(int(m.sum()), k, dtype=np.int64))
                sub_c.append(np.array([pos_in_cols[int(x)] for x in idx[m]], dtype=np.int64))
                sub_v.append(v[m].astype(np.float32))
            if not sub_r:
                continue
            M = sp.csr_matrix((np.concatenate(sub_v),
                               (np.concatenate(sub_r), np.concatenate(sub_c))),
                              shape=(len(rs), len(cols)))
            M = _normalise(M)
            dense = np.asarray(M.todense(), dtype=np.float32)
            for gi, ck in gene_to_col.items():
                out[start:start + len(rs), gi] = dense[:, ck].astype(np.float16)
            done += len(rs)
            if done % (chunk * 5) == 0:
                print(f"  [cache] {done:,}/{n_use:,} cells", flush=True)
    np.savez_compressed(npz_path, X=out, genes=np.array(genes))
    print(f"[cache] wrote {npz_path} {out.shape}", flush=True)
