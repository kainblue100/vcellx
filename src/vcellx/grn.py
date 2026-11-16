"""vcellx · gene regulatory graph from public chromatin-contact data.

Construction rule (deliberately narrow and auditable):

  node  = a gene on the expression gene axis
  edge  A-B  exists when a Hi-C loop has one anchor overlapping A's promoter
            window and the other anchor overlapping B's promoter window
  weight     = the loop's `observed` count from the juicer_tools loop file

Nothing here looks at the expression data, and nothing looks at the held-out
targets' response. The graph is built before training from public assays only.

Two controls are produced alongside:
  shuffle  degree-matched topology randomisation, so "more neighbours seen" is
           held fixed and only the wiring changes
  degree   the per-node degree vector, for reporting
"""
from __future__ import annotations

import gzip
import json
import os

import numpy as np
import scipy.sparse as sp


def read_loops(path: str, min_fdr: float = 0.05, use_col: str = "observed"):
    """Read juicer_tools bedpe loops -> list of (chr1, s1, e1, chr2, s2, e2, w)."""
    loops = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if header is None and p[0] == "chr1":
                header = p
                continue
            if len(p) < 12:
                continue
            try:
                x1, x2 = int(p[1]), int(p[2])
                y1, y2 = int(p[4]), int(p[5])
            except ValueError:
                continue
            w = 1.0
            if use_col == "observed" and len(p) > 11:
                try:
                    w = float(p[11])
                except ValueError:
                    w = 1.0
            # fdr filter, if present
            if header and "fdrBL" in header:
                k = header.index("fdrBL")
                if len(p) > k:
                    try:
                        if float(p[k]) > min_fdr:
                            continue
                    except ValueError:
                        pass
            loops.append((p[0], x1, x2, p[3], y1, y2, w))
    return loops


def _which_genes(chrom_starts, lo, hi):
    """Indices of TSS entries in [lo, hi) for one chromosome."""
    if chrom_starts is None:
        return np.zeros(0, dtype=np.int64)
    st = chrom_starts
    return np.where((st >= lo) & (st < hi))[0]


def build_graph(genes, tss, loops, promoter_pad: int = 1000,
                topk: int = 10, min_weight: float = 0.0):
    """Return symmetric CSR adjacency over gene indices plus a summary dict."""
    G = len(genes)
    gidx = {}
    for i, g in enumerate(genes):
        if g in tss:
            gidx[g] = i
    by_chrom = {}
    for g, i in gidx.items():
        c, t, _ = tss[g]
        by_chrom.setdefault(c, []).append((t, i))
    chrom_arr = {}
    for c, v in by_chrom.items():
        v.sort()
        chrom_arr[c] = (np.array([x[0] for x in v], dtype=np.int64),
                        np.array([x[1] for x in v], dtype=np.int64))

    rows, cols, vals = [], [], []
    n_same = 0
    for c1, a1, b1, c2, a2, b2, w in loops:
        if w <= min_weight:
            continue
        for (ca, lo1, hi1, cb, lo2, hi2) in ((c1, a1, b1, c2, a2, b2),
                                             (c2, a2, b2, c1, a1, b1)):
            ch_a = chrom_arr.get(ca)
            ch_b = chrom_arr.get(cb)
            if ch_a is None or ch_b is None:
                continue
            ia = _which_genes(ch_a[0], lo1 - promoter_pad, hi1 + promoter_pad)
            ib = _which_genes(ch_b[0], lo2 - promoter_pad, hi2 + promoter_pad)
            if len(ia) == 0 or len(ib) == 0:
                continue
            for x in ch_a[1][ia]:
                for y in ch_b[1][ib]:
                    if x == y:
                        continue
                    rows.append(int(x)); cols.append(int(y)); vals.append(float(w))
                    if ca == cb:
                        n_same += 1

    if not rows:
        raise SystemExit("graph has no edges: check promoter windows / loop file")

    A = sp.coo_matrix((vals, (rows, cols)), shape=(G, G)).tocsr()
    A.sum_duplicates()
    A = A.maximum(A.T)              # symmetrise

    if topk > 0:
        A = _topk(A, topk)

    deg = np.asarray((A > 0).sum(axis=1)).ravel()
    summary = {
        "n_genes": int(G),
        "n_genes_with_tss": int(len(gidx)),
        "n_edges": int((A > 0).nnz // 2) if A.nnz else 0,
        "isolated_genes": int((deg == 0).sum()),
        "degree_median": float(np.median(deg)),
        "degree_max": int(deg.max()),
        "weight_sum": float(A.sum()),
        "loops_used": int(len(loops)),
        "promoter_pad": promoter_pad,
        "topk": topk,
        "source": "K562 in-situ Hi-C loops (ENCODE), juicer_tools observed",
    }
    return A.tocsr(), deg, summary


def _topk(A: sp.csr_matrix, k: int) -> sp.csr_matrix:
    A = A.tocsr()
    rows, cols, vals = [], [], []
    for i in range(A.shape[0]):
        s, e = A.indptr[i], A.indptr[i + 1]
        if s == e:
            continue
        idx = A.indices[s:e]
        v = A.data[s:e]
        if len(v) > k:
            keep = np.argpartition(-v, k)[:k]
            idx, v = idx[keep], v[keep]
        rows.extend([i] * len(idx)); cols.extend(idx.tolist()); vals.extend(v.tolist())
    A2 = sp.coo_matrix((vals, (rows, cols)), shape=A.shape).tocsr()
    A2 = A2.maximum(A2.T)
    return A2


def shuffle_degree_matched(A: sp.csr_matrix, seed: int = 42, n_swap: int | None = None):
    """Randomise targets while keeping each node's degree.

    Constrained transpositions: pick two nodes u,v with different targets,
    pick a target of u and a target of v, and swap them if the result introduces
    no duplicate edges. The out-degree of every node is preserved exactly.
    """
    A = A.tocoo()
    # directed edge list on the symmetrised graph
    rows, cols = A.row.copy(), A.col.copy()
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    n_edges = len(rows)
    if n_swap is None:
        n_swap = max(1000, n_edges * 20)

    nb = {}
    for r, c in zip(rows, cols):
        nb.setdefault(int(r), set()).add(int(c))

    done = 0
    tries = 0
    while done < n_swap and tries < n_swap * 40:
        tries += 1
        e1 = rng.integers(n_edges)
        e2 = rng.integers(n_edges)
        u1, t1 = int(rows[e1]), int(cols[e1])
        u2, t2 = int(rows[e2]), int(cols[e2])
        if t1 == t2 or u1 == u2:
            continue
        if t2 in nb[u1] or t1 in nb[u2]:
            continue
        nb[u1].discard(t1); nb[u1].add(t2)
        nb[u2].discard(t2); nb[u2].add(t1)
        rows[e1], cols[e1] = u1, t2
        rows[e2], cols[e2] = u2, t1
        done += 1

    S = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=A.shape).tocsr()
    S = S.maximum(S.T)
    return S, {"swaps": int(done), "attempts": int(tries)}
