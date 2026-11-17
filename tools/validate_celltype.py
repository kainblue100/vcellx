"""The correct form of Test A: aggregate by cell type, not by random pools.

Two earlier designs were wrong and are recorded so the mistake is not repeated:
  per-cell      -> ATAC nonzero 13%, sparsity swamps the signal (median r=0.004)
  random pools  -> each pool has the same cell-type mix, so between-pool
                   variance is mostly counting noise (median r=-0.011)

Aggregating by cell type keeps the biological variance that is supposed to make
promoter accessibility and expression move together.
"""
import gzip
import os

import numpy as np
import scipy.sparse as sp

B = "/mnt/storage/bzhang/AIVC/paired-trimodal-curation"
WORK = "/mnt/storage/bzhang/AIVC/vcellx/work"
RNA = os.path.join(B, "GSE303006-metadata-v1/GSM9115533_mesc.RNAcounts.gene.total.format.tsv.gz")
ATAC = os.path.join(B, "GSE303006-mESC-v1/GSM9115537_mesc.atac.fragment.tsv.gz")
QC = os.path.join(B, "GSE303006-metadata-v1/GSE303006_charm_metadata_qcpass.tsv.gz")
MGTF = os.path.join(WORK, "gencode.vM25.basic.annotation.gtf.gz")
PROM_UP, PROM_DN = 2000, 500
MIN_CELLS = 10


def read_tss(path, want):
    out = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[2] != "gene":
                continue
            a = p[8]
            if 'gene_type "' in a and a.split('gene_type "')[1].split('"')[0] not in ("protein_coding", "lncRNA"):
                continue
            if 'gene_name "' not in a:
                continue
            n = a.split('gene_name "')[1].split('"')[0]
            if n not in want or n in out:
                continue
            s, e = int(p[3]), int(p[4])
            out[n] = (p[0], s if p[6] == "+" else e)
    return out


def corr_cols(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    sx = np.sqrt((X ** 2).sum(0)); sy = np.sqrt((Y ** 2).sum(0))
    ok = (sx > 0) & (sy > 0)
    r = np.full(X.shape[1], np.nan, np.float32)
    r[ok] = (X[:, ok] * Y[:, ok]).sum(0) / (sx[ok] * sy[ok])
    return r


def main():
    with gzip.open(RNA, "rt", encoding="utf-8", errors="replace") as f:
        cells = f.readline().rstrip("\n").split("\t")[1:]
        genes, rows = [], []
        for line in f:
            p = line.rstrip("\n").split("\t")
            genes.append(p[0]); rows.append(np.array(p[1:], np.float32))
    M = np.vstack(rows)

    # QC: cellname -> celltype
    ct = {}
    with gzip.open(QC, "rt", encoding="utf-8", errors="replace") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        ci, ti = hdr.index("cellname"), hdr.index("celltype")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) > max(ci, ti):
                ct[p[ci]] = p[ti]

    keep, keep_ct = [], []
    for c in cells:
        k = "R1" + c if ("R1" + c) in ct else (c if c in ct else None)
        if k is not None:
            keep.append(c); keep_ct.append(ct[k])
    keep_ct = np.array(keep_ct)
    print(f"匹配细胞 {len(keep)}；细胞类型 {len(set(keep_ct))} 种", flush=True)
    uniq, counts = np.unique(keep_ct, return_counts=True)
    for u, n in zip(uniq, counts):
        print(f"  {u:<24s} {n:>4d}", flush=True)
    keep_ct_use = [u for u, n in zip(uniq, counts) if n >= MIN_CELLS]
    print(f"用于聚合的类型数（>={MIN_CELLS} 细胞）: {len(keep_ct_use)}", flush=True)

    tss = read_tss(MGTF, set(genes))
    glist = list(tss)
    gidx = {g: i for i, g in enumerate(glist)}
    by_chrom = {}
    for g in glist:
        c, t = tss[g]
        by_chrom.setdefault(c, []).append((t - PROM_UP, t + PROM_DN, gidx[g]))
    for c in by_chrom:
        by_chrom[c].sort()
    starts = {c: np.array([x[0] for x in v]) for c, v in by_chrom.items()}

    cidx = {c: i for i, c in enumerate(keep)}
    rr, cc_ = [], []
    with gzip.open(ATAC, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            ci = cidx.get(p[6])
            if ci is None:
                continue
            v = by_chrom.get(p[0])
            if v is None:
                continue
            try:
                mid = (int(p[1]) + int(p[2])) // 2
            except ValueError:
                continue
            st = starts[p[0]]
            k = int(np.searchsorted(st, mid, side="right")) - 1
            if k < 0:
                continue
            lo, hi, gi = v[k]
            if lo <= mid < hi:
                rr.append(ci); cc_.append(gi)
    A = sp.csr_matrix((np.ones(len(rr), np.float32), (rr, cc_)),
                      shape=(len(keep), len(glist)))
    cell_pos = {c: i for i, c in enumerate(cells)}
    R = M[:, [cell_pos[c] for c in keep]].T[:, [gidx[g] for g in glist]]
    print(f"ATAC {A.shape}  RNA {R.shape}", flush=True)

    # ---- aggregate by cell type ----
    Ab, Rb = [], []
    for t in keep_ct_use:
        sel = np.where(keep_ct == t)[0]
        Ab.append(np.asarray(A[sel].sum(0)).ravel())
        Rb.append(np.asarray(R[sel].sum(0)).ravel())
    Ab = np.vstack(Ab); Rb = np.vstack(Rb)
    print(f"聚合后 {Ab.shape[0]} 种类型 x {Ab.shape[1]} 基因", flush=True)

    r = corr_cols(Ab, Rb)
    fin = np.isfinite(r)
    print("\n=== Test A（按细胞类型聚合）ATAC vs RNA 逐基因相关 ===", flush=True)
    print(f"  可计算 {int(fin.sum())} 个基因")
    print(f"  中位 r = {np.nanmedian(r):.4f}")
    print(f"  均值 r = {np.nanmean(r):.4f}")
    print(f"  r > 0.5 比例 = {np.mean(r[fin] > 0.5)*100:.1f}%")
    print(f"  r > 0.3 比例 = {np.mean(r[fin] > 0.3)*100:.1f}%")

    rng = np.random.default_rng(0)
    perm = rng.permutation(Ab.shape[1])
    rn = corr_cols(Ab[:, perm], Rb)
    print(f"\n  打乱对照：中位 r = {np.nanmedian(rn):.4f}  均值 = {np.nanmean(rn):.4f}", flush=True)

    np.savez_compressed(os.path.join(WORK, "paired_celltype.npz"),
                        genes=np.array(glist), r=r, r_null=rn,
                        celltypes=np.array(keep_ct_use),
                        atac_bulk=Ab.astype(np.float32), rna_bulk=Rb.astype(np.float32))
    print(f"\n  已保存 {WORK}/paired_celltype.npz")


if __name__ == "__main__":
    main()
