"""Disambiguate the weak per-cell ATAC->RNA correlation.

The per-cell test gave median r = 0.0039, which could mean either
  (a) promoter accessibility genuinely carries little information about that
      gene's expression in this tissue, or
  (b) single-cell sparsity (ATAC nonzero 13%) drowns the signal.

Test: pool cells into pseudo-replicates, form pseudobulk ATAC and RNA per
replicate, and correlate across replicates. Sparsity averages out while real
biological variation remains. If r jumps, (b) was the explanation.
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
N_REP = 20


def read_mouse_tss(path, want):
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


def main():
    # ---- RNA ----
    with gzip.open(RNA, "rt", encoding="utf-8", errors="replace") as f:
        cells = f.readline().rstrip("\n").split("\t")[1:]
        genes, rows = [], []
        for line in f:
            p = line.rstrip("\n").split("\t")
            genes.append(p[0]); rows.append(np.array(p[1:], np.float32))
    M = np.vstack(rows)                                  # genes x cells

    # ---- QC matched cells ----
    qc = set()
    with gzip.open(QC, "rt", encoding="utf-8", errors="replace") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        ci = hdr.index("cellname")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) > ci:
                qc.add(p[ci])
    keep = [c for c in cells if ("R1" + c) in qc or c in qc]
    cell_pos = {c: i for i, c in enumerate(cells)}

    tss = read_mouse_tss(MGTF, set(genes))
    glist = list(tss)
    gidx = {g: i for i, g in enumerate(glist)}
    by_chrom = {}
    for g in glist:
        c, t = tss[g]
        by_chrom.setdefault(c, []).append((t - PROM_UP, t + PROM_DN, gidx[g]))
    for c in by_chrom:
        by_chrom[c].sort()
    starts = {c: np.array([x[0] for x in v]) for c, v in by_chrom.items()}

    # ---- ATAC ----
    cidx = {c: i for i, c in enumerate(keep)}
    rr, cc_ = [], []
    n = 0
    with gzip.open(ATAC, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            n += 1
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
    R = M[:, [cell_pos[c] for c in keep]].T[:, [gidx[g] for g in glist]]
    print(f"ATAC {A.shape}  RNA {R.shape}", flush=True)

    # ---- pseudobulk replicates ----
    rng = np.random.default_rng(0)
    order = rng.permutation(len(keep))
    groups = np.array_split(order, N_REP)
    Ab = np.vstack([np.asarray(A[g].sum(0)).ravel() for g in groups])   # reps x genes
    Rb = np.vstack([np.asarray(R[g].sum(0)).ravel() for g in groups])

    def corr_cols(X, Y):
        X = X - X.mean(0, keepdims=True)
        Y = Y - Y.mean(0, keepdims=True)
        sx = np.sqrt((X ** 2).sum(0)); sy = np.sqrt((Y ** 2).sum(0))
        ok = (sx > 0) & (sy > 0)
        r = np.full(X.shape[1], np.nan, np.float32)
        r[ok] = (X[:, ok] * Y[:, ok]).sum(0) / (sx[ok] * sy[ok])
        return r

    r_bulk = corr_cols(Ab, Rb)
    print("\n=== 伪批量（20 组）ATAC vs RNA，逐基因相关 ===", flush=True)
    print(f"  可计算 {int(np.isfinite(r_bulk).sum())} 个基因")
    print(f"  中位 r = {np.nanmedian(r_bulk):.4f}")
    print(f"  均值 r = {np.nanmean(r_bulk):.4f}")
    print(f"  r > 0.5 的比例 = {np.mean(r_bulk[np.isfinite(r_bulk)] > 0.5)*100:.1f}%")
    print(f"  r > 0.3 的比例 = {np.mean(r_bulk[np.isfinite(r_bulk)] > 0.3)*100:.1f}%")

    perm = rng.permutation(len(glist))
    r_null = corr_cols(Ab[:, perm], Rb)
    print(f"\n  打乱对照：中位 r = {np.nanmedian(r_null):.4f}  "
          f"均值 r = {np.nanmean(r_null):.4f}")

    np.savez_compressed(os.path.join(WORK, "paired_pseudobulk.npz"),
                        genes=np.array(glist), r=r_bulk, r_null=r_null,
                        atac_bulk=Ab.astype(np.float32), rna_bulk=Rb.astype(np.float32))
    print(f"\n  已保存 {WORK}/paired_pseudobulk.npz")


if __name__ == "__main__":
    main()
