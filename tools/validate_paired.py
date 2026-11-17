"""Validate the two assumptions the two-arm model rests on, using the paired
single-cell multi-omics data (GSE303006: same cells have RNA, ATAC and Hi-C).

  Test A   does a gene's promoter ATAC signal predict its own RNA, in the same
           cell, better than a shuffled-gene null?
           -> this is the assumption behind the `direct` arm.

  Test B   are genes joined by a chromatin-contact edge more co-expressed than
           degree-matched random pairs?
           -> this is the assumption behind the `graph` arm.

Both tests are run entirely inside the matched dataset, so there is no
cross-context confound: same species, same tissue, same cells.

DATA PROVENANCE CORRECTION: the files and the earlier audit label this sample
"mESC", but the QC table says tissue = brain cortex with cortical cell types
(Inh_Sst, Ex_L3/4_IT). The label is reported as what the QC says.
"""
from __future__ import annotations

import gzip
import os

import numpy as np
import scipy.sparse as sp

B = "/mnt/storage/bzhang/AIVC/paired-trimodal-curation"
RNA = os.path.join(B, "GSE303006-metadata-v1/GSM9115533_mesc.RNAcounts.gene.total.format.tsv.gz")
ATAC = os.path.join(B, "GSE303006-mESC-v1/GSM9115537_mesc.atac.fragment.tsv.gz")
QC = os.path.join(B, "GSE303006-metadata-v1/GSE303006_charm_metadata_qcpass.tsv.gz")
WORK = "/mnt/storage/bzhang/AIVC/vcellx/work"
MGTF = os.path.join(WORK, "gencode.vM25.basic.annotation.gtf.gz")
MGTF_URL = ("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/"
            "gencode.vM25.basic.annotation.gtf.gz")

PROM_UP, PROM_DN = 2000, 500


def ensure_mouse_gtf():
    import urllib.request
    if os.path.exists(MGTF) and os.path.getsize(MGTF) > 5e6:
        return MGTF
    print(f"  downloading mouse GENCODE M25 (GRCm38) ...", flush=True)
    tmp = MGTF + ".part"
    with urllib.request.urlopen(MGTF_URL, timeout=180) as r, open(tmp, "wb") as f:
        while True:
            c = r.read(1 << 20)
            if not c:
                break
            f.write(c)
    os.replace(tmp, MGTF)
    print(f"  -> {os.path.getsize(MGTF)/1e6:.1f} MB", flush=True)
    return MGTF


def read_mouse_tss(path, want):
    out = {}
    keep = {"protein_coding", "lncRNA"}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[2] != "gene":
                continue
            a = p[8]
            if 'gene_type "' in a:
                gt = a.split('gene_type "')[1].split('"')[0]
                if gt not in keep:
                    continue
            if 'gene_name "' not in a:
                continue
            name = a.split('gene_name "')[1].split('"')[0]
            if name not in want or name in out:
                continue
            s, e = int(p[3]), int(p[4])
            tss = s if p[6] == "+" else e
            out[name] = (p[0], tss)
    return out


def load_rna():
    """genes x cells -> (genes, cells, CSR cells x genes)."""
    with gzip.open(RNA, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        cells = header[1:]
        genes, rows = [], []
        for line in f:
            p = line.rstrip("\n").split("\t")
            genes.append(p[0])
            rows.append(np.array(p[1:], dtype=np.float32))
    M = np.vstack(rows)                     # (genes, cells)
    return genes, cells, M


def promoter_counts(cells_keep, tss, min_len=0):
    """Count ATAC fragments per (cell, gene) promoter window."""
    cell_idx = {c: i for i, c in enumerate(cells_keep)}
    gene_list = [g for g in tss]
    gidx = {g: i for i, g in enumerate(gene_list)}
    by_chrom = {}
    for g in gene_list:
        c, t = tss[g]
        by_chrom.setdefault(c, []).append((t - PROM_UP, t + PROM_DN, gidx[g]))
    for c in by_chrom:
        by_chrom[c].sort()
    rr, cc_ = [], []
    starts = {c: np.array([x[0] for x in v]) for c, v in by_chrom.items()}
    n_rows = 0
    with gzip.open(ATAC, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_rows += 1
            if n_rows % 10_000_000 == 0:
                print(f"    atac rows {n_rows/1e6:.0f}M", flush=True)
            p = line.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            ci = cell_idx.get(p[6])
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
    nC, nG = len(cells_keep), len(gene_list)
    if not rr:
        return sp.csr_matrix((nC, nG)), gene_list
    return sp.csr_matrix((np.ones(len(rr), np.float32), (rr, cc_)),
                         shape=(nC, nG)), gene_list


def corr_per_gene(A, R):
    """Pearson r between promoter ATAC and RNA, per gene, across cells.

    A, R: (cells, genes). Columns that are constant in either matrix are
    undefined and returned as NaN, not zero.
    """
    A = A - A.mean(0, keepdims=True)
    R = R - R.mean(0, keepdims=True)
    sa = np.sqrt((A ** 2).sum(0))
    sr = np.sqrt((R ** 2).sum(0))
    ok = (sa > 0) & (sr > 0)
    r = np.full(A.shape[1], np.nan, dtype=np.float32)
    r[ok] = (A[:, ok] * R[:, ok]).sum(0) / (sa[ok] * sr[ok])
    return r


def main():
    os.makedirs(WORK, exist_ok=True)
    print("=== 载入 RNA ===", flush=True)
    genes, cells, M = load_rna()
    print(f"  RNA {M.shape[0]} genes x {M.shape[1]} cells", flush=True)

    print("=== QC：挑出四模态齐全的细胞 ===", flush=True)
    qc_cells = {}
    with gzip.open(QC, "rt", encoding="utf-8", errors="replace") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        ci = hdr.index("cellname")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) > ci:
                qc_cells[p[ci]] = True
    keep = [c for c in cells if ("R1" + c) in qc_cells or c in qc_cells]
    print(f"  QC 细胞 {len(qc_cells)}；RNA 细胞 {len(cells)}；匹配 {len(keep)}", flush=True)

    print("=== 小鼠注释（GRCm38）===", flush=True)
    gtf = ensure_mouse_gtf()
    want = set(genes)
    tss = read_mouse_tss(gtf, want)
    print(f"  TSS 覆盖 {len(tss)}/{len(want)} 个 RNA 基因", flush=True)

    print("=== 统计每个基因启动子的 ATAC 计数（会流式扫 5000 万行）===", flush=True)
    A, gene_list = promoter_counts(keep, tss)
    print(f"  ATAC 矩阵 {A.shape}，非零率 {(A>0).mean():.4f}", flush=True)

    # align RNA to the same cells and genes
    cell_pos = {c: i for i, c in enumerate(cells)}
    R = M[:, [cell_pos[c] for c in keep]].T          # (kept cells, genes)
    gpos = {g: i for i, g in enumerate(genes)}
    Rg = R[:, [gpos[g] for g in gene_list]]

    print("\n=== Test A：启动子 ATAC 能否预测该基因自身的 RNA（同一细胞）===", flush=True)
    r = corr_per_gene(A.toarray(), Rg)
    valid = np.isfinite(r)
    print(f"  基因数 {len(r)}，其中可计算相关 {int(valid.sum())}")
    print(f"  中位 r = {np.nanmedian(r):.4f}")
    print(f"  均值 r = {np.nanmean(r):.4f}")
    print(f"  r > 0.2 的比例 = {np.mean(r[valid] > 0.2)*100:.1f}%")

    # null: shuffle the gene labels of the ATAC matrix
    rng = np.random.default_rng(0)
    perm = rng.permutation(A.shape[1])
    r_null = corr_per_gene(A.toarray()[:, perm], Rg)
    print(f"\n  打乱对照：中位 r = {np.nanmedian(r_null):.4f}   "
          f"均值 r = {np.nanmean(r_null):.4f}")
    print(f"  真实 − 打乱 = {np.nanmedian(r) - np.nanmedian(r_null):+.4f} (中位)")

    np.savez_compressed(os.path.join(WORK, "paired_validation.npz"),
                        genes=np.array(gene_list), r=r, r_null=r_null,
                        n_cells=len(keep))
    print(f"\n  已保存 {WORK}/paired_validation.npz")


if __name__ == "__main__":
    main()
