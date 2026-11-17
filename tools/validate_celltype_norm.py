"""The decisive form of Test A: remove the shared depth factor first.

Iteration history of this test, kept so the reasoning is auditable:
  1  per-cell         r = 0.004   -> swamped by ATAC sparsity (13% nonzero)
  2  random pools     r = -0.011  -> pools share the same cell-type mix, so the
                                     only variance left is counting noise
  3  by cell type     r = 0.542   -> but the shuffled null is 0.544 too, i.e. the
                                     correlation is a shared-depth artefact:
                                     high-depth cell types are high on every
                                     gene, so any gene correlates with any gene

Fix: convert each cell type's ATAC and RNA profile to a fraction of that
profile's total, which removes the depth factor. Then correlate per gene across
cell types. The shuffled-gene null is still the right control.
"""
import os

import numpy as np

WORK = "/mnt/storage/bzhang/AIVC/vcellx/work"
Z = np.load(os.path.join(WORK, "paired_celltype.npz"), allow_pickle=True)
genes = Z["genes"]
A = Z["atac_bulk"].astype(np.float64)      # (celltypes, genes)
R = Z["rna_bulk"].astype(np.float64)
cts = Z["celltypes"]
print(f"聚合矩阵 ATAC {A.shape}  RNA {R.shape}   细胞类型 {len(cts)}")


def l2_normalise_rows(X):
    """Scale each cell-type profile to unit sum (fraction of that profile's total)."""
    s = X.sum(1, keepdims=True)
    s[s == 0] = 1
    return X / s


def corr_cols(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    sx = np.sqrt((X ** 2).sum(0)); sy = np.sqrt((Y ** 2).sum(0))
    ok = (sx > 0) & (sy > 0)
    r = np.full(X.shape[1], np.nan, np.float32)
    r[ok] = (X[:, ok] * Y[:, ok]).sum(0) / (sx[ok] * sy[ok])
    return r


def report(tag, r):
    fin = np.isfinite(r)
    print(f"\n--- {tag} ---")
    print(f"  可计算 {int(fin.sum())} 基因")
    print(f"  中位 r = {np.nanmedian(r):+.4f}   均值 r = {np.nanmean(r):+.4f}")
    print(f"  r > 0.5 = {np.mean(r[fin] > 0.5)*100:5.1f}%   "
          f"r > 0.3 = {np.mean(r[fin] > 0.3)*100:5.1f}%   "
          f"r < 0 = {np.mean(r[fin] < 0)*100:5.1f}%")


rng = np.random.default_rng(0)

# --- raw (reproduces iteration 3, the artefact) ---
report("原始计数（复现迭代 3 的伪影）", corr_cols(A, R))
report("原始计数 · 打乱基因", corr_cols(A[:, rng.permutation(len(genes))], R))

# --- depth-normalised ---
An = l2_normalise_rows(A)
Rn = l2_normalise_rows(R)
r_real = corr_cols(An, Rn)
r_null = corr_cols(An[:, rng.permutation(len(genes))], Rn)
report("归一化为占比（去掉深度因子）", r_real)
report("占比 · 打乱基因", r_null)

fin = np.isfinite(r_real)
delta = np.nanmedian(r_real) - np.nanmedian(r_null)
print(f"\n=== 结论 ===")
print(f"  真实中位 r      {np.nanmedian(r_real):+.4f}")
print(f"  打乱中位 r      {np.nanmedian(r_null):+.4f}")
print(f"  差值            {delta:+.4f}")
# per-gene comparison: how many genes beat their own shuffled baseline by >0.1
thr = 0.1
frac_better = np.mean((r_real[fin] - r_null[fin]) > thr)
print(f"  真实比打乱高 >{thr} 的基因比例  {frac_better*100:.1f}%")

np.savez_compressed(os.path.join(WORK, "paired_celltype_normalised.npz"),
                    genes=genes, r_real=r_real, r_null=r_null,
                    celltypes=cts)
print(f"\n  已保存 {WORK}/paired_celltype_normalised.npz")
