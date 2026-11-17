"""Statistical summary of the depth-normalised Test A.

Answers one question precisely: is the per-gene difference between the real and
the shuffled correlation distribution shifted away from zero, and by how much?
"""
import os

import numpy as np
from scipy import stats

WORK = "/mnt/storage/bzhang/AIVC/vcellx/work"
Z = np.load(os.path.join(WORK, "paired_celltype_normalised.npz"), allow_pickle=True)
r = Z["r_real"].astype(np.float64)
rn = Z["r_null"].astype(np.float64)
genes = Z["genes"]

ok = np.isfinite(r) & np.isfinite(rn)
d = r[ok] - rn[ok]
print(f"可比较基因 {ok.sum():,}\n")

print("=== 真实 vs 打乱（逐基因配对）===")
print(f"  中位差   {np.median(d):+.5f}")
print(f"  均值差   {d.mean():+.5f}")
print(f"  标准差   {d.std():.4f}")
t = stats.ttest_1samp(d, 0.0)
print(f"  t 检验   t = {t.statistic:+.2f}   p = {t.pvalue:.3e}")
w = stats.wilcoxon(d) if len(d) < 20000 else stats.wilcoxon(d[:20000])
print(f"  Wilcoxon W = {w.statistic:.0f}   p = {w.pvalue:.3e}")

print("\n=== 效应量分布 ===")
for thr in (0.0, 0.05, 0.1, 0.2, 0.3):
    print(f"  真实 − 打乱 > {thr:.2f}: {np.mean(d > thr)*100:5.1f}%")

print("\n=== 一个直觉式的 sanity check ===")
print("  如果表观特征真的普遍携带表达信息，差值分布应当明显右移。")
print(f"  实际均值差 {d.mean():+.5f}，中位差 {np.median(d):+.5f}。")

# how many genes would pass a Bonferroni-ish bar if we had per-gene nulls
print("\n=== 是否可以认为'多数基因无信号' ===")
print(f"  差值落在 ±0.05 内的比例: {np.mean(np.abs(d) < 0.05)*100:.1f}%")
print(f"  差值 < -0.05 的比例    : {np.mean(d < -0.05)*100:.1f}%")
