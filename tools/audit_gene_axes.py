"""Inspect gene axes and check harmonisation. Handles categorical columns."""
import os

import h5py
import numpy as np
import pandas as pd

B = "/mnt/storage/bzhang/AIVC"


def read_col(group, name):
    """Read an h5ad var/obs column that may be a dataset or a categorical group."""
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
    v = node[:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in v])


print("=" * 74)
print("Tabula Sapiens v2 (cellxgene all-cells)")
print("=" * 74)
p = os.path.join(B, "Tabula-Sapiens-v2-CELLxGENE/Tabula_Sapiens_v2_All_Cells.h5ad")
with h5py.File(p, "r") as f:
    var = f["var"]
    ens = read_col(var, "_index")
    sym = read_col(var, "feature_name")
    print(f"genes: {len(ens):,}")
    print("  ensembl[:4] :", list(ens[:4]))
    print("  symbol [:4] :", list(sym[:4]) if sym is not None else None)
    print("  symbols not null:", int(sum(1 for s in (sym if sym is not None else []) if s)))
    X = f["X"]
    print("  X encoding:", dict(X.attrs).get("encoding-type"),
          " shape:", tuple(int(v) for v in X.attrs["shape"]))
    print("  has raw:", "raw" in f, "| layers:", list(f["layers"].keys())[:5])
    tab_syms = set(s for s in (sym if sym is not None else []) if s)

print("\n" + "=" * 74)
print("Orion / Tahoe gene metadata")
print("=" * 74)
gm = pd.read_parquet(os.path.join(B, "X-Atlas-Orion/metadata/gene_metadata.parquet"))
gm2 = pd.read_parquet(os.path.join(B, "Tahoe-100M/metadata/gene_metadata.parquet"))
print("Orion:", gm.shape, list(gm.columns))
print("Tahoe:", gm2.shape, list(gm2.columns))

print("\n" + "=" * 74)
print("Arc panel + overlap")
print("=" * 74)
arc = [l.strip() for l in open(os.path.join(
    B, "arc-institute-virtual-cell-dataset/gene_names.csv")) if l.strip()]
arc_set = set(arc)
orion = set(gm["gene_name"].astype(str))
tahoe = set(gm2["gene_symbol"].astype(str))

print(f"Arc genes           : {len(arc_set):,}")
print(f"Tabula symbols      : {len(tab_syms):,}")
print(f"Orion symbols       : {len(orion):,}")
print(f"Tahoe symbols       : {len(tahoe):,}")
for nm, s in [("Tabula", tab_syms), ("Orion", orion), ("Tahoe", tahoe)]:
    inter = arc_set & s
    print(f"Arc ∩ {nm:<8s}: {len(inter):>6,}  ({len(inter)/len(arc_set)*100:5.1f}% of Arc)")
all_ext = tab_syms | orion | tahoe
print(f"Arc ∩ any external  : {len(arc_set & all_ext):,} ({len(arc_set & all_ext)/len(arc_set)*100:.1f}%)")
print(f"external ∩ any      : {len(tab_syms | orion | tahoe):,}")

# A shared panel usable for pretraining and for the perturbation model
shared = sorted(arc_set & (tab_syms | orion | tahoe))
print(f"\nshared panel (Arc ∩ external): {len(shared):,}")
print("  examples:", shared[:10])
