"""Inspect the .mcool resolution groups properly (bins/pixels may be groups)."""
import h5py

P = "/mnt/storage/bzhang/AIVC/epigenomics/raw/4dn/HFFc6/4DNFIB59T7NN.mcool"


def describe(node, name, depth=0, maxdepth=2):
    pad = "  " * (depth + 1)
    if isinstance(node, h5py.Dataset):
        print(f"{pad}{name}: DATASET shape={node.shape} dtype={node.dtype}")
    else:
        print(f"{pad}{name}: GROUP keys={list(node.keys())}")
        if depth < maxdepth:
            for k in node.keys():
                describe(node[k], k, depth + 1, maxdepth)


with h5py.File(P, "r") as f:
    for res in ("100000", "500000", "1000000"):
        g = f["resolutions"][res]
        print(f"=== resolution {res} ===")
        describe(g, res, 0, 2)
        # try the compound-dataset case
        for nm in ("bins", "pixels"):
            n = g[nm]
            if isinstance(n, h5py.Dataset):
                print(f"  {nm} fields:", n.dtype.names)
                print(f"  {nm}[:3]:", n[:3])
        print()
