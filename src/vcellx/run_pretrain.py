"""Stage P driver: build the gene panel, cache Tabula onto it, pretrain."""
from __future__ import annotations

import argparse
import os
import time

B = "/mnt/storage/bzhang/AIVC"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=["panel", "cache", "train", "all"])
    ap.add_argument("--work", default=os.path.join(B, "vcellx/work"))
    ap.add_argument("--panel", type=int, default=4096)
    ap.add_argument("--cache-cells", type=int, default=400_000)
    ap.add_argument("--train-cells", type=int, default=150_000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)

    from . import pretrain

    panel_npz = os.path.join(a.work, f"panel{a.panel}.npz")
    cache_npz = os.path.join(a.work, f"tabula_panel{a.panel}_{a.cache_cells//1000}k.npz")

    arc_genes = [l.strip() for l in open(os.path.join(
        B, "arc-institute-virtual-cell-dataset/gene_names.csv")) if l.strip()]

    if a.step in ("panel", "all"):
        t0 = time.time()
        genes, _, _ = pretrain.tabula_panel(a.panel, arc_genes, panel_npz)
        print(f"[panel] {time.time()-t0:.0f}s  {len(genes)} genes", flush=True)

    z = __import__("numpy").load(panel_npz, allow_pickle=True)
    genes = z["genes"]

    if a.step in ("cache", "all"):
        t0 = time.time()
        pretrain.cache_tabula(cache_npz, genes, max_cells=a.cache_cells)
        print(f"[cache] {time.time()-t0:.0f}s", flush=True)

    if a.step in ("train", "all"):
        t0 = time.time()
        pretrain.pretrain(
            None, genes, os.path.join(a.work, "foundation"),
            max_cells=a.train_cells, epochs=a.epochs, device=a.device,
            d=a.dim, hidden=a.hidden, threads=a.threads,
            sources=[(cache_npz, "tabula-sapiens-v2")],
        )
        print(f"[train] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
