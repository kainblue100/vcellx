"""Sweep GRN construction parameters to see whether a usable graph is reachable
from the K562 Hi-C loop calls over a 2000-gene HVG panel.

The first attempt gave 14 edges and 1973 isolated genes, which is not a network.
This measures the trade-off between promoter window width, loop FDR threshold
and the resulting degree distribution.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/storage/bzhang/AIVC/vcellx/src")
from vcellx.epi import ensure_gencode, read_tss          # noqa: E402
from vcellx import grn                                    # noqa: E402

B = "/mnt/storage/bzhang/AIVC"
WORK = os.path.join(B, "vcellx/work")
K562 = os.path.join(B, "epigenomics/k562-encode-matched-bulk-v1")

z = np.load(os.path.join(WORK, "arc_hvg2000.npz"), allow_pickle=True)
genes = [str(g) for g in z["genes"]]
print(f"panel genes: {len(genes)}", flush=True)

gtf = ensure_gencode(WORK)
tss = read_tss(gtf, genes)
print(f"with TSS: {len(tss)}", flush=True)

loop_files = ["ENCFF757IIX.bedpe.gz", "ENCFF134HIZ.bedpe.gz",
              "ENCFF440HUR.bedpe.gz", "ENCFF693XIL.bedpe.gz"]

for fdr in (0.05, 0.2, 1.0):
    loops = []
    for fn in loop_files:
        p = os.path.join(K562, fn)
        if os.path.exists(p):
            loops += grn.read_loops(p, min_fdr=fdr)
    print(f"\n=== fdr<={fdr}: {len(loops):,} loops ===", flush=True)

    for pad in (1000, 5000, 25000, 100000):
        try:
            A, deg, s = grn.build_graph(genes, tss, loops, promoter_pad=pad, topk=10)
        except SystemExit as e:
            print(f"  pad={pad:>7}: {e}", flush=True)
            continue
        iso = s["isolated_genes"]
        print(f"  pad={pad:>7}  edges={s['n_edges']:>6}  "
              f"isolated={iso:>4}/{len(genes)}  "
              f"deg_median={s['degree_median']:.0f}  deg_max={s['degree_max']}", flush=True)
