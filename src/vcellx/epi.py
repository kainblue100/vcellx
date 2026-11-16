"""vcellx · epigenomic features per gene.

Builds, for every gene on the expression gene axis, a small vector of
regulatory features measured in a **public** cell line:

  promoter ATAC accessibility   (K562 ATAC-seq peak calls)
  promoter TF occupancy         (K562 CTCF / SP1 / USF1 ChIP-seq)
  promoter CpG methylation      (K562 WGBS, optional)

HONESTY NOTE, and it is not a footnote: the expression data is the Arc VCC
training set, whose cell context is not stated in its README. The public
epigenomic assays available here are K562. So this prior is a
*cross-context human regulatory prior*, not a matched network for the
expression data. Any conclusion about "does the GRN help" carries that
caveat, and the write-up says so in the results section rather than hiding it.

Gene -> promoter mapping needs coordinates, so we take TSS from the GENCODE
basic annotation.
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import urllib.request

import numpy as np

GENCODE_URL = ("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/"
               "gencode.v44.basic.annotation.gtf.gz")

PROMOTER_UP = 1000
PROMOTER_DOWN = 1000


# --------------------------------------------------------------------------
# download helpers
# --------------------------------------------------------------------------
def fetch(url: str, dest: str, expect_min_bytes: int = 1 << 20) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) >= expect_min_bytes:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"  downloading {os.path.basename(dest)} ...", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    print(f"  -> {os.path.getsize(dest)/1e6:.1f} MB", flush=True)
    return dest


def ensure_gencode(cache_dir: str) -> str:
    dest = os.path.join(cache_dir, "gencode.v44.basic.annotation.gtf.gz")
    return fetch(GENCODE_URL, dest, expect_min_bytes=10 << 20)


# --------------------------------------------------------------------------
# TSS extraction
# --------------------------------------------------------------------------
def read_tss(gtf_gz: str, gene_symbols) -> dict:
    """gene symbol -> (chrom, tss, strand). First occurrence wins.

    Only protein_coding and lncRNA genes are kept; the expression axis is
    gene symbols, so we match on gene_name.
    """
    want = set(gene_symbols)
    out = {}
    keep_types = {"protein_coding", "lncRNA", "miRNA", "snRNA", "snoRNA"}
    with gzip.open(gtf_gz, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            attrs = parts[8]
            if 'gene_type "' not in attrs:
                # older format fallback
                gt = None
            else:
                gt = attrs.split('gene_type "')[1].split('"')[0]
            if gt is not None and gt not in keep_types:
                continue
            if 'gene_name "' not in attrs:
                continue
            name = attrs.split('gene_name "')[1].split('"')[0]
            if name not in want or name in out:
                continue
            chrom = parts[0]
            start, end = int(parts[3]), int(parts[4])
            strand = parts[6]
            tss = start if strand == "+" else end
            out[name] = (chrom, tss, strand)
    return out


# --------------------------------------------------------------------------
# peak parsing
# --------------------------------------------------------------------------
def read_bed_gz(path: str, min_score: float = 0.0) -> dict:
    """chrom -> (starts, ends, scores) int/float arrays. Standard BED first 5 cols."""
    acc = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(("#", "track", "browser")) or not line.strip():
                continue
            p = line.split("\t")
            if len(p) < 3:
                continue
            try:
                s, e = int(p[1]), int(p[2])
            except ValueError:
                continue
            sc = 1.0
            if len(p) >= 5:
                try:
                    sc = float(p[4])
                except ValueError:
                    sc = 1.0
            if sc < min_score:
                continue
            acc.setdefault(p[0], []).append((s, e, sc))
    out = {}
    for c, v in acc.items():
        a = np.array(v, dtype=np.float64)
        order = np.argsort(a[:, 0])
        a = a[order]
        out[c] = (a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2].astype(np.float32))
    return out


def _overlap_sum(starts, ends, scores, lo, hi):
    """Sum of peak scores overlapping [lo, hi) and the number of peaks."""
    if starts is None or len(starts) == 0:
        return 0.0, 0
    i0 = np.searchsorted(starts, hi, side="left")
    i1 = np.searchsorted(starts, lo, side="left")
    sel = slice(max(0, i1 - 1), max(0, i0))
    s, e, w = starts[sel], ends[sel], scores[sel]
    if len(s) == 0:
        return 0.0, 0
    ov = np.maximum(0, np.minimum(e, hi) - np.maximum(s, lo))
    hit = ov > 0
    return float(w[hit].sum()), int(hit.sum())


# --------------------------------------------------------------------------
# feature matrix
# --------------------------------------------------------------------------
def build_features(genes, tss_map, peak_sets: dict, out_path: str):
    """(n_genes, n_features) float32 + availability mask + feature names.

    Features per gene, in this order:
      log1p(ATAC promoter signal), ATAC peak count in promoter,
      then for each TF: log1p(signal), peak count,
      log1p(gene-to-TSS distance proxy is not used; coordinates are only for mapping)
    """
    names = ["atac_sig", "atac_n"]
    for tf in peak_sets:
        names += [f"{tf}_sig", f"{tf}_n"]

    G = len(genes)
    F = np.zeros((G, len(names)), dtype=np.float32)
    ok = np.zeros(G, dtype=bool)
    missing = []

    for gi, g in enumerate(genes):
        info = tss_map.get(g)
        if info is None:
            missing.append(g)
            continue
        chrom, tss, _ = info
        lo, hi = tss - PROMOTER_UP, tss + PROMOTER_DOWN
        row = []
        have_any = False
        for key, peaks in peak_sets.items():
            st, en, sc = peaks.get(chrom, (None, None, None))
            sig, n = _overlap_sum(st, en, sc, lo, hi)
            row += [np.log1p(sig), float(n)]
            if n > 0:
                have_any = True
        F[gi] = row
        ok[gi] = have_any

    np.savez_compressed(out_path, X=F, ok=ok,
                        names=np.array(names), genes=np.array(genes))
    print(f"  genes with TSS: {G - len(missing)}/{G}; "
          f"with >=1 peak in promoter: {int(ok.sum())}", flush=True)
    if missing:
        print(f"  no TSS for {len(missing)} genes, e.g. {missing[:6]}", flush=True)
    return F, ok, names


def run(genes, workdir: str, atac_url: str | None = None,
        k562_dir: str | None = None, out_name: str = "epi_features.npz"):
    """Convenience driver used by the CLI."""
    os.makedirs(workdir, exist_ok=True)
    gtf = ensure_gencode(workdir)
    tss = read_tss(gtf, genes)
    print(f"  TSS found for {len(tss)}/{len(genes)} requested genes", flush=True)

    peak_sets = {}
    # ATAC: prefer an explicit URL, else a local file in k562_dir
    if atac_url:
        ap = fetch(atac_url, os.path.join(workdir, "k562_atac.bed.gz"), 1 << 20)
        peak_sets["atac"] = read_bed_gz(ap)
    elif k562_dir:
        cand = os.path.join(k562_dir, "ENCFF855PCP.bed.gz")   # K562 ATAC-seq
        if os.path.exists(cand):
            peak_sets["atac"] = read_bed_gz(cand)

    if k562_dir:
        tf_files = {"ctcf": "ENCFF769AUF.bed.gz",
                    "sp1": "ENCFF528QDU.bed.gz",
                    "usf1": "ENCFF679COG.bed.gz"}
        for tf, fn in tf_files.items():
            p = os.path.join(k562_dir, fn)
            if os.path.exists(p):
                peak_sets[tf] = read_bed_gz(p)

    if not peak_sets:
        raise SystemExit("no peak files found")
    print("  peak sets: " + ", ".join(
        f"{k}({sum(len(v[0]) for v in peak_sets[k].values()):,} peaks)" for k in peak_sets),
        flush=True)

    return build_features(genes, tss, peak_sets, os.path.join(workdir, out_name))
