"""Diff the remote manifest against what is actually on disk, per file.

Reports, for each dataset:
  present and byte-identical
  present but WRONG SIZE (truncated / partial)
  MISSING
"""
import json
import os
import sys

MANIFEST = sys.argv[1] if len(sys.argv) > 1 else "remote_manifest.json"
# local root per dataset key
ROOTS = {
    "arc": "/mnt/storage/bzhang/AIVC/arc-institute-virtual-cell-dataset",
    "tahoe": "/mnt/storage/bzhang/AIVC/Tahoe-100M",
    "orion": "/mnt/storage/bzhang/AIVC/X-Atlas-Orion",
}

M = json.load(open(MANIFEST))
for key, root in ROOTS.items():
    if key not in M:
        print(f"\n### {key}: not in manifest")
        continue
    files = M[key]["files"]
    print(f"\n### {key}  ->  {root}")
    print(f"  expected {len(files):,} files, {sum(files.values())/1e9:.2f} GB")
    ok = bad = miss = 0
    bad_list, miss_list = [], []
    got = 0
    for rel, size in files.items():
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            miss += 1
            if len(miss_list) < 12:
                miss_list.append(rel)
            continue
        a = os.path.getsize(p)
        got += a
        if a == size:
            ok += 1
        else:
            bad += 1
            if len(bad_list) < 12:
                bad_list.append((rel, a, size))
    print(f"  identical : {ok:,}")
    print(f"  wrong size: {bad:,}")
    print(f"  missing   : {miss:,}")
    print(f"  bytes on disk for expected paths: {got/1e9:.2f} GB")
    if bad_list:
        print("  -- wrong size samples (path, local, expected) --")
        for r, a, e in bad_list:
            print(f"     {r}  {a:,} vs {e:,}  ({(a/e*100 if e else 0):.1f}%)")
    if miss_list:
        print("  -- missing samples --")
        for r in miss_list:
            print(f"     {r}")

    # anything on disk that is not in the manifest
    have = set()
    for dp, _, fns in os.walk(root):
        for fn in fns:
            have.add(os.path.relpath(os.path.join(dp, fn), root).replace("\\", "/"))
    extra = have - set(files)
    print(f"  extra files on disk: {len(extra):,}")
    for r in sorted(extra)[:8]:
        print(f"     + {r}")
