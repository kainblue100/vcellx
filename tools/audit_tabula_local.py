import hashlib
import json
import os
import sys

B = "/mnt/storage/bzhang/AIVC"
man = os.path.join(B, "manifests/tabula-cellxgene-1c88f927-bcbb-4bb1-9881-281842945a2d.json")
sha = os.path.join(B, "manifests/tabula-cellxgene-1c88f927-bcbb-4bb1-9881-281842945a2d.sha256")
logf = os.path.join(B, "logs/tabula.log")
target = os.path.join(B, "Tabula-Sapiens-v2-CELLxGENE/Tabula_Sapiens_v2_All_Cells.h5ad")

print("=" * 72)
print("cellxgene manifest")
print("=" * 72)
try:
    print(open(man).read()[:1500])
except Exception as e:
    print("ERR", e)

print("\n" + "=" * 72)
print("recorded sha256")
print("=" * 72)
try:
    rec = open(sha).read().strip()
    print(rec)
except Exception as e:
    rec = None
    print("ERR", e)

print("\n" + "=" * 72)
print("download log")
print("=" * 72)
try:
    print(open(logf).read()[-2000:])
except Exception as e:
    print("ERR", e)

print("\n" + "=" * 72)
print("file")
print("=" * 72)
if os.path.isfile(target):
    sz = os.path.getsize(target)
    print(f"size: {sz:,} bytes ({sz/1e9:.2f} GB)")
    want = rec.split()[0] if rec else None
    if want and len(want) == 64:
        print("hashing (this takes a minute on 45 GB)...", flush=True)
        h = hashlib.sha256()
        with open(target, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        got = h.hexdigest()
        print(f"sha256 recorded: {want}")
        print(f"sha256 computed: {got}")
        print("MATCH" if got == want else "*** MISMATCH ***")
    else:
        print("no usable recorded sha256")
else:
    print("missing")
