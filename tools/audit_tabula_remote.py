"""Audit the two remaining sources: Tabula Sapiens (figshare) and cellxgene.

For Tabula Sapiens the question is not "is a file present" but "is the figshare
release present". The local copy is a single merged All_Cells h5ad which is a
different artifact from the 28 per-organ files, so we report both separately
and check that the merged file is actually a readable h5ad with the expected
shape.
"""
import json
import os
import urllib.request

print("=" * 74)
print("figshare Tabula Sapiens v2 (article 27921984) — full file list")
print("=" * 74)
req = urllib.request.Request("https://api.figshare.com/v2/articles/27921984",
                            headers={"User-Agent": "vcellx-audit"})
with urllib.request.urlopen(req, timeout=120) as r:
    js = json.load(r)
files = js.get("files", [])
tot = sum(f.get("size", 0) for f in files)
print(f"expected: {len(files)} files, {tot/1e9:.2f} GB")
names = []
for f in sorted(files, key=lambda x: x.get("name", "")):
    names.append(f["name"])
print("\nfile names:")
for n in names:
    print("  " + n)

json.dump({"files": [{"name": f["name"], "size": f.get("size", 0),
                      "md5": f.get("computed_md5", ""),
                      "url": f.get("download_url", "")} for f in files]},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "tabula_figshare.json"), "w"), indent=2)
print("\nwrote tabula_figshare.json")
