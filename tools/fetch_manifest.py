"""Fetch the complete remote file manifest for each HF dataset, following
pagination, and write path -> size so the server can diff it against disk."""

import json
import os
import urllib.parse
import urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote_manifest.json")

REPOS = {
    "arc": "cyrilzakka/arc-institute-virtual-cell-dataset",
    "tahoe": "tahoebio/Tahoe-100M",
    "orion": "Xaira-Therapeutics/X-Atlas-Orion",
}


def fetch_all(repo):
    """Paginate the HF tree API until it stops returning files."""
    base = f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true"
    files = {}
    cursor = None
    pages = 0
    while True:
        url = base + (f"&cursor={urllib.parse.quote(cursor)}" if cursor else "")
        req = urllib.request.Request(url, headers={"User-Agent": "vcellx-audit"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.load(r)
            link = r.headers.get("Link") or r.headers.get("link")
        pages += 1
        for x in data:
            if x.get("type") == "file":
                files[x["path"]] = int(x.get("size", 0) or 0)
        # HF returns a Link header with rel="next"
        nxt = None
        if link:
            for part in link.split(","):
                if 'rel="next"' in part:
                    seg = part.split("<")[1].split(">")[0]
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(seg).query)
                    nxt = (q.get("cursor") or [None])[0]
        if not nxt or nxt == cursor or pages > 200:
            break
        cursor = nxt
    return files, pages


out = {}
for key, repo in REPOS.items():
    print(f"fetching {repo} ...", flush=True)
    try:
        files, pages = fetch_all(repo)
        tot = sum(files.values())
        out[key] = {"repo": repo, "files": files}
        print(f"  pages={pages}  files={len(files):,}  total={tot/1e9:.2f} GB", flush=True)
        # show shard index range for parquet shards, to detect gaps
        shards = sorted(p for p in files if p.endswith(".parquet"))
        if shards:
            print(f"  parquet shards: {len(shards):,}", flush=True)
            print(f"    first: {shards[0]}", flush=True)
            print(f"    last : {shards[-1]}", flush=True)
    except Exception as e:
        print(f"  ERROR {type(e).__name__}: {e}", flush=True)

json.dump(out, open(OUT, "w"))
print(f"\nwrote {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB)", flush=True)
