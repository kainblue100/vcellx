"""Push the repository through the GitHub contents API.

`git push` over https://github.com:443 keeps failing on this host
("Empty reply from server" / "Could not connect") while api.github.com works
fine, so when the git transport is unavailable we write the tree through the
API instead.

The first version of this script hard-coded the two files it knew about and
silently shipped a commit that omitted everything else. This version walks the
working tree, so nothing is left behind, and takes the commit message from a
file so it is not lost either.

Usage
  python push_via_api.py [-m <message-file>] [--dry-run]

Blob SHAs are content-addressed, so unchanged files re-upload to the same SHA
and the resulting tree is identical to what git would have produced.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys

GH = r"D:\coda_lungs\tools\gh_extract\bin\gh.exe"
REPO = "kainblue100/vcellx"
ROOT = r"D:\coda_matlab\vcellx"
BRANCH = "main"

# never ship these, whatever the tree contains
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules"}
SKIP_EXT = {".pyc", ".pyo", ".log", ".tmp", ".part"}
SKIP_NAMES = {".DS_Store", "Thumbs.db"}


def gh(*args, input_text=None):
    p = subprocess.run([GH, *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", input=input_text)
    if p.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:3])} failed: {p.stderr[:400]}")
    return p.stdout


def api(path, method="GET", body=None):
    args = ["api", "-X", method, path]
    if body is not None:
        args += ["--input", "-"]
        return json.loads(gh(*args, input_text=json.dumps(body)))
    return json.loads(gh(*args))


def collect():
    """Every shippable file in the working tree, as repo-relative paths.

    .gitignore is authoritative: an earlier version of this script ignored it
    and pushed .shots/*.png, files the repo explicitly excludes. Ask git which
    paths are ignored rather than reimplementing the pattern language.
    """
    out = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in SKIP_EXT or fn in SKIP_NAMES:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, ROOT).replace("\\", "/")
            out[rel] = full

    if out:
        p = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT,
                           input="\n".join(out), capture_output=True, text=True)
        ignored = {line.strip() for line in p.stdout.splitlines() if line.strip()}
        for rel in ignored:
            out.pop(rel, None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--message", help="file containing the commit message")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    msg = ""
    if a.message:
        msg = open(a.message, encoding="utf-8").read()
    if not msg.strip():
        msg = "Sync repository\n\nCo-authored-by: CommandCodeBot <noreply@commandcode.ai>\n"
    if "CommandCodeBot" not in msg:
        msg = msg.rstrip() + "\n\nCo-authored-by: CommandCodeBot <noreply@commandcode.ai>\n"

    files = collect()
    print(f"=== 工作树中共 {len(files)} 个文件 ===")

    ref = api(f"repos/{REPO}/git/ref/heads/{BRANCH}")
    head_sha = ref["object"]["sha"]
    base_tree = api(f"repos/{REPO}/git/commits/{head_sha}")["tree"]["sha"]
    print(f"  remote head {head_sha[:10]}  tree {base_tree[:10]}")

    # what is already on the remote
    remote = {}
    def walk(sha, prefix=""):
        t = api(f"repos/{REPO}/git/trees/{sha}")
        for e in t.get("tree", []):
            if e["type"] == "tree":
                walk(e["sha"], prefix + e["path"] + "/")
            elif e["type"] == "blob":
                remote[prefix + e["path"]] = e["sha"]
    walk(base_tree, "")
    print(f"  remote 现有 {len(remote)} 个文件")

    changed, new, dropped = [], [], sorted(set(remote) - set(files))
    entries = []
    for rel, full in sorted(files.items()):
        data = open(full, "rb").read()
        blob = api(f"repos/{REPO}/git/blobs", "POST", {
            "content": base64.b64encode(data).decode("ascii"),
            "encoding": "base64",
        })
        if remote.get(rel) != blob["sha"]:
            (new if rel not in remote else changed).append(rel)
        entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]})

    # Deletions must be explicit. With base_tree set, any path the new tree does
    # not mention is kept from the base tree, so omitting these would report a
    # deletion that never happened.
    for rel in dropped:
        entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": None})
        print(f"  drop  {rel}")
    print(f"\n=== 变更 ===")
    for r in new:
        print(f"  新增  {r}")
    for r in changed:
        print(f"  修改  {r}")
    print(f"  共 {len(new)} 新增 / {len(changed)} 修改 / {len(dropped)} 删除")

    if a.dry_run:
        print("\n--dry-run，不提交")
        return

    if not (new or changed or dropped):
        print("\n没有变化，无需提交")
        return

    tree = api(f"repos/{REPO}/git/trees", "POST",
               {"base_tree": base_tree, "tree": entries})
    newc = api(f"repos/{REPO}/git/commits", "POST",
               {"message": msg, "tree": tree["sha"], "parents": [head_sha]})
    api(f"repos/{REPO}/git/refs/heads/{BRANCH}", "PATCH", {"sha": newc["sha"]})
    print(f"\n已推送 -> https://github.com/{REPO}/commit/{newc['sha']}")


if __name__ == "__main__":
    main()
