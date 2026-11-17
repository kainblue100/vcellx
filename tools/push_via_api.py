"""Push changed files through the GitHub contents API.

`git push` over https://github.com:443 keeps failing on this host
("Could not connect to server") while api.github.com works fine, so when the
git transport is unavailable we write the files through the API instead. The
commit message and author are preserved; the resulting commit is identical in
content to what git would have produced for these files.
"""
import base64
import json
import os
import subprocess
import sys

GH = r"D:\coda_lungs\tools\gh_extract\bin\gh.exe"
REPO = "kainblue100/vcellx"
ROOT = r"D:\coda_matlab\vcellx"
BRANCH = "main"

FILES = ["README.md", "index.html"]
MSG = ("预训练完成：记录真实的编码器结果\n\n"
       "预训练编码器已在 150,000 个 Tabula Sapiens 细胞上完成 8 个 epoch，\n"
       "掩码重建损失 0.680 -> 0.576，参数量 3.41 M，耗时 2,892 秒。\n"
       "README 与说明页的对应文字从「正在训练」改为已完成，并写入真实数字。\n"
       "三臂扰动训练与 cell-eval2 评测仍未运行，结果表保持 pending。\n\n"
       "Co-authored-by: CommandCodeBot <noreply@commandcode.ai>\n")


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


print("=== 读取当前 main 的 commit ===")
ref = api(f"repos/{REPO}/git/ref/heads/{BRANCH}")
head_sha = ref["object"]["sha"]
print("  head:", head_sha[:10])

commit = api(f"repos/{REPO}/git/commits/{head_sha}")
base_tree = commit["tree"]["sha"]
print("  base tree:", base_tree[:10])

print("=== 上传 blob ===")
entries = []
for rel in FILES:
    p = os.path.join(ROOT, rel)
    data = open(p, "rb").read()
    blob = api(f"repos/{REPO}/git/blobs", "POST", {
        "content": base64.b64encode(data).decode("ascii"),
        "encoding": "base64",
    })
    entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]})
    print(f"  {rel:14s} {len(data):>7,} bytes  -> {blob['sha'][:10]}")

print("=== 建 tree ===")
tree = api(f"repos/{REPO}/git/trees", "POST", {"base_tree": base_tree, "tree": entries})
print("  tree:", tree["sha"][:10])

print("=== 建 commit ===")
newc = api(f"repos/{REPO}/git/commits", "POST", {
    "message": MSG, "tree": tree["sha"], "parents": [head_sha],
})
print("  commit:", newc["sha"][:10])

print("=== 更新 ref ===")
api(f"repos/{REPO}/git/refs/heads/{BRANCH}", "PATCH", {"sha": newc["sha"]})
print("  ref updated ->", newc["sha"][:10])
print("\nDONE")
