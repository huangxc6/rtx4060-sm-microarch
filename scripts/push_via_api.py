"""在 github.com:443 不可达时，用 api.github.com 的 Git Data API 推送本地 HEAD。

原理：GitHub 的 blob 是内容寻址的，未变的文件可以直接沿用远端已有的 blob sha，
只上传变化的部分，再拼一棵新树、建一个提交、移动引用。
"""
from __future__ import annotations

import base64
import json
import subprocess
import time
import sys
import urllib.request
from pathlib import Path

ROOT = Path(r"E:\2026\4060-sm-microarch")
OWNER, REPO, BRANCH = "huangxc6", "rtx4060-sm-microarch", "main"
API = f"https://api.github.com/repos/{OWNER}/{REPO}"


def token() -> str:
    out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        sys.exit("拿不到 gh token")
    return out.stdout.strip()


TOK = token()


def api(path: str, method: str = "GET", body: dict | None = None):
    url = f"{API}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"token {TOK}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "dsh-pusher",
        "Content-Type": "application/json",
    })
    last = None
    for attempt in range(1, 7):          # 网络会间歇性重置 TLS，必须重试
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404, 422):
                sys.exit(f"{method} {path} -> {e.code}: {e.read().decode()[:400]}")
            last = e
        except Exception as e:            # URLError / SSLError / 超时
            last = e
        time.sleep(3 * attempt)
        req = urllib.request.Request(url, data=data, method=method, headers=req.headers)
    sys.exit(f"{method} {path} 重试 6 次仍失败: {last}")


def git(*args: str) -> str:
    """文本输出（强制 utf-8，避免 Windows 下按 GBK 解码中文路径/内容）"""
    out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        sys.exit(f"git {args} 失败: {out.stderr}")
    return out.stdout


def git_bytes(*args: str) -> bytes:
    """二进制输出（blob 内容可能是任意字节，不能当文本解码）"""
    out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True)
    if out.returncode != 0:
        sys.exit(f"git {args} 失败: {out.stderr[:300]}")
    return out.stdout


def main() -> int:
    ref = api(f"/git/ref/heads/{BRANCH}")
    remote_head = ref["object"]["sha"]
    remote_tree = api(f"/git/commits/{remote_head}")["tree"]["sha"]
    print(f"远端 HEAD {remote_head[:8]}  tree {remote_tree[:8]}")

    # 远端已有路径 -> blob sha
    remote = {}
    stack = [(remote_tree, "")]
    while stack:
        sha, prefix = stack.pop()
        for e in api(f"/git/trees/{sha}?recursive=1" if sha == remote_tree else f"/git/trees/{sha}")["tree"]:
            p = f"{prefix}{e['path']}"
            if e["type"] == "blob":
                remote[p] = e["sha"]
            elif e["type"] == "tree":
                stack.append((e["sha"], p + "/"))
    print(f"远端 {len(remote)} 个文件")

    # 本地 HEAD 的完整清单
    local = {}
    for line in git("ls-tree", "-r", "HEAD").splitlines():
        meta, path = line.split("\t")
        mode, _type, sha = meta.split()
        local[path] = (mode, sha)

    changed = {p: v for p, v in local.items() if remote.get(p) != v[1]}
    deleted = [p for p in remote if p not in local]
    print(f"需要上传 {len(changed)} 个、删除 {len(deleted)} 个")

    entries = []
    for i, (path, (mode, sha)) in enumerate(sorted(changed.items()), 1):
        if sha in remote.values():          # 内容已存在于远端（重命名等）
            entries.append({"path": path, "mode": mode, "type": "blob", "sha": sha})
            continue
        data = base64.b64encode(git_bytes("cat-file", "-p", sha)).decode()
        r = api("/git/blobs", "POST", {"content": data, "encoding": "base64"})
        entries.append({"path": path, "mode": mode, "type": "blob", "sha": r["sha"]})
        print(f"  [{i}/{len(changed)}] {path}")
    for p in deleted:
        entries.append({"path": p, "mode": "100644", "type": "blob", "sha": None})

    if not entries:
        print("没有需要推送的改动")
        return 0

    tree = api("/git/trees", "POST", {"base_tree": remote_tree, "tree": entries})
    msg = git("log", "-1", "--pretty=%s%n%n%b").strip()
    commit = api("/git/commits", "POST", {"message": msg, "tree": tree["sha"], "parents": [remote_head]})
    api(f"/git/refs/heads/{BRANCH}", "PATCH", {"sha": commit["sha"], "force": True})
    print(f"\n已推送: {commit['sha'][:8]}  {msg.splitlines()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
