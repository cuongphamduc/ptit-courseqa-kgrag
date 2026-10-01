"""Stage 9b — create the three public locations and upload the packages built by 09_build_release.py.

Credentials and owners come from the environment (never written to disk or logs):
  ZENODO_TOKEN   personal access token, scopes deposit:write + deposit:actions   (ZENODO_SANDBOX=1 -> sandbox.zenodo.org)
  HF_TOKEN       Hugging Face token with write access;  HF_OWNER = user or organisation name
  GH_TOKEN       GitHub token allowed to create repositories (classic: `repo`; fine-grained: Administration + Contents
                 read/write);  GH_OWNER = user or organisation name;  GH_PRIVATE=1 creates a private repository
Order:
  reserve        create a Zenodo draft and pre-reserve its DOI          -> work/release/zenodo_state.json
  build          rebuild the packages with HF_OWNER / GH_OWNER / DOI filled in (runs 09_build_release.py)
  hf             create <HF_OWNER>/PTIT-CourseQA (dataset) and upload the folder
  github         create <GH_OWNER>/ptit-courseqa-kgrag, commit and push the code
  zenodo         upload the two zip files and the metadata to the draft (NOT published)
  publish        publish the Zenodo draft (irreversible: the DOI becomes permanent) — only after the author confirms
usage: 10_upload_release.py reserve build hf github zenodo     and later:  10_upload_release.py publish
"""
import json
import os
import subprocess
import sys

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL = os.path.join(ROOT, "work", "release")
STATE = os.path.join(REL, "zenodo_state.json")
ZEN = "https://sandbox.zenodo.org/api" if os.environ.get("ZENODO_SANDBOX") == "1" else "https://zenodo.org/api"
HF_REPO = "PTIT-CourseQA"
GH_REPO = "ptit-courseqa-kgrag"


def env(k):
    v = os.environ.get(k)
    if not v:
        sys.exit(f"missing environment variable {k}")
    return v


def zen(method, url, **kw):
    r = requests.request(method, url if url.startswith("http") else ZEN + url,
                         headers={"Authorization": f"Bearer {env('ZENODO_TOKEN')}"}, timeout=600, **kw)
    if r.status_code >= 400:
        sys.exit(f"Zenodo {method} {url}: {r.status_code} {r.text[:500]}")
    return r.json() if r.text else {}


def state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def reserve():
    st = state()
    if st.get("id"):
        print("draft exists:", st)
        return
    d = zen("POST", "/deposit/depositions", json={"metadata": {"prereserve_doi": True}})
    st = {"id": d["id"], "doi": d["metadata"]["prereserve_doi"]["doi"], "bucket": d["links"]["bucket"],
          "html": d["links"]["html"], "api": ZEN}
    json.dump(st, open(STATE, "w"), indent=1)
    print("reserved DOI", st["doi"], st["html"])


def build():
    e = dict(os.environ, HF_OWNER=env("HF_OWNER"), GH_OWNER=env("GH_OWNER"), ZENODO_DOI=state()["doi"])
    subprocess.run([sys.executable, os.path.join(ROOT, "work", "scripts", "09_build_release.py")], env=e, check=True)
    for p in ("huggingface", "github", "zenodo"):
        out = subprocess.run(["grep", "-rlE", "\\{\\{(HF_OWNER|GH_OWNER|ZENODO_DOI)\\}\\}", os.path.join(REL, p)], capture_output=True, text=True).stdout
        bad = [l for l in out.split() if l.endswith((".md", ".cff", ".json")) and "outputs" not in l]
        if bad:
            sys.exit(f"unfilled placeholders: {bad}")


def hf():
    from huggingface_hub import HfApi
    api = HfApi(token=env("HF_TOKEN"))
    rid = f"{env('HF_OWNER')}/{HF_REPO}"
    api.create_repo(rid, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(repo_id=rid, repo_type="dataset", folder_path=os.path.join(REL, "huggingface", HF_REPO),
                      commit_message="PTIT-CourseQA v1.0.0")
    print("uploaded https://huggingface.co/datasets/" + rid)


def github():
    tok, owner = env("GH_TOKEN"), env("GH_OWNER")
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    me = requests.get("https://api.github.com/user", headers=h, timeout=60).json()["login"]
    url = "https://api.github.com/user/repos" if owner == me else f"https://api.github.com/orgs/{owner}/repos"
    body = {"name": GH_REPO, "private": os.environ.get("GH_PRIVATE") == "1",
            "description": "PTIT-CourseQA benchmark and KG-RAG evaluation for Vietnamese small language models"}
    r = requests.post(url, headers=h, json=body, timeout=60)
    if r.status_code not in (201, 422):  # 422: already exists
        sys.exit(f"GitHub create: {r.status_code} {r.text[:300]}")
    d = os.path.join(REL, "github", GH_REPO)
    git = lambda *a: subprocess.run(["git", "-C", d, *a], check=True)
    if not os.path.isdir(os.path.join(d, ".git")):
        git("init", "-b", "main")
    git("add", "-A")
    if subprocess.run(["git", "-C", d, "diff", "--cached", "--quiet"]).returncode:
        git("-c", "user.name=Pham Duc Cuong", "-c", "user.email=cuongpd@ptit.edu.vn", "commit", "-m",
            "PTIT-CourseQA v1.0.0: benchmark code and KG-RAG evaluation pipeline\n\n"
            "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>")
    import base64
    auth = base64.b64encode(f"x-access-token:{tok}".encode()).decode()
    git("-c", f"http.extraheader=Authorization: Basic {auth}", "push", "-u",
        f"https://github.com/{owner}/{GH_REPO}.git", "main")
    print(f"pushed https://github.com/{owner}/{GH_REPO}")


def zenodo():
    st = state()
    z = os.path.join(REL, "zenodo")
    for f in sorted(os.listdir(z)):
        if f.endswith(".zip"):
            with open(os.path.join(z, f), "rb") as fh:
                zen("PUT", f"{st['bucket']}/{f}", data=fh)
            print("uploaded", f)
    meta = json.load(open(os.path.join(z, "zenodo_metadata.json")))
    meta["metadata"]["prereserve_doi"] = {"doi": st["doi"], "recid": st["id"]}
    zen("PUT", f"/deposit/depositions/{st['id']}", json=meta)
    print("draft ready (not published):", st["html"])


def publish():
    st = state()
    d = zen("POST", f"/deposit/depositions/{st['id']}/actions/publish")
    print("published", d.get("doi"), d.get("links", {}).get("record_html"))


if __name__ == "__main__":
    steps = {"reserve": reserve, "build": build, "hf": hf, "github": github, "zenodo": zenodo, "publish": publish}
    for a in sys.argv[1:] or sys.exit(__doc__):
        steps[a]()
