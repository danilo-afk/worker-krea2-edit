import os
import json, sys, re, requests
def key():
    for l in open('os.environ.get("RUNPOD_KEY_FILE","")'):
        m = re.search(r'(rpa_[A-Za-z0-9_-]{20,})', l)
        if m and m.group(1).startswith(os.environ.get('RUNPOD_KEY_PREFIX','rpa_')): return m.group(1)
    for l in open('os.environ.get("RUNPOD_KEY_FILE","")'):
        m = re.search(r'(rpa_[A-Za-z0-9_-]{20,})', l)
        if m: return m.group(1)
    raise SystemExit("sem chave runpod")
K = key()
REST = "https://rest.runpod.io/v1"
H = {"Authorization": f"Bearer {K}", "Content-Type": "application/json"}
def rest(method, path, body=None):
    r = requests.request(method, REST + path, headers=H, json=body, timeout=60)
    try: return r.status_code, r.json()
    except Exception: return r.status_code, r.text[:500]
def job(ep, path, body=None, method="POST"):
    r = requests.request(method, f"https://api.runpod.ai/v2/{ep}/{path}", headers=H, json=body, timeout=120)
    return r.status_code, r.json()
if __name__ == "__main__":
    what = sys.argv[1]
    if what == "vols":
        print(json.dumps(rest("GET","/networkvolumes")[1], indent=1)[:3000])
    elif what == "eps":
        s, d = rest("GET","/endpoints")
        for e in d: print(e.get("id"), e.get("name"), e.get("gpuTypeIds"), e.get("networkVolumeId"), e.get("workersMax"), e.get("computeType"))
