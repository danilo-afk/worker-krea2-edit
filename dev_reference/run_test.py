import sys, json, time, base64, pathlib
sys.path.insert(0,'.')
import rp
EP="qd7bf3ra5erp63"
D="/Users/danilo/Downloads/cena_2_memorias"
def img(p, name):
    return {"name": name, "image": "data:image/jpeg;base64," + base64.b64encode(open(p,'rb').read()).decode()}
mode = sys.argv[1]
if mode == "restore":
    inp = {"prompt": "Enhance this portrait photo: same person, same framing and background, sharp photorealistic detail, natural skin texture, crisp hair strands, clear eyes, soft studio light.",
           "images": [img(f"{D}/REF-maia-rosto-recortado.jpg","a.jpg")], "steps": 10, "ref_boost": 4, "seed": 7}
elif mode == "vamp":
    inp = {"prompt": "Enhance this portrait photo: same man, same framing, remove the dark UI frame around the image so the grey backdrop fills the canvas, sharp photorealistic detail, natural skin texture. Keep the pointed ears, goatee and clothing.",
           "images": [img(f"{D}/REF-vampiro-rosto-ampliado.jpg","a.jpg")], "steps": 10, "ref_boost": 4, "seed": 7, "aspect_ratio":"1:1"}
elif mode == "two":
    inp = {"prompt": "Place this man standing next to the woman on a rooftop at night, city lights blurred behind them, cinematic soft light. Keep both faces exactly as in the references.",
           "images": [img(f"{D}/REF-maia-rosto-recortado.jpg","a.jpg"), img(f"{D}/REF-vampiro-rosto-ampliado.jpg","b.jpg")], "steps": 10, "ref_boost": 4, "ref_boost_a": 2, "seed": 7, "aspect_ratio":"3:4"}
if len(sys.argv)>2 and sys.argv[2]=="refine": inp["refine"]=True
s,d = rp.job(EP,"run",{"input":inp}); print("submit",s,d)
jid=d["id"]; t0=time.time(); last=None
while True:
    s,st = rp.job(EP,f"status/{jid}",method="GET")
    if st.get("status")!=last: print(int(time.time()-t0),"s", st.get("status")); last=st.get("status")
    if st.get("status") in ("COMPLETED","FAILED","CANCELLED","TIMED_OUT"): break
    time.sleep(10)
out=st.get("output") or {}
if st.get("status")!="COMPLETED":
    print(json.dumps(st)[:3000]); sys.exit(1)
print("exec ms", st.get("executionTime"), "delay ms", st.get("delayTime"))
if "error" in out: print(json.dumps(out)[:3000]); sys.exit(1)
for i,im in enumerate(out.get("images",[])):
    p=pathlib.Path(f"{D}/krea2_{mode}{'_refine' if inp.get('refine') else ''}_{i}.png"); p.write_bytes(base64.b64decode(im["data"])); print("saved",p)
