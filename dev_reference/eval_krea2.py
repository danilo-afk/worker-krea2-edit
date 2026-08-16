"""Eval Krea2 Identity Edit: matriz prompt(PT/EN) x ref_boost -> juiz VLM (identidade, obediência) + diff perceptual."""
import os, sys, json, time, base64, re, urllib.request, io
sys.path.insert(0, os.path.dirname(__file__)); import rp
EP="qd7bf3ra5erp63"
KEY = os.environ["GEMINI_API_KEY"]  # juiz VLM (Google AI Studio)
A = os.environ.get("EVAL_REF_IMAGE", "ref.png")  # retrato de referência
CASES = [
  ("pt", "ela na praia corpo inteiro e biquine brasileiro azul"),
  ("en", "Place this same woman on a sunny beach, full body shot, wearing a blue Brazilian bikini. Keep her face exactly the same."),
]
BOOSTS = [4, 2, 1]
def b64(p): return base64.b64encode(open(p,'rb').read()).decode()
def run(prompt, boost):
    inp={"prompt":prompt,"images":[{"name":"a.png","image":"data:image/png;base64,"+b64(A)}],"ref_boost":boost,"seed":11,"aspect_ratio":"3:4"}
    s,d=rp.job(EP,"run",{"input":inp}); jid=d["id"]
    while True:
        s,st=rp.job(EP,f"status/{jid}",method="GET")
        if st.get("status") in ("COMPLETED","FAILED","CANCELLED","TIMED_OUT"): break
        time.sleep(5)
    if st["status"]!="COMPLETED": return None, st
    return base64.b64decode(st["output"]["images"][0]["data"]), st
def judge(out_png, prompt):
    body={"contents":[{"parts":[
      {"text":"You are a strict image-edit judge. IMAGE 1 = reference person. IMAGE 2 = edited output. Prompt used: '"+prompt+"'. Answer ONLY JSON: {\"identity\": 0-10 (same face as IMAGE 1?), \"prompt_adherence\": 0-10 (does IMAGE 2 follow the prompt: beach, full body, BLUE bikini?), \"blue_bikini\": true/false, \"beach\": true/false, \"full_body\": true/false, \"notes\": short}"},
      {"inline_data":{"mime_type":"image/png","data":b64(A)}},
      {"inline_data":{"mime_type":"image/png","data":base64.b64encode(out_png).decode()}}]}],
      "generationConfig":{"responseMimeType":"application/json"}}
    req=urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent",data=json.dumps(body).encode(),headers={"Content-Type":"application/json","x-goog-api-key":KEY})
    r=json.load(urllib.request.urlopen(req,timeout=120))
    return json.loads(r["candidates"][0]["content"]["parts"][0]["text"])
def phash_diff(a_bytes, b_bytes):
    from PIL import Image
    def ph(b):
        im=Image.open(io.BytesIO(b)).convert("L").resize((16,16)); px=list(im.getdata()); m=sum(px)/len(px); return [1 if p>m else 0 for p in px]
    x,y=ph(a_bytes),ph(b_bytes); return sum(i!=j for i,j in zip(x,y))/len(x)
results=[]
for lang,prompt in CASES:
    for boost in BOOSTS:
        t=time.time(); out,st=run(prompt,boost)
        if out is None: results.append({"lang":lang,"boost":boost,"error":str(st)[:200]}); continue
        p=f"eval_{lang}_rb{boost}.png"; open(p,'wb').write(out)
        j=judge(out,prompt); d=phash_diff(open(A,'rb').read(),out)
        results.append({"lang":lang,"boost":boost,"exec_s":round(st.get("executionTime",0)/1000,1),"diff":round(d,2),**j,"file":p})
        print(json.dumps(results[-1],ensure_ascii=False),flush=True)
json.dump(results,open("eval_krea2_results.json","w"),indent=1,ensure_ascii=False)
