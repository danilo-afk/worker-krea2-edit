# worker-krea2-edit

RunPod Serverless + ComfyUI: **Krea 2 Turbo (fp8) + LoRA Identity Edit v1.2** — edição de imagem por referência
(restore/re-stage de pessoa, compor pessoa em cena, edições locais) via nós `comfyui-krea2edit`.

- Custom node: fork `danilo-afk/comfyui-krea2edit` (pinado por SHA no Dockerfile).
- Modelos (públicos, ~20,5GB) no network volume — **semear UMA vez fora do serverless** com `scripts/seed_volume.sh` (pod da mesma região, volume em `/workspace`); o `start.sh` só valida e falha rápido se faltar (`KREA_BOOTSTRAP_DOWNLOAD=1` reativa o download no worker): `Comfy-Org/Krea-2`
  (`krea2_turbo_fp8_scaled`, `qwen3vl_4b_fp8_scaled`, `qwen_image_vae`) + `conradlocke/krea2-identity-edit` v1.2.
- GPU: RTX 4090 (24GB) serve.

## Input (modo-prompt)
```json
{"input": {"prompt": "...", "images": [{"name":"a.png","image":"data:image/png;base64,..."}, {"name":"b.png","image":"..."}],
 "seed": 42, "steps": 10, "ref_boost": 4, "grounding_px": 768, "aspect_ratio": "3:4",
 "refine": false, "refine_steps": 4, "refine_denoise": 0.2}}
```
`images[0]` = **A** (base/cena, define aspecto), `images[1]` = **B** (pessoa) — ordem fixa do LoRA. Aceita
`image_urls`/`reference_images` (URLs) no lugar de `images`. `workflow` (API format) cru também é aceito.

## Output
`{"images": [{"filename", "type": "base64", "data"}]}`.

## POC validada (2026-08-16)
Endpoint `qd7bf3ra5erp63` (RTX 4090, volume `p79ybse2ph` US-IL-1). Warm: 22–45s/img; cold (volume semeado): ~4,5 min.
`dev_reference/run_test.py restore|vamp|two [refine]` (RUNPOD_KEY_FILE aponta p/ arquivo com a chave `rpa_…`).
Aprendizados: LoRA preserva o resto da imagem quase pixel-a-pixel (moldura/borda NÃO some — cropar antes);
2 refs = identidades fortes (~1MP); `refine` deixa a pele "dura" (opt-in). Após build novo: recycle
(`workersMax 0→1`) — o Job API pode responder `ENDPOINT_PAUSED` por ~1 min após o PATCH (propagação).
