# worker-krea2-edit

RunPod Serverless + ComfyUI: **Krea 2 Turbo (fp8) + LoRA Identity Edit v1.2** — edição de imagem por referência
(restore/re-stage de pessoa, compor pessoa em cena, edições locais) via nós `comfyui-krea2edit`.

- Custom node: fork `danilo-afk/comfyui-krea2edit` (pinado por SHA no Dockerfile).
- Modelos (públicos, ~20,5GB) baixados pelo `start.sh` pro network volume: `Comfy-Org/Krea-2`
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
