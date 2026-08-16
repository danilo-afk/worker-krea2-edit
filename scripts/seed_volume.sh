#!/usr/bin/env bash
# Semeia o network volume com os modelos do Krea 2 Identity Edit — rodar UMA vez
# num POD (CPU serve) da MESMA região do volume, com o volume montado em /workspace.
# Ex.: VOLUME=/workspace bash scripts/seed_volume.sh
# Não usa GPU: evita pagar GPU parada no cold start do serverless.
set -euo pipefail
VOLUME="${VOLUME:-/workspace}"
LORA="${KREA_LORA_NAME:-krea2_identity_edit_v1_2.safetensors}"
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 -c "import huggingface_hub, hf_transfer" 2>/dev/null || pip install -q "huggingface_hub[hf_transfer]" hf_transfer

dl() { # repo file dest
  local repo="$1" file="$2" dest="$3"
  if [ -s "$dest" ]; then echo "ok (existe): $dest"; return; fi
  mkdir -p "$(dirname "$dest")"
  python3 - "$repo" "$file" "$dest" <<'PY'
import sys, shutil, tempfile, os
from huggingface_hub import hf_hub_download
repo, file, dest = sys.argv[1:4]
tmp = tempfile.mkdtemp(dir=os.path.dirname(dest))
p = hf_hub_download(repo_id=repo, filename=file, local_dir=tmp)
shutil.move(p, dest); shutil.rmtree(tmp, ignore_errors=True)
print("baixado:", dest, os.path.getsize(dest))
PY
}
dl Comfy-Org/Krea-2 diffusion_models/krea2_turbo_fp8_scaled.safetensors "$VOLUME/models/diffusion_models/krea2_turbo_fp8_scaled.safetensors"
dl Comfy-Org/Krea-2 text_encoders/qwen3vl_4b_fp8_scaled.safetensors     "$VOLUME/models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors"
dl Comfy-Org/Krea-2 vae/qwen_image_vae.safetensors                       "$VOLUME/models/vae/qwen_image_vae.safetensors"
dl conradlocke/krea2-identity-edit "$LORA"                               "$VOLUME/models/loras/$LORA"
touch "$VOLUME/.krea2_models_ready"
echo "volume semeado: $VOLUME"; du -sh "$VOLUME/models"
