#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Ensure ComfyUI-Manager runs in offline network mode
comfy-manager-set-mode offline || echo "worker-krea2-edit - Could not set ComfyUI-Manager network_mode" >&2

# ============ Auto-download modelos para Network Volume ============
VOLUME="/runpod-volume"
if [ -d "$VOLUME" ]; then
  echo "worker-krea2-edit: Network volume detectado em $VOLUME"

  # Evita corrida entre múltiplos workers escrevendo os mesmos modelos no volume.
  LOCK_FILE="$VOLUME/.model-bootstrap.lock"
  if command -v flock >/dev/null 2>&1; then
    exec 9>"$LOCK_FILE"
    if ! flock -w 1800 9; then
      echo "worker-krea2-edit: timeout aguardando lock de bootstrap ($LOCK_FILE)" >&2
      exit 1
    fi
    echo "worker-krea2-edit: lock de bootstrap adquirido"
  else
    echo "worker-krea2-edit: flock não encontrado; bootstrap seguirá sem lock" >&2
  fi

  validate_safetensors_coverage() {
    local file="$1"
    python - "$file" <<'PY'
import json
import os
import struct
import sys

path = sys.argv[1]
size = os.path.getsize(path)
if size < 8:
    raise SystemExit("arquivo menor que 8 bytes")

with open(path, "rb") as f:
    header_len_raw = f.read(8)
    if len(header_len_raw) != 8:
        raise SystemExit("falha ao ler tamanho do header")
    header_len = struct.unpack("<Q", header_len_raw)[0]
    if header_len <= 0 or (8 + header_len) > size:
        raise SystemExit(
            f"header inválido: header_len={header_len}, file_size={size}"
        )
    header_bytes = f.read(header_len)
    if len(header_bytes) != header_len:
        raise SystemExit("header incompleto")

try:
    header = json.loads(header_bytes)
except Exception as e:
    raise SystemExit(f"header JSON inválido: {e}")

max_end = 0
for key, value in header.items():
    if key == "__metadata__":
        continue
    if not isinstance(value, dict):
        raise SystemExit(f"tensor {key} inválido: entrada não é dict")
    data_offsets = value.get("data_offsets")
    if not isinstance(data_offsets, list) or len(data_offsets) != 2:
        raise SystemExit(f"tensor {key} sem data_offsets válidos")
    start, end = data_offsets
    if not isinstance(start, int) or not isinstance(end, int):
        raise SystemExit(f"tensor {key} offsets não inteiros")
    if start < 0 or end < start:
        raise SystemExit(f"tensor {key} offsets inválidos: {start}, {end}")
    max_end = max(max_end, end)

required_size = 8 + header_len + max_end
if required_size > size:
    raise SystemExit(
        f"arquivo incompleto: required_size={required_size}, file_size={size}"
    )
PY
  }

  check_size() {
    local file="$1" min="$2"
    if [ -f "$file" ]; then
      local size
      size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
      if [ "$size" -lt "$min" ]; then
        echo "worker-krea2-edit: $file corrompido (${size} bytes < ${min}), re-baixando..."
        rm -f "$file"
        return 1
      fi
      if [[ "$file" == *.safetensors ]]; then
        if ! validate_safetensors_coverage "$file" >/tmp/worker_safetensors_check.log 2>&1; then
          echo "worker-krea2-edit: $file inválido (safetensors), re-baixando..."
          cat /tmp/worker_safetensors_check.log >&2 || true
          rm -f "$file"
          return 1
        fi
      fi
    else
      return 1
    fi
    return 0
  }

  # Download rápido via hf_hub_download + hf_transfer (100-200 MB/s vs ~13 MB/s do wget).
  hf_fast_download() {
    local file="$1" url="$2"
    HF_HUB_ENABLE_HF_TRANSFER=1 python - "$file" "$url" <<'PYHF'
import os, sys, shutil, tempfile
file, url = sys.argv[1], sys.argv[2]
try:
    rest = url.split("huggingface.co/", 1)[1]
    repo_part, path_part = rest.split("/resolve/", 1)
    rev, filename = path_part.split("/", 1)
except Exception as e:
    print("worker-krea2-edit: hf parse falhou:", e); sys.exit(3)
try:
    from huggingface_hub import hf_hub_download
except Exception as e:
    print("worker-krea2-edit: huggingface_hub ausente:", e); sys.exit(4)
tmp = tempfile.mkdtemp(dir=os.path.dirname(file) or "/tmp")
try:
    p = hf_hub_download(repo_id=repo_part, filename=filename, revision=rev, local_dir=tmp)
    os.makedirs(os.path.dirname(file), exist_ok=True)
    shutil.move(p, file)
    print("worker-krea2-edit: hf_transfer OK ->", file)
except Exception as e:
    print("worker-krea2-edit: hf_hub_download falhou:", e); sys.exit(5)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
PYHF
  }

  download_with_validation() {
    local file="$1" min="$2" url="$3" label="$4"
    local max_attempts=3
    local attempt=1
    while [ "$attempt" -le "$max_attempts" ]; do
      echo "worker-krea2-edit: Baixando ${label} (tentativa ${attempt}/${max_attempts})..."
      mkdir -p "$(dirname "$file")"
      rm -f "$file"
      # 1) rápido: hf_hub_download + hf_transfer
      if hf_fast_download "$file" "$url" && check_size "$file" "$min"; then
        return 0
      fi
      rm -f "$file"
      # 2) fallback: wget (nunca fica pior que o método antigo)
      if wget --progress=dot:giga -O "$file" "$url" && check_size "$file" "$min"; then
        return 0
      fi
      echo "worker-krea2-edit: Falha ao validar ${label} na tentativa ${attempt}" >&2
      attempt=$((attempt + 1))
    done
    echo "worker-krea2-edit: ERRO ao baixar ${label} após ${max_attempts} tentativas." >&2
    return 1
  }

  # ============ Krea 2 Turbo (fp8, Comfy-Org) + Identity Edit LoRA ============
  # Todos públicos (não-gated). ~20,5GB no total; cabe na RTX 4090 (24GB).
  KREA_UNET="$VOLUME/models/diffusion_models/krea2_turbo_fp8_scaled.safetensors"
  KREA_UNET_URL="https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors"
  if ! check_size "$KREA_UNET" 13000000000; then
    download_with_validation "$KREA_UNET" 13000000000 "$KREA_UNET_URL" "Krea 2 Turbo fp8" || exit 1
  fi
  KREA_TE="$VOLUME/models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors"
  KREA_TE_URL="https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors"
  if ! check_size "$KREA_TE" 5200000000; then
    download_with_validation "$KREA_TE" 5200000000 "$KREA_TE_URL" "Qwen3-VL 4B fp8 (text encoder)" || exit 1
  fi
  KREA_VAE="$VOLUME/models/vae/qwen_image_vae.safetensors"
  KREA_VAE_URL="https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors"
  if ! check_size "$KREA_VAE" 250000000; then
    download_with_validation "$KREA_VAE" 250000000 "$KREA_VAE_URL" "Qwen Image VAE" || exit 1
  fi
  # LoRA Identity Edit v1.2 (conradlocke) — 1,8GB. Versão configurável via env.
  KREA_LORA_NAME="${KREA_LORA_NAME:-krea2_identity_edit_v1_2.safetensors}"
  KREA_LORA="$VOLUME/models/loras/$KREA_LORA_NAME"
  KREA_LORA_URL="https://huggingface.co/conradlocke/krea2-identity-edit/resolve/main/$KREA_LORA_NAME"
  if ! check_size "$KREA_LORA" 400000000; then
    download_with_validation "$KREA_LORA" 400000000 "$KREA_LORA_URL" "Krea2 Identity Edit LoRA" || exit 1
  fi
  # ==========================================================================

  echo "worker-krea2-edit: Modelos prontos no volume!"
else
  echo "worker-krea2-edit: Sem network volume, usando modelos do container"
fi
# ===================================================================

echo "worker-krea2-edit: Starting ComfyUI"

: "${COMFY_LOG_LEVEL:=DEBUG}"
: "${COMFY_STARTUP_LOG:=/tmp/comfyui.log}"
: "${GPU_READY_MAX_ATTEMPTS:=180}"
: "${GPU_READY_SLEEP_SECONDS:=2}"
mkdir -p "$(dirname "$COMFY_STARTUP_LOG")"
: > "$COMFY_STARTUP_LOG"
echo "worker-krea2-edit: startup log em $COMFY_STARTUP_LOG"

export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_MODULE_LOADING=LAZY

print_gpu_snapshot() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "worker-krea2-edit: nvidia-smi snapshot:"
    nvidia-smi -L || true
    nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader || true
  else
    echo "worker-krea2-edit: nvidia-smi não encontrado" >&2
  fi
}

print_comfy_failure_excerpt() {
  local log_file="$1"
  echo "worker-krea2-edit: últimas linhas do ComfyUI log:" >&2
  if [ -f "$log_file" ]; then
    tail -n 120 "$log_file" >&2 || true
  else
    echo "worker-krea2-edit: log não encontrado em $log_file" >&2
  fi
}

wait_for_gpu_ready() {
  local attempt=1
  while [ "$attempt" -le "$GPU_READY_MAX_ATTEMPTS" ]; do
    if command -v nvidia-smi >/dev/null 2>&1; then
      if nvidia-smi -L >/dev/null 2>&1; then
        echo "worker-krea2-edit: GPU pronta (tentativa ${attempt}/${GPU_READY_MAX_ATTEMPTS})"
        return 0
      fi
    else
      # fallback sem nvidia-smi
      return 0
    fi
    echo "worker-krea2-edit: aguardando GPU ficar disponível (${attempt}/${GPU_READY_MAX_ATTEMPTS})..."
    sleep "$GPU_READY_SLEEP_SECONDS"
    attempt=$((attempt + 1))
  done
  echo "worker-krea2-edit: GPU indisponível após ${GPU_READY_MAX_ATTEMPTS} tentativas" >&2
  return 1
}

start_comfy_supervisor() {
  local comfy_args=("$@")
  local max_fast_failures="${COMFY_MAX_FAST_FAILURES:-5}"
  local fast_failure_window_s="${COMFY_FAST_FAILURE_WINDOW_S:-45}"
  (
    local attempt=1
    local fast_failures=0
    while true; do
      local started_at
      started_at=$(date +%s)
      echo "worker-krea2-edit: iniciando ComfyUI (attempt ${attempt})"
      python -u /comfyui/main.py "${comfy_args[@]}" >> "$COMFY_STARTUP_LOG" 2>&1
      local code=$?
      local ended_at
      ended_at=$(date +%s)
      local runtime_s=$((ended_at - started_at))
      echo "worker-krea2-edit: ComfyUI saiu com código ${code} (attempt ${attempt}, runtime ${runtime_s}s)" | tee -a "$COMFY_STARTUP_LOG"
      print_comfy_failure_excerpt "$COMFY_STARTUP_LOG"

      if [ "$runtime_s" -le "$fast_failure_window_s" ]; then
        fast_failures=$((fast_failures + 1))
      else
        fast_failures=0
      fi

      if [ "$fast_failures" -ge "$max_fast_failures" ]; then
        echo "worker-krea2-edit: ComfyUI falhou rapidamente ${fast_failures} vezes; abortando worker para evitar loop infinito" | tee -a "$COMFY_STARTUP_LOG" >&2
        exit 1
      fi

      attempt=$((attempt + 1))
      sleep 4
      wait_for_gpu_ready || true
    done
  ) &
}

print_gpu_snapshot
wait_for_gpu_ready

if [ "$SERVE_API_LOCALLY" == "true" ]; then
    start_comfy_supervisor --disable-auto-launch --disable-metadata --lowvram --listen --verbose "${COMFY_LOG_LEVEL}" --log-stdout

    echo "worker-krea2-edit: Starting RunPod Handler"
    python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    start_comfy_supervisor --disable-auto-launch --disable-metadata --lowvram --verbose "${COMFY_LOG_LEVEL}" --log-stdout

    echo "worker-krea2-edit: Starting RunPod Handler"
    python -u /handler.py
fi
