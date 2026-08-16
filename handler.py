import runpod
from runpod.serverless.utils import rp_upload
import json
import urllib.request
import urllib.parse
import time
import os
import requests
import base64
from io import BytesIO
import websocket
import uuid
import tempfile
import socket
import traceback
import logging
import subprocess
import re
import hashlib
try:
    from PIL import Image
except ImportError:
    Image = None

from network_volume import (
    is_network_volume_debug_enabled,
    run_network_volume_diagnostics,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COMFY_API_AVAILABLE_INTERVAL_MS = int(os.environ.get("COMFY_API_AVAILABLE_INTERVAL_MS", 100))
COMFY_API_AVAILABLE_MAX_RETRIES = int(os.environ.get("COMFY_API_AVAILABLE_MAX_RETRIES", 1800))
WEBSOCKET_RECONNECT_ATTEMPTS = int(os.environ.get("WEBSOCKET_RECONNECT_ATTEMPTS", 5))
WEBSOCKET_RECONNECT_DELAY_S = int(os.environ.get("WEBSOCKET_RECONNECT_DELAY_S", 3))
MAX_INLINE_VIDEO_BYTES = int(os.environ.get("MAX_INLINE_VIDEO_BYTES", 4_000_000))
COMFY_STARTUP_LOG = os.environ.get("COMFY_STARTUP_LOG", "/tmp/comfyui.log")
WORKFLOW_EVENT_IDLE_TIMEOUT_S = int(os.environ.get("WORKFLOW_EVENT_IDLE_TIMEOUT_S", 180))
GEMMA_NODE_IDLE_TIMEOUT_S = int(os.environ.get("GEMMA_NODE_IDLE_TIMEOUT_S", 900))
CHECKPOINT_NODE_IDLE_TIMEOUT_S = int(os.environ.get("CHECKPOINT_NODE_IDLE_TIMEOUT_S", 1200))
SAMPLER_NODE_IDLE_TIMEOUT_S = int(os.environ.get("SAMPLER_NODE_IDLE_TIMEOUT_S", 1800))
DECODE_NODE_IDLE_TIMEOUT_S = int(os.environ.get("DECODE_NODE_IDLE_TIMEOUT_S", 900))

if os.environ.get("WEBSOCKET_TRACE", "false").lower() == "true":
    websocket.enableTrace(True)

COMFY_HOST = "127.0.0.1:8188"
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"

OOM_PATTERNS = [
    re.compile(r"out of memory", re.IGNORECASE),
    re.compile(r"cuda.*oom", re.IGNORECASE),
    re.compile(r"cublas_status_alloc_failed", re.IGNORECASE),
    re.compile(r"memoryerror", re.IGNORECASE),
    re.compile(r"std::bad_alloc", re.IGNORECASE),
    re.compile(r"insufficient memory", re.IGNORECASE),
    re.compile(r"not enough memory", re.IGNORECASE),
    re.compile(r"killed process", re.IGNORECASE),
]

GPU_PATTERNS = [
    re.compile(r"cuda", re.IGNORECASE),
    re.compile(r"cudnn", re.IGNORECASE),
    re.compile(r"cublas", re.IGNORECASE),
    re.compile(r"nvidia", re.IGNORECASE),
    re.compile(r"device-side assert", re.IGNORECASE),
    re.compile(r"illegal memory access", re.IGNORECASE),
    re.compile(r"driver", re.IGNORECASE),
]


def _detect_image_format(blob):
    if blob[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if blob[:2] == b"\xff\xd8":
        return "jpeg"
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return "webp"
    if blob[:3] == b"GIF":
        return "gif"
    return "unknown"


def _summarize_bytes(blob):
    return {
        "size_bytes": len(blob),
        "sha256_16": hashlib.sha256(blob).hexdigest()[:16],
        "magic_hex_16": blob[:16].hex(),
        "detected_format": _detect_image_format(blob),
    }


def _get_idle_timeout_for_node_class(node_class):
    # Loaders (unet fp8 13GB / qwen3vl 5GB do volume NFS) e o encode grounded (VLM)
    # ficam minutos sem evento no cold start — tolerar.
    if node_class in ("CLIPLoader", "Krea2EditGroundedEncode", "LTXVGemmaCLIPModelLoader", "LTXAVTextEncoderLoader"):
        return GEMMA_NODE_IDLE_TIMEOUT_S
    if node_class in ("CheckpointLoaderSimple", "UNETLoader", "LoraLoaderModelOnly", "Krea2EditModelPatch"):
        return CHECKPOINT_NODE_IDLE_TIMEOUT_S
    if node_class in ("SamplerCustomAdvanced", "KSampler"):
        return SAMPLER_NODE_IDLE_TIMEOUT_S
    # LTX-2 usa LTXVSpatioTemporalTiledVAEDecode; LTX-2.3 usa LTXVTiledVAEDecode/LTXVAudioVAEDecode.
    if node_class in (
        "LTXVSpatioTemporalTiledVAEDecode",
        "LTXVTiledVAEDecode",
        "LTXVAudioVAEDecode",
    ):
        return DECODE_NODE_IDLE_TIMEOUT_S
    return WORKFLOW_EVENT_IDLE_TIMEOUT_S


def _probe_image_path(path):
    info = {
        "path": path,
        "exists": os.path.exists(path),
    }
    if not info["exists"]:
        return info

    with open(path, "rb") as fh:
        raw = fh.read()
    info.update(_summarize_bytes(raw))

    try:
        result = subprocess.run(
            ["file", "-b", path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        info["file_cmd"] = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    except Exception as exc:
        info["file_cmd_error"] = str(exc)

    if Image is None:
        info["pillow"] = "unavailable"
        return info

    try:
        with Image.open(path) as img:
            img.load()
            info["pillow"] = {
                "format": img.format,
                "mode": img.mode,
                "size": list(img.size),
            }
    except Exception as exc:
        info["pillow_error"] = f"{type(exc).__name__}: {exc}"

    return info


def _preflight_loadimage_inputs(workflow):
    load_nodes = _extract_loadimage_nodes(workflow)
    checks = []
    for node in load_nodes:
        expected = node.get("expected_image")
        if not isinstance(expected, str):
            continue
        path = os.path.join("/comfyui/input", expected)
        probe = _probe_image_path(path)
        checks.append({
            "node_id": node.get("node_id"),
            "expected_image": expected,
            "probe": probe,
        })
    return checks


def _safe_dict_keys(value):
    if isinstance(value, dict):
        return sorted(list(value.keys()))
    return []


def _build_runtime_diagnostics(parts):
    """
    Classifica mensagens de erro em categorias úteis para retorno de API.
    """
    if parts is None:
        return None

    if isinstance(parts, str):
        text_parts = [parts]
    elif isinstance(parts, (list, tuple)):
        text_parts = [str(p) for p in parts if p is not None and str(p).strip()]
    else:
        text_parts = [str(parts)]

    joined = " | ".join(text_parts).strip()
    if not joined:
        return None

    for pattern in OOM_PATTERNS:
        if pattern.search(joined):
            return {
                "category": "GPU_OOM",
                "matched": pattern.pattern,
                "message": joined[:1200],
            }

    for pattern in GPU_PATTERNS:
        if pattern.search(joined):
            return {
                "category": "GPU_RUNTIME",
                "matched": pattern.pattern,
                "message": joined[:1200],
            }

    return {
        "category": "UNKNOWN",
        "message": joined[:1200],
    }


def _read_comfy_log_tail(max_lines=120):
    if not COMFY_STARTUP_LOG or not os.path.exists(COMFY_STARTUP_LOG):
        return []
    try:
        with open(COMFY_STARTUP_LOG, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        return [line.rstrip("\n") for line in lines[-max_lines:] if line.strip()]
    except Exception as exc:
        return [f"(falha ao ler log de startup: {exc})"]


def _extract_loadimage_nodes(workflow):
    """Retorna lista de nós LoadImage com nome esperado do arquivo."""
    load_nodes = []
    if not isinstance(workflow, dict):
        return load_nodes

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        if class_type != "LoadImage":
            continue
        inputs = node.get("inputs", {})
        expected_image = inputs.get("image") if isinstance(inputs, dict) else None
        load_nodes.append({"node_id": str(node_id), "expected_image": expected_image})

    return load_nodes


def _build_workflow_node_lookup(workflow):
    lookup = {}
    if not isinstance(workflow, dict):
        return lookup
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        lookup[str(node_id)] = node.get("class_type") or "unknown"
    return lookup


def _log_ws_event(prefix, payload):
    print(f"worker-krea2-edit - {prefix}:", json.dumps(payload, ensure_ascii=False))


def _resolve_model_probe(path_hint, categories):
    if not isinstance(path_hint, str) or not path_hint:
        return {"path_hint": path_hint, "resolved": None, "exists": False}

    candidates = []
    if os.path.isabs(path_hint):
        candidates.append(path_hint)
    else:
        for category in categories:
            candidates.append(os.path.join("/runpod-volume/models", category, path_hint))
            candidates.append(os.path.join("/comfyui/models", category, path_hint))

    for candidate in candidates:
        if os.path.exists(candidate):
            info = {
                "path_hint": path_hint,
                "resolved": candidate,
                "exists": True,
                "size_bytes": os.path.getsize(candidate),
            }
            try:
                with open(candidate, "rb") as fh:
                    info["sha256_16"] = hashlib.sha256(fh.read(1024 * 1024)).hexdigest()[:16]
            except Exception as exc:
                info["sha256_error"] = str(exc)
            return info

    return {"path_hint": path_hint, "resolved": None, "exists": False, "candidates": candidates}


def _preflight_gemma_loader(workflow):
    results = []
    if not isinstance(workflow, dict):
        return results

    for node_id, node in workflow.items():
        if not isinstance(node, dict) or node.get("class_type") != "LTXVGemmaCLIPModelLoader":
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            inputs = {}
        gemma_path = inputs.get("gemma_path")
        ltxv_path = inputs.get("ltxv_path")
        results.append({
            "node_id": str(node_id),
            "gemma_path": _resolve_model_probe(gemma_path, ["text_encoders", "LLM"]),
            "ltxv_path": _resolve_model_probe(ltxv_path, ["checkpoints"]),
            "max_length": inputs.get("max_length"),
        })
    return results


def _log_job_diagnostics(job_id, job_input, workflow, input_images):
    """Logs curtos para diagnosticar se I2V recebeu imagens corretamente."""
    input_keys = _safe_dict_keys(job_input)
    image_count = len(input_images) if isinstance(input_images, list) else 0
    image_names = []
    image_payload_sizes = []

    if isinstance(input_images, list):
        for image in input_images:
            if not isinstance(image, dict):
                continue
            name = image.get("name")
            payload = image.get("image")
            if isinstance(name, str):
                image_names.append(name)
            if isinstance(payload, str):
                image_payload_sizes.append(len(payload))

    load_nodes = _extract_loadimage_nodes(workflow)
    expected_names = [
        item.get("expected_image")
        for item in load_nodes
        if isinstance(item.get("expected_image"), str)
    ]

    print(
        "worker-krea2-edit - Job input summary:",
        json.dumps(
            {
                "job_id": job_id,
                "input_keys": input_keys,
                "workflow_node_count": len(workflow) if isinstance(workflow, dict) else 0,
                "load_image_nodes": load_nodes,
                "images_count": image_count,
                "image_names": image_names,
                "image_payload_sizes": image_payload_sizes,
            },
            ensure_ascii=False,
        ),
    )

    if load_nodes and image_count == 0:
        print(
            "worker-krea2-edit - WARNING: Workflow possui LoadImage, mas input.images veio vazio/ausente."
        )

    if expected_names and image_names:
        missing = [name for name in expected_names if name not in image_names]
        extra = [name for name in image_names if name not in expected_names]
        if missing or extra:
            print(
                "worker-krea2-edit - WARNING: Nomes de imagem não batem com LoadImage:",
                json.dumps({"expected": expected_names, "received": image_names, "missing": missing, "extra": extra}),
            )


def _comfy_server_status():
    """Verifica se o servidor ComfyUI HTTP está acessível."""
    try:
        resp = requests.get(f"http://{COMFY_HOST}/", timeout=5)
        return {"reachable": resp.status_code == 200, "status_code": resp.status_code}
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


def _attempt_websocket_reconnect(ws_url, max_attempts, delay_s, initial_error):
    """Tenta reconectar ao WebSocket após desconexão."""
    print(f"worker-krea2-edit - Websocket fechou: {initial_error}. Reconectando...")
    last_error = initial_error
    for attempt in range(max_attempts):
        srv_status = _comfy_server_status()
        if not srv_status["reachable"]:
            print(f"worker-krea2-edit - ComfyUI HTTP inacessível – abortando reconexão")
            raise websocket.WebSocketConnectionClosedException(
                "ComfyUI HTTP unreachable during websocket reconnect"
            )
        print(f"worker-krea2-edit - Tentativa {attempt + 1}/{max_attempts}...")
        try:
            new_ws = websocket.WebSocket()
            new_ws.connect(ws_url, timeout=10)
            print(f"worker-krea2-edit - Websocket reconectado.")
            return new_ws
        except (websocket.WebSocketException, ConnectionRefusedError, socket.timeout, OSError) as err:
            last_error = err
            print(f"worker-krea2-edit - Tentativa {attempt + 1} falhou: {err}")
            if attempt < max_attempts - 1:
                time.sleep(delay_s)

    raise websocket.WebSocketConnectionClosedException(
        f"Falha ao reconectar. Último erro: {last_error}"
    )


def convert_video_to_mp4(video_bytes, filename):
    """Converte vídeo para MP4 via ffmpeg se necessário."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".mp4":
        return video_bytes, filename

    src_path = None
    mp4_path = None
    try:
        src_path = os.path.join(tempfile.gettempdir(), f"src_{uuid.uuid4().hex}{ext}")
        mp4_path = os.path.join(tempfile.gettempdir(), f"out_{uuid.uuid4().hex}.mp4")
        with open(src_path, "wb") as f:
            f.write(video_bytes)

        result = subprocess.run(
            ["ffmpeg", "-i", src_path, "-c:v", "libx264", "-preset", "fast",
             "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-y", mp4_path],
            capture_output=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"worker-krea2-edit - ffmpeg error: {result.stderr.decode()}")
            return video_bytes, filename

        with open(mp4_path, "rb") as f:
            mp4_bytes = f.read()

        new_filename = os.path.splitext(filename)[0] + ".mp4"
        print(f"worker-krea2-edit - Convertido {filename} -> {new_filename}")
        return mp4_bytes, new_filename
    except Exception as e:
        print(f"worker-krea2-edit - Conversão falhou: {e}")
        return video_bytes, filename
    finally:
        for p in [src_path, mp4_path]:
            if p and os.path.exists(p):
                os.remove(p)


def upload_binary_artifact(job_id, payload_bytes, filename, default_ext):
    """Upload de artefato binário e retorna URL pública."""
    tmp_path = None
    try:
        file_ext = os.path.splitext(filename)[1] or default_ext
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            tmp.write(payload_bytes)
            tmp_path = tmp.name
        return rp_upload.upload_image(job_id, tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _has_output_node(workflow):
    """Validação mínima: workflow precisa ter ao menos um output node conhecido."""
    if not isinstance(workflow, dict):
        return False

    output_nodes = {
        "VHS_VideoCombine",
        "SaveVideo",
        "SaveWEBM",
        "SaveImage",
        "PreviewImage",
    }
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") in output_nodes:
            return True
    return False


# ======================= Krea 2 Identity Edit (modo-prompt) =======================
# Grafo montado PROGRAMATICAMENTE (API format). Receita = workflow oficial do
# comfyui-krea2edit (v1.2) + 2º passe opcional "detailer" (Benji): er_sde, 4 steps,
# denoise baixo após upscale lanczos 1.5x. Sem SeedVR2.
KREA_UNET_NAME = os.environ.get("KREA_UNET_NAME", "krea2_turbo_fp8_scaled.safetensors")
KREA_TE_NAME = os.environ.get("KREA_TE_NAME", "qwen3vl_4b_fp8_scaled.safetensors")
KREA_VAE_NAME = os.environ.get("KREA_VAE_NAME", "qwen_image_vae.safetensors")
KREA_LORA_NAME = os.environ.get("KREA_LORA_NAME", "krea2_identity_edit_v1_2.safetensors")
KREA_MAX_AREA = int(os.environ.get("KREA_MAX_AREA", str(1024 * 1024)) or 1024 * 1024)

# aspect_ratio -> (W, H) ~1MP, múltiplos de 16 (sweet spot do LoRA; 2 pessoas ≤1.5MP).
_ASPECT_WH = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
}


def _url_to_data_uri(url):
    """Baixa uma URL http(s) e devolve data URI base64 (para input.images)."""
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    ct = resp.headers.get("Content-Type", "image/png").split(";")[0]
    return f"data:{ct};base64," + base64.b64encode(resp.content).decode()


def _image_size(data_uri):
    if Image is None:
        return None
    try:
        b64 = data_uri.split(",", 1)[1] if data_uri.startswith("data:") else data_uri
        return Image.open(BytesIO(base64.b64decode(b64))).size
    except Exception:
        return None


def _fit_area_dims(w, h, area):
    """(w,h) com o mesmo aspecto, ~area px, múltiplos de 16."""
    import math
    s = math.sqrt(area / float(w * h))
    return max(16, int(round(w * s / 16)) * 16), max(16, int(round(h * s / 16)) * 16)


def _out_dims(job_input, images):
    """Resolução de saída: aspect_ratio explícito > width/height > aspecto da imagem A > 1:1."""
    ar = (job_input.get("aspect_ratio") or "").strip()
    if ar in _ASPECT_WH:
        return _ASPECT_WH[ar]
    w, h = job_input.get("width"), job_input.get("height")
    if w and h:
        return _fit_area_dims(int(w), int(h), min(int(w) * int(h), KREA_MAX_AREA))
    if images:
        sz = _image_size(images[0]["image"])
        if sz:
            return _fit_area_dims(sz[0], sz[1], KREA_MAX_AREA)
    return _ASPECT_WH["1:1"]


def build_workflow_from_prompt(job_input):
    """Modo-prompt: {prompt, images:[A(,B)], ...} -> grafo Krea2 Identity Edit.

    A = imagem base/cena (obrigatória; também define o aspecto se não informado).
    B = pessoa/sujeito (opcional; ordem FIXA do LoRA: cena 1ª, pessoa 2ª).
    Params: seed, steps(10), cfg(1), sampler(euler), scheduler(simple), ref_boost(4),
    ref_boost_a(1), grounding_px(768), lora_strength(1), aspect_ratio|width/height,
    refine(bool, default False), refine_steps(4), refine_denoise(0.2), refine_scale(1.5),
    negative_prompt (só importa com cfg>1).
    Retorna (workflow, images) — images = [{name, image}] p/ upload.
    """
    prompt = (job_input.get("prompt") or "").strip()
    images = list(job_input.get("images") or [])
    urls = job_input.get("image_urls") or job_input.get("reference_images") or []
    if isinstance(urls, str):
        urls = [urls]
    for i, u in enumerate(urls):
        if u:
            images.append({"name": f"ref_{i}.png", "image": _url_to_data_uri(u)})
    if not images:
        raise ValueError("Krea2 Identity Edit exige ao menos 1 imagem (A = base/cena)")
    if not prompt:
        raise ValueError("prompt (instrução de edição) obrigatório")
    images = images[:2]
    for i, im in enumerate(images):
        im["name"] = im.get("name") or f"ref_{i}.png"

    def _f(k, d):
        v = job_input.get(k)
        return float(v) if v not in (None, "") else d

    def _i(k, d):
        v = job_input.get(k)
        return int(v) if v not in (None, "") else d

    # Seed aleatória por padrão: re-rodar o nó gera variação (seed fixa = mesma
    # imagem + cache do ComfyUI). `seed` explícito continua determinístico.
    import random as _random
    seed = _i("seed", _random.randint(0, 2**31 - 1))
    steps = _i("steps", 10)
    cfg = _f("cfg", 1.0)
    sampler = job_input.get("sampler") or "euler"
    scheduler = job_input.get("scheduler") or "simple"
    ref_boost = _f("ref_boost", 4.0)
    ref_boost_a = _f("ref_boost_a", 1.0)
    grounding_px = _i("grounding_px", 768)
    lora_strength = _f("lora_strength", 1.0)
    W, H = _out_dims(job_input, images)
    two = len(images) == 2

    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": KREA_UNET_NAME, "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": KREA_LORA_NAME, "strength_model": lora_strength}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": KREA_TE_NAME, "type": "krea2", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": KREA_VAE_NAME}},
        "10": {"class_type": "LoadImage", "inputs": {"image": images[0]["name"], "upload": "image"}},
        "12": {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["4", 0]}},
        "30": {"class_type": "EmptySD3LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "20": {"class_type": "Krea2EditGroundedEncode",
               "inputs": {"clip": ["3", 0], "prompt": prompt, "image": ["10", 0],
                          "grounding_px": grounding_px, "system_prompt": ""}},
        # negativo grounded (recipe oficial); irrelevante com cfg 1, importa com raw/cfg>1
        "21": {"class_type": "Krea2EditGroundedEncode",
               "inputs": {"clip": ["3", 0], "prompt": (job_input.get("negative_prompt") or ""),
                          "image": ["10", 0], "grounding_px": grounding_px, "system_prompt": ""}},
        "22": {"class_type": "Krea2EditModelPatch",
               "inputs": {"model": ["2", 0], "source_latent": ["12", 0], "ref_boost": ref_boost,
                          "ref_boost_a": ref_boost_a, "fit_mode": "fit", "vae": ["4", 0],
                          "source_image": ["10", 0], "target_latent": ["30", 0]}},
        "40": {"class_type": "KSampler",
               "inputs": {"model": ["22", 0], "seed": seed, "steps": steps, "cfg": cfg,
                          "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0,
                          "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["30", 0]}},
        "41": {"class_type": "VAEDecode", "inputs": {"samples": ["40", 0], "vae": ["4", 0]}},
    }
    if two:
        wf["11"] = {"class_type": "LoadImage", "inputs": {"image": images[1]["name"], "upload": "image"}}
        wf["13"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["4", 0]}}
        wf["20"]["inputs"]["image_b"] = ["11", 0]
        wf["21"]["inputs"]["image_b"] = ["11", 0]
        wf["22"]["inputs"]["source_latent_b"] = ["13", 0]
        wf["22"]["inputs"]["source_image_b"] = ["11", 0]

    last = "41"
    if str(job_input.get("refine", "")).lower() in ("1", "true", "yes"):
        # 2º passe (detailer): upscale lanczos + er_sde poucos steps, denoise baixo.
        wf["50"] = {"class_type": "ImageScaleBy",
                    "inputs": {"image": ["41", 0], "upscale_method": "lanczos",
                               "scale_by": _f("refine_scale", 1.5)}}
        wf["51"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["50", 0], "vae": ["4", 0]}}
        wf["52"] = {"class_type": "KSampler",
                    "inputs": {"model": ["22", 0], "seed": seed + 1, "steps": _i("refine_steps", 4),
                               "cfg": 1.0, "sampler_name": "er_sde", "scheduler": "simple",
                               "denoise": _f("refine_denoise", 0.2),
                               "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["51", 0]}}
        wf["53"] = {"class_type": "VAEDecode", "inputs": {"samples": ["52", 0], "vae": ["4", 0]}}
        last = "53"
    wf["90"] = {"class_type": "SaveImage", "inputs": {"images": [last, 0], "filename_prefix": "krea2_edit"}}
    print(f"worker-krea2-edit - krea2 graph: {W}x{H} imgs={len(images)} steps={steps} "
          f"ref_boost={ref_boost} refine={last=='53'}")
    return wf, images
# ================================================================================


def validate_input(job_input):
    """Valida input do job."""
    if job_input is None:
        return None, "Please provide input"

    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    workflow = job_input.get("workflow")
    # Modo-prompt: sem `workflow` mas com `prompt` (ou imagens) → constrói o workflow Krea2 Identity Edit.
    if workflow is None and (job_input.get("prompt") or job_input.get("images") or job_input.get("image_urls") or job_input.get("reference_images")):
        try:
            workflow, built_images = build_workflow_from_prompt(job_input)
            if job_input.get("images") is None:
                job_input["images"] = built_images
            else:
                job_input["images"] = built_images
        except Exception as e:
            return None, f"Falha ao montar workflow do prompt: {e}"

    if workflow is None:
        return None, "Missing 'workflow' parameter"
    if not _has_output_node(workflow):
        return None, "Workflow sem output node reconhecido (ex: VHS_VideoCombine/SaveVideo)"

    images = job_input.get("images")
    if images is not None:
        if not isinstance(images, list) or not all(
            "name" in image and "image" in image for image in images
        ):
            return None, "'images' must be a list of objects with 'name' and 'image' keys"

    comfy_org_api_key = job_input.get("comfy_org_api_key")
    return {"workflow": workflow, "images": images, "comfy_org_api_key": comfy_org_api_key}, None


def check_server(url, retries=500, delay=50):
    """Verifica se o servidor ComfyUI está acessível."""
    print(f"worker-krea2-edit - Verificando API em {url}...")
    for i in range(retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"worker-krea2-edit - API acessível")
                return True
        except requests.Timeout:
            pass
        except requests.RequestException:
            pass
        time.sleep(delay / 1000)

    print(f"worker-krea2-edit - Falha ao conectar em {url} após {retries} tentativas.")
    return False


def upload_images(images):
    """Grava imagens decodificadas em /comfyui/input e usa upload HTTP apenas como fallback."""
    if not images:
        return {"status": "success", "message": "No images to upload", "details": []}

    responses = []
    upload_errors = []
    print(f"worker-krea2-edit - Uploading {len(images)} imagem(ns)...")

    for image in images:
        blob = None
        try:
            name = image["name"]
            image_data_uri = image["image"]
            if "," in image_data_uri:
                base64_data = image_data_uri.split(",", 1)[1].strip()
            else:
                base64_data = image_data_uri.strip()

            blob = base64.b64decode(base64_data)
            print(
                "worker-krea2-edit - Decoded image summary:",
                json.dumps({"name": name, **_summarize_bytes(blob)}, ensure_ascii=False),
            )

            input_dir = "/comfyui/input"
            os.makedirs(input_dir, exist_ok=True)
            dst_path = os.path.join(input_dir, name)

            if Image is None:
                with open(dst_path, "wb") as fh:
                    fh.write(blob)
            else:
                with Image.open(BytesIO(blob)) as img:
                    img.load()
                    normalized = img
                    if img.mode not in ("RGB", "RGBA"):
                        normalized = img.convert("RGBA" if "A" in img.getbands() else "RGB")
                    with BytesIO() as buf:
                        normalized.save(buf, format="PNG")
                        normalized_blob = buf.getvalue()
                with open(dst_path, "wb") as fh:
                    fh.write(normalized_blob)

            probe = _probe_image_path(dst_path)
            print(
                "worker-krea2-edit - Saved image probe:",
                json.dumps({"name": name, "probe": probe}, ensure_ascii=False),
            )
            if probe.get("pillow_error"):
                raise ValueError(f"imagem salva inválida: {probe['pillow_error']}")

            file_size = os.path.getsize(dst_path)
            responses.append(f"Saved OK: {name} ({file_size} bytes)")
        except Exception as e:
            # Loga o erro ANTES de tentar o fallback
            print(f"worker-krea2-edit - Primary save FAILED ({type(e).__name__}): {e}")

            # Fallback para endpoint /upload/image
            if blob is None:
                upload_errors.append(f"base64 decode falhou para {image.get('name', '?')}: {e}")
                continue
            try:
                detected_format = _detect_image_format(blob)
                if detected_format == "jpeg":
                    fallback_mime = "image/jpeg"
                elif detected_format == "webp":
                    fallback_mime = "image/webp"
                elif detected_format == "gif":
                    fallback_mime = "image/gif"
                else:
                    fallback_mime = "image/png"

                files = {
                    "image": (image.get("name", "input.png"), BytesIO(blob), fallback_mime),
                    "overwrite": (None, "true"),
                }
                response = requests.post(f"http://{COMFY_HOST}/upload/image", files=files, timeout=30)
                response.raise_for_status()
                fallback_resp = response.json() if response.content else {}

                # Verifica se o arquivo realmente ficou acessível após o upload
                dst_path = os.path.join("/comfyui/input", image.get("name", "input.png"))
                if os.path.exists(dst_path):
                    probe = _probe_image_path(dst_path)
                    print(
                        "worker-krea2-edit - Fallback image probe:",
                        json.dumps({"name": image.get("name"), "probe": probe, "response": fallback_resp}, ensure_ascii=False),
                    )
                else:
                    print(f"worker-krea2-edit - AVISO: fallback upload OK mas arquivo nao encontrado em {dst_path}. Resp: {fallback_resp}")

                responses.append(f"Upload OK (fallback): {image.get('name', 'unknown')}")
                print(f"worker-krea2-edit - Upload OK (fallback): {image.get('name', 'unknown')}")
            except Exception as fallback_error:
                error_msg = (
                    f"Erro no upload de {image.get('name', 'unknown')}: primary={e}; "
                    f"fallback={fallback_error}"
                )
                print(f"worker-krea2-edit - {error_msg}")
                upload_errors.append(error_msg)

    if upload_errors:
        return {"status": "error", "message": "Algumas imagens falharam", "details": upload_errors}
    return {"status": "success", "message": "Todas as imagens enviadas", "details": responses}


def queue_workflow(workflow, client_id, comfy_org_api_key=None):
    """Enfileira workflow no ComfyUI."""
    payload = {"prompt": workflow, "client_id": client_id}

    key_from_env = os.environ.get("COMFY_ORG_API_KEY")
    effective_key = comfy_org_api_key if comfy_org_api_key else key_from_env
    if effective_key:
        payload["extra_data"] = {"api_key_comfy_org": effective_key}

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"http://{COMFY_HOST}/prompt", data=data, headers=headers, timeout=30)

    if response.status_code == 400:
        print(f"worker-krea2-edit - ComfyUI 400: {response.text}")
        try:
            error_data = response.json()
            error_message = "Workflow validation failed"
            error_details = []

            if "error" in error_data:
                error_info = error_data["error"]
                if isinstance(error_info, dict):
                    error_message = error_info.get("message", error_message)
                else:
                    error_message = str(error_info)

            if "node_errors" in error_data:
                for node_id, node_error in error_data["node_errors"].items():
                    if isinstance(node_error, dict):
                        for error_type, error_msg in node_error.items():
                            error_details.append(f"Node {node_id} ({error_type}): {error_msg}")
                    else:
                        error_details.append(f"Node {node_id}: {node_error}")

            if error_details:
                raise ValueError(f"{error_message}:\n" + "\n".join(f"  {d}" for d in error_details))
            else:
                raise ValueError(f"{error_message}. Raw: {response.text}")
        except (json.JSONDecodeError, KeyError):
            raise ValueError(f"ComfyUI validation failed: {response.text}")

    response.raise_for_status()
    return response.json()


def get_history(prompt_id):
    """Recupera histórico do prompt."""
    response = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def get_file_data(filename, subfolder, file_type):
    """Busca dados de arquivo do endpoint /view do ComfyUI."""
    data = {"filename": filename, "subfolder": subfolder, "type": file_type}
    url_values = urllib.parse.urlencode(data)
    try:
        response = requests.get(f"http://{COMFY_HOST}/view?{url_values}", timeout=120)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"worker-krea2-edit - Erro ao buscar {filename}: {e}")
        return None


def handler(job):
    """Handler principal: edição de imagem por referência (Krea 2 Identity Edit)."""
    if is_network_volume_debug_enabled():
        run_network_volume_diagnostics()

    job_input = job["input"]
    job_id = job["id"]

    validated_data, error_message = validate_input(job_input)
    if error_message:
        diagnostics = _build_runtime_diagnostics(error_message)
        if diagnostics and diagnostics.get("category") in {"GPU_OOM", "GPU_RUNTIME"}:
            print("worker-krea2-edit - Runtime diagnostics:", json.dumps(diagnostics))
        return {"error": error_message, "diagnostics": diagnostics}

    workflow = validated_data["workflow"]
    input_images = validated_data.get("images")
    _log_job_diagnostics(job_id, job_input, workflow, input_images)

    if not check_server(
        f"http://{COMFY_HOST}/",
        COMFY_API_AVAILABLE_MAX_RETRIES,
        COMFY_API_AVAILABLE_INTERVAL_MS,
    ):
        message = f"ComfyUI ({COMFY_HOST}) inacessível após múltiplas tentativas."
        startup_tail = _read_comfy_log_tail()
        diagnostics = _build_runtime_diagnostics([message] + startup_tail)
        if startup_tail:
            print("worker-krea2-edit - Comfy startup log tail:\n" + "\n".join(startup_tail[-40:]))
        return {
            "error": message,
            "details": {
                "comfy_startup_log_tail": startup_tail[-120:],
                "wait_ms": COMFY_API_AVAILABLE_MAX_RETRIES * COMFY_API_AVAILABLE_INTERVAL_MS,
            },
            "diagnostics": diagnostics,
        }

    if input_images:
        upload_result = upload_images(input_images)
        if upload_result["status"] == "error":
            diagnostics = _build_runtime_diagnostics(upload_result.get("details"))
            return {
                "error": "Falha no upload de imagens",
                "details": upload_result["details"],
                "diagnostics": diagnostics,
            }
        preflight_checks = _preflight_loadimage_inputs(workflow)
        if preflight_checks:
            print(
                "worker-krea2-edit - LoadImage preflight:",
                json.dumps(preflight_checks, ensure_ascii=False),
            )
            invalid_checks = [
                check for check in preflight_checks
                if not check.get("probe", {}).get("exists")
                or check.get("probe", {}).get("pillow_error")
            ]
            if invalid_checks:
                diagnostics = _build_runtime_diagnostics(json.dumps(invalid_checks, ensure_ascii=False))
                return {
                    "error": "Imagem de entrada inválida antes do queue_workflow",
                    "details": invalid_checks,
                    "diagnostics": diagnostics,
                }

    gemma_preflight = _preflight_gemma_loader(workflow)
    if gemma_preflight:
        print(
            "worker-krea2-edit - Gemma loader preflight:",
            json.dumps(gemma_preflight, ensure_ascii=False),
        )

    ws = None
    client_id = str(uuid.uuid4())
    prompt_id = None
    output_data = []
    video_data = []
    audio_data = []
    errors = []
    history_output_summary = []
    workflow_nodes = _build_workflow_node_lookup(workflow)

    try:
        ws_url = f"ws://{COMFY_HOST}/ws?clientId={client_id}"
        print(f"worker-krea2-edit - Conectando ao websocket: {ws_url}")
        ws = websocket.WebSocket()
        ws.connect(ws_url, timeout=10)
        print(f"worker-krea2-edit - Websocket conectado")

        try:
            queued_workflow = queue_workflow(
                workflow, client_id,
                comfy_org_api_key=validated_data.get("comfy_org_api_key"),
            )
            prompt_id = queued_workflow.get("prompt_id")
            if not prompt_id:
                raise ValueError(f"prompt_id ausente na resposta: {queued_workflow}")
            print(f"worker-krea2-edit - Workflow enfileirado: {prompt_id}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise ValueError(f"Erro ao enfileirar workflow: {e}")

        print(f"worker-krea2-edit - Aguardando execução ({prompt_id})...")
        execution_done = False
        last_event_at = time.time()
        last_queue_remaining = None
        current_node_id = None
        while True:
            try:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    message_type = message.get("type")
                    data = message.get("data", {})
                    if message_type == "status":
                        status_data = message.get("data", {}).get("status", {})
                        queue_remaining = status_data.get("exec_info", {}).get("queue_remaining", "N/A")
                        if queue_remaining != last_queue_remaining:
                            print(f"worker-krea2-edit - Queue: {queue_remaining} restantes")
                            last_queue_remaining = queue_remaining
                        last_event_at = time.time()
                    elif message_type == "execution_start":
                        if data.get("prompt_id") == prompt_id:
                            _log_ws_event("Execution start", {"prompt_id": prompt_id})
                            last_event_at = time.time()
                    elif message_type == "execution_cached":
                        if data.get("prompt_id") == prompt_id:
                            _log_ws_event(
                                "Execution cached",
                                {
                                    "prompt_id": prompt_id,
                                    "nodes": data.get("nodes", []),
                                },
                            )
                            last_event_at = time.time()
                    elif message_type == "executing":
                        if data.get("node") is None and data.get("prompt_id") == prompt_id:
                            print(f"worker-krea2-edit - Execução finalizada: {prompt_id}")
                            execution_done = True
                            break
                        if data.get("prompt_id") == prompt_id:
                            current_node_id = str(data.get("node")) if data.get("node") is not None else None
                            current_node_class = workflow_nodes.get(current_node_id, "unknown") if current_node_id else None
                            _log_ws_event(
                                "Executing node",
                                {
                                    "prompt_id": prompt_id,
                                    "node_id": current_node_id,
                                    "class_type": current_node_class,
                                    "idle_timeout_s": _get_idle_timeout_for_node_class(current_node_class),
                                },
                            )
                            last_event_at = time.time()
                    elif message_type == "progress":
                        value = data.get("value")
                        max_value = data.get("max")
                        progress_payload = {
                            "prompt_id": prompt_id,
                            "node_id": current_node_id,
                            "class_type": workflow_nodes.get(current_node_id, "unknown") if current_node_id else None,
                            "value": value,
                            "max": max_value,
                        }
                        _log_ws_event("Progress", progress_payload)
                        last_event_at = time.time()
                    elif message_type == "executed":
                        if data.get("prompt_id") == prompt_id:
                            node_id = str(data.get("node")) if data.get("node") is not None else None
                            output = data.get("output")
                            _log_ws_event(
                                "Executed node",
                                {
                                    "prompt_id": prompt_id,
                                    "node_id": node_id,
                                    "class_type": workflow_nodes.get(node_id, "unknown") if node_id else None,
                                    "output_keys": _safe_dict_keys(output),
                                },
                            )
                            last_event_at = time.time()
                    elif message_type == "execution_success":
                        if data.get("prompt_id") == prompt_id:
                            _log_ws_event("Execution success", {"prompt_id": prompt_id})
                            last_event_at = time.time()
                    elif message_type == "execution_interrupted":
                        if data.get("prompt_id") == prompt_id:
                            errors.append("Execution interrupted pelo ComfyUI")
                            _log_ws_event("Execution interrupted", {"prompt_id": prompt_id})
                            break
                    elif message_type == "execution_error":
                        if data.get("prompt_id") == prompt_id:
                            error_details = f"Node: {data.get('node_type')}, ID: {data.get('node_id')}, Msg: {data.get('exception_message')}"
                            print(f"worker-krea2-edit - Erro de execução: {error_details}")
                            errors.append(f"Execution error: {error_details}")
                            break
                    else:
                        if data.get("prompt_id") == prompt_id:
                            _log_ws_event("WS event", {"type": message_type, "data_keys": _safe_dict_keys(data)})
                            last_event_at = time.time()
                else:
                    continue
            except websocket.WebSocketTimeoutException:
                idle_for = int(time.time() - last_event_at)
                current_node_class = workflow_nodes.get(current_node_id, "unknown") if current_node_id else None
                idle_limit = _get_idle_timeout_for_node_class(current_node_class)
                if idle_for >= idle_limit:
                    current_node_class = workflow_nodes.get(current_node_id, "unknown") if current_node_id else None
                    raise ValueError(
                        f"Sem eventos do websocket por {idle_for}s após enqueue do prompt {prompt_id}. "
                        f"Último node conhecido: {current_node_id or 'none'} ({current_node_class or 'unknown'}). "
                        f"Limite aplicado: {idle_limit}s."
                    )
                continue
            except websocket.WebSocketConnectionClosedException as closed_err:
                try:
                    ws = _attempt_websocket_reconnect(
                        ws_url, WEBSOCKET_RECONNECT_ATTEMPTS,
                        WEBSOCKET_RECONNECT_DELAY_S, closed_err,
                    )
                    continue
                except websocket.WebSocketConnectionClosedException as reconn_err:
                    raise reconn_err
            except json.JSONDecodeError:
                print(f"worker-krea2-edit - JSON inválido no websocket.")

        if not execution_done and not errors:
            raise ValueError("Loop de monitoramento terminou sem confirmação.")

        print(f"worker-krea2-edit - Buscando histórico de {prompt_id}...")
        history = get_history(prompt_id)

        if prompt_id not in history:
            error_msg = f"Prompt {prompt_id} não encontrado no histórico."
            if not errors:
                diagnostics = _build_runtime_diagnostics(error_msg)
                return {"error": error_msg, "diagnostics": diagnostics}
            errors.append(error_msg)
            diagnostics = _build_runtime_diagnostics(errors)
            return {"error": "Job falhou", "details": errors, "diagnostics": diagnostics}

        outputs = history.get(prompt_id, {}).get("outputs", {})
        print(f"worker-krea2-edit - Processando {len(outputs)} nós de saída...")

        for node_id, node_output in outputs.items():
            if not isinstance(node_output, dict):
                history_output_summary.append(
                    {"node_id": str(node_id), "keys": [], "warning": "node_output not dict"}
                )
                continue

            if isinstance(node_output, dict):
                history_output_summary.append(
                    {"node_id": str(node_id), "keys": sorted(list(node_output.keys()))}
                )

            def normalize_output_entries(raw_value, label):
                entries = []
                if isinstance(raw_value, list):
                    for idx, item in enumerate(raw_value):
                        if isinstance(item, dict):
                            entries.append(item)
                        else:
                            print(
                                "worker-krea2-edit - Ignorando item de saída não estruturado:",
                                json.dumps(
                                    {
                                        "label": label,
                                        "index": idx,
                                        "python_type": type(item).__name__,
                                        "value": item,
                                    },
                                    ensure_ascii=False,
                                ),
                            )
                elif isinstance(raw_value, dict):
                    entries.append(raw_value)
                elif raw_value is not None:
                    print(
                        "worker-krea2-edit - Ignorando saída não estruturada:",
                        json.dumps(
                            {
                                "label": label,
                                "python_type": type(raw_value).__name__,
                                "value": raw_value,
                            },
                            ensure_ascii=False,
                        ),
                    )
                return entries

            # Imagens
            if "images" in node_output:
                for image_info in normalize_output_entries(node_output["images"], "images"):
                    filename = image_info.get("filename")
                    subfolder = image_info.get("subfolder", "")
                    img_type = image_info.get("type")

                    if img_type == "temp":
                        continue
                    if not filename:
                        continue

                    image_bytes = get_file_data(filename, subfolder, img_type)
                    if image_bytes:
                        if os.environ.get("BUCKET_ENDPOINT_URL"):
                            try:
                                file_ext = os.path.splitext(filename)[1] or ".png"
                                with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                                    tmp.write(image_bytes)
                                    tmp_path = tmp.name
                                s3_url = rp_upload.upload_image(job_id, tmp_path)
                                os.remove(tmp_path)
                                output_data.append({"filename": filename, "type": "s3_url", "data": s3_url})
                            except Exception as e:
                                errors.append(f"Erro S3 upload {filename}: {e}")
                        else:
                            b64 = base64.b64encode(image_bytes).decode("utf-8")
                            output_data.append({"filename": filename, "type": "base64", "data": b64})
                    else:
                        errors.append(f"Falha ao ler imagem {filename} via /view")

            # Vídeos (VHS usa "gifs"; SaveVideo pode expor "videos" ou "animated")
            video_entries = []
            if any(key in node_output for key in ("gifs", "videos", "animated", "video")):
                print(
                    "worker-krea2-edit - Video output raw summary:",
                    json.dumps(
                        {
                            "node_id": str(node_id),
                            "gifs_type": type(node_output.get("gifs")).__name__ if "gifs" in node_output else None,
                            "videos_type": type(node_output.get("videos")).__name__ if "videos" in node_output else None,
                            "animated_type": type(node_output.get("animated")).__name__ if "animated" in node_output else None,
                            "video_type": type(node_output.get("video")).__name__ if "video" in node_output else None,
                            "animated_preview": node_output.get("animated"),
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                )
            video_entries.extend(normalize_output_entries(node_output.get("gifs"), "gifs"))
            video_entries.extend(normalize_output_entries(node_output.get("videos"), "videos"))
            video_entries.extend(normalize_output_entries(node_output.get("animated"), "animated"))
            video_entries.extend(normalize_output_entries(node_output.get("video"), "video"))

            if video_entries:
                for vid_info in video_entries:
                    filename = vid_info.get("filename")
                    subfolder = vid_info.get("subfolder", "")
                    vid_type = vid_info.get("type", "output")

                    if not filename:
                        continue

                    vid_bytes = get_file_data(filename, subfolder, vid_type)
                    if vid_bytes:
                        vid_bytes, filename = convert_video_to_mp4(vid_bytes, filename)
                        try:
                            uploaded_url = upload_binary_artifact(job_id, vid_bytes, filename, ".mp4")
                            print(
                                "worker-krea2-edit - Video upload ok:",
                                json.dumps(
                                    {
                                        "filename": filename,
                                        "bytes": len(vid_bytes),
                                        "url": uploaded_url,
                                    },
                                    ensure_ascii=False,
                                ),
                            )
                            video_data.append({"filename": filename, "type": "s3_url", "data": uploaded_url})
                        except Exception as e:
                            print(f"worker-krea2-edit - Upload remoto de vídeo falhou ({filename}): {e}. Fallback base64.")
                            if len(vid_bytes) > MAX_INLINE_VIDEO_BYTES:
                                errors.append(
                                    f"Vídeo {filename} excede limite inline ({len(vid_bytes)} bytes > {MAX_INLINE_VIDEO_BYTES}) "
                                    "e upload remoto falhou"
                                )
                            else:
                                b64 = base64.b64encode(vid_bytes).decode("utf-8")
                                video_data.append({"filename": filename, "type": "base64", "data": b64})
                    else:
                        errors.append(f"Falha ao ler vídeo {filename} via /view")

            # Áudio (LTX-2 pode gerar áudio sincronizado)
            if "audio" in node_output:
                for audio_info in normalize_output_entries(node_output["audio"], "audio"):
                    filename = audio_info.get("filename")
                    subfolder = audio_info.get("subfolder", "")
                    audio_type = audio_info.get("type", "temp")

                    if not filename:
                        continue

                    audio_bytes = get_file_data(filename, subfolder, audio_type)
                    if audio_bytes:
                        b64 = base64.b64encode(audio_bytes).decode("utf-8")
                        audio_data.append({"filename": filename, "type": "base64", "data": b64})
                    else:
                        errors.append(f"Falha ao ler áudio {filename} via /view")

    except websocket.WebSocketException as e:
        print(f"worker-krea2-edit - WebSocket Error: {e}")
        print(traceback.format_exc())
        diagnostics = _build_runtime_diagnostics(str(e))
        if diagnostics and diagnostics.get("category") in {"GPU_OOM", "GPU_RUNTIME"}:
            print("worker-krea2-edit - Runtime diagnostics:", json.dumps(diagnostics))
        return {"error": f"WebSocket error: {e}", "diagnostics": diagnostics}
    except requests.RequestException as e:
        print(f"worker-krea2-edit - HTTP Error: {e}")
        print(traceback.format_exc())
        diagnostics = _build_runtime_diagnostics(str(e))
        if diagnostics and diagnostics.get("category") in {"GPU_OOM", "GPU_RUNTIME"}:
            print("worker-krea2-edit - Runtime diagnostics:", json.dumps(diagnostics))
        return {"error": f"HTTP error: {e}", "diagnostics": diagnostics}
    except ValueError as e:
        print(f"worker-krea2-edit - Value Error: {e}")
        print(traceback.format_exc())
        diagnostics = _build_runtime_diagnostics(str(e))
        if diagnostics and diagnostics.get("category") in {"GPU_OOM", "GPU_RUNTIME"}:
            print("worker-krea2-edit - Runtime diagnostics:", json.dumps(diagnostics))
        return {"error": str(e), "diagnostics": diagnostics}
    except Exception as e:
        print(f"worker-krea2-edit - Unexpected Error: {e}")
        print(traceback.format_exc())
        diagnostics = _build_runtime_diagnostics(str(e))
        if diagnostics and diagnostics.get("category") in {"GPU_OOM", "GPU_RUNTIME"}:
            print("worker-krea2-edit - Runtime diagnostics:", json.dumps(diagnostics))
        return {"error": f"Erro inesperado: {e}", "diagnostics": diagnostics}
    finally:
        if ws and ws.connected:
            ws.close()

    final_result = {}

    if output_data:
        final_result["images"] = output_data

    if video_data:
        final_result["video"] = video_data[0]["data"]
        final_result["video_filename"] = video_data[0]["filename"]
        print(
            "worker-krea2-edit - Final video selected:",
            json.dumps(
                {
                    "filename": video_data[0]["filename"],
                    "type": video_data[0]["type"],
                    "url": video_data[0]["data"] if video_data[0]["type"] == "s3_url" else None,
                },
                ensure_ascii=False,
            ),
        )
        if len(video_data) > 1:
            final_result["videos"] = video_data

    if audio_data:
        final_result["audio"] = audio_data[0]["data"]
        final_result["audio_filename"] = audio_data[0]["filename"]

    if errors:
        final_result["errors"] = errors
        diagnostics = _build_runtime_diagnostics(errors)
        if diagnostics:
            final_result["diagnostics"] = diagnostics
            if diagnostics.get("category") in {"GPU_OOM", "GPU_RUNTIME"}:
                print("worker-krea2-edit - Runtime diagnostics:", json.dumps(diagnostics))

    has_output = output_data or video_data or audio_data
    if not has_output and errors:
        diagnostics = _build_runtime_diagnostics(errors)
        if diagnostics and diagnostics.get("category") in {"GPU_OOM", "GPU_RUNTIME"}:
            print("worker-krea2-edit - Runtime diagnostics:", json.dumps(diagnostics))
        return {"error": "Job falhou sem output", "details": errors, "diagnostics": diagnostics}
    elif not has_output and not errors:
        diagnostics = _build_runtime_diagnostics(history_output_summary)
        return {
            "error": "Job concluído sem mídia no histórico do ComfyUI",
            "details": history_output_summary,
            "diagnostics": diagnostics,
        }

    print(f"worker-krea2-edit - Job concluído: {len(output_data)} img, {len(video_data)} vid, {len(audio_data)} audio")
    return final_result


if __name__ == "__main__":
    print("worker-krea2-edit - Starting handler...")
    runpod.serverless.start({"handler": handler})
