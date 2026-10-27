"""Clip generation (MiniMax-H3 I2V via ComfyUI: approved image = first frame) + final stitch.
Speed: 4-step Turbo LoRA + sage attention, int8 text encoder/VAE when downloaded, content-hash clip cache (unchanged scenes are never
re-rendered), draft 544p mode, and a weight pre-read into the OS page cache so ComfyUI's cold load does not wait on truenas."""
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "product_video"))
from generate import COMFY, URL, _call, ensure_server  # noqa: E402  (shared ComfyUI launcher: sage attention, port from env)

CLIENT_ID = uuid.uuid4().hex
PROGRESS = {"phase": "", "value": 0, "max": 0}  # live view of the running ComfyUI job (fed by its websocket)
PHASES = {"unet": "loading video model", "lora": "loading video model", "clip": "loading text encoder", "vae": "loading VAE",
          "avae": "loading VAE", "img": "loading first frame", "cond": "loading text encoder + encoding prompt (slow on the first run)", "sample": "sampling",
          "dec": "decoding video", "adec": "decoding audio", "video": "muxing", "save": "saving clip"}


def progress_listener():
    """Background thread: follow ComfyUI's websocket for node + step progress of our prompts."""
    from websockets.sync.client import connect

    def run():
        while True:
            try:
                with connect(URL.replace("http", "ws") + f"/ws?clientId={CLIENT_ID}", max_size=None, open_timeout=5) as ws:
                    for raw in ws:
                        if not isinstance(raw, str):
                            continue  # binary preview frames
                        m = json.loads(raw)
                        t, d = m.get("type"), m.get("data", {})
                        if t == "executing":
                            node = d.get("node")
                            PROGRESS.update(phase=PHASES.get(node, node or ""), value=0, max=0)
                        elif t == "progress":
                            PROGRESS.update(value=d.get("value", 0), max=d.get("max", 0))
                        elif t in ("execution_success", "execution_error"):
                            PROGRESS.update(phase="", value=0, max=0)
            except Exception:
                time.sleep(3)  # ComfyUI not up yet / restarted
    threading.Thread(target=run, daemon=True).start()

W = Path("/mnt/truenas/video_gen_models/ComfyUI-MiniMax-H3")
DIT = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
LORA_4 = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"  # trained shifts video 6 / audio 3
LORA_8 = "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
LORA = LORA_4

# Quality presets, measured on the L40S (warm, 5 s clip, new prompt each clip; see bench/results.jsonl):
#   final 768p, Comfy Kitchen INT8 attention   ~72 s  (closest to full-precision attention: 27.6 dB vs 27.4 dB for sage)
#   fast  768p, + sol-attn sparse attention    ~65 s  (slightly further from full precision: 25.4 dB)
#   draft 544p, Comfy Kitchen attention        ~34 s
#   hq    768p, 8-step Turbo LoRA, dense        ~137 s (most natural skin/hair, stable identity; only if the LoRA file exists)
#   (4-step v1.2 LoRA was tested too: softer and drifted in colour, so v1.0 stays the default)
PRESETS = {
    "final": dict(size="final", lora=LORA_4, steps=4, sparse=None),
    "fast":  dict(size="final", lora=LORA_4, steps=4, sparse=1.3),
    "draft": dict(size="draft", lora=LORA_4, steps=4, sparse=None),
    "hq":    dict(size="final", lora=LORA_8, steps=8, sparse=None),
}


def available_presets():
    return [q for q, p in PRESETS.items() if (W / "loras" / p["lora"]).exists()]


def _pick(folder, fast, slow):
    return fast if (W / folder / fast).exists() else slow  # int8 is half the bytes to load; fall back until downloaded


def text_encoder():
    return _pick("text_encoders", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "qwen3vl_32b_minimax_h3_bf16.safetensors")


def video_vae():
    return _pick("vae", "minimax_h3_video_vae_int8_convrot.safetensors", "minimax_h3_video_vae_fp16.safetensors")


SIZES = {  # (aspect, quality) -> (w, h); 768p is the model's native canvas
    ("16:9", "final"): (1344, 768), ("16:9", "draft"): (960, 544),
    ("9:16", "final"): (768, 1344), ("9:16", "draft"): (544, 960),
}


def prefetch():
    """Read the video weights once so they sit in the page cache (background thread at studio start)."""
    def run():
        for f in [W / "text_encoders" / text_encoder(), W / "diffusion_models" / DIT, W / "vae" / video_vae(), W / "loras" / LORA_4]:
            try:
                with open(f, "rb") as fh:
                    while fh.read(64 << 20):
                        pass
            except OSError:
                pass
    threading.Thread(target=run, daemon=True).start()


def comfy_free():
    try:
        _call("/free", {"unload_models": True, "free_memory": True})
    except Exception:
        pass  # server not up yet: nothing to free


def clip_key(image_path, prompt, quality, aspect, seed):
    h = hashlib.sha1(Path(image_path).read_bytes())
    # every setting that changes the pixels is in the key, so clips made with older settings are not reused
    h.update(json.dumps([prompt, quality, aspect, seed, DIT, PRESETS[quality], text_encoder(), video_vae(), "ck-attention"]).encode())
    return h.hexdigest()[:20]


def _graph(prompt, image, w, h, seed, prefix, preset):
    model = ["attn", 0]
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": DIT, "weight_dtype": "default"}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0], "lora_name": preset["lora"], "strength_model": 1.0}},
        "shift": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["lora", 0], "shift_video": 6.0, "shift_audio": 3.0}},
        # Comfy Kitchen INT8 attention: faster than sage on the L40S and closer to full-precision output
        "attn": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["shift", 0], "attention": "comfy kitchen attention"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": text_encoder(), "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": video_vae()}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "img": {"class_type": "LoadImage", "inputs": {"image": image}},
        "cond": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt, "width": w, "height": h,
                                                                  "length": 124, "first_frame": ["img", 0]}},
        "guider": {"class_type": "BasicGuider", "inputs": {"model": "MODEL", "conditioning": ["cond", 0]}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "sigmas": {"class_type": "BasicScheduler", "inputs": {"model": "MODEL", "scheduler": "simple", "steps": preset["steps"], "denoise": 1.0}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                                                                    "sigmas": ["sigmas", 0], "latent_image": ["cond", 1]}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "adec": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]}},
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["dec", 0], "fps": 24, "audio": ["adec", 0]}},
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"}},
    }
    if preset["sparse"]:  # sol-attn block-sparse attention (compiled kernel available on the L40S)
        g["sparse"] = {"class_type": "BlockSparseAttention", "inputs": {"model": model, "selection": "sol-attn", "selection.tau": preset["sparse"],
                       "start_percent": 0.2, "end_percent": 1.0, "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256,
                       "sink_conditioning": "exact_kv_and_rows", "verbose": False}}
        model = ["sparse", 0]
    g["guider"]["inputs"]["model"] = model
    g["sigmas"]["inputs"]["model"] = model
    return g


def make_clip(image_path, prompt, quality, aspect, seed, out_path, retries=3):
    """Render one 5 s clip starting from image_path. Retries with a new seed on failure."""
    preset = PRESETS[quality]
    w, h = SIZES[(aspect, preset["size"])]
    err = None
    for attempt in range(retries):
        try:
            PROGRESS.update(phase="starting video model (first run loads weights)", value=0, max=0)
            ensure_server()
            name = f"sb_{uuid.uuid4().hex[:10]}.png"
            shutil.copy(image_path, COMFY / "input" / name)
            pid = _call("/prompt", {"prompt": _graph(prompt, name, w, h, seed + attempt * 1000, f"studio/{uuid.uuid4().hex[:8]}", preset),
                                    "client_id": CLIENT_ID})["prompt_id"]
            while pid not in (hist := _call(f"/history/{pid}")):
                time.sleep(2)
            r = hist[pid]
            if r["status"]["status_str"] != "success":
                raise RuntimeError(json.dumps(r["status"]["messages"][-1])[:400])
            f = next(v for o in r["outputs"].values() for v in o.get("images", []))
            shutil.copy(COMFY / "output" / f.get("subfolder", "") / f["filename"], out_path)
            return
        except Exception as e:
            err = e
            time.sleep(3)
    raise RuntimeError(f"clip failed after {retries} tries: {err}")


def _duration(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]))


def stitch(clips, out, sound=True, fade=0.25):
    """Join clips with a short video (+audio) crossfade; sound=False drops audio entirely."""
    if len(clips) == 1:
        subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-i", str(clips[0]), *([] if sound else ["-an"]), "-c:v", "copy",
                               *(["-c:a", "copy"] if sound else []), str(out)])
        return
    durs = [_duration(c) for c in clips]
    inputs = [x for c in clips for x in ("-i", str(c))]
    v, a, acc, parts = "[0:v]", "[0:a]", durs[0], []
    for i in range(1, len(clips)):
        parts.append(f"{v}[{i}:v]xfade=transition=fade:duration={fade}:offset={acc - fade:.3f}[v{i}]")
        if sound:
            parts.append(f"{a}[{i}:a]acrossfade=d={fade}[a{i}]")
        v, a, acc = f"[v{i}]", f"[a{i}]", acc + durs[i] - fade
    maps = ["-map", v] + (["-map", a, "-c:a", "aac", "-b:a", "192k"] if sound else ["-an"])
    subprocess.check_call(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(parts), *maps,
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(out)])
