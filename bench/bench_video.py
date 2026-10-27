"""Benchmark MiniMax-H3 I2V settings on this GPU: same first frame, prompt and seed; per-phase timing from ComfyUI's websocket.
Usage (from video_gen_test/, ComfyUI on :8199 started with --use-sage-attention):
    .venv/bin/python storyboard_studio/bench/bench_video.py <config> [<config> ...]
Writes storyboard_studio/bench/out/<config>.mp4 and appends one JSON line per run to bench/results.jsonl."""
import json
import shutil
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
URL, CID = "http://127.0.0.1:8199", uuid.uuid4().hex
IMAGE = ROOT / "ref_image.png"  # 1344x768 portrait of the woman: faces show end-of-clip morphing clearly
PROMPT = ("The young woman with red hair in the yellow raincoat slowly turns her head toward the camera and gives a gentle smile, "
          "rain falls softly, her clear umbrella and hair move slightly in the breeze, neon reflections shimmer on the wet street. "
          "0-2s: she looks off to the side. 2-4s: she turns to face the camera. 4-5s: a warm smile, she holds still. "
          "Camera: slow push-in. Audio: soft rain, distant city hum.")
SEED = 1234
LORAS = {"v1.0": "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
         "v1.2": "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors",
         "8step": "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"}

# config name -> settings. attn: "server" (= --use-sage-attention), "pytorch", "ck" (comfy kitchen int8); sparse: sol-attn tau or None
CONFIGS = {
    "sage":        dict(te="int8", vae="int8", attn="server", sparse=None, lora="v1.0", steps=4),
    "pytorch":     dict(te="int8", vae="int8", attn="pytorch", sparse=None, lora="v1.0", steps=4),
    "ck":          dict(te="int8", vae="int8", attn="ck", sparse=None, lora="v1.0", steps=4),
    "sparse_ck":   dict(te="int8", vae="int8", attn="ck", sparse=1.3, lora="v1.0", steps=4),
    "sparse_pt":   dict(te="int8", vae="int8", attn="pytorch", sparse=1.3, lora="v1.0", steps=4),
    "sage_bf16te": dict(te="bf16", vae="fp16", attn="server", sparse=None, lora="v1.0", steps=4),  # the setup used so far
    "v12_ck":      dict(te="int8", vae="int8", attn="ck", sparse=None, lora="v1.2", steps=4),
    "v12_pytorch": dict(te="int8", vae="int8", attn="pytorch", sparse=None, lora="v1.2", steps=4),
    "8step_ck":    dict(te="int8", vae="int8", attn="ck", sparse=None, lora="8step", steps=8),
    "8step_sparse_ck": dict(te="int8", vae="int8", attn="ck", sparse=1.3, lora="8step", steps=8),
    "draft_ck":    dict(te="int8", vae="int8", attn="ck", sparse=None, lora="v1.0", steps=4, w=960, h=544),
}


def call(path, data=None):
    req = urllib.request.Request(URL + path, json.dumps(data).encode() if data else None, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def graph(c, image_name, prefix):
    te = {"int8": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "bf16": "qwen3vl_32b_minimax_h3_bf16.safetensors"}[c["te"]]
    vae = {"int8": "minimax_h3_video_vae_int8_convrot.safetensors", "fp16": "minimax_h3_video_vae_fp16.safetensors"}[c["vae"]]
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0], "lora_name": LORAS[c["lora"]], "strength_model": 1.0}},
        "shift": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["lora", 0], "shift_video": 6.0, "shift_audio": 3.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "img": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "cond": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": PROMPT, "width": c.get("w", 1344),
                                                                  "height": c.get("h", 768), "length": 124, "first_frame": ["img", 0]}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "adec": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]}},
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["dec", 0], "fps": 24, "audio": ["adec", 0]}},
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"}},
    }
    model = ["shift", 0]
    if c["attn"] != "server":
        g["attn"] = {"class_type": "ModelAttentionBackend", "inputs": {"model": model,
                     "attention": {"pytorch": "pytorch attention", "ck": "comfy kitchen attention"}[c["attn"]]}}
        model = ["attn", 0]
    if c["sparse"]:
        g["sparse"] = {"class_type": "BlockSparseAttention", "inputs": {"model": model, "selection": "sol-attn", "selection.tau": c["sparse"],
                       "start_percent": 0.2, "end_percent": 1.0, "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256,
                       "sink_conditioning": "exact_kv_and_rows", "verbose": False}}
        model = ["sparse", 0]
    g["guider"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["cond", 0]}}
    g["sigmas"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple", "steps": c["steps"], "denoise": 1.0}}
    g["sample"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                                                                    "sigmas": ["sigmas", 0], "latent_image": ["cond", 1]}}
    return g


def run(name):
    c = CONFIGS[name.removesuffix("_fresh")]
    fresh = name.endswith("_fresh")  # new prompt text each run -> the text encoder must really run (no ComfyUI cache)
    events = []  # (time, kind, detail)
    done = threading.Event()

    def listen():
        with connect(URL.replace("http", "ws") + f"/ws?clientId={CID}", max_size=None) as ws:
            for raw in ws:
                if not isinstance(raw, str):
                    continue
                m = json.loads(raw)
                t, d = m.get("type"), m.get("data", {})
                if t == "executing":
                    events.append((time.time(), "node", d.get("node")))
                elif t == "progress":
                    events.append((time.time(), "step", d.get("value")))
                elif t in ("execution_success", "execution_error"):
                    events.append((time.time(), t, ""))
                    done.set()
                    return

    th = threading.Thread(target=listen, daemon=True)
    th.start()
    time.sleep(0.5)
    img = f"bench_{uuid.uuid4().hex[:6]}.png"
    shutil.copy(IMAGE, ROOT / "ComfyUI" / "input" / img)
    t0 = time.time()
    g = graph(c, img, f"bench/{name}")
    if fresh:
        g["cond"]["inputs"]["prompt"] = PROMPT + f" Variation {uuid.uuid4().hex[:6]}."
    pid = call("/prompt", {"prompt": g, "client_id": CID})["prompt_id"]
    done.wait(3600)
    total = time.time() - t0
    hist = call(f"/history/{pid}")[pid]
    ok = hist["status"]["status_str"] == "success"

    def first(kind, detail=None):
        return next((t for t, k, d in events if k == kind and (detail is None or d == detail)), None)

    steps = [t for t, k, d in events if k == "step"]
    s_cond, s_sample, s_dec = first("node", "cond"), first("node", "sample"), first("node", "adec")
    res = {"config": name, **c, "fresh": fresh, "ok": ok, "total_s": round(total, 1),
           "load+encode_s": round(s_sample - t0, 1) if s_sample else None,
           "sample_init_s": round(steps[0] - s_sample, 1) if steps and s_sample else None,  # model to GPU + LoRA patch before step 1
           "per_step_s": round((steps[-1] - steps[0]) / max(1, len(steps) - 1), 1) if len(steps) > 1 else None,
           "decode_s": round(time.time() - s_dec, 1) if s_dec else None,
           "error": None if ok else json.dumps(hist["status"]["messages"][-1])[:300]}
    if ok:
        f = next(v for o in hist["outputs"].values() for v in o.get("images", []))
        shutil.copy(ROOT / "ComfyUI" / "output" / f.get("subfolder", "") / f["filename"], OUT / f"{name}.mp4")
    with open(Path(__file__).resolve().parent / "results.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
