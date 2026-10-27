"""FLUX.2 Klein image worker for the storyboard studio. Runs in the image project's venv (it has the right diffusers);
the studio server starts it and talks to it over localhost HTTP. Stdlib only besides torch/diffusers/PIL.

POST /gen      {"prompt", "out", "width", "height", "seed", "refs": [paths]}  -> {"ok": true, "out": path}
POST /offload  move the pipeline to CPU RAM so the GPU is free for the video model (next /gen moves it back)
GET  /health
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

MODEL = os.environ.get("FLUX_MODEL", "/mnt/truenas/image_model/flux_model")
pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
on_gpu = False
lock = threading.Lock()  # one GPU job at a time


def to_gpu():
    global on_gpu
    if not on_gpu:
        pipe.to("cuda")
        on_gpu = True


def offload():
    global on_gpu
    if on_gpu:
        pipe.to("cpu")
        torch.cuda.empty_cache()
        on_gpu = False


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        self._send(200, {"ok": True, "on_gpu": on_gpu})

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with lock:
                if self.path == "/offload":
                    offload()
                    return self._send(200, {"ok": True})
                to_gpu()
                refs = [Image.open(p).convert("RGB") for p in req.get("refs", [])]
                img = pipe(prompt=req["prompt"], height=int(req.get("height", 768)), width=int(req.get("width", 1344)),
                           num_inference_steps=4, guidance_scale=1.0,
                           generator=torch.Generator("cuda").manual_seed(int(req.get("seed", 0))),
                           **({"image": refs} if refs else {})).images[0]
                img.save(req["out"])
            self._send(200, {"ok": True, "out": req["out"]})
        except Exception as e:  # report, keep serving
            self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8021
    print(f"flux worker ready on 127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
