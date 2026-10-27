"""Storyboard Studio: prompt -> Gemini storyboard -> FLUX first frames (edit / regenerate / upload / add / delete / reorder)
-> editable video prompts -> MiniMax-H3 I2V clips -> final video.
Run from video_gen_test/:  .venv/bin/python storyboard_studio/server.py   then open http://<host>:7870"""
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
load_dotenv(HERE.parent / "product_video" / ".env")  # shared GEMINI_API_KEY if the studio has none
sys.path.insert(0, str(HERE))
import planner  # noqa: E402
import video  # noqa: E402

PROJECTS = HERE / "projects"
PROJECTS.mkdir(exist_ok=True)
FLUX_PY = os.environ.get("FLUX_PYTHON", "/mnt/newdisk/neel/Text_to_image_model/venv/bin/python")
FLUX_PORT = int(os.environ.get("FLUX_PORT", "8021"))
MAX_REFS = 4
ALLOWED_IMG = {".png", ".jpg", ".jpeg", ".webp"}

app = FastAPI(title="Storyboard Studio")
lock = threading.RLock()
jobs: "queue.Queue[tuple]" = queue.Queue()
state = {"gpu": None, "current": "", "started": 0.0, "sub": "", "recent": [], "avg": {}, "video_warm": False}  # gpu: model owning the GPU
COLD_LOAD_SECS = 600  # first clip of a session also loads the video weights
LABELS = {"plan": "Planning storyboard (Gemini)", "fill": "Writing new scene (Gemini) + image", "image": "Generating image",
          "clip": "Rendering clip", "final": "Rendering final video"}
DEFAULT_SECS = {"plan": 30, "fill": 40, "image": 8, "clip": 80, "final": 600}  # clip: measured ~72 s warm at 768p


# ---------- project store ----------
def pdir(pid):
    d = (PROJECTS / pid).resolve()
    if d.parent != PROJECTS.resolve() or not d.exists():
        raise HTTPException(404, "project not found")
    return d


def load(pid):
    return json.loads((pdir(pid) / "project.json").read_text())


def save(p):
    f = PROJECTS / p["id"] / "project.json"
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(p, indent=1))
    tmp.replace(f)


def update(pid, fn):
    """Read-modify-write under the lock; fn mutates the project dict."""
    with lock:
        p = load(pid)
        fn(p)
        save(p)
        return p


def scene(p, sid):
    for s in p["scenes"]:
        if s["id"] == sid:
            return s
    raise HTTPException(404, "scene not found")


def set_scene(pid, sid, **kw):
    def f(p):
        try:
            scene(p, sid).update(kw)
        except HTTPException:
            pass  # scene deleted while its job ran
    update(pid, f)


def new_scene(d):
    return {"id": uuid.uuid4().hex[:8], "image_prompt": d.get("image_prompt", ""), "video_prompt": d.get("video_prompt", ""),
            "refs": d.get("refs", []), "image": None, "history": [], "clip": None, "seed": uuid.uuid4().int % 2**31, "status": ""}


# ---------- GPU owners ----------
flux_proc = None


def flux_call(path, data=None, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{FLUX_PORT}{path}", json.dumps(data or {}).encode() if data is not None else None,
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def ensure_flux():
    global flux_proc
    try:
        flux_call("/health", timeout=3)
        return
    except Exception:
        pass
    if flux_proc is None or flux_proc.poll() is not None:
        flux_proc = subprocess.Popen([FLUX_PY, str(HERE / "flux_worker.py"), str(FLUX_PORT)],
                                     stdout=open(HERE / "flux_worker.log", "a"), stderr=subprocess.STDOUT)
    for _ in range(300):
        time.sleep(2)
        try:
            flux_call("/health", timeout=3)
            return
        except Exception:
            if flux_proc.poll() is not None:
                raise RuntimeError("FLUX worker exited; see storyboard_studio/flux_worker.log")
    raise RuntimeError("FLUX worker did not start")


def use_gpu(kind):
    """Only one model fits on the GPU: park the other one in CPU RAM before switching."""
    if state["gpu"] == kind:
        return
    if kind == "image":
        video.comfy_free()
    else:
        try:
            flux_call("/offload", {})
        except Exception:
            pass
    state["gpu"] = kind


# ---------- jobs ----------
def gen_image(pid, sid, mode, instruction=""):
    p = load(pid)
    s = scene(p, sid)
    d = PROJECTS / pid
    w, h = video.SIZES[(p["aspect"], "final")]
    refs = [str(d / p["refs"][i - 1]["path"]) for i in s["refs"] if 1 <= i <= len(p["refs"])]
    if mode == "edit" and s["image"]:
        prompt = (f"Edit the first reference image: {instruction}. Keep everything else exactly the same: composition, framing, subject, "
                  f"product, lighting and colors. Photorealistic, no text.")
        refs = [str(d / s["image"])] + refs
    else:
        prompt = s["image_prompt"]
    out = f"img/{sid}_{uuid.uuid4().hex[:6]}.png"
    ensure_flux()
    use_gpu("image")
    r = flux_call("/gen", {"prompt": prompt, "out": str(d / out), "width": w, "height": h, "seed": int(time.time() * 1000) % 2**31, "refs": refs})
    if not r.get("ok"):
        raise RuntimeError(r.get("error"))

    def f(p):
        s = scene(p, sid)
        if s["image"]:
            s["history"].append(s["image"])
        s.update(image=out, clip=None, status="", future=[])
    update(pid, f)


def gen_clip(pid, sid):
    p = load(pid)
    s = scene(p, sid)
    if not s["image"]:
        raise RuntimeError("scene has no image yet")
    d = PROJECTS / pid
    key = video.clip_key(d / s["image"], s["video_prompt"], p["quality"], p["aspect"], s["seed"])
    out = d / "clips" / f"{key}.mp4"
    if not out.exists():  # cache: unchanged image+prompt+settings is never rendered twice
        use_gpu("video")
        video.make_clip(str(d / s["image"]), s["video_prompt"], p["quality"], p["aspect"], s["seed"], str(out))
        state["video_warm"] = True
    set_scene(pid, sid, clip=f"clips/{key}.mp4", status="")


def render_final(pid):
    p = load(pid)
    for attempt in range(3):  # pass 1 checks every scene's cache key; later passes catch scenes edited during the render
        todo = load(pid)["scenes"] if attempt == 0 else \
            [s for s in load(pid)["scenes"] if not s["clip"] or not (PROJECTS / pid / s["clip"]).exists()]
        if not todo:
            break
        for i, s in enumerate(todo):
            state["sub"] = f"clip {i + 1}/{len(todo)}"
            set_scene(pid, s["id"], status="rendering clip")
            gen_clip(pid, s["id"])
    state["sub"] = "joining clips"
    p = load(pid)
    if any(not s["clip"] for s in p["scenes"]):
        raise RuntimeError("scenes kept changing during the render; press Render final video again")
    d = PROJECTS / pid
    out = f"final_{time.strftime('%H%M%S')}.mp4"
    video.stitch([d / s["clip"] for s in p["scenes"]], d / out, sound=p["sound"])
    update(pid, lambda p: p.update(final=out, status=""))


def worker():
    while True:
        kind, pid, sid, args = jobs.get()
        state.update(current=f"{kind} {sid or ''}".strip(), kind=kind, pid=pid, sid=sid, started=time.time(), sub="")
        ok = "done"
        try:
            if sid:
                set_scene(pid, sid, status={"image": "generating image", "clip": "rendering clip"}.get(kind, kind))
            if kind == "plan":
                p = load(pid)
                d = PROJECTS / pid
                scenes = planner.plan(p["brief"], p["n"], [str(d / r["path"]) for r in p["refs"]], [r["label"] for r in p["refs"]])
                update(pid, lambda p: p.update(scenes=[new_scene(s) for s in scenes], status=""))
                for s in load(pid)["scenes"]:
                    set_scene(pid, s["id"], status="queued")
                    jobs.put(("image", pid, s["id"], {"mode": "regenerate"}))
            elif kind == "fill":
                p = load(pid)
                ids = [s["id"] for s in p["scenes"]]
                i = ids.index(sid)
                filled = planner.fill_scene(p["brief"], p["scenes"][i - 1] if i > 0 else None,
                                            p["scenes"][i + 1] if i + 1 < len(ids) else None, args.get("idea", ""), len(p["refs"]))
                set_scene(pid, sid, **filled)
                gen_image(pid, sid, "regenerate")
            elif kind == "image":
                gen_image(pid, sid, args.get("mode", "regenerate"), args.get("instruction", ""))
            elif kind == "clip":
                gen_clip(pid, sid)
            elif kind == "final":
                update(pid, lambda p: p.update(status="rendering final video"))
                render_final(pid)
            if sid:
                set_scene(pid, sid, status="")
        except HTTPException:
            pass
        except Exception as e:
            traceback.print_exc()
            msg = f"error: {e}"[:300]
            ok = msg
            try:
                if sid:
                    set_scene(pid, sid, status=msg)
                update(pid, lambda p: p.update(status=msg if not sid else p.get("status", "")))
            except Exception:
                pass
        finally:
            took = time.time() - state["started"]
            if ok == "done":  # running average per job kind -> ETA for the next one
                a = state["avg"].get(kind)
                state["avg"][kind] = took if a is None else 0.7 * a + 0.3 * took
            state["recent"] = ([{"label": job_label(kind, pid, sid), "secs": round(took, 1), "result": ok}] + state["recent"])[:10]
            state.update(current="", kind=None, sub="")
            jobs.task_done()


def job_label(kind, pid, sid):
    label = LABELS.get(kind, kind)
    if sid:
        try:
            ids = [s["id"] for s in load(pid)["scenes"]]
            label += f" · scene {ids.index(sid) + 1}" if sid in ids else ""
        except Exception:
            pass
    return label


# ---------- API ----------
@app.get("/api/progress")
def progress():
    cur = None
    if state.get("kind"):
        k = state["kind"]
        elapsed = time.time() - state["started"]
        expect = state["avg"].get(k, DEFAULT_SECS.get(k, 60))
        if k == "final":
            expect = state["avg"].get("clip", DEFAULT_SECS["clip"]) * max(1, len(load(state["pid"])["scenes"])) + 10
        if k in ("clip", "final") and not state["video_warm"]:
            expect += COLD_LOAD_SECS
        step = video.PROGRESS if k in ("clip", "final") else {"phase": "", "value": 0, "max": 0}
        cur = {"label": job_label(k, state["pid"], state["sid"]), "sub": state["sub"], "elapsed": round(elapsed),
               "eta": max(0, round(expect - elapsed)), "pct": min(99, round(100 * elapsed / expect)) if expect else 0,
               "phase": step["phase"], "step": step["value"], "steps": step["max"], "project": state["pid"]}
    queued = [job_label(k, p, s) for k, p, s, _ in list(jobs.queue)]
    return {"current": cur, "queue": queued, "recent": state["recent"], "gpu": state["gpu"]}



async def save_upload(f: UploadFile, dest_dir: Path, stem: str) -> str:
    ext = Path(f.filename or "").suffix.lower()
    if ext not in ALLOWED_IMG:
        raise HTTPException(400, "upload a PNG, JPG or WebP image")
    data = await f.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(400, "image too large (max 20 MB)")
    rel = f"{dest_dir.name}/{stem}_{uuid.uuid4().hex[:6]}.png"
    tmp = dest_dir / f"upload{ext}"
    tmp.write_bytes(data)
    try:
        Image.open(tmp).convert("RGB").save(dest_dir.parent / rel)  # re-encode: validates and normalises
    except Exception:
        raise HTTPException(400, "could not read that image")
    finally:
        tmp.unlink(missing_ok=True)
    return rel


@app.post("/api/projects")
async def create(brief: str = Form(...), seconds: int = Form(30), aspect: str = Form("16:9"), labels: str = Form("[]"),
                 files: list[UploadFile] = File(default=[])):
    if not brief.strip():
        raise HTTPException(400, "write a prompt")
    if aspect not in ("16:9", "9:16"):
        raise HTTPException(400, "aspect must be 16:9 or 9:16")
    files = [f for f in files if f.filename]
    if len(files) > MAX_REFS:
        raise HTTPException(400, f"max {MAX_REFS} reference photos")
    pid = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:4]
    d = PROJECTS / pid
    for sub in ("refs", "img", "clips"):
        (d / sub).mkdir(parents=True)
    lab = json.loads(labels or "[]")
    refs = [{"path": await save_upload(f, d / "refs", f"ref{i + 1}"), "label": (lab[i] if i < len(lab) and lab[i] else f"reference {i + 1}")}
            for i, f in enumerate(files)]
    n = max(1, min(36, round(seconds / planner.SCENE_SECONDS)))
    p = {"id": pid, "brief": brief.strip(), "n": n, "aspect": aspect, "quality": "final", "sound": True, "refs": refs,
         "scenes": [], "final": None, "status": "planning storyboard with Gemini"}
    save(p)
    jobs.put(("plan", pid, None, {}))
    return p


@app.get("/api/options")
def options():
    return {"qualities": video.available_presets()}


@app.get("/api/projects")
def list_projects():
    out = []
    for f in sorted(PROJECTS.glob("*/project.json"), reverse=True)[:30]:
        p = json.loads(f.read_text())
        out.append({"id": p["id"], "brief": p["brief"][:90], "scenes": len(p["scenes"])})
    return out


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    p = load(pid)
    p["queue"] = jobs.qsize()
    p["working_on"] = state["current"]
    return p


@app.patch("/api/projects/{pid}")
def patch_project(pid: str, body: dict):
    def f(p):
        if body.get("quality") in video.available_presets():
            p["quality"] = body["quality"]
        if isinstance(body.get("sound"), bool):
            p["sound"] = body["sound"]
    return update(pid, f)


@app.patch("/api/projects/{pid}/scenes/{sid}")
def patch_scene(pid: str, sid: str, body: dict):
    def f(p):
        s = scene(p, sid)
        for k in ("image_prompt", "video_prompt"):
            if isinstance(body.get(k), str):
                s[k] = body[k]
                if k == "video_prompt":
                    s["clip"] = None
        if isinstance(body.get("refs"), list):
            s["refs"] = sorted({int(r) for r in body["refs"] if 1 <= int(r) <= len(p["refs"])})
    return update(pid, f)


@app.post("/api/projects/{pid}/scenes/{sid}/image")
def image_job(pid: str, sid: str, body: dict):
    p = load(pid)
    s = scene(p, sid)
    mode = body.get("mode", "regenerate")
    if mode == "edit" and not (body.get("instruction") or "").strip():
        raise HTTPException(400, "describe the edit")
    if mode == "edit" and not s["image"]:
        raise HTTPException(400, "no image to edit yet")
    set_scene(pid, sid, status="queued")
    jobs.put(("image", pid, sid, {"mode": mode, "instruction": (body.get("instruction") or "").strip()[:500]}))
    return {"ok": True}


@app.post("/api/projects/{pid}/scenes/{sid}/upload")
async def upload_image(pid: str, sid: str, file: UploadFile = File(...)):
    d = pdir(pid)
    rel = await save_upload(file, d / "img", f"{sid}_upload")
    w, h = video.SIZES[(load(pid)["aspect"], "final")]
    from PIL import ImageOps
    ImageOps.fit(Image.open(d / rel).convert("RGB"), (w, h), Image.LANCZOS).save(d / rel)  # frame must match the video canvas

    def f(p):
        s = scene(p, sid)
        if s["image"]:
            s["history"].append(s["image"])
        s.update(image=rel, clip=None, status="", future=[])
    return update(pid, f)


@app.post("/api/projects/{pid}/scenes/{sid}/undo")
def undo(pid: str, sid: str):
    def f(p):
        s = scene(p, sid)
        if not s["history"]:
            raise HTTPException(400, "nothing to undo")
        s.setdefault("future", []).append(s["image"])
        s.update(image=s["history"].pop(), clip=None)
    return update(pid, f)


@app.post("/api/projects/{pid}/scenes/{sid}/redo")
def redo(pid: str, sid: str):
    def f(p):
        s = scene(p, sid)
        if not s.get("future"):
            raise HTTPException(400, "nothing to redo")
        s["history"].append(s["image"])
        s.update(image=s["future"].pop(), clip=None)
    return update(pid, f)


@app.post("/api/projects/{pid}/scenes")
def add_scene(pid: str, body: dict):
    """Insert a scene after `after` (or at the start). Blank prompts -> Gemini writes a fitting shot from `idea`."""
    s = new_scene(body)

    def f(p):
        ids = [x["id"] for x in p["scenes"]]
        at = ids.index(body["after"]) + 1 if body.get("after") in ids else 0
        p["scenes"].insert(at, s)
    update(pid, f)
    if not s["image_prompt"].strip():
        set_scene(pid, s["id"], status="queued")
        jobs.put(("fill", pid, s["id"], {"idea": (body.get("idea") or "")[:500]}))
    elif body.get("generate", True):
        set_scene(pid, s["id"], status="queued")
        jobs.put(("image", pid, s["id"], {"mode": "regenerate"}))
    return load(pid)


@app.delete("/api/projects/{pid}/scenes/{sid}")
def delete_scene(pid: str, sid: str):
    return update(pid, lambda p: p.update(scenes=[s for s in p["scenes"] if s["id"] != sid]))


@app.post("/api/projects/{pid}/scenes/{sid}/move")
def move_scene(pid: str, sid: str, body: dict):
    def f(p):
        ids = [s["id"] for s in p["scenes"]]
        i = ids.index(sid) if sid in ids else -1
        j = i + (1 if body.get("dir", 1) > 0 else -1)
        if i >= 0 and 0 <= j < len(ids):
            p["scenes"][i], p["scenes"][j] = p["scenes"][j], p["scenes"][i]
    return update(pid, f)


@app.post("/api/projects/{pid}/scenes/{sid}/clip")
def clip_job(pid: str, sid: str):
    if not scene(load(pid), sid)["image"]:
        raise HTTPException(400, "generate or upload an image first")
    set_scene(pid, sid, status="queued")
    jobs.put(("clip", pid, sid, {}))
    return {"ok": True}


@app.post("/api/projects/{pid}/render")
def render(pid: str):
    p = load(pid)
    if not p["scenes"] or any(not s["image"] for s in p["scenes"]):
        raise HTTPException(400, "every scene needs an image first")
    update(pid, lambda p: p.update(status="queued final render"))
    jobs.put(("final", pid, None, {}))
    return {"ok": True}


app.mount("/files", StaticFiles(directory=PROJECTS), name="files")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


if __name__ == "__main__":
    import uvicorn
    for f in PROJECTS.glob("*/project.json"):  # queued/running jobs died with the old process
        p = json.loads(f.read_text())
        busy = lambda t: bool(t) and not t.startswith("error")
        if busy(p.get("status")) or any(busy(x.get("status")) for x in p["scenes"]):
            p["status"] = "" if busy(p.get("status")) else p["status"]
            for x in p["scenes"]:
                x["status"] = "" if busy(x.get("status")) else x["status"]
            save(p)
    threading.Thread(target=worker, daemon=True).start()
    video.prefetch()  # warm the page cache with the video weights while the user edits images
    video.progress_listener()
    threading.Thread(target=ensure_flux, daemon=True).start()
    uvicorn.run(app, host=os.environ.get("STUDIO_HOST", "0.0.0.0"), port=int(os.environ.get("STUDIO_PORT", "7870")))
