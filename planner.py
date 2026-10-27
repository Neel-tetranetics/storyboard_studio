"""Gemini storyboard: brief (+ optional reference photos) -> N scenes, each with a first-frame image prompt and a video (motion) prompt."""
import os
import time

from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel

SCENE_SECONDS = 5


class Scene(BaseModel):
    image_prompt: str
    video_prompt: str
    refs: list[int]  # 1-based reference photos this frame should contain


class Plan(BaseModel):
    style: str
    scenes: list[Scene]


SYSTEM = """You are a senior commercial director, cinematographer and prompt engineer.
Turn the brief into exactly {n} consecutive shots of ~{sec} seconds each (total ~{total}s). Each shot is made in two steps:
1) An image model renders the FIRST FRAME from `image_prompt`.
2) A quantized (fp8), step-distilled image-to-video model animates that frame using `video_prompt`. It cannot invent missing detail, so be explicit.

Reference photos: {tags}. If a shot shows a referenced subject, list its number in `refs` and describe it in words in the image prompt
("the watch from the reference image: navy steel case, sunburst dial..."). Never invent logos or text.

image_prompt (80-140 words): a single still frame. Shot size and angle, lens and depth of field, subject and exact pose, every visible
object with materials and colors, background, lighting direction and color temperature, mood. Photorealistic commercial look unless the
brief asks for another style. Exactly one product unit unless the brief says otherwise. Describe what IS there; avoid negations.

video_prompt (80-140 words): what happens starting FROM that frame. Re-state the subject briefly, then a timeline in ~{sec}s with 2-3 beats
("0-2s: ..., 2-4s: ..., 4-5s: ..."), exactly ONE smooth camera move (slow push-in, orbit, slider, crane), physical detail of motion
(light sweeps, reflections, fabric, water, particles), and end with one short audio line (ambience, foley, music mood; no dialogue unless asked).
Keep everything physically plausible: the product stays solid, opaque and unchanged.

Build an arc: hook -> reveal -> details/benefits -> lifestyle -> hero end shot. Keep palette, light and wardrobe consistent;
put that shared look in `style` (1-2 sentences). Do not repeat `style` inside the prompts.
Return JSON: {{"style": str, "scenes": [{{"image_prompt": str, "video_prompt": str, "refs": [int]}} x {n}]}}."""


def _client():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is empty (storyboard_studio/.env or product_video/.env)")
    return genai.Client(api_key=key)


def fill_scene(brief: str, before: dict | None, after: dict | None, idea: str, n_refs: int, retries: int = 6) -> dict:
    """Write one new shot that fits between two existing shots (for 'add scene here')."""
    ctx = (f"Brief: {brief}\nShot before: {before['video_prompt'] if before else '(start of film)'}\n"
           f"Shot after: {after['video_prompt'] if after else '(end of film)'}\nIdea for the new shot: {idea or 'choose what fits best'}")
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM.format(n=1, sec=SCENE_SECONDS, total=SCENE_SECONDS, tags=f"{n_refs} reference photo(s)"),
        response_mime_type="application/json", response_schema=Plan, temperature=0.8)
    client, model, err = _client(), os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"), None
    for attempt in range(retries):
        try:
            p = client.models.generate_content(model=model, contents=[ctx], config=config).parsed
            if p and p.scenes:
                s = p.scenes[0]
                return {"image_prompt": f"{p.style.strip()} {s.image_prompt.strip()}", "video_prompt": f"{p.style.strip()} {s.video_prompt.strip()}",
                        "refs": sorted({r for r in s.refs if 1 <= r <= n_refs})}
        except Exception as e:
            err = e
        time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f"Gemini scene fill failed: {err}")


def plan(brief: str, n: int, ref_paths: list[str], ref_labels: list[str], retries: int = 8) -> list[dict]:
    tags = ", ".join(f"reference {i + 1} = {lab}" for i, lab in enumerate(ref_labels)) or "none"
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM.format(n=n, sec=SCENE_SECONDS, total=n * SCENE_SECONDS, tags=tags),
        response_mime_type="application/json", response_schema=Plan, temperature=0.8)
    contents = [f"Brief: {brief}"]
    for i, p in enumerate(ref_paths):
        contents += [f"Reference {i + 1} ({ref_labels[i]}):", Image.open(p).convert("RGB")]
    client, model, err = _client(), os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"), None
    for attempt in range(retries):
        try:
            p = client.models.generate_content(model=model, contents=contents, config=config).parsed
            if p and len(p.scenes) == n:
                style = p.style.strip()
                return [{"image_prompt": f"{style} {s.image_prompt.strip()}", "video_prompt": f"{style} {s.video_prompt.strip()}",
                         "refs": sorted({r for r in s.refs if 1 <= r <= len(ref_paths)})} for s in p.scenes]
            err = f"got {len(p.scenes) if p else 0} scenes, wanted {n}"
        except Exception as e:  # 503 high demand, 429, bad JSON -> back off and retry
            err = e
        time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f"Gemini planning failed after {retries} tries: {err}")
