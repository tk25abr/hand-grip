"""First guess about an object BEFORE touching it: a vision-language model looks at the wrist-camera crop.

Source chain (the dashboard shows which one answered):
  1. "gemini"      Gemini via the Google GenAI SDK Interactions API (needs GEMINI_API_KEY)
  2. "local gemma" Gemma behind an Ollama-compatible endpoint, if GEMMA_URL (or OLLAMA_HOST) is set
  3. "depth-only"  a geometric prior computed from the depth measurement alone (no model, no network)

Called ONCE per object during TARE, never inside the 10 ms control loop. Model answers are cached on
disk by image (results/gemini_cache.json): the picture of each object is the same in every Monte-Carlo
run, only the hidden properties change, so 30 runs do not make 450 calls.

Environment:
  GEMINI_API_KEY   the only place the Gemini key is read from (never written anywhere)
  GEMINI_MODEL     default gemini-3.8-flash
  GEMINI_TIMEOUT   seconds, default 20
  GEMMA_URL        e.g. http://localhost:11434 (OLLAMA_HOST is accepted too)
  GEMMA_MODEL      default gemma3:4b
"""
import fcntl
import hashlib
import json
import math
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import numpy as np

import perception
from scene import MAX_OPENING

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "results", "gemini_cache.json")
PROMPT_VERSION = "v1"
CACHE_MAX_DIFF = 6.0        # mean |pixel difference| (0-255) under which two crops count as the same picture
SURFACES = ("dry", "wet", "slippery", "grippy")
GRIP_LIMIT_N = 120.0        # strongest squeeze the gripper can make

PROMPT = """You are the vision module of a robot with a two-finger parallel-jaw gripper (rubber pads, \
opens 136 mm). The image is a crop from its wrist camera looking down at ONE object on a table, before \
touching it. The scene is a physics simulation: objects are drawn as simple shapes (sphere, ellipsoid, \
cylinder, box) with realistic colour and size, so use colour, size, shape and transparency as clues.

The depth camera measured: height @H mm, width across the jaws @W mm, length @L mm@CUT, \
top shape: @SHAPE.

Say what the object most likely is and estimate its properties for grasping. Reply with JSON only, no \
other text, exactly these keys:
{"object": str, "material": str, "mass_g": [lo, hi], "fragile": bool, "max_safe_squeeze_n": [lo, hi], \
"surface": "dry|wet|slippery|grippy", "friction_guess": [lo, hi], \
"graspable_by_parallel_jaw_14cm": bool, "notes": str}

max_safe_squeeze_n: squeeze force per jaw in newtons above which the object may crack, bruise or crumple \
(use [150, 300] if a 120 N gripper cannot damage it). friction_guess: static friction coefficient between \
the object's surface and rubber pads. Ranges should honestly cover your uncertainty. notes: one short \
sentence."""


# ---------------------------------------------------------------------------------------- parsing
def extract_json(text):
    """First balanced {...} block in the model's text (ignores code fences and prose around it)."""
    if not text:
        raise ValueError("empty answer")
    start = text.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    block = text[start:i + 1]
                    try:
                        return json.loads(block)
                    except json.JSONDecodeError:
                        fixed = re.sub(r",\s*([}\]])", r"\1", block)        # trailing commas
                        try:
                            return json.loads(fixed)
                        except json.JSONDecodeError:
                            break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object found in the answer")


def _num(v):
    if isinstance(v, bool):
        raise ValueError
    if isinstance(v, (int, float)):
        f = float(v)
    else:
        f = float(re.findall(r"-?\d+(?:\.\d+)?", str(v))[0])
    if not math.isfinite(f):
        raise ValueError
    return f


def _range(v, lo_lim, hi_lim):
    """[lo, hi] from a list, a single number or a "lo-hi" string; sorted, clamped, never zero-width."""
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        a, b = _num(v[0]), _num(v[1])
    elif isinstance(v, (list, tuple)) and len(v) == 1:
        a = b = _num(v[0])
    elif isinstance(v, str) and len(re.findall(r"\d+(?:\.\d+)?", v)) >= 2:
        a, b = (float(x) for x in re.findall(r"\d+(?:\.\d+)?", v)[:2])
    else:
        a = b = _num(v)
    a, b = sorted((float(np.clip(a, lo_lim, hi_lim)), float(np.clip(b, lo_lim, hi_lim))))
    if b - a < 1e-9:
        a, b = max(lo_lim, a * 0.8), min(hi_lim, b * 1.2)
    return [round(a, 3), round(b, 3)]


def _bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    s = str(v).strip().lower()
    if s in ("true", "yes", "y", "1"):
        return True
    if s in ("false", "no", "n", "0"):
        return False
    raise ValueError


def validate(raw, prior):
    """Model JSON -> clean guess. Bad or missing fields are filled from the depth-only prior and listed."""
    out, bad = {}, []
    checks = [("object", lambda v: str(v).strip()[:40] or None),
              ("material", lambda v: str(v).strip()[:30] or None),
              ("mass_g", lambda v: _range(v, 0.5, 50000)),
              ("fragile", _bool),
              ("max_safe_squeeze_n", lambda v: _range(v, 0.5, 500)),
              ("surface", lambda v: next(s for s in SURFACES if s in str(v).lower())),
              ("friction_guess", lambda v: _range(v, 0.05, 2.5)),
              ("graspable_by_parallel_jaw_14cm", _bool),
              ("notes", lambda v: str(v).strip()[:160])]
    for key, fn in checks:
        try:
            val = fn(raw[key])
            if val is None:
                raise ValueError
            out[key] = val
        except (KeyError, ValueError, TypeError, IndexError, StopIteration):
            out[key] = prior[key]
            bad.append(key)
    if bad:
        out["notes"] = (out.get("notes", "") + f" [filled from depth prior: {', '.join(bad)}]").strip()
    out["invalid_fields"] = bad
    return out


# ---------------------------------------------------------------------------------------- sources
def depth_prior(m):
    """No model: what depth alone says. Size is measured; material, mass, strength and grip are unknown,
    so the ranges are wide and the controller has to learn the rest by touch."""
    if not m["found"]:
        return dict(object="nothing seen", material="unknown", mass_g=[1.0, 5000.0], fragile=True,
                    max_safe_squeeze_n=[3.0, 120.0], surface="dry", friction_guess=[0.3, 1.0],
                    graspable_by_parallel_jaw_14cm=False,
                    notes="depth only: no object found above the table at this slot")
    vol = perception.volume_m3(m)
    mass = [max(2.0, vol * 20 * 1000), max(10.0, vol * 3000 * 1000)]      # density 20 .. 3000 kg/m^3
    h, w, l = (round(m[k] * 1000) for k in ("height_m", "width_m", "length_m"))
    fits = m["width_m"] <= MAX_OPENING - 0.006 and m["height_m"] >= 0.004
    # strength unknown: assume it may be damaged above ~25 N (ripe tomato ~17 N, egg ~36 N on average);
    # light objects need far less than that, so this mostly matters for heavy ones
    return dict(object=f"unknown {m['shape'].split(' (')[0].replace(',', '')} object", material="unknown",
                mass_g=[round(mass[0], 1), round(mass[1], 1)], fragile=False,
                max_safe_squeeze_n=[25.0, GRIP_LIMIT_N], surface="dry", friction_guess=[0.3, 1.0],
                graspable_by_parallel_jaw_14cm=bool(fits),
                notes=f"depth only: {h}x{w}x{l} mm measured; material unknown, touch decides")


def _prompt(m):
    cut = " (runs off the image, so length is a lower bound)" if m.get("cut_off") else ""
    return (PROMPT.replace("@H", f"{m['height_m'] * 1000:.0f}").replace("@W", f"{m['width_m'] * 1000:.0f}")
            .replace("@L", f"{m['length_m'] * 1000:.0f}").replace("@CUT", cut).replace("@SHAPE", m["shape"]))


def gemini_model():
    return os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")


def ask_gemini(jpeg_b64, prompt, model):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("no GEMINI_API_KEY")
    from google import genai                      # imported lazily: the depth-only path needs no SDK
    client = genai.Client(api_key=key)
    it = client.interactions.create(
        model=model,
        input=[{"type": "image", "data": jpeg_b64, "mime_type": "image/jpeg"},
               {"type": "text", "text": prompt}],
        generation_config={"thinking_level": "low"},
    )
    return it.output_text


def gemma_url():
    url = os.environ.get("GEMMA_URL") or os.environ.get("OLLAMA_HOST")
    if url and not url.startswith("http"):
        url = "http://" + url
    return url.rstrip("/") if url else None


def ask_gemma(jpeg_b64, prompt, model):
    url = gemma_url()
    if not url:
        raise RuntimeError("no GEMMA_URL / OLLAMA_HOST configured")
    body = json.dumps(dict(model=model, stream=False, format="json",
                           messages=[dict(role="user", content=prompt, images=[jpeg_b64])])).encode()
    req = urllib.request.Request(url + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=float(os.environ.get("GEMMA_TIMEOUT", 45))) as r:
        return json.loads(r.read())["message"]["content"]


def _with_timeout(fn, seconds, *args):
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn, *args)
    try:
        return fut.result(timeout=seconds)
    except FutureTimeout:
        raise RuntimeError(f"timed out after {seconds:.0f} s")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------------------- cache
class Cache:
    """Answers keyed by (source, model, prompt version, picture). A picture matches when its 16x16
    thumbnail differs from a stored one by less than CACHE_MAX_DIFF grey levels on average, which
    absorbs sub-pixel jitter of the hand between runs; different objects differ by far more."""

    def __init__(self, path=CACHE_PATH):
        self.path = path

    def _load(self):
        try:
            with open(self.path) as f:
                return json.load(f).get("entries", [])
        except (OSError, ValueError):
            return []

    def find(self, source, model, thumb):
        best, best_d = None, CACHE_MAX_DIFF
        for e in self._load():
            if e.get("source") != source or e.get("model") != model or e.get("prompt") != PROMPT_VERSION:
                continue
            t = np.array(e["thumb"], np.int16)
            if t.shape != thumb.shape:
                continue
            d = float(np.mean(np.abs(t - thumb)))
            if d < best_d:
                best, best_d = e, d
        return best

    def add(self, entry):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path + ".lock", "w") as lock:          # several bench processes may write at once
            fcntl.flock(lock, fcntl.LOCK_EX)
            entries = self._load()
            entries.append(entry)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(dict(version=1, note="Model answers cached by wrist-camera picture. No API keys here.",
                               entries=entries), f)
            os.replace(tmp, self.path)
            fcntl.flock(lock, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------------------- main entry
def first_guess(rgb, m, use_cache=True, cache=None):
    """One guess per object. Returns the validated JSON fields plus: source, model, latency_s (this call),
    model_latency_s (the original model call, also when served from cache), cached, fallback (why a
    better source was skipped), depth (the measurement the guess was made with)."""
    cache = cache or Cache()
    prior = depth_prior(m)
    img = perception.crop(rgb, m["bbox"])
    thumb = perception.thumbnail(img)
    img_hash = hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest()[:12]
    prompt = _prompt(m)
    depth = dict(found=m["found"], height_mm=round(m["height_m"] * 1000), width_mm=round(m["width_m"] * 1000),
                 length_mm=round(m["length_m"] * 1000), shape=m["shape"], cut_off=m["cut_off"])
    skipped = []
    sources = [("gemini", gemini_model(), ask_gemini, float(os.environ.get("GEMINI_TIMEOUT", 20)))]
    sources.append(("local gemma", os.environ.get("GEMMA_MODEL", "gemma3:4b"), ask_gemma,
                    float(os.environ.get("GEMMA_TIMEOUT", 45))))
    for source, model, ask, timeout in sources:
        t0 = time.time()
        hit = cache.find(source, model, thumb) if use_cache else None   # cached answer works offline too
        if hit is None and source == "gemini" and not os.environ.get("GEMINI_API_KEY"):
            skipped.append("gemini: no GEMINI_API_KEY")
            continue
        if hit is None and source == "local gemma" and not gemma_url():
            skipped.append("gemma: no GEMMA_URL")
            continue
        if hit is not None:
            g = validate(hit["parsed"], prior)
            return dict(g, source=source, model=model, latency_s=round(time.time() - t0, 3),
                        model_latency_s=hit.get("latency_s"), cached=True, fallback="; ".join(skipped) or None,
                        depth=depth, image_hash=img_hash)
        try:
            text = _with_timeout(ask, timeout, perception.jpeg_b64(img), prompt, model)
            dt = time.time() - t0
            raw = extract_json(text)
        except Exception as e:                     # network, quota, timeout, bad JSON ... -> next source
            skipped.append(f"{source}: {type(e).__name__}: {str(e)[:80]}")
            continue
        g = validate(raw, prior)
        if use_cache:
            cache.add(dict(source=source, model=model, prompt=PROMPT_VERSION, image_hash=img_hash,
                           thumb=thumb.tolist(), answer=text[:2000], parsed=raw, latency_s=round(dt, 3),
                           created=time.strftime("%Y-%m-%d %H:%M:%S")))
        return dict(g, source=source, model=model, latency_s=round(dt, 3), model_latency_s=round(dt, 3),
                    cached=False, fallback="; ".join(skipped) or None, depth=depth, image_hash=img_hash)
    return dict(prior, invalid_fields=[], source="depth-only", model=None, latency_s=0.0, model_latency_s=None,
                cached=False, fallback="; ".join(skipped) or None, depth=depth, image_hash=img_hash)