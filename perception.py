"""Wrist-camera perception BEFORE touching: depth -> object height / width / shape, RGB crop for the VLM.

Allowed inputs only (no ground truth):
  - the wrist camera image (RGB + depth, with sensor noise)
  - the hand position from the gantry encoders (sensors["gripper_xyz"]) and the finger opening
    (sensors["aperture_mm"]) - to cut the robot's own fingers out of the depth image
  - the camera mount (scene.WRIST_CAM_*), the taught table height and the taught slot x/y

The camera looks down and ~20 deg forward, so neighbouring slots are also in view. The object is the
biggest blob of depth points standing out of the table inside a window around the taught slot.
"""
import base64
import math

import cv2
import numpy as np

from objects import TABLE_Z
from scene import (WRIST_CAM_POS, WRIST_CAM_XYAXES, WRIST_CAM_FOVY, FINGER_HALF, PAD_HALF, PAD_Z,
                   MAX_OPENING)

WINDOW = (0.105, 0.12)      # half-size (x, y) of the search area around the taught slot (< half the slot spacing)
MIN_RISE = 0.004            # depth points more than 4 mm above the table belong to an object
MIN_PIXELS = 40             # fewer object pixels than this = nothing there
TOP_BAND = 0.0025           # points within 2.5 mm of the highest point = "top surface"


def cam_rotation():
    """Columns = camera x (image right), y (image up), z (backwards) axes in the hand/world frame."""
    x = np.array(WRIST_CAM_XYAXES[:3], float)
    y = np.array(WRIST_CAM_XYAXES[3:], float)
    x /= np.linalg.norm(x)
    y -= x * (x @ y)
    y /= np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def backproject(depth, hand_xyz):
    """Planar depth image (m) -> world xyz of every pixel. The hand does not rotate (Cartesian gantry)."""
    h, w = depth.shape
    f = (h / 2) / math.tan(math.radians(WRIST_CAM_FOVY) / 2)
    u = (np.arange(w) + 0.5 - w / 2) / f
    v = (np.arange(h) + 0.5 - h / 2) / f
    xc = u[None, :] * depth
    yc = -v[:, None] * depth
    pc = np.stack([xc, yc, -depth], axis=-1)
    return pc @ cam_rotation().T + (np.asarray(hand_xyz, float) + np.array(WRIST_CAM_POS))


def _finger_mask(pts, hand_xyz, aperture_m):
    """Pixels that see the robot's own fingers/pads (known geometry, open to `aperture_m`)."""
    rel = pts - np.asarray(hand_xyz, float)
    inner = aperture_m / 2 - 0.004
    outer = aperture_m / 2 + 2 * PAD_HALF[1] + 2 * FINGER_HALF[1] + 0.004
    tip = PAD_Z - PAD_HALF[2] - 0.004
    ay = np.abs(rel[..., 1])
    return (np.abs(rel[..., 0]) < FINGER_HALF[0] + 0.004) & (ay > inner) & (ay < outer) & (rel[..., 2] > tip)


def measure(rgb, depth, slot, sensors):
    """Returns a dict with found, height_m, width_m (across the jaws = y), length_m (x), shape,
    cut_off (object runs off the image edge, so length is a lower bound), bbox (pixel crop), n_px."""
    hand = np.asarray(sensors["gripper_xyz"], float)
    aperture = float(sensors.get("aperture_mm", MAX_OPENING * 1000)) / 1000
    d = cv2.medianBlur(depth.astype(np.float32), 5)
    pts = backproject(d, hand)
    rel = pts - np.array([slot["x"], slot["y"], 0.0])
    mask = ((np.abs(rel[..., 0]) < WINDOW[0]) & (np.abs(rel[..., 1]) < WINDOW[1])
            & (pts[..., 2] > TABLE_Z + MIN_RISE) & (pts[..., 2] < hand[2] - 0.05)
            & ~_finger_mask(pts, hand, aperture))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = dict(found=False, height_m=0.0, width_m=0.0, length_m=0.0, shape="nothing", cut_off=False,
               bbox=None, n_px=0, flat_top=0.0, fill=0.0)
    if n <= 1:
        return out
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < MIN_PIXELS:
        return out
    m = lab == k
    p = pts[m]
    top = float(np.percentile(p[:, 2], 98))
    y0, y1 = np.percentile(p[:, 1], [0.3, 99.7])
    x0, x1 = np.percentile(p[:, 0], [0.3, 99.7])
    width, length = float(y1 - y0), float(x1 - x0)
    top_px = m & (pts[..., 2] > top - TOP_BAND)
    flat_top = float(top_px.sum() / m.sum())
    # outline of the top surface in the image: filled area / min-area rectangle (box ~1, disc ~0.785)
    fill = 0.0
    cs, _ = cv2.findContours(top_px.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cs:
        c = max(cs, key=cv2.contourArea)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        fill = float(cv2.contourArea(c) / max(1.0, rw * rh))
    if flat_top < 0.45:
        shape = "round, domed (sphere/ellipsoid)"
    elif fill > 0.9:
        shape = "box (flat faces)"
    else:
        shape = "round, flat top (cylinder)"
    x, y, w, h = (int(v) for v in stats[k, :4])
    H, W = depth.shape
    cut = y + h >= H - 1 or x <= 0 or x + w >= W - 1
    out.update(found=True, height_m=top - TABLE_Z, width_m=width, length_m=length, shape=shape,
               cut_off=bool(cut), bbox=(x, y, w, h), n_px=int(m.sum()), flat_top=flat_top, fill=fill)
    return out


def volume_m3(m):
    """Rough volume from the measured box and the shape class (for the depth-only mass prior)."""
    v = m["height_m"] * m["width_m"] * max(m["length_m"], m["width_m"] if m["cut_off"] else 0.0)
    if m["shape"].startswith("box"):
        return v
    if "cylinder" in m["shape"]:
        return v * math.pi / 4
    return v * math.pi / 6


def crop(rgb, bbox, margin=0.35, min_px=64, out_px=384):
    """RGB crop around the object (neighbours mostly cut away), upscaled for the VLM."""
    H, W = rgb.shape[:2]
    if bbox is None:
        x, y, w, h = W // 4, H // 2, W // 2, H // 2       # nothing found: bottom-centre, where the slot is
    else:
        x, y, w, h = bbox
    side = max(w, h, min_px) * (1 + 2 * margin)
    cx, cy = x + w / 2, y + h / 2
    x0, x1 = int(max(0, cx - side / 2)), int(min(W, cx + side / 2))
    y0, y1 = int(max(0, cy - side / 2)), int(min(H, cy + side / 2))
    c = rgb[y0:y1, x0:x1]
    s = out_px / max(c.shape[:2])
    return cv2.resize(c, (max(1, round(c.shape[1] * s)), max(1, round(c.shape[0] * s))),
                      interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)


def jpeg_b64(img_rgb, quality=90):
    ok, buf = cv2.imencode(".jpg", img_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def thumbnail(img_rgb, n=16):
    """Small colour thumbnail used as the cache key (same object = same picture, robust to 1-px jitter)."""
    return cv2.resize(img_rgb, (n, n), interpolation=cv2.INTER_AREA).astype(np.int16)
