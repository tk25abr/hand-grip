"""Sensor panel + video frame. The same panel is drawn inside the MuJoCo window (interactive)
and next to the camera views in recorded videos."""
import numpy as np
import cv2

import perception
from objects import public_slot
from trial import OUTCOME_COLORS

FONT = cv2.FONT_HERSHEY_SIMPLEX
G = 9.81


def txt(img, s, xy, scale=0.45, col=(230, 230, 230), th=1):
    cv2.putText(img, str(s), (int(xy[0]), int(xy[1])), FONT, scale, col, th, cv2.LINE_AA)


def bar(img, x, y, w, h, frac, col, bg=(55, 58, 64)):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    f = float(np.clip(frac, 0, 1))
    if f > 0:
        cv2.rectangle(img, (x, y), (x + int(w * f), y + h), col, -1)


def sensor_row(img, y, label, value, frac, col, w):
    txt(img, label, (12, y + 11), 0.42, (190, 195, 205))
    bar(img, 175, y, w - 300, 13, frac, col)
    txt(img, value, (w - 118, y + 11), 0.45, (255, 255, 255))


def panel(runner, w=460, h=720):
    """Live sensor readout for the current trial + results so far."""
    img = np.full((h, w, 3), 24, np.uint8)
    sim, pol = runner.sim, runner.policy
    s = sim.read_sensors()
    y = 22
    txt(img, pol.name[:52], (12, y), 0.48, (255, 210, 120)); y += 22
    if runner.finished:
        txt(img, "ALL TRIALS DONE", (12, y), 0.6, (255, 255, 255), 2); y += 26
    else:
        sl = runner.slot
        txt(img, f"Trial {runner.k + 1}/{len(runner.order)}   object (truth): {sl['name']}", (12, y), 0.5, (255, 255, 255)); y += 20
        brk = "-" if sl["max_force"] >= 5000 else f"{sl['max_force']:.0f} N"
        stf = "rigid" if sl["stiff"] >= 1000 else f"{sl['stiff']:.1f} N/mm"
        txt(img, f"truth: {sl['mass']*1000:.0f} g | mu {sl['mu']:.2f} | breaks {brk} | {stf}", (12, y), 0.4, (150, 150, 160)); y += 17
        ident = runner.label or ("looking..." if getattr(pol, "uses_camera", False) else "unknown - no camera, no AI")
        txt(img, f"robot thinks it is: {ident}"[:58], (12, y), 0.45, (150, 200, 255)); y += 20
        txt(img, f"phase: {runner.phase}", (12, y), 0.5, (255, 230, 120), 1); y += 18
        dl = _wrap(runner.decision, 62)[:2]
        for i, l in enumerate(dl):
            txt(img, l, (12, y), 0.4, (200, 200, 200)); y += 15
        y += 1

    y += 6
    cv2.line(img, (10, y), (w - 10, y), (70, 70, 80), 1); y += 10
    txt(img, "GRIPPER SENSORS (what the robot measures)", (12, y + 4), 0.42, (160, 165, 175)); y += 16
    tare = runner.tare if runner.tare is not None else s["wrist_force"][2]
    load_g = max(0.0, (s["wrist_force"][2] - tare) / G * 1000) if runner.phase not in ("ABOVE", "TARE") else 0.0
    rows = [
        ("grip command", f"{s['grip_cmd']:5.1f} N", max(0, s["grip_cmd"]) / 60, (120, 170, 255)),
        ("tactile pad L", f"{s['touch_l']:5.1f} N", s["touch_l"] / 60, (255, 140, 60)),
        ("tactile pad R", f"{s['touch_r']:5.1f} N", s["touch_r"] / 60, (255, 140, 60)),
        ("pad pressure", f"{s['pressure_kpa']:5.0f} kPa", s["pressure_kpa"] / 50, (255, 110, 90)),
        ("slip risk (shear/normal)", f"{s['shear_ratio']:5.2f}", s["shear_ratio"] / 1.0, (245, 200, 60)),
        ("finger gap = width", f"{s['aperture_mm']:5.0f} mm", s["aperture_mm"] / 140, (150, 220, 150)),
        ("wrist load = weight", f"{load_g:5.0f} g", load_g / 3000, (90, 210, 230)),
        ("hand height", f"{(s['gripper_xyz'][2] - 0.40) * 100:5.1f} cm", (s["gripper_xyz"][2] - 0.40) / 0.6, (180, 180, 200)),
        ("wrist accel", f"{np.linalg.norm(s['wrist_acc']) - G:+5.1f} m/s2", abs(np.linalg.norm(s["wrist_acc"]) - G) / 5, (200, 160, 230)),
    ]
    for lab, val, frac, col in rows:
        sensor_row(img, y, lab, val, frac, col, w)
        y += 19
    if not runner.finished:
        lift = runner.judge.max_lift * 100 if runner.judge else 0
        cur = (sim.truth_obj_pos(runner.slot["idx"])[2] - runner.judge.rest_z) * 100
        txt(img, f"object lift (truth, for scoring): {cur:5.1f} cm   max {lift:4.1f} cm", (12, y + 10), 0.4, (150, 150, 160))
    y += 20

    # time plot: last 6 s of pad force + load
    ph, pw = 92, w - 24
    x0, y0 = 12, y
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (34, 36, 42), -1)
    ser = runner.series[-600:]
    if len(ser) > 2:
        t1 = ser[-1]["t"]
        def line(key, scale, col):
            pts = [(x0 + pw - int((t1 - r["t"]) / 6.0 * pw), y0 + ph - int(np.clip(r[key] / scale, 0, 1) * (ph - 4)) - 2)
                   for r in ser if t1 - r["t"] <= 6.0]
            if len(pts) > 1:
                cv2.polylines(img, [np.array(pts, np.int32)], False, col, 1, cv2.LINE_AA)
        line("grip_cmd", 60, (120, 170, 255))
        line("touch_l", 60, (255, 140, 60))
        line("load_g", 3000, (90, 210, 230))
        line("obj_lift", 0.25, (120, 230, 120))
    txt(img, "last 6 s:  grip cmd  pad force  weight  lift", (x0 + 4, y0 + 12), 0.36, (170, 170, 180))
    y = y0 + ph + 14

    # results table
    txt(img, "RESULTS", (12, y + 4), 0.45, (160, 165, 175)); y += 12
    cols = (12, 170, 262, 330)
    for c, hname in zip(cols, ("object", "result", "peak N", "weight g")):
        txt(img, hname, (c, y + 10), 0.37, (140, 140, 150))
    y += 15
    done = {r["idx"]: r for r in runner.results}
    row_h = max(12, min(15, (h - y - 26) // max(1, len(runner.order))))
    for i in runner.order:
        sl = sim.slots[i]
        r = done.get(i)
        cells = [sl["name"][:20]]
        if r:
            col = OUTCOME_COLORS[r["result"]]
            cells += [r["result"], f"{r['peak_n']:.0f}", f"{r['weight_g']:.0f}" if r["weight_g"] else "-"]
        elif not runner.finished and i == runner.slot["idx"]:
            col = (255, 255, 255); cells += ["running..."]
        else:
            col = (95, 95, 105)
        for c, v in zip(cols, cells):
            txt(img, v, (c, y + 10), 0.37, col)
        y += row_h
    n_ok = sum(r["result"] == "SUCCESS" for r in runner.results)
    n_dec = sum(r["result"] == "DECLINED" for r in runner.results)
    txt(img, f"success {n_ok}/{len(runner.results)} done" + (f"   declined {n_dec}" if n_dec else ""),
        (12, h - 10), 0.5, (255, 255, 255), 1)
    return img


SOURCE_COLORS = {"gemini": (110, 230, 130), "local gemma": (250, 210, 90), "depth-only": (255, 150, 70)}


def _wrap(s, n):
    out, line = [], ""
    for w in str(s).split():
        if len(line) + len(w) + 1 > n and line:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    return out + ([line] if line else [])


def _rng(v, f="{:.0f}"):
    return f"{f.format(v[0])}-{f.format(v[1])}"


def brain_panel(runner, w=430, h=300):
    """Smart policy: LOOK (depth + vision model guess, source, latency) -> FEEL (touch) -> DECIDE (log).
    Returns None for policies without a camera guess."""
    pol = runner.policy
    g = getattr(pol, "guess", None)
    if runner.finished or not getattr(pol, "uses_camera", False):
        return None
    img = np.full((h, w, 3), 20, np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (80, 80, 95), 1)
    if g is None or runner.guess is None:
        txt(img, "LOOK: waiting for the wrist camera...", (10, 22), 0.48, (200, 200, 210))
        return img
    src = g["source"]
    col = SOURCE_COLORS.get(src, (230, 230, 230))
    if g.get("cached"):
        lat = f"cached ({g['model_latency_s']:.1f} s call)" if g.get("model_latency_s") else "cached"
    elif src == "depth-only":
        lat = "no model"
    else:
        lat = f"{g['latency_s']:.1f} s"
    txt(img, "LOOK", (10, 20), 0.5, (255, 255, 255), 1)
    txt(img, f"source: {src.upper()}" + (f" [{g['model']}]" if g.get("model") else ""), (62, 20), 0.45, col, 1)
    txt(img, lat, (w - 8 - 8 * len(lat), 20), 0.4, (180, 180, 190))
    x0, y = 10, 30
    if getattr(pol, "crop", None) is not None:
        th = cv2.resize(pol.crop, (84, 84), interpolation=cv2.INTER_AREA)
        img[y:y + 84, x0:x0 + 84] = th
        x0 += 92
    d = g["depth"]
    rows = [(f"thinks: {g['object']}", (150, 200, 255)),
            (f"{g['material']} | {_rng(g['mass_g'])} g | fragile {'YES' if g['fragile'] else 'no'}", (220, 220, 225)),
            (f"safe squeeze {_rng(g['max_safe_squeeze_n'])} N | {g['surface']} | mu {_rng(g['friction_guess'], '{:.2f}')}",
             (220, 220, 225)),
            (f"depth: {d['height_mm']} mm tall, {d['width_mm']} mm wide", (170, 175, 185))]
    if src == "depth-only" and g.get("fallback"):
        rows.append(("(" + g["fallback"].replace("gemini: ", "").replace("gemma: ", "")[:44] + ")", (150, 150, 160)))
    for s, c in rows:
        txt(img, s[:42 if x0 > 10 else 52], (x0, y + 13), 0.4, c)
        y += 17
    y = max(y, 30 + 84) + 8
    felt = getattr(pol, "felt", {})
    parts = []
    if "stiffness" in felt:
        parts.append(f"{'soft' if felt.get('soft') else 'firm'} {felt['stiffness']:.0f} N/mm" if felt.get("soft")
                     else "firm (rigid)")
    if "weight_g" in felt:
        parts.append(f"weight {felt['weight_g']:.0f} g")
    if "friction" in felt:
        parts.append(f"friction {felt['friction']:.2f}")
    txt(img, "FEEL", (10, y + 12), 0.5, (255, 255, 255), 1)
    txt(img, " | ".join(parts) if parts else "not touched yet", (62, y + 12), 0.42, (120, 230, 200))
    y += 22
    txt(img, "DECIDE", (10, y + 12), 0.5, (255, 255, 255), 1)
    y += 18
    room = max(1, (h - y - 4) // 15)
    lines = []
    for _, s in reversed(getattr(pol, "log", [])):          # newest entries that fit, never half an entry
        entry = [("- " if i == 0 else "  ") + l for i, l in enumerate(_wrap(s, 58))]
        if len(lines) + len(entry) > room:
            break
        lines = entry + lines
    for l in lines:
        dec = "DECLINE" in l or "needs ~" in l or "still slipping" in l
        txt(img, l, (10, y + 11), 0.38, (255, 170, 120) if dec else (215, 215, 220))
        y += 15
    return img


def _label(img, lines, x, y, col, scale=0.42):
    """Filled label box with one bold first line (like a detector tag)."""
    h = int(16 * scale / 0.42)
    wmax = max(cv2.getTextSize(l, FONT, scale, 2 if i == 0 else 1)[0][0] for i, l in enumerate(lines)) + 8
    H, W = img.shape[:2]
    x = int(np.clip(x, 0, max(0, W - wmax)))
    y = int(np.clip(y, 0, max(0, H - h * len(lines))))
    cv2.rectangle(img, (x, y), (x + wmax, y + h * len(lines) + 3), (15, 15, 18), -1)
    cv2.rectangle(img, (x, y), (x + wmax, y + h * len(lines) + 3), col, 1)
    for i, l in enumerate(lines):
        txt(img, l, (x + 4, y + h * (i + 1) - 3), scale if i == 0 else scale * 0.9,
            col if i == 0 else (225, 225, 230), 2 if i == 0 else 1)


def camera_views(runner, cw, ch):
    """Wrist RGB + depth with the AI detector drawn on top: the box comes from the live depth image
    (perception.measure, same code the robot uses), the labels + % from the vision model (display only)."""
    sim, pol = runner.sim, runner.policy
    rgb = sim.render("wrist_cam", cw, ch).copy()
    depth = sim.render("wrist_cam", cw, ch, depth=True)
    dn = (np.clip(1 - (depth - 0.05) / 0.5, 0, 1) * 255).astype(np.uint8)
    dcol = np.ascontiguousarray(cv2.applyColorMap(dn, cv2.COLORMAP_TURBO)[:, :, ::-1])
    if runner.finished or not getattr(pol, "uses_camera", False):
        return rgb, dcol
    m = perception.measure(rgb, depth, public_slot(runner.slot), sim.read_sensors())
    det = getattr(pol, "detection", None) if runner.guess is not None else None
    g = getattr(pol, "guess", None) if runner.guess is not None else None
    if runner.guess is None:
        col, lines = (230, 230, 230), ["detecting..."]
    elif det:
        col = SOURCE_COLORS["gemini"]
        lines = [f"{lab} {p * 100:.0f}%" for lab, p in det["top3"]]
    elif g and g["source"] != "depth-only":
        col, lines = SOURCE_COLORS.get(g["source"], (230, 230, 230)), [f"{g['object']}", "(no % cached)"]
    else:
        col, lines = SOURCE_COLORS["depth-only"], ["unknown object", "depth only, no model"]
    if m["found"]:
        x, y, w, h = m["bbox"]
        for im in (rgb, dcol):
            cv2.rectangle(im, (x, y), (x + w, y + h), col, 2)
        _label(rgb, lines, x, y - 16 * len(lines) - 4 if y > 16 * len(lines) + 4 else y + h + 2, col)
        _label(dcol, [f"h {m['height_m'] * 1000:.0f} mm  w {m['width_m'] * 1000:.0f} mm"], x, y + h + 2,
               (255, 255, 255), 0.38)
    elif runner.phase not in ("ABOVE", "TARE"):
        _label(rgb, [lines[0], "target between the fingers"], 4, ch - 40, col, 0.38)
    return rgb, dcol


def _cam_block(runner, W, ch):
    cw = (W - 6) // 2
    rgb, dcol = camera_views(runner, cw, ch)
    row = np.full((ch + 8, W, 3), 18, np.uint8)
    row[4:4 + ch, 0:cw] = rgb
    row[4:4 + ch, W - cw:W] = dcol
    txt(row, "wrist camera + AI detector", (6, 20), 0.42, (255, 255, 255))
    txt(row, "wrist depth", (W - cw + 6, 20), 0.42, (255, 255, 255))
    return row


def live_overlay(runner, vw, vh):
    """ONE image for the MuJoCo window (brain + cameras with detector + sensors), scaled to the window,
    drawn on the left. (Several separate images do not all show up in every mujoco version.)
    Picks one column or two columns, whichever gives bigger text."""
    W = 460
    b = brain_panel(runner, w=W, h=300)
    sens = panel(runner, W, 720)
    # two columns: [brain + tall cameras] | [sensors]
    top = b.shape[0] if b is not None else 0
    left = [b] if b is not None else []
    left.append(_cam_block(runner, W, 720 - top - 8))
    two = np.hstack([np.vstack(left), np.full((720, 6, 3), 18, np.uint8), sens])
    # one column: brain / cameras / sensors
    one_parts = ([b] if b is not None else []) + [_cam_block(runner, W, 170), sens]
    s2 = min((vh - 20) / two.shape[0], 0.62 * vw / two.shape[1])
    one_h = sum(p.shape[0] for p in one_parts)
    s1 = min((vh - 20) / one_h, 0.5 * vw / W)
    img, s = (two, s2) if s2 >= s1 else (np.vstack(one_parts), s1)
    if s < 0.3:
        return None
    H, Wd = img.shape[:2]
    out = cv2.resize(img, (int(Wd * s), int(H * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return np.ascontiguousarray(out)


def video_frame(runner, W=1280, H=720):
    sim = runner.sim
    img = np.full((H, W, 3), 18, np.uint8)
    main = sim.render("main", 820, 461)
    img[0:461, 0:820] = main
    txt(img, "BENCH VIEW", (10, 20), 0.5, (255, 255, 255))
    y0 = 461 + 8
    cw, ch = 266, 200
    close = sim.render("close", cw, ch)
    rgb, dcol = camera_views(runner, cw, ch)
    rgb_lab = "wrist camera + AI detector" if getattr(runner.policy, "uses_camera", False) else "wrist camera RGB"
    for k, (im, lab) in enumerate([(close, "close-up"), (rgb, rgb_lab), (dcol, "wrist depth")]):
        x = 8 + k * (cw + 7)
        img[y0:y0 + ch, x:x + cw] = im
        txt(img, lab, (x + 6, y0 + 16), 0.42, (255, 255, 255))
    if not getattr(runner.policy, "uses_camera", False):
        x = 8 + cw + 7
        cv2.rectangle(img, (x, y0 + 22), (x + cw, y0 + 42), (15, 15, 15), -1)
        txt(img, "NOT used by this policy", (x + 6, y0 + 37), 0.45, (255, 180, 80))
    b = brain_panel(runner)
    if b is not None:                         # smart policy: LOOK / FEEL / DECIDE over the bench view
        bh, bw = b.shape[:2]
        roi = img[28:28 + bh, 8:8 + bw]
        img[28:28 + bh, 8:8 + bw] = (0.15 * roi + 0.85 * b).astype(np.uint8)
    img[0:H, 820:W] = panel(runner, W - 820, H)
    return img