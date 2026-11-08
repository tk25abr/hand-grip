"""Stiffness calibration: a virtual compression test for every object, in the real gripper scene.

MuJoCo contacts are 'soft constraints' whose stiffness depends on the contact time constant AND on
the masses involved, so one setting cannot give every object its real N/mm. For each object we:
  1. squeeze it between the pads at 5 N and at 25 N for a range of contact time constants,
  2. read the change in finger gap (the robot's own encoders, noise-free here),
  3. stiffness k = 20 N / change in gap (mm)  -> a curve k(time constant) per object,
then for any target stiffness (nominal or sampled) we interpolate the time constant.
Limit: very stiff light objects (egg ~140 N/mm) top out at what the solver allows at a 1 ms step;
that cap is reported. Below the encoder noise (0.2 mm) the difference is not measurable anyway.

  python calibrate.py            -> calibration.json + results/compression_test.png + table
"""
import json
import os
import sys

import numpy as np
import mujoco

from objects import nominal_params, slots, TABLE_Z, _CATALOG
from scene import HOME, PAD_Z, PAD_HALF, FINGER_Y0, FINGER_HALF, TC_MIN, TC_DEFAULT

HERE = os.path.dirname(os.path.abspath(__file__))
CAL_PATH = os.path.join(HERE, "calibration.json")
GRID = [0.002 * (2 ** (i / 2)) for i in range(11)]      # 2 ms ... 64 ms
INNER0 = 2 * (FINGER_Y0 - FINGER_HALF[1] - 2 * PAD_HALF[1])


def _true_gap(sim):
    m, d = sim.model, sim.data
    return INNER0 - d.qpos[m.joint("j_l").qposadr[0]] - d.qpos[m.joint("j_r").qposadr[0]]


def _place_hand(sim, s):
    """Teleport the carriage so the pads straddle the object at mid-height (a lab compression rig)."""
    hz = TABLE_Z + max(s["h"] / 2, PAD_HALF[2] + 0.002) - PAD_Z
    tgt = np.array([s["x"], s["y"], hz]) - np.array(HOME)
    m, d = sim.model, sim.data
    for j, v in zip(("gx", "gy", "gz"), tgt):
        d.qpos[m.joint(j).qposadr[0]] = v
    d.ctrl[0:3] = tgt
    mujoco.mj_forward(m, d)


def squeeze_curve(idx, tc, forces=(5.0, 25.0), hold_s=1.0):
    """Returns finger gaps (m) at each squeeze force for object idx with contact time constant tc."""
    from sim import Sim
    tcs = [TC_DEFAULT] * len(_CATALOG)
    tcs[idx] = tc
    sim = Sim(nominal_params(), tcs)
    s = sim.slots[idx]
    sim.grip(-10)
    _place_hand(sim, s)
    sim.step(300)
    sim.grip(8.0)                       # close until the pads touch (5 N alone closes too slowly)
    for _ in range(40):
        sim.step(50)
        if all(sim.pad_contacts()[k][2] > 0 for k in ("l", "r")):
            break
    gaps = []
    for f in forces:
        sim.grip(f)
        sim.step(int(hold_s / sim.model.opt.timestep))
        gaps.append(_true_gap(sim))
    ok = np.all(np.isfinite(sim.data.qpos)) and all(sim.pad_contacts()[k][2] > 0 for k in ("l", "r"))
    return np.array(gaps), ok


def measure_k(idx, tc):
    g, ok = squeeze_curve(idx, tc)
    dg = (g[0] - g[1]) * 1000
    return (20.0 / dg if dg > 1e-4 else 1e4) if ok else None


def calibrate_object(idx):
    s = slots()[idx]
    if s["width"] > INNER0 - 0.004:
        return s["name"], dict(skipped="wider than the hand opens - not graspable, default softness")
    ks = []
    for tc in GRID:
        k = measure_k(idx, tc)
        ks.append(k)
    pts = [(tc, k) for tc, k in zip(GRID, ks) if k is not None and k > 0]
    return s["name"], dict(tc=[p[0] for p in pts], k=[p[1] for p in pts])


def load():
    if not os.path.exists(CAL_PATH):
        return None
    with open(CAL_PATH) as f:
        return json.load(f)


def solref_tcs(params, cal=None):
    """Contact time constant per object so that each has its (sampled) real stiffness."""
    cal = cal or load()
    out, capped = [], []
    for c, p in zip(_CATALOG, params):
        entry = (cal or {}).get(c["name"], {})
        if "tc" not in entry or p["stiff"] >= 1000:
            out.append(TC_MIN if p["stiff"] >= 1000 else TC_DEFAULT)
            continue
        tc, k = np.array(entry["tc"]), np.array(entry["k"])
        order = np.argsort(k)
        lk, ltc = np.log(k[order]), np.log(tc[order])
        target = np.log(p["stiff"])
        if target >= lk[-1]:
            capped.append(c["name"])
        v = float(np.exp(np.interp(target, lk, ltc)))
        out.append(float(np.clip(v, TC_MIN, GRID[-1])))
    return out


def compression_plot(cal, path):
    """Simulated force-deformation vs published data for the objects that have real measurements."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sim import Sim
    names = {c["name"]: i for i, c in enumerate(_CATALOG)}
    refs = {
        "egg": dict(points=([12.03, 15.26, 20.61, 25.83, 30.91, 49.54], [0.100, 0.127, 0.133, 0.187, 0.222, 0.347]),
                    label="egg: measured (Mendelu 2004)"),
        "tomato": dict(band=(3, 9), label="tomato: 3-9 N/mm (Ruiz Altisent)"),
        "tennis ball": dict(band=(10.9, 14.3), label="tennis ball: ITF 5.59-7.37 mm @ 80 N"),
        "strawberry": dict(band=(0.8, 3.0), label="strawberry: rough range (puncture data)"),
    }
    params = nominal_params()
    tcs = solref_tcs(params, cal)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    forces = np.linspace(2, 40, 12)
    for ax, (nm, ref) in zip(axes, refs.items()):
        idx = names[nm]
        sim_tcs = [TC_DEFAULT] * len(_CATALOG)
        sim_tcs[idx] = tcs[idx]
        sim = Sim(params, sim_tcs)
        s = sim.slots[idx]
        sim.grip(-10)
        _place_hand(sim, s)
        sim.step(300)
        sim.grip(8.0)
        for _ in range(40):
            sim.step(50)
            if all(sim.pad_contacts()[k][2] > 0 for k in ("l", "r")):
                break
        sim.grip(forces[0]); sim.step(800)
        g0 = _true_gap(sim)
        defl = []
        fmax = min(40, s["max_force"] * 1.5)
        fs = forces[forces <= fmax]
        for f in fs:
            sim.grip(f); sim.step(500)
            defl.append((g0 - _true_gap(sim)) * 1000)
        ax.plot(fs - forces[0], defl, "o-", color="#1f6feb", label="simulation (finger encoders)")
        xs = np.linspace(0, fs[-1] - forces[0], 20)
        if "band" in ref:
            lo, hi = ref["band"]
            ax.fill_between(xs, xs / hi, xs / lo, color="#f0a020", alpha=0.3, label=ref["label"])
        else:
            F, d = ref["points"]
            ax.plot(np.array(F), d, "s", color="#d03030", label=ref["label"])
            ax.set_xlim(0, 50)
        ax.set_title(f"{nm}: target {params[idx]['stiff']:g} N/mm")
        ax.set_xlabel("squeeze force (N)")
        ax.set_ylabel("deformation (mm)")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle("Virtual compression test vs published measurements", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    from multiprocessing import Pool
    with Pool(max(1, os.cpu_count() or 1)) as pool:
        res = pool.map(calibrate_object, range(len(_CATALOG)))
    cal = dict(res)
    with open(CAL_PATH, "w") as f:
        json.dump(cal, f, indent=1)
    params = nominal_params()
    tcs = solref_tcs(params, cal)
    print(f"\n{'object':18s} {'target N/mm':>12s} {'max possible':>13s}  time-const")
    for c, p, tc in zip(_CATALOG, params, tcs):
        e = cal.get(c["name"], {})
        kmax = max(e["k"]) if "k" in e else float("nan")
        tgt = "rigid" if p["stiff"] >= 1000 else f"{p['stiff']:g}"
        flag = "  (capped)" if "k" in e and p["stiff"] < 1000 and p["stiff"] > kmax else ""
        print(f"{c['name']:18s} {tgt:>12s} {kmax:13.1f}  {tc*1000:6.1f} ms{flag}" + ("  skipped" if "skipped" in e else ""))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    compression_plot(cal, os.path.join(HERE, "results", "compression_test.png"))
    print("saved calibration.json, results/compression_test.png")


if __name__ == "__main__":
    main()
