"""Grip Lab — run the 15-object pick test.

  Watch it live (MuJoCo window with the sensor panel inside it):
      macOS:   mjpython run.py
      Linux/Windows:  python run.py
  Record a video (no window):
      python run.py --mode record --out baseline.mp4          (Linux servers: MUJOCO_GL=egl)
  Full dashboard live in a browser window, no MuJoCo window (plain python, any OS):
      python run.py --mode web --policy smart
  Just the numbers, fast:
      python run.py --mode bench

  Many runs with object properties drawn from their real-world ranges (objects_sources.md):
      python run.py --mode bench --runs 30

  Smart policy (wrist camera + Gemini first guess + touch/weight feedback), any mode:
      export GEMINI_API_KEY=...          (without it: local Gemma if GEMMA_URL is set, else depth-only)
      python run.py --mode bench --policy smart
      python run.py --mode record --policy smart --out smart.mp4
      mjpython run.py --policy smart

  Options:  --policy baseline|smart
  
            --force 40      squeeze force of the baseline gripper (N)
            --model NAME    Gemini model for the smart policy (default gemini-3.8-flash)
            --objects "egg,tomato soup can,smartphone"   run only some objects
            --sample 7      use randomly drawn object properties (seed 7) instead of nominal ones

  Keys in the live window:
      N  skip to next object      P  pause / resume      R  restart all 15
      U  baseline force +5 N      J  baseline force -5 N      H  panels inside the 3D window on/off
  The full dashboard opens in a browser window next to it (http://localhost:8765, --no-web to skip).
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from sim import Sim
from trial import TrialRunner, CTRL_DT, OUTCOMES
from policies import BaselinePolicy, SmartPolicy
from objects import nominal_params, sample_params
import calibrate

HERE = os.path.dirname(os.path.abspath(__file__))


def make_policy(args):
    if args.policy == "smart":
        return SmartPolicy(use_cache=not args.no_cache, learn=not getattr(args, "no_learn", False))
    return BaselinePolicy(force=args.force)


def tag(args):
    return "smart" if args.policy == "smart" else "baseline"


def make(args, seed=None):
    if seed is None and getattr(args, "sample", None) is not None:
        seed = args.sample
    params = nominal_params() if seed is None else sample_params(seed)
    cal = calibrate.load()
    if cal is None:
        print("No calibration.json - run `python calibrate.py` first for real object stiffness (using defaults now)")
    sim = Sim(params, calibrate.solref_tcs(params, cal), seed=seed or 0)
    order = None
    if args.objects:
        want = [w.strip().lower() for w in args.objects.split(",")]
        order = [s["idx"] for s in sim.slots if s["name"].lower() in want]
    return sim, TrialRunner(sim, make_policy(args), order=order)


def save_results(runner, tag):
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    keys = ["idx", "name", "result", "why", "grip_n", "peak_n", "weight_g", "true_g",
            "aperture_mm", "true_width_mm", "lift_cm", "slip_mm", "label", "source", "micro_weight_g"]
    with open(os.path.join(out, f"{tag}_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in runner.results:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k in keys})
    with open(os.path.join(out, f"{tag}_sensor_log.csv"), "w", newline="") as f:
        w = None
        for idx, ser in runner.all_series.items():
            for row in ser:
                row = dict(object=runner.sim.slots[idx]["name"], **row)
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    w.writeheader()
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    lines = [f"# {runner.policy.name}", "",
             "| # | object | result | why | grip N | peak N | weight measured / true (g) | width by touch / true (mm) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in runner.results:
        wm = f"{r['weight_g']:.0f}" if r["weight_g"] else "-"
        ap = f"{r['aperture_mm']:.0f}" if r["aperture_mm"] is not None else "-"
        lines.append(f"| {r['idx']+1} | {r['name']} | **{r['result']}** | {r['why']} | {r['grip_n']:.0f} | "
                     f"{r['peak_n']:.0f} | {wm} / {r['true_g']:.0f} | {ap} / {r['true_width_mm']:.0f} |")
    n_ok = sum(r["result"] == "SUCCESS" for r in runner.results)
    counts = {}
    for r in runner.results:
        counts[r["result"]] = counts.get(r["result"], 0) + 1
    lines += ["", f"**Success: {n_ok}/{len(runner.results)}** — " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))]
    with open(os.path.join(out, f"{tag}_results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSuccess {n_ok}/{len(runner.results)}  {counts}")
    print(f"saved: results/{tag}_results.csv, results/{tag}_results.md, results/{tag}_sensor_log.csv")


def _one_run(job):
    args, seed = job
    sim, runner = make(args, seed)
    runner.log_series = False
    n = round(CTRL_DT / sim.model.opt.timestep)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        while not runner.finished:
            runner.tick()
            sim.step(n)
    return [dict(name=r["name"], result=r["result"], source=r.get("source"), why=r["why"]) for r in runner.results]


def warm_cache(args):
    """Smart policy: ask the vision model once per object up front (one process), so the parallel runs
    below all read the same cached answers instead of racing to make 15 x N calls."""
    from guess_all import look_all
    params = nominal_params()
    sim = Sim(params, calibrate.solref_tcs(params, calibrate.load()))
    res = look_all(sim, make_policy(args))
    srcs = {}
    for _, g in res:
        srcs[g["source"]] = srcs.get(g["source"], 0) + 1
    print(f"first guesses ready: {srcs}", flush=True)


def monte_carlo(args):
    from multiprocessing import get_context
    t0 = time.time()
    if args.policy == "smart":
        warm_cache(args)
    jobs = [(args, 1000 + i) for i in range(args.runs)]
    # spawn, not fork: the parent may already hold an OpenGL/EGL context (camera rendering in
    # warm_cache), and forked children deadlock on it on Linux
    with get_context("spawn").Pool(max(1, os.cpu_count() or 1)) as pool:
        runs = []
        for i, r in enumerate(pool.imap_unordered(_one_run, jobs), 1):
            runs.append(r)
            print(f"  run {i}/{args.runs} done ({time.time() - t0:.0f} s)", flush=True)
    names = [r["name"] for r in runs[0]]
    outcomes = OUTCOMES
    per = {n: {o: 0 for o in outcomes} for n in names}
    totals = []
    sources = {}
    for run in runs:
        for r in run:
            per[r["name"]][r["result"]] += 1
            if r.get("source"):
                sources[r["source"]] = sources.get(r["source"], 0) + 1
        totals.append(sum(r["result"] == "SUCCESS" for r in run))
    N = len(runs)
    src_line = [f"First-guess source: " + ", ".join(f"{k} {v}" for k, v in sources.items()), ""] if sources else []
    lines = [f"# {make_policy(args).name}", "", *src_line,
             f"{N} runs. Each run draws every object's mass, friction, damage force and stiffness from its "
             f"real-world range (see objects_sources.md).", "",
             f"**Objects picked up successfully per run: {np.mean(totals):.1f} / {len(names)}  "
             f"(sd {np.std(totals):.1f}, min {min(totals)}, max {max(totals)})**", "",
             "| object | success | " + " | ".join(o.lower() for o in outcomes[1:]) + " |",
             "|---|---|" + "---|" * (len(outcomes) - 1)]
    for n in names:
        c = per[n]
        lines.append(f"| {n} | **{100 * c['SUCCESS'] / N:.0f}%** | " +
                     " | ".join(f"{100 * c[o] / N:.0f}%" if c[o] else "" for o in outcomes[1:]) + " |")
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    name = "smart_montecarlo" if args.policy == "smart" else f"baseline_{args.force:.0f}N_montecarlo"
    with open(os.path.join(out, f"{name}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out, f"{name}_runs.json"), "w") as f:
        json.dump(dict(policy=args.policy, force=args.force, seeds=[1000 + i for i in range(args.runs)],
                       runs=runs), f)
    print("\n".join(lines))
    print(f"\nsaved results/{name}.md, results/{name}_runs.json  ({time.time() - t0:.0f} s)")


def bench(args):
    if args.runs > 1:
        return monte_carlo(args)
    sim, runner = make(args)
    n = round(CTRL_DT / sim.model.opt.timestep)
    t0 = time.time()
    while not runner.finished:
        runner.tick()
        sim.step(n)
    print(f"(sim {sim.time:.0f} s, wall {time.time() - t0:.0f} s)")
    save_results(runner, tag(args))


def record(args):
    import imageio.v2 as imageio
    from dashboard import video_frame
    sim, runner = make(args)
    n = round(CTRL_DT / sim.model.opt.timestep)
    fps = 30
    every = max(1, round(1 / fps / CTRL_DT))
    writer = imageio.get_writer(args.out, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    tick = 0
    tail = 0
    while tail < 60:
        runner.tick()
        sim.step(n)
        if tick % every == 0:
            writer.append_data(video_frame(runner))
            if runner.finished:
                tail += 1
        tick += 1
    writer.close()
    save_results(runner, tag(args))
    print(f"video: {os.path.abspath(args.out)}")


def interactive(args):
    import mujoco
    import mujoco.viewer
    from dashboard import live_overlay

    sim, runner = make(args)
    n = round(CTRL_DT / sim.model.opt.timestep)
    state = dict(paused=False, skip=False, restart=False, dforce=0.0, overlay=False)
    web = None
    if not args.no_web:
        from live_web import LiveDashboard
        from dashboard import video_frame
        web = LiveDashboard(args.port)

    def on_key(key):
        k = chr(key) if 32 <= key < 127 else ""
        if k == "P":
            state["paused"] = not state["paused"]
        elif k == "N":
            state["skip"] = True
        elif k == "R":
            state["restart"] = True
        elif k == "U":
            state["dforce"] += 5
        elif k == "J":
            state["dforce"] -= 5
        elif k == "H":
            state["overlay"] = not state["overlay"]

    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key,
                                      show_left_ui=False, show_right_ui=False) as v:
        v.cam.lookat[:] = [0.0, 0.05, 0.5]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.9, 90, -32
        has_images = hasattr(v, "set_images")
        if not has_images:
            print("Your mujoco is too old for the in-window panel: pip install -U mujoco")
        frame = 0
        while v.is_running():
            t0 = time.time()
            with v.lock():
                if state["restart"]:
                    state["restart"] = False
                    old_force = getattr(runner.policy, "force", None)
                    sim2, runner2 = make(args)
                    if old_force is not None:
                        runner2.policy.force = old_force
                    # keep the same model/data objects the viewer is showing
                    sim.data.qpos[:] = sim2.data.qpos
                    sim.data.qvel[:] = 0
                    sim.crushed.clear()
                    for i, sl in enumerate(sim.slots):
                        sim.reset_object(i)
                        sim.peak_force[i] = 0.0
                        sim.model.geom_rgba[sim.obj_geom[i]] = [float(c) for c in sl["rgba"].split()]
                    runner = TrialRunner(sim, runner2.policy, order=runner2.order)
                if state["dforce"] and not hasattr(runner.policy, "force"):
                    state["dforce"] = 0.0                   # U/J only change the baseline's fixed force
                if state["dforce"]:
                    runner.policy.force = float(np.clip(runner.policy.force + state["dforce"], 5, 120))
                    print(f"baseline force now {runner.policy.force:.0f} N (applies from the next grasp)")
                    state["dforce"] = 0.0
                if state["skip"] and not runner.finished:
                    state["skip"] = False
                    sim.grip(-10)
                    sim.reset_object(runner.slot["idx"])
                    runner._next_trial()
                if not state["paused"]:
                    for _ in range(2):                      # 2 control ticks per frame -> ~real time at 50 fps
                        runner.tick()
                        sim.step(n)
            if has_images and frame % 3 == 0:
                vp = v.viewport                             # one image: brain + cameras/detector + sensors
                ov = live_overlay(runner, vp.width, vp.height) if state["overlay"] else None
                if ov is None:
                    v.clear_images()
                else:
                    oh, ow = ov.shape[:2]
                    v.set_images([(mujoco.MjrRect(10, vp.height - oh - 10, ow, oh), ov)])
            if web is not None and frame % 2 == 0:       # full dashboard (same frame as the video) -> browser
                web.publish(video_frame(runner))
            v.sync()
            frame += 1
            time.sleep(max(0.0, 0.02 - (time.time() - t0)))
    if web is not None:
        web.close()
    if runner.results:
        save_results(runner, tag(args))


def web_only(args):
    """No MuJoCo window: run the trials and show the full dashboard in a browser window (plain python)."""
    from live_web import LiveDashboard
    from dashboard import video_frame
    sim, runner = make(args)
    n = round(CTRL_DT / sim.model.opt.timestep)
    web = LiveDashboard(args.port)
    t_start, tail = time.time(), 0
    while tail < 300:
        for _ in range(3):                                  # 30 ms of sim per frame
            runner.tick()
            sim.step(n)
        web.publish(video_frame(runner))
        ahead = sim.time - (time.time() - t_start)          # never faster than real time
        if ahead > 0:
            time.sleep(ahead)
        if runner.finished:
            tail += 1
    save_results(runner, tag(args))
    print("done - the browser window keeps the last frame")
    time.sleep(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["interactive", "record", "bench", "web"], default="interactive")
    ap.add_argument("--policy", choices=["baseline", "smart"], default="baseline")
    ap.add_argument("--force", type=float, default=40.0, help="baseline: fixed squeeze force (N)")
    ap.add_argument("--objects", help="comma-separated object names")
    ap.add_argument("--out", help="record: video file (default baseline.mp4 / smart.mp4)")
    ap.add_argument("--runs", type=int, default=1, help="bench: number of runs with sampled object properties")
    ap.add_argument("--sample", type=int, help="draw object properties with this seed instead of nominal")
    ap.add_argument("--model", help="smart: Gemini model name (default $GEMINI_MODEL or gemini-3.8-flash)")
    ap.add_argument("--no-cache", action="store_true", help="smart: ignore results/gemini_cache.json")
    ap.add_argument("--no-learn", action="store_true", help="smart: switch off the learning layer (yield sensing, limit challenge, memory)")
    ap.add_argument("--no-web", action="store_true", help="interactive: no live dashboard in the browser")
    ap.add_argument("--port", type=int, default=8765, help="live dashboard port (browser window)")
    args = ap.parse_args()
    if args.model:
        os.environ["GEMINI_MODEL"] = args.model          # read by vlm.py, inherited by bench processes
    args.out = args.out or f"{tag(args)}.mp4"
    {"interactive": interactive, "record": record, "bench": bench, "web": web_only}[args.mode](args)