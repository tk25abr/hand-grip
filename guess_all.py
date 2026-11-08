"""Milestone check: the first guess for all 15 objects, before touching anything.

The gantry visits each slot exactly like a real trial (ABOVE -> TARE), the smart policy measures the
object in the wrist depth image and asks the vision model once (Gemini -> local Gemma -> depth-only).
Running this once also fills results/gemini_cache.json, so a live demo afterwards has no API wait.

  python guess_all.py                         # all 15
  python guess_all.py --objects "egg,smartphone"
  python guess_all.py --no-cache              # force fresh model calls

The "check" line shows ground truth for the reader only; it is never given to the policy.
"""
import argparse
import json
import os
import time

import numpy as np

import calibrate
from objects import nominal_params
from policies import SmartPolicy
from sim import Sim
from trial import TrialRunner, CTRL_DT

HERE = os.path.dirname(os.path.abspath(__file__))


def look_all(sim, policy, order=None, on_guess=None):
    """Drive the gantry above each slot like a trial and call the runner's own TARE perception step."""
    runner = TrialRunner(sim, policy, order=order, log_series=False)
    n = round(CTRL_DT / sim.model.opt.timestep)
    out = []
    while not runner.finished:
        if runner.phase == "TARE" and sim.time - runner.phase_t0 > 0.3:
            g = runner.perceive(sim.read_sensors())
            out.append((runner.slot["idx"], g))
            if on_guess:
                on_guess(runner.slot, g)
            sim.reset_object(runner.slot["idx"])
            runner._next_trial()
            continue
        runner.tick()
        sim.step(n)
    return out


def _r(v, f="{:.0f}"):
    return f"{f.format(v[0])}-{f.format(v[1])}"


def show(slot, g):
    d = g["depth"]
    src = g["source"] + (f" [{g['model']}]" if g.get("model") else "")
    lat = f"cache hit ({g['latency_s']:.2f} s; model call took {g['model_latency_s']:.1f} s)" if g["cached"] \
        else f"{g['latency_s']:.2f} s"
    print(f"[{slot['idx'] + 1:2d}] depth: h {d['height_mm']} mm, w {d['width_mm']} mm, l {d['length_mm']} mm"
          f"{' (cut off)' if d['cut_off'] else ''}, {d['shape']}")
    print(f"     source: {src}, {lat}" + (f"   skipped: {g['fallback']}" if g.get("fallback") else ""))
    print(f"     guess: {g['object']} | {g['material']} | {_r(g['mass_g'])} g | fragile {'yes' if g['fragile'] else 'no'}"
          f" | safe squeeze {_r(g['max_safe_squeeze_n'])} N | {g['surface']} | mu {_r(g['friction_guess'], '{:.2f}')}"
          f" | graspable {'yes' if g['graspable_by_parallel_jaw_14cm'] else 'NO'}")
    print(f"     notes: {g['notes']}")
    brk = "never" if slot["max_force"] >= 5000 else f"{slot['max_force']:.0f} N"
    print(f"     check (truth, not given to the robot): {slot['name']}, {slot['mass'] * 1000:.0f} g, "
          f"h {slot['h'] * 1000:.0f} w {slot['width'] * 1000:.0f} mm, breaks {brk}, mu {slot['mu']:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", help="comma-separated object names")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    params = nominal_params()
    sim = Sim(params, calibrate.solref_tcs(params, calibrate.load()))
    order = None
    if args.objects:
        want = [w.strip().lower() for w in args.objects.split(",")]
        order = [s["idx"] for s in sim.slots if s["name"].lower() in want]
    print(f"GEMINI_API_KEY {'set' if os.environ.get('GEMINI_API_KEY') else 'NOT set'} | "
          f"Gemma endpoint: {os.environ.get('GEMMA_URL') or os.environ.get('OLLAMA_HOST') or 'not configured'}\n")
    t0 = time.time()
    res = look_all(sim, SmartPolicy(use_cache=not args.no_cache), order, on_guess=show)
    srcs = {}
    for _, g in res:
        srcs[g["source"]] = srcs.get(g["source"], 0) + 1
    print(f"\n{len(res)} objects, sources: {srcs}, wall {time.time() - t0:.0f} s")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "first_guess.json"), "w") as f:
        json.dump([dict(idx=i, **{k: v for k, v in g.items()}) for i, g in res], f, indent=1, default=str)
    print("saved results/first_guess.json")


if __name__ == "__main__":
    main()
