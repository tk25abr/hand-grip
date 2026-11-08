"""Does the robot get better with experience?  Runs the 15 objects several times in a row, keeping the
smart policy's memory between passes (each pass is a fresh bench with the same objects), and compares it
with the same policy with the learning layer switched off.

  python learning_bench.py                       # 3 random seeds, 2 passes, learning on vs off
  python learning_bench.py --seeds 5 --passes 3
  python learning_bench.py --stress              # make the wet jar heavy AND fragile: the hard case

What changes between pass 1 and pass 2 is only what the robot itself sensed and remembered
(the squeeze that held an object, the squeeze at which it gave way, its weight). The simulator's
ground truth is used here only to score the outcome, as everywhere else.
"""
import argparse
import contextlib
import io
import time

import numpy as np

import calibrate
import objects
from objects import sample_params, nominal_params
from policies import SmartPolicy
from sim import Sim
from trial import TrialRunner, CTRL_DT

OUTCOMES = ["SUCCESS", "DECLINED", "CRUSHED", "SLIPPED", "DROPPED", "MISSED", "COLLISION"]


def stress_catalog():
    """The wet jar becomes a heavy, fragile, rigid ceramic jar that breaks around 30 N but needs ~42 N to
    hold: the one object where the vision model's cautious guess (~36 N) is RIGHT."""
    for c in objects._CATALOG:
        if c["name"] == "wet glass jar":
            c["break_n"] = (25.0, 30.0, 35.0)
            c["note"] = "STRESS TEST: heavy and fragile ceramic jar"


def run_pass(policy, params, seed):
    cal = calibrate.load()
    sim = Sim(params, calibrate.solref_tcs(params, cal), seed=seed)
    runner = TrialRunner(sim, policy, log_series=False)
    n = round(CTRL_DT / sim.model.opt.timestep)
    with contextlib.redirect_stdout(io.StringIO()):
        while not runner.finished:
            runner.tick()
            sim.step(n)
    return runner.results, sim.time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--nominal", action="store_true", help="nominal object properties instead of random draws")
    args = ap.parse_args()
    if args.stress:
        stress_catalog()
    rows = {}
    t0 = time.time()
    for learn in (False, True):
        for seed in range(args.seeds):
            params = nominal_params() if args.nominal else sample_params(2000 + seed)
            policy = SmartPolicy(learn=learn)                # one policy = one memory, kept across the passes
            for p in range(args.passes):
                res, sim_t = run_pass(policy, params, seed)
                row = rows.setdefault((learn, p), dict(n=0, t=0.0, c={o: 0 for o in OUTCOMES}, res=[]))
                row["n"] += 1
                row["t"] += sim_t
                row["res"] += res
                for r in res:
                    row["c"][r["result"]] += 1
        print(f"  {'learning ON ' if learn else 'learning OFF'} done ({time.time() - t0:.0f} s)", flush=True)
    print()
    title = "STRESS (wet jar = heavy + fragile)" if args.stress else "random objects"
    print(f"{title}: {args.seeds} seeds x 15 objects, memory kept across {args.passes} passes\n")
    print("| learning | pass | picked up / 15 | refused | crushed | slipped or dropped | sim seconds per run |")
    print("|---|---|---|---|---|---|---|")
    for (learn, p), r in sorted(rows.items()):
        n = r["n"]
        c = r["c"]
        print(f"| {'on' if learn else 'off'} | {p + 1} | {c['SUCCESS'] / n:.1f} | {c['DECLINED'] / n:.1f} | "
              f"{c['CRUSHED'] / n:.2f} | {(c['SLIPPED'] + c['DROPPED']) / n:.2f} | {r['t'] / n:.0f} |")
    print()
    for learn in (True,):
        for p in range(args.passes):
            bad = [(x["name"], x["result"]) for x in rows[(learn, p)]["res"] if x["result"] in ("CRUSHED",)]
            if bad:
                names = {}
                for nm, _ in bad:
                    names[nm] = names.get(nm, 0) + 1
                print(f"learning on, pass {p + 1}: crushed {names}")


if __name__ == "__main__":
    main()
