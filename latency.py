"""How long does each part take?  Runs the 15 objects once with the smart policy and times every part of the
robot's own computation on THIS computer (CPU only). For the PDF's "where does each component run" table.

  python latency.py

Timed: depth -> object (perception.measure), first guess (vlm.first_guess: from the offline cache, the
depth-only prior, or a live Gemini call if GEMINI_API_KEY is set), the camera render itself (simulation, not
part of the robot), and the policy's decision inside the 10 ms control loop (adjust).
"""
import contextlib
import io
import statistics as st
import time

import calibrate
import perception
import vlm
from objects import nominal_params
from policies import SmartPolicy
from sim import Sim
from trial import TrialRunner, CTRL_DT

T = {"depth to object (perception.measure)": [], "first guess (vlm.first_guess)": [],
     "control-loop decision (policy.adjust, every 10 ms)": [], "wrist camera render (simulation)": []}


def timed(key, fn):
    def wrap(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            T[key].append(time.perf_counter() - t0)
    return wrap


def main():
    perception.measure = timed("depth to object (perception.measure)", perception.measure)
    vlm.first_guess = timed("first guess (vlm.first_guess)", vlm.first_guess)
    params = nominal_params()
    sim = Sim(params, calibrate.solref_tcs(params, calibrate.load()))
    sim.read_wrist_camera = timed("wrist camera render (simulation)", sim.read_wrist_camera)
    policy = SmartPolicy()
    policy.adjust = timed("control-loop decision (policy.adjust, every 10 ms)", policy.adjust)
    runner = TrialRunner(sim, policy, log_series=False)
    n = round(CTRL_DT / sim.model.opt.timestep)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        while not runner.finished:
            runner.tick()
            sim.step(n)
    print(f"15 objects, {sim.time:.0f} s of robot time, {time.time() - t0:.0f} s on this computer\n")
    print("| part | calls | mean | max |")
    print("|---|---|---|---|")
    for k, v in T.items():
        if v:
            print(f"| {k} | {len(v)} | {st.mean(v) * 1000:.2f} ms | {max(v) * 1000:.2f} ms |")
    print("\nThe control loop has a 10 ms budget per tick; the decision must stay far below it.")


if __name__ == "__main__":
    main()
