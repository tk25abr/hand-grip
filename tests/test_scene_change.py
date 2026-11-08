"""Mid-run scene change: the held object gets 2.5x heavier and the robot is not told. It must notice it
through the wrist load and either re-grip (holds on) or carry it back down - never drop or crush it.
The bench restores the object's mass afterwards."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import calibrate                                   # noqa: E402
from objects import nominal_params                 # noqa: E402
from policies import SmartPolicy                   # noqa: E402
from sim import Sim                                # noqa: E402
from trial import TrialRunner, CTRL_DT             # noqa: E402


def test_load_change_regrip_or_safe_putdown(monkeypatch):
    for k in ("GEMINI_API_KEY", "GEMMA_URL", "OLLAMA_HOST"):
        monkeypatch.delenv(k, raising=False)
    p = nominal_params()
    sim = Sim(p, calibrate.solref_tcs(p, calibrate.load()))
    names = ["tomato soup can", "steel weight 3kg"]
    order = [s["idx"] for s in sim.slots if s["name"] in names]
    mass0 = {i: sim.slots[i]["mass"] for i in order}
    pol = SmartPolicy(use_cache=True)            # cached Gemini answers in results/, works offline
    runner = TrialRunner(sim, pol, order=order, log_series=False)
    runner.change = dict(factor=2.5, names=None, at=0.4)
    n = round(CTRL_DT / sim.model.opt.timestep)
    seen = []
    while not runner.finished:
        runner.tick()
        sim.step(n)
        seen += [m for _, m in pol.log if "load rose" in m and m not in seen]
    res = {r["name"]: r["result"] for r in runner.results}
    assert set(res.values()) <= {"SUCCESS", "DECLINED"}, res          # never dropped, slipped or crushed
    assert len(seen) == 2, seen                                       # noticed the change both times
    assert all(abs(sim.slots[i]["mass"] - mass0[i]) < 1e-12 for i in order)   # bench restored
