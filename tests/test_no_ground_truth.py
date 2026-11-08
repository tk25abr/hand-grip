"""Hard rule: the smart policy may only use public_slot(slot) (taught x/y), the sensors dict from
sim.read_sensors() and the wrist camera image. It must NEVER read sim.slots ground truth
(mass, mu, max_force, stiff, name, ...). This test fails if it does.

Two checks:
  1. static  - the policy's source files do not import the simulator or touch ground-truth fields
  2. dynamic - every sim.slots entry is replaced by a guard that raises if it is read while any
               policy method is on the call stack; then real trials are run.
A deliberately cheating policy is used to prove the guard actually catches cheating.

  python -m pytest tests/ -q
"""
import ast
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import calibrate                                   # noqa: E402
from objects import nominal_params                 # noqa: E402
from policies import SmartPolicy                   # noqa: E402
from sim import Sim                                # noqa: E402
from trial import TrialRunner, CTRL_DT             # noqa: E402
from guess_all import look_all                     # noqa: E402

POLICY_FILES = ["policies.py", "perception.py", "vlm.py"]
ALLOWED_SLOT_KEYS = {"idx", "x", "y"}
# ("shape" is also a key of the robot's own depth measurement, so the static check skips it;
#  the dynamic guard below still catches slot["shape"])
TRUTH_KEYS = {"mass", "mu", "max_force", "stiff", "name", "size", "rgba", "note", "h", "width"}
FORBIDDEN_MODULES = {"sim", "mujoco", "calibrate", "trial", "run", "dashboard", "gc", "inspect", "ctypes"}
FORBIDDEN_ATTRS = {"slots", "truth_obj_pos", "crushed", "peak_force", "_CATALOG", "qpos", "xpos",
                   "sensordata", "geom_rgba", "_f_filt"}
OBJECTS_ALLOWED = {"TABLE_Z", "public_slot"}


# --------------------------------------------------------------------------------------------- static
@pytest.mark.parametrize("fname", POLICY_FILES)
def test_policy_source_has_no_ground_truth_access(fname):
    tree = ast.parse(open(os.path.join(ROOT, fname)).read(), fname)
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in FORBIDDEN_MODULES or a.name == "objects":
                    problems.append(f"line {node.lineno}: import {a.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in FORBIDDEN_MODULES:
                problems.append(f"line {node.lineno}: from {node.module} import ...")
            if mod == "objects":
                bad = [a.name for a in node.names if a.name not in OBJECTS_ALLOWED]
                if bad:
                    problems.append(f"line {node.lineno}: from objects import {bad}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            problems.append(f"line {node.lineno}: .{node.attr}")
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                and node.slice.value in TRUTH_KEYS:
            problems.append(f"line {node.lineno}: [{node.slice.value!r}]")
    assert not problems, f"{fname} touches ground truth:\n  " + "\n  ".join(problems)


# --------------------------------------------------------------------------------------------- dynamic
class GuardedSlot(dict):
    """A sim.slots entry that raises if a ground-truth key is read from inside a policy call."""
    policy = None
    violations = []

    def _check(self, key):
        if key in ALLOWED_SLOT_KEYS or GuardedSlot.policy is None:
            return
        f = sys._getframe(1)
        while f is not None:
            if f.f_locals.get("self") is GuardedSlot.policy:
                GuardedSlot.violations.append((key, f.f_code.co_name))
                raise AssertionError(f"policy read ground truth slot[{key!r}] in {f.f_code.co_name}()")
            f = f.f_back

    def __getitem__(self, k):
        self._check(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        self._check(k)
        return super().get(k, default)

    def items(self):
        self._check("<items>")
        return super().items()

    def values(self):
        self._check("<values>")
        return super().values()

    def __iter__(self):
        self._check("<iter>")
        return super().__iter__()

    def copy(self):
        self._check("<copy>")
        return super().copy()


class ArgSpy:
    """Wraps a policy: records exactly what the runner hands it."""

    def __init__(self, inner):
        self.inner, self.slot_keys, self.sensor_keys = inner, set(), set()

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def _note(self, slot, sensors):
        self.slot_keys |= set(dict.keys(slot))
        if sensors is not None:
            self.sensor_keys |= set(sensors.keys())

    def identify(self, slot, rgb=None, depth=None, sensors=None):
        self._note(slot, sensors)
        return self.inner.identify(slot, rgb, depth, sensors)

    def plan(self, slot, sensors):
        self._note(slot, sensors)
        return self.inner.plan(slot, sensors)

    def adjust(self, phase, slot, sensors):
        self._note(slot, sensors)
        return self.inner.adjust(phase, slot, sensors)


def _guarded_sim():
    p = nominal_params()
    sim = Sim(p, calibrate.solref_tcs(p, calibrate.load()))
    sim.slots = [GuardedSlot(s) for s in sim.slots]
    return sim


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Ground-truth access does not depend on which model answers; keep the test offline."""
    for k in ("GEMINI_API_KEY", "GEMMA_URL", "OLLAMA_HOST"):
        monkeypatch.delenv(k, raising=False)
    GuardedSlot.violations = []
    yield
    GuardedSlot.policy = None


def test_identify_never_reads_ground_truth():
    sim = _guarded_sim()
    pol = SmartPolicy(use_cache=False)
    GuardedSlot.policy = pol
    spy = ArgSpy(pol)
    res = look_all(sim, spy)
    assert len(res) == 15
    assert not GuardedSlot.violations
    assert spy.slot_keys <= ALLOWED_SLOT_KEYS, spy.slot_keys
    assert spy.sensor_keys == set(sim.read_sensors().keys())


def test_full_trials_never_read_ground_truth():
    """Whole pick sequence (plan + adjust every 10 ms) on a few very different objects."""
    if not hasattr(SmartPolicy, "plan"):
        pytest.skip("SmartPolicy.plan() not implemented yet")
    sim = _guarded_sim()
    pol = SmartPolicy(use_cache=False)
    GuardedSlot.policy = pol
    spy = ArgSpy(pol)
    runner = TrialRunner(sim, spy, order=[0, 6, 9, 12], log_series=False)
    n = round(CTRL_DT / sim.model.opt.timestep)
    while not runner.finished:
        runner.tick()
        sim.step(n)
    assert len(runner.results) == 4
    assert not GuardedSlot.violations
    assert spy.slot_keys <= ALLOWED_SLOT_KEYS, spy.slot_keys


def test_guard_catches_a_cheating_policy():
    """Proof the test has teeth: a policy that peeks at sim.slots must be caught."""
    sim = _guarded_sim()

    class Cheater(SmartPolicy):
        def identify(self, slot, rgb=None, depth=None, sensors=None):
            return {"object": sim.slots[slot["idx"]]["name"]}       # reads ground truth -> must fail

    pol = Cheater(use_cache=False)
    GuardedSlot.policy = pol
    with pytest.raises(AssertionError, match="ground truth"):
        look_all(sim, pol, order=[0])
    assert GuardedSlot.violations == [("name", "identify")]
