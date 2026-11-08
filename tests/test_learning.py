"""Tests for the learning layer: memory, yield sensing, and the limit challenge. None of them needs
the simulator's ground truth."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory import ObjectMemory, THUMB_DIFF       # noqa: E402
from policies import SmartPolicy, YIELD_TICKS     # noqa: E402


def thumb(v):
    return np.full((16, 16, 3), v, dtype=np.int16)


def test_memory_recalls_similar_looking_object_only():
    m = ObjectMemory()
    m.record(thumb(100), held_n=12.0, weight_g=600.0)
    assert m.recall(thumb(103))["held_n"] == 12.0                 # same look (difference 3 < THUMB_DIFF)
    assert m.recall(thumb(100 + THUMB_DIFF + 5)) is None          # different look
    assert len(m) == 1


def test_memory_crush_limit_only_moves_down_held_only_up():
    m = ObjectMemory()
    m.record(thumb(50), crush_n=30.0, held_n=10.0)
    m.record(thumb(50), crush_n=45.0, held_n=4.0)                 # a later, weaker observation
    r = m.recall(thumb(50))
    assert r["crush_n"] == 30.0 and r["held_n"] == 10.0
    m.record(thumb(50), crush_n=20.0, held_n=15.0)
    r = m.recall(thumb(50))
    assert r["crush_n"] == 20.0 and r["held_n"] == 15.0


def test_memory_saves_and_loads(tmp_path):
    p = str(tmp_path / "mem.json")
    m = ObjectMemory(p)
    m.record(thumb(70), held_n=8.0)
    m.save()
    again = ObjectMemory(p)
    assert again.recall(thumb(70))["held_n"] == 8.0


def _policy_in_microlift():
    p = SmartPolicy(use_cache=True, learn=True, memory=ObjectMemory())
    p._reset()
    p.thumb = thumb(10)
    p.F, p.cap, p.k, p.soft = 20.0, 60.0, 30.0, False
    return p


def _s(aperture, touch=2.0):
    return dict(aperture_mm=aperture, touch_l=touch, touch_r=touch)


def test_yield_is_sensed_from_a_sudden_aperture_drop_and_remembered():
    p = _policy_in_microlift()
    for _ in range(14):                                           # steady hold, sensor noise only
        assert p._yield_watch("MICROLIFT", _s(60.0 + np.random.normal(0, 0.2))) is None
    out = None
    for _ in range(YIELD_TICKS + 2):                              # the object gives way: fingers close 5 mm
        out = p._yield_watch("MICROLIFT", _s(55.0)) or out
    assert out is not None and "abort" in out
    assert p.memory.recall(thumb(10))["crush_n"] == 20.0


def test_slow_elastic_closing_is_not_a_yield():
    p = _policy_in_microlift()
    ap = 60.0
    for i in range(60):                                           # squeeze ramps up, gap shrinks 0.02 mm per tick
        p.F = 20.0 + 0.8 * i
        ap -= 0.02
        assert p._yield_watch("WEIGH", _s(ap)) is None


def test_limit_challenge_needs_a_firm_object_and_no_known_crush():
    p = _policy_in_microlift()
    assert p._may_escalate(70.0) is True                          # within 2x of a 60 N guess, firm, nothing known
    assert p._may_escalate(130.0) is False                        # more than 2x the guess
    p.soft = True
    assert p._may_escalate(70.0) is False                         # felt soft: touch wins over the plan
    p.soft = False
    p.mem = dict(crush_n=30.0)
    assert p._may_escalate(70.0) is False                         # it already gave way once


def test_no_learning_switch_keeps_old_behaviour():
    p = SmartPolicy(learn=False)
    assert p.memory is None
    p._reset()
    assert p._may_escalate(70.0) is False and p._relax_cap() is False


def test_object_falling_out_is_not_a_yield():
    p = _policy_in_microlift()
    for _ in range(14):
        p._yield_watch("MICROLIFT", _s(60.0))
    out = None
    for _ in range(YIELD_TICKS + 3):                              # fingers snap shut on air: pads unloaded
        out = p._yield_watch("MICROLIFT", _s(48.0, touch=0.0)) or out
    assert out is None and p.memory.recall(thumb(10)) is None


def test_remembered_friction_is_the_measured_one():
    p = _policy_in_microlift()
    p.slipped, p.mu_obs, p.mu_est, p.F = True, 0.2, 0.5, 40.0
    p._remember(held=True)
    r = p.memory.recall(thumb(10))
    assert abs(r["mu_measured"] - 0.17) < 1e-9 and r["held_n"] == 40.0
