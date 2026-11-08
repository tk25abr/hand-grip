"""Grasping policies. The runner asks a policy three things:

  identify(slot, rgb, depth, sensors) -> once, before touching (TARE): what is it? (camera policies only
                                         get rgb/depth; others get None)
  plan(slot, sensors)                 -> before touching: how hard to squeeze, how fast to lift
  adjust(phase, slot, sensors)        -> every control tick while holding: optional new squeeze force

`slot` is what the robot is allowed to know about a slot: taught position (like an industrial robot that
was shown where things are). No name, mass or material. `sensors` = sim.read_sensors().

BaselinePolicy = the "dumb" industrial gripper: one fixed force for everything, no camera,
no AI, no feedback. This is the comparison point for the smart version.

SmartPolicy = wrist camera (RGB + depth) + a vision model's first guess before touching, then corrected
by the gripper's own sensors while grasping. It never sees sim.slots (tests/test_no_ground_truth.py).
"""
import math

import numpy as np

import perception
import vlm
from objects import TABLE_Z
from scene import MAX_OPENING, PAD_HALF, PAD_Z, PALM_ABOVE_PAD

G = 9.81
DT = 0.01               # control tick (s)
SF = 1.5                # safety factor on the squeeze needed to hold
A_LIFT = 5.0            # m/s^2 budget for lifting (measured peak at lift start: 4.6); shake spikes -> slip reflex
F_MIN, F_MAX = 1.0, 100.0
F_APPROACH = 8.0        # free closing (fingers move ~5 cm/s)
F_SEEK = 1.0            # last 10 mm before the expected width: touch gently
SEEK_MARGIN_MM = 10.0
SOFT_K = 4.0            # N/mm: below this the object gives visibly under the fingers
SOFT_UNKNOWN_N = 5.0    # soft + material unknown (depth only): could be fruit or paper -> assume <= 5 N
PROBE_MAX, PROBE_RATE = 3.0, 8.0       # stiffness probe: up to 3 N at 8 N/s
CLEAR_MIN = 0.004       # pads never lower than 4 mm above the table
CARRY_MIN = 0.18        # grip low enough that the object is carried at least 18 cm above the table
BUMP = 1.25             # squeeze test step during the micro-lift
SLIP_UP = 1.15          # slip reflex step


class BaselinePolicy:
    uses_camera = False

    def __init__(self, force=40.0, lift_speed=0.15, grasp_clearance=0.012):
        self.force = force
        self.lift_speed = lift_speed
        self.grasp_clearance = grasp_clearance   # pads always go to 12 mm above the table

    @property
    def name(self):
        return f"BASELINE: fixed {self.force:.0f} N, no camera, no AI, no feedback"

    def identify(self, slot, rgb=None, depth=None, sensors=None):
        return None   # cannot know what the object is

    def plan(self, slot, sensors):
        return dict(grip_force=self.force, lift_speed=self.lift_speed,
                    grasp_clearance=self.grasp_clearance,
                    reason=f"same {self.force:.0f} N and same grasp height for every object")

    def adjust(self, phase, slot, sensors):
        return None   # open loop: never changes the squeeze


class SmartPolicy:
    """Look first (depth size + VLM guess), then feel (stiffness, weight, slip) and correct."""
    uses_camera = True

    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        self.percept = None     # depth measurement of the current object
        self.guess = None       # first guess (VLM or depth-only prior) for the current object
        self.crop = None        # what the vision model was shown (for the dashboard)
        self.log = []           # decisions with reasons, for the dashboard
        self.felt = {}          # what touch measured: stiffness, weight, friction

    @property
    def name(self):
        return "SMART: camera + vision guess, corrected by touch"

    # ------------------------------------------------------------------ helpers
    def _say(self, text, t=None):
        self.log.append((t, text))
        return text

    def _f(self, f, say=None, **extra):
        """Command a squeeze (always within [0, cap])."""
        self.F = float(min(max(f, 0.0), self.cap))
        out = dict(force=self.F, **extra)
        if say:
            out["say"] = self._say(say, self.t)
        return out

    def _reset(self):
        self.log, self.felt = [], {}
        self.F, self.cap, self.t = 0.0, F_MAX, 0.0
        self.cstate, self.buf, self.hist, self.target, self.hits = "approach", [], [], F_SEEK, 0
        self.soft, self.k = False, None
        self.z0, self.ml, self.win, self.ref, self.waits, self.r_held = None, "rise", [], None, 0, 0.0
        self.slipped, self.mu_obs, self.wbuf, self.weighed = False, None, [], None
        self.n_slip = {}

    # ------------------------------------------------------------------ before touching
    def identify(self, slot, rgb=None, depth=None, sensors=None):
        """Once per object, before touching: measure size from depth, then ask the vision model once."""
        self._reset()
        if rgb is None or depth is None:
            raise ValueError("SmartPolicy needs the wrist camera (uses_camera=True)")
        self.percept = perception.measure(rgb, depth, slot, sensors)
        self.crop = perception.crop(rgb, self.percept["bbox"], out_px=192)
        self.guess = vlm.first_guess(rgb, self.percept, use_cache=self.use_cache)
        return self.guess

    def plan(self, slot, sensors):
        g, m = self.guess, self.percept
        self.t = sensors["t"]
        self.tare = sensors["wrist_force"][2]          # own tare: empty hand, same moment as the runner's

        def decline(why):
            self._say(f"DECLINE before touching: {why}", self.t)
            return dict(decline=why, grip_force=0.0, reason=f"DECLINE: {why}")

        if not m["found"]:
            return decline("depth sees nothing standing on the table here")
        w, h = m["width_m"], m["height_m"]
        if w > MAX_OPENING - 0.006:
            return decline(f"{w * 1000:.0f} mm wide, the hand opens only {MAX_OPENING * 1000:.0f} mm")
        if h < 0.006:
            return decline(f"only {h * 1000:.0f} mm tall, too thin for the pads")
        self.fragile = bool(g["fragile"])
        self.safe_lo = g["max_safe_squeeze_n"][0]
        self.cap = min(F_MAX, 0.6 * self.safe_lo if self.fragile else self.safe_lo)
        if self.cap < 0.5:
            return decline(f"may be damaged above {self.safe_lo:.1f} N, too delicate for this gripper")
        mu_lo, mu_hi = g["friction_guess"]
        if g["surface"] in ("wet", "slippery"):
            mu_lo = min(mu_lo, 0.25)                   # wet / slippery: be conservative
        self.mu_lo = self.mu_est = mu_lo
        m_lo, m_hi = (v / 1000 for v in g["mass_g"])
        self.m_guess = (m_lo, m_hi)
        f_micro = SF * math.sqrt(m_lo * m_hi) * G / (2 * (mu_lo + mu_hi) / 2)
        self.F_micro = float(np.clip(f_micro, F_MIN, self.cap))
        self.w_mm = w * 1000
        # pads at mid-height, never below 4 mm, low enough to carry it high (the hand is at its carrying
        # height right now, in TARE), but always high enough that the palm clears the top
        carry = sensors["gripper_xyz"][2] - TABLE_Z + (PAD_Z - PAD_HALF[2])    # pad bottom while carrying
        palm_min = h - PALM_ABOVE_PAD + 0.015
        clearance = max(CLEAR_MIN, min(h / 2 - PAD_HALF[2], carry - CARRY_MIN), palm_min)
        why_h = ("lowest grip where the palm clears the top" if clearance == palm_min
                 else "pads low for a flat object" if clearance == CLEAR_MIN
                 else "low, so it is carried high" if clearance == carry - CARRY_MIN else "pads at mid-height")
        lift = 0.08 if self.fragile or m_hi > 1.0 else 0.15
        cap_why = f"0.6 x {self.safe_lo:.0f} N (fragile)" if self.fragile else f"safe {self.safe_lo:.0f} N"
        self._say(f"depth {h * 1000:.0f} mm tall, {w * 1000:.0f} mm wide -> pads {clearance * 1000:.0f} mm up "
                  f"({why_h})", self.t)
        self._say(f"guess: {g['object']}, {g['mass_g'][0]:.0f}-{g['mass_g'][1]:.0f} g -> start "
                  f"{self.F_micro:.1f} N, cap {self.cap:.0f} N ({cap_why})", self.t)
        return dict(grip_force=F_APPROACH, lift_speed=lift, grasp_clearance=clearance, micro_lift=True,
                    close_time=4.0,
                    reason=f"{g['object']} ({g['source']}): pads {clearance * 1000:.0f} mm up, "
                           f"start {self.F_micro:.1f} N, cap {self.cap:.0f} N")

    # ------------------------------------------------------------------ every 10 ms
    def adjust(self, phase, slot, s):
        self.t = s["t"]
        if phase == "CLOSE":
            return self._close(s)
        if phase == "MICROLIFT":
            return self._microlift(s)
        if phase == "WEIGH":
            return self._weigh(s)
        if phase in ("LIFT", "HOLD", "SHAKE"):
            return self._slip_reflex(phase, s)
        return None

    def _close(self, s):
        """Fast approach -> gentle touch -> stiffness probe -> ramp to the micro-lift squeeze."""
        ap, fl, fr = s["aperture_mm"], s["touch_l"], s["touch_r"]
        st = self.cstate
        if st == "approach":
            self.hits = self.hits + 1 if max(fl, fr) > 0.3 else 0     # 2 ticks in a row: not sensor noise
            if ap < self.w_mm + SEEK_MARGIN_MM or self.hits >= 2:
                self.cstate = "seek"
                return self._f(F_SEEK)
            return dict(force=F_APPROACH)                  # free space: not limited by the cap
        if st == "seek":
            if min(fl, fr) > 0.2:
                self.cstate, self.buf = "contact", []
            elif ap < 6.0:                                 # fully closed (4 mm) and never touched
                return dict(abort=self._say("pads closed on nothing: the object is not where depth saw it",
                                            self.t))
            elif ap > self.w_mm + SEEK_MARGIN_MM + 15 and max(fl, fr) < 0.15:
                self.cstate, self.hits = "approach", 0     # false alarm far from the object: close fast again
                return dict(force=F_APPROACH)
            return self._f(F_SEEK)
        if st == "contact":                                # settle at the touch force, remember gap + force
            self.buf.append((ap, (fl + fr) / 2))
            if len(self.buf) >= 5:
                self.g0, self.f0 = (float(np.mean(v)) for v in zip(*self.buf))
                self.cstate, self.hist = "probe", []
                self.felt["width_mm"] = self.g0
            return self._f(F_SEEK)
        if st == "probe":
            self.hist.append((ap, (fl + fr) / 2))
            g, f = (float(np.mean(v)) for v in zip(*self.hist[-5:]))
            dg, dF = self.g0 - g, f - self.f0
            probe_max = min(PROBE_MAX, self.cap)
            if dg >= 0.6 and dF < 0.15:          # gap shrank but force did not rise: object shifting, not
                self.g0, self.f0 = g, f            # being squeezed -> restart the stiffness reference here
                return self._f(min(self.F + PROBE_RATE * DT, probe_max))
            soft = dg >= 0.6 and dF / dg < SOFT_K
            timed_out = self.F >= probe_max - 1e-9 and len(self.hist) > 25
            if timed_out and dF < 0.5:          # pads never loaded up: no proof it is firm -> treat as soft
                soft = True
            if soft or dF >= 1.5 or timed_out:
                self.soft = soft
                self.k = dF / max(dg, 0.05)
                self.felt["stiffness"] = self.k
                self.felt["soft"] = soft
                self.cstate = "ramp"
                if soft:
                    # trust the vision cap only if vision already expected something delicate;
                    # vision said "firm" but the fingers feel it give -> touch wins, material unknown
                    known = self.guess["source"] != "depth-only" and self.fragile
                    base = self.safe_lo if known else min(self.safe_lo, SOFT_UNKNOWN_N)
                    self.cap = min(self.cap, 0.6 * base)
                    msg = (f"soft: {self.k:.1f} N/mm (gives {dg:.1f} mm for +{dF:.1f} N) -> cap "
                           f"0.6 x {base:.0f} = {self.cap:.1f} N"
                           + ("" if known else " (touch overrides vision: material unknown, fruit? paper?)"))
                    if not known and self.guess["source"] != "depth-only":
                        # vision was wrong about the material, so its mass guess is not trusted either:
                        # start from what the depth size alone allows (light foam/paper .. dense)
                        pr = vlm.depth_prior(self.percept)
                        m_lo, m_hi = (v / 1000 for v in pr["mass_g"])
                        self.m_guess = (m_lo, m_hi)
                        self.F_micro = float(np.clip(SF * math.sqrt(m_lo * m_hi) * G / sum(pr["friction_guess"]),
                                                     F_MIN, self.cap))
                        msg += f"; vision's mass guess dropped -> start {self.F_micro:.1f} N"
                else:
                    msg = f"firm: gives only {max(dg, 0):.2f} mm for +{dF:.1f} N (> {dF / max(dg, 0.2):.0f} N/mm)"
                self.target = float(np.clip(self.F_micro, F_MIN, self.cap))
                return self._f(min(self.F, self.target) if soft else self.F, say=msg)
            return self._f(min(self.F + PROBE_RATE * DT, probe_max))
        if st == "ramp":
            rate = 10.0 if self.soft else 80.0
            step = rate * DT
            f = self.F + float(np.clip(self.target - self.F, -step, step))
            if abs(f - min(self.target, self.cap)) < 1e-9:
                self.cstate, self.buf = "hold", []
            return self._f(f)
        if st == "hold":                                   # 0.1 s at the target, then the hand may lift
            self.buf.append(1)
            if len(self.buf) >= 10:
                return dict(self._f(self.F), done=True)
            return self._f(self.F)
        return None

    def _microlift(self, s):
        """Rising 1 cm/s. Squeeze test: grip 25 % harder and compare the pad forces before and after.
          held    -> the lifting (tangential) force stays = weight while the measured squeeze grows
          sliding -> the lifting force grows with the squeeze: it sits at the friction limit, and that
                     shear/normal ratio IS the object's friction (the pads also chatter). Tighten, test again.
        Soft objects build up the squeeze slowly, so the test waits until the squeeze has really grown."""
        z = s["gripper_xyz"][2]
        if self.z0 is None:
            self.z0 = z
        if self.ml == "rise":
            if z - self.z0 > 0.004:
                self.ml, self.win = "ref", []
            return self._f(self.F)
        if self.ml not in ("ref", "bump"):
            return dict(self._f(self.F), done=True)
        self.win.append((s["shear_ratio"], s["touch_l"], s["touch_r"]))
        if len(self.win) < 8:
            return self._f(self.F)
        both = [r for r, fl, fr in self.win if r > 0 and fl > 0.2 and fr > 0.2]
        steady = len(both) >= 7                             # a pad losing contact = chatter = sliding
        touching = [r for r, _, _ in self.win if r > 0]
        r = float(np.median(touching)) if len(touching) >= 3 else None
        n = float(np.mean([fl + fr for _, fl, fr in self.win]))   # measured squeeze, both pads
        if all(fl < 0.2 and fr < 0.2 for _, fl, fr in self.win) and s["aperture_mm"] < self.w_mm - 10:
            # fingers closed past the object's width and feel nothing: never squeeze harder on air
            return dict(abort=self._say("nothing between the pads: lost it or never had it", self.t))
        tang = (r or 0.0) * n                               # tangential force = what holds it up
        self.win = []
        if self.ml == "ref":
            self.ref, self.F_before = (r, n, tang), self.F
            if self.F >= self.cap - 1e-9:
                self.ml = "ok"
                return dict(self._f(self.F, say=f"already at the cap {self.cap:.1f} N: no squeeze test"),
                            done=True)
            self.ml, self.waits = "bump", 0
            return self._f(self.F * BUMP)
        r0, n0, t0 = self.ref
        if steady and r is not None and r0 is not None:
            gn = n / max(n0, 1e-6) - 1
            if gn < 0.08:                                  # squeeze has not grown measurably (yet)
                if self.waits < (8 if self.soft else 4):   # soft objects take time to build it up
                    self.waits += 1
                    return self._f(self.F)
                if self.F >= self.cap - 1e-9:
                    return dict(abort=self._say(f"cannot confirm a secure grip at {self.cap:.1f} N", self.t))
                self.ref, self.F_before, self.waits = (r, n, tang), self.F, 0
                return self._f(self.F * BUMP)               # no proof either way: squeeze a bit more
            gt = tang / max(t0, 1e-6) - 1
            if gt < 0.4 * gn:                              # lifting force did not follow the squeeze
                self.ml = "ok"
                self.r_held = r                            # proven: friction >= weight / squeeze
                return dict(self._f(self.F, say=f"squeeze test {self.F_before:.1f} -> {self.F:.1f} N: squeeze "
                                                f"+{gn * 100:.0f} %, lifting force {gt * 100:+.0f} % -> held"),
                            done=True)
        self.slipped = True                                 # lifting force followed the squeeze -> sliding
        if r is not None:
            self.mu_obs = r
            self.felt["friction"] = r
        say = None
        if self.n_slip.get("micro", 0) == 0:
            mu = f"friction ~{r:.2f}" if r is not None else "pads chattering"
            say = f"squeeze test: lifting force grew with the squeeze -> it is sliding ({mu}); tightening"
        self.n_slip["micro"] = self.n_slip.get("micro", 0) + 1
        if self.F >= self.cap - 1e-9:
            return dict(abort=self._say(f"still slipping at {self.cap:.0f} N, the most it may take", self.t))
        self.ref, self.F_before, self.waits = (r, n, tang), self.F, 0
        return self._f(self.F * BUMP, say=say)

    def _weigh(self, s):
        """Still in the air for 0.3 s: weigh it, compute the squeeze it needs, compare with the cap."""
        self.wbuf.append(s["wrist_force"][2] - self.tare)
        if self.weighed is not None:
            return self.weighed
        if self.ml != "ok":
            self.weighed = dict(abort=self._say("could not confirm a secure grip in the micro-lift", self.t))
            return self.weighed
        if len(self.wbuf) < 30:
            return self._f(self.F)
        m = max(0.0, float(np.mean(self.wbuf[10:])) / G)
        self.felt["weight_g"] = m * 1000
        measured = self.slipped and self.mu_obs is not None
        # slid at some point: its friction was measured at the limit; held from the start: the ratio it
        # held at is a proven lower bound on its friction
        mu = 0.85 * self.mu_obs if measured else max(self.mu_lo, self.r_held)
        self.mu_est = max(0.05, mu)
        f_req = SF * m * (G + A_LIFT) / (2 * self.mu_est)
        self.felt["needs_n"] = f_req
        lo, hi = (v * 1000 for v in self.m_guess)
        src = self.guess["source"].replace("depth-only", "depth prior").replace("gemini", "Gemini")
        ratio = m * 1000 / math.sqrt(lo * hi) if lo > 0 else 1.0
        cmp_ = f" ({ratio:.1f}x the guess)" if ratio > 1.5 or ratio < 0.67 else ""
        self._say(f"{src} guessed {lo:.0f}-{hi:.0f} g, wrist says {m * 1000:.0f} g{cmp_}", self.t)
        mu_why = (f"measured friction {self.mu_obs:.2f} x 0.85" if measured else
                  f"friction >= {self.mu_est:.2f} ({'proven by the squeeze test' if self.r_held > self.mu_lo else 'guess'})")
        fmt = lambda f: f"{f:.0f}" if f >= 10 else f"{f:.1f}"
        if f_req > self.cap:
            harm = "crack" if self.fragile else "be damaged"
            self.weighed = dict(abort=self._say(f"needs ~{fmt(f_req)} N to hold, but it may {harm} above "
                                                f"~{fmt(self.cap)} N", self.t))
            return self.weighed
        new = max(F_MIN, f_req)
        lift = 0.08 if (self.fragile or self.soft or m > 1.0) else 0.15
        msg = f"weight {m * 1000:.0f} g, {mu_why} -> grip {self.F:.1f} -> {min(new, self.cap):.1f} N"
        self.weighed = dict(self._f(new, say=msg), done=True, lift_speed=lift)
        return self.weighed

    def _slip_reflex(self, phase, s):
        """Shear/normal close to the friction estimate -> squeeze 15 % harder (never above the cap)."""
        sr = s["shear_ratio"]
        if sr > 0.8 * self.mu_est and self.F < self.cap - 1e-9:
            n = self.n_slip.get(phase, 0)
            self.n_slip[phase] = n + 1
            old = self.F
            say = None
            if n == 0:
                say = f"{phase}: slip risk {sr:.2f} > 0.8 x {self.mu_est:.2f} -> grip {old:.1f} -> {old * SLIP_UP:.1f} N"
            return self._f(old * SLIP_UP, say=say)
        return None