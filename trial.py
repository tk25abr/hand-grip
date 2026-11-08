"""Runs pick-and-lift trials over the 15 slots with a given policy, and judges each one.

One trial:  ABOVE -> TARE -> DESCEND -> CLOSE [-> MICROLIFT -> WEIGH] -> LIFT -> HOLD -> SHAKE -> PLACE
            -> RELEASE -> RISE
  TARE      : wrist force read with an empty hand (so the weight reading starts at zero);
              a camera policy also gets the wrist RGB + depth here: identify(slot, rgb, depth, sensors)
  MICROLIFT : (if the plan asks for it) lift 1 cm slowly - up to 3 cm while the policy is still tightening
  WEIGH     : hold still 0.3 s -> wrist load = weight; the policy may then put it back and DECLINE
  HOLD      : object held still in the air -> wrist force change = measured weight
  SHAKE     : quick sideways move -> tests whether the grip survives acceleration

The policy only gets sensor readings + the taught slot position. The Judge uses ground truth
(object pose, damage) to score the outcome — that's the evaluation, not the controller.

adjust() may return None, a float (new squeeze force) or a dict with any of
  force: new squeeze (N) · abort: reason (put it down, open, DECLINED) · done: this phase can end early
  lift_speed: m/s for the main lift · say: a decision to show on the dashboard
"""
import math
import numpy as np

from scene import HOME, PAD_Z, PAD_HALF
from objects import TABLE_Z, public_slot

G = 9.81
CTRL_DT = 0.01          # control tick, s
CAM_RES = (640, 480)    # wrist camera resolution handed to camera policies
HOVER_Z = TABLE_Z + 0.45  # hand height when carrying (object ~20 cm above the table)
PAD_BOTTOM = -PAD_Z + PAD_HALF[2]   # distance from hand frame down to the bottom of the pads
MICRO_LIFT = 0.01       # micro-lift: 1 cm ...
MICRO_MAX = 0.03        # ... or up to 3 cm while the policy is still tightening its grip
MICRO_SPEED = 0.01      # m/s
MICRO_HOLD = 0.3        # s still in the air before reading the weight

OUTCOME_COLORS = {"SUCCESS": (70, 200, 110), "CRUSHED": (235, 60, 60), "SLIPPED": (245, 160, 40),
                  "DROPPED": (245, 160, 40), "MISSED": (150, 150, 160), "COLLISION": (200, 120, 255),
                  "DECLINED": (110, 170, 255), "RUNNING": (230, 230, 230)}
OUTCOMES = ["SUCCESS", "CRUSHED", "SLIPPED", "DROPPED", "MISSED", "COLLISION", "DECLINED"]


class Judge:
    """Ground-truth scoring of one trial."""

    def __init__(self, sim, idx):
        self.sim, self.idx = sim, idx
        self.rest_z = sim.truth_obj_pos(idx)[2]
        self.max_lift = 0.0
        self.lifted = False
        self.dropped = False
        self.max_slip = 0.0
        self.rel0 = None
        self.landed_on_top = False
        self.contact_after_close = None
        self.weight_g = None
        self.aperture_mm = None
        self.declined = None      # reason, if the robot refused this object
        self.tried = False        # the fingers started closing on it
        self.touched = False      # both pads were on it at some point while grasping

    def tick(self, phase, s):
        p = self.sim.truth_obj_pos(self.idx)
        hand_z = s["gripper_xyz"][2]
        lift = p[2] - self.rest_z
        self.max_lift = max(self.max_lift, lift)
        if phase == "DESCEND" and s["hand_collision"]:
            self.landed_on_top = True
        if phase in ("CLOSE", "MICROLIFT", "WEIGH"):
            self.tried = True
            if all(s["contact"]):
                self.touched = True
        if phase in ("HOLD", "SHAKE") and lift > 0.15:
            self.lifted = True
        if self.lifted and phase in ("HOLD", "SHAKE") and lift < 0.05:
            self.dropped = True
        if phase == "LIFT" and self.rel0 is None and lift > 0.005:
            self.rel0 = hand_z - p[2]
        if self.rel0 is not None and phase in ("LIFT", "HOLD", "SHAKE") and lift > 0.01:
            self.max_slip = max(self.max_slip, (hand_z - p[2]) - self.rel0)

    def verdict(self):
        crushed = self.idx in self.sim.crushed
        peak = self.sim.peak_force[self.idx]
        if self.landed_on_top and not self.lifted:
            res = "MISSED"
            why = "hand landed on top of it (wider than the hand opens)"
        elif self.landed_on_top:
            res = "COLLISION"
            why = "lifted it, but first rammed its top with the palm (too tall for a fixed grasp height)"
        elif crushed:
            res = "CRUSHED"
            why = f"squeeze reached its damage limit ({self.sim.slots[self.idx]['max_force']:.0f} N for this one)"
        elif self.declined and (self.touched or not self.tried):   # tried and never touched = a miss
            res = "DECLINED"
            why = f"robot refused safely: {self.declined}"
        elif not self.contact_after_close:
            res = "MISSED"
            why = "pads never touched it (too thin / below the pads)"
        elif self.dropped:
            res = "DROPPED"
            why = "lifted, then fell out during hold/shake"
        elif not self.lifted:
            res = "SLIPPED"
            why = f"slid out of the grip while lifting (rose only {self.max_lift*100:.1f} cm)"
        elif self.max_slip > 0.03:
            res = "SLIPPED"
            why = f"held on, but slid {self.max_slip*1000:.0f} mm down inside the fingers (unsafe grip)"
        else:
            res = "SUCCESS"
            why = f"lifted {self.max_lift*100:.0f} cm, slipped {self.max_slip*1000:.0f} mm in the hand"
        return res, why, peak


class TrialRunner:
    PHASES = ["ABOVE", "TARE", "DESCEND", "CLOSE", "MICROLIFT", "WEIGH", "LIFT", "HOLD", "SHAKE", "PLACE",
              "RELEASE", "RISE", "PUTDOWN", "DECLINE", "DONE"]

    def __init__(self, sim, policy, order=None, log_series=True):
        self.sim, self.policy = sim, policy
        self.order = list(order) if order is not None else list(range(len(sim.slots)))
        self.k = -1
        self.results = []
        self.series = []        # per-tick sensor log of the current trial (for plots / CSV)
        self.all_series = {}
        self.log_series = log_series
        self.finished = False
        self.t_next = 0.0
        self.change = None      # scene change: dict(factor=2.5, names={...} or None, at=0.4 s into HOLD)
        self._next_trial()

    # ------------------------------------------------------------------ helpers
    @property
    def slot(self):
        return self.sim.slots[self.order[self.k]]

    def _set_phase(self, ph):
        self.phase = ph
        self.phase_t0 = self.sim.time

    def _next_trial(self):
        if self.k >= 0:
            self.all_series[self.slot["idx"]] = self.series
        self.k += 1
        if self.k >= len(self.order):
            self.finished = True
            self.phase = "DONE"
            self.sim.grip(-10)
            self.goal = np.array(HOME, float)
            return
        s = self.slot
        self.series = []
        self.judge = Judge(self.sim, s["idx"])
        self.plan = None
        self.label = None
        self.guess = None
        self.tare = None
        self.weight_g = None
        self.micro_weight_g = None
        self.max_cmd = 0.0
        self.grasp_z = None
        self.load_t = None                       # sim time of a mid-run load change in this trial
        self.decision = "moving to next slot"
        self.sim.grip(-10)                       # open
        self.cmd = np.array(self.sim.target, float)
        self.goal = np.array([s["x"], s["y"], HOVER_Z])
        self.speed = 0.5
        self._set_phase("ABOVE")

    def _move(self):
        """Move the commanded target towards the goal at self.speed (m/s)."""
        d = self.goal - self.cmd
        n = np.linalg.norm(d)
        step = self.speed * CTRL_DT
        self.cmd = self.goal.copy() if n <= step else self.cmd + d / n * step
        self.sim.move_to(*self.cmd)
        return n <= 1e-4

    def _settled(self, tol=0.004):
        return np.linalg.norm(self.sim.gripper_pos() - self.goal) < tol

    def _clearance(self):
        return self.plan.get("grasp_clearance", 0.012) if self.plan else 0.012

    # ------------------------------------------------------------------ main tick
    def tick(self):
        """Call every CTRL_DT seconds of sim time."""
        if self.finished:
            self._move()
            return
        s = self.sim.read_sensors()
        sl = self.slot
        ph = self.phase
        t_in = self.sim.time - self.phase_t0
        arrived = self._move()
        if any(s["contact"]):                    # strongest squeeze commanded while touching the object
            self.max_cmd = max(self.max_cmd, s["grip_cmd"])

        if ph == "ABOVE" and arrived and self._settled():
            self._set_phase("TARE")
        elif ph == "TARE" and t_in > 0.3:
            self.tare = s["wrist_force"][2]
            self.perceive(s)
            self.plan = self.policy.plan(public_slot(sl), s)
            self.decision = self.plan["reason"]
            if self.plan.get("decline"):
                self.judge.declined = self.plan["decline"]
                self.decision = f"DECLINED: {self.plan['decline']}"
                self._set_phase("DECLINE")
            else:
                self.goal = np.array([sl["x"], sl["y"], TABLE_Z + PAD_BOTTOM + self._clearance()])
                self.speed = 0.15
                self._set_phase("DESCEND")
        elif ph == "DECLINE" and t_in > 1.0:          # refused before touching: show why, move on
            self._finish_trial()
            return
        elif ph == "DESCEND" and (arrived and self._settled() or t_in > 4):
            if abs(s["gripper_xyz"][2] - self.goal[2]) > 0.006:   # something stopped the hand on the way down
                self.judge.landed_on_top = True
            self.sim.grip(self.plan["grip_force"])
            self._set_phase("CLOSE")
        elif ph == "CLOSE":
            r = self._adjust(s)
            if r["abort"]:
                self._abort(r["abort"])
            elif t_in > self.plan.get("close_time", 0.8) or (r["done"] and t_in > 0.05):
                self.judge.contact_after_close = all(s["contact"])
                self.judge.aperture_mm = s["aperture_mm"]
                self.grasp_z = self.cmd[2]
                if self.plan.get("micro_lift"):
                    self.goal = np.array([sl["x"], sl["y"], self.grasp_z + MICRO_MAX])
                    self.speed = MICRO_SPEED
                    self._set_phase("MICROLIFT")
                else:
                    self._start_lift()
        elif ph == "MICROLIFT":
            r = self._adjust(s)
            if r["abort"]:
                self._abort(r["abort"])
            elif (r["done"] and self.cmd[2] - self.grasp_z >= MICRO_LIFT - 1e-9) or arrived:
                self.goal = self.cmd.copy()               # stop and hold still
                self._set_phase("WEIGH")
        elif ph == "WEIGH":
            r = self._adjust(s)
            if t_in > 0.1:
                self.micro_weight_g = (s["wrist_force"][2] - self.tare) / G * 1000
            if r["abort"]:
                self._abort(r["abort"])
            elif (r["done"] and t_in >= MICRO_HOLD) or t_in > 1.5:
                self._start_lift()
        elif ph == "LIFT":
            self._adjust(s)
            if arrived and self._settled(0.01):
                self._set_phase("HOLD")
        elif ph == "HOLD":
            c = self.change
            if (c and self.load_t is None and t_in > c.get("at", 0.4)
                    and (not c.get("names") or sl["name"].lower() in c["names"])):
                self.add_load(c["factor"])
            r = self._adjust(s)
            if r["abort"]:                           # the policy decided to carry it back down
                self._abort(r["abort"])
                return self._log_tick(s)
            if t_in > 0.5:
                self.weight_g = max(0.0, (s["wrist_force"][2] - self.tare) / G * 1000)
            if t_in > 1.2 and (self.load_t is None or self.sim.time - self.load_t > 1.8):
                self.judge.weight_g = self.weight_g
                self.goal = np.array([sl["x"] + 0.12, sl["y"], HOVER_Z])
                self.speed = 0.6
                self._set_phase("SHAKE")
        elif ph == "SHAKE":
            self._adjust(s)
            if arrived and abs(self.cmd[0] - (sl["x"] + 0.12)) < 1e-6 and self.goal[0] != sl["x"]:
                self.goal = np.array([sl["x"], sl["y"], HOVER_Z])
            elif arrived and self.goal[0] == sl["x"]:
                self.goal = np.array([sl["x"], sl["y"], TABLE_Z + PAD_BOTTOM + self._clearance() + 0.004])
                self.speed = 0.15
                self._set_phase("PLACE")
        elif ph == "PLACE" and (arrived and self._settled(0.006) or t_in > 3):
            self.sim.grip(-10)
            self._set_phase("RELEASE")
        elif ph == "PUTDOWN" and (arrived and self._settled(0.003) or t_in > 3):
            self.sim.grip(-10)
            self._set_phase("RELEASE")
        elif ph == "RELEASE" and t_in > 0.5:
            self.goal = np.array([sl["x"], sl["y"], HOVER_Z])
            self.speed = 0.4
            self._set_phase("RISE")
        elif ph == "RISE" and arrived:
            self._finish_trial()
            return

        self._log_tick(s)

    def add_load(self, factor):
        """Scene change while the object is in the hand (scheduled with .change, or the L key live)."""
        if self.finished or self.phase not in ("LIFT", "HOLD", "SHAKE") or self.load_t is not None:
            return False
        self.sim.add_load(self.slot["idx"], factor)
        self.load_t = self.sim.time
        self.decision = f"SCENE CHANGE: object now {factor:g}x heavier (robot not told)"
        print(f"  scene change: {self.slot['name']} is now {factor:g}x heavier", flush=True)
        return True

    def _log_tick(self, s):
        sl = self.slot
        self.judge.tick(self.phase, s)
        if self.log_series:
            self.series.append(dict(t=self.sim.time, phase=self.phase, grip_cmd=s["grip_cmd"],
                                    touch_l=s["touch_l"], touch_r=s["touch_r"],
                                    load_g=(s["wrist_force"][2] - self.tare) / G * 1000 if self.tare is not None else 0.0,
                                    aperture_mm=s["aperture_mm"], shear=s["shear_ratio"],
                                    hand_z=s["gripper_xyz"][2] - TABLE_Z,
                                    obj_lift=self.sim.truth_obj_pos(sl["idx"])[2] - self.judge.rest_z))

    def _start_lift(self):
        sl = self.slot
        self.goal = np.array([sl["x"], sl["y"], HOVER_Z])
        self.speed = self.plan.get("lift_speed", 0.15)
        self._set_phase("LIFT")

    def _abort(self, reason):
        """Policy refuses mid-grasp: put the object back down if it is off the table, open, DECLINED."""
        self.judge.declined = reason
        self.decision = f"DECLINED: {reason}"
        if self.phase in ("MICROLIFT", "WEIGH") and self.grasp_z is not None:
            sl = self.slot
            self.goal = np.array([sl["x"], sl["y"], self.grasp_z + 0.001])
            self.speed = 0.02
            self._set_phase("PUTDOWN")
        elif self.phase in ("LIFT", "HOLD", "SHAKE") and self.grasp_z is not None:
            sl = self.slot                               # in the air: carry it back down, then let go
            self.goal = np.array([sl["x"], sl["y"], self.grasp_z + 0.002])
            self.speed = 0.08
            self._set_phase("PUTDOWN")
        else:
            self.sim.grip(-10)
            self._set_phase("RELEASE")

    def perceive(self, s):
        """TARE: a camera policy gets the wrist RGB + depth (and the sensor readings) before touching.
        Called once per object; this is where a vision model may be asked, never in the control loop."""
        rgb = depth = None
        if getattr(self.policy, "uses_camera", False):
            rgb, depth = self.sim.read_wrist_camera(*CAM_RES)
        g = self.policy.identify(public_slot(self.slot), rgb, depth, s)
        self.guess = g if isinstance(g, dict) else None
        self.label = f"{g.get('object')} ({g.get('source')})" if isinstance(g, dict) else g
        return g

    def _adjust(self, s):
        r = self.policy.adjust(self.phase, public_slot(self.slot), s)
        out = dict(force=None, abort=None, done=False)
        if isinstance(r, dict):
            out.update({k: r[k] for k in ("force", "abort", "done", "lift_speed") if k in r})
            if r.get("lift_speed") and self.plan is not None:
                self.plan["lift_speed"] = r["lift_speed"]
            if r.get("say"):
                self.decision = r["say"]
        elif r is not None:
            out["force"] = float(r)
        if out["force"] is not None:
            self.sim.grip(out["force"])
        return out

    def _finish_trial(self):
        res, why, peak = self.judge.verdict()
        sl = self.slot
        g = self.guess or {}
        self.results.append(dict(idx=sl["idx"], name=sl["name"], result=res, why=why,
                                 peak_n=peak,
                                 grip_n=self.max_cmd or (self.plan or {}).get("grip_force", 0.0),
                                 weight_g=self.judge.weight_g if self.judge.lifted else None,
                                 true_g=sl["mass"] * 1000,
                                 aperture_mm=self.judge.aperture_mm, true_width_mm=sl["width"] * 1000,
                                 lift_cm=self.judge.max_lift * 100, slip_mm=self.judge.max_slip * 1000,
                                 label=self.label, source=g.get("source"),
                                 micro_weight_g=self.micro_weight_g))
        print(f"  [{sl['idx']+1:2d}/15] {sl['name']:18s} {res:8s} {why}")
        self.sim.reset_object(sl["idx"])
        self._next_trial()

    def current_result(self):
        return "RUNNING"