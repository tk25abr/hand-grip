"""MuJoCo wrapper: moves the gantry, drives the fingers, reads every sensor.

read_sensors() returns ONLY what physical sensors on the gripper would measure.
Ground truth (object poses, true masses, damage limits) is kept separate in truth_*() and is
used for physics side-effects (crushing) and evaluation — never by a policy.
"""
import math
import numpy as np
import mujoco

from scene import build_mjcf, HOME, PAD_AREA, MAX_TRAVEL, FINGER_Y0, FINGER_HALF, PAD_HALF, PAD_Z
from objects import slots, TABLE_Z

G = 9.81


class Sim:
    def __init__(self, params=None, tc=None, seed=0):
        """params: per-object properties (objects.nominal_params / sample_params); tc: contact time
        constants from the stiffness calibration (calibrate.solref_tcs)."""
        self.slots = slots(params)
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self.slots, tc))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.obj_body = [m.body(f"body{s['idx']}").id for s in self.slots]
        self.obj_geom = [m.geom(f"obj{s['idx']}").id for s in self.slots]
        self.pad = {"l": m.geom("pad_l").id, "r": m.geom("pad_r").id}
        self.finger_geoms = {m.geom("finger_l").id, m.geom("finger_r").id, m.geom("palm").id,
                             self.pad["l"], self.pad["r"]}
        self.table_geom = m.geom("table").id
        self.target = np.array(HOME, float)
        self.grip_cmd = 0.0
        self.crushed = set()
        self.peak_force = {i: 0.0 for i in range(len(self.slots))}
        self._f_filt = np.zeros(len(self.slots))   # 30 ms low-pass, like a real force sensor
        self._sensor = {n: (m.sensor(n).adr[0], m.sensor(n).dim[0]) for n in
                        ["wrist_force", "wrist_torque", "wrist_acc", "touch_l", "touch_r", "pos_l", "pos_r"]}
        self._f_hist = []
        mujoco.mj_forward(m, self.data)
        self.renderer = None
        self._renderers = {}
        self._rng = np.random.default_rng(seed)

    # ---------------- actuation ----------------
    def move_to(self, x, y, z):
        self.target[:] = (x, y, z)
        self.data.ctrl[0] = x - HOME[0]
        self.data.ctrl[1] = y - HOME[1]
        self.data.ctrl[2] = z - HOME[2]

    def grip(self, force_n):
        """Squeeze force per finger in N (negative = open)."""
        self.grip_cmd = float(force_n)
        self.data.ctrl[3] = force_n
        self.data.ctrl[4] = force_n

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            self._apply_damage()

    @property
    def time(self):
        return self.data.time

    # ---------------- sensors (what the robot is allowed to know) ----------------
    def _s(self, name):
        a, d = self._sensor[name]
        return self.data.sensordata[a:a + d].copy()

    def gripper_pos(self):
        return self.data.site("wrist").xpos.copy()

    def pad_contacts(self):
        """Normal and tangential (shear) force on each pad — what a tactile skin measures."""
        out = {"l": [0.0, 0.0, 0], "r": [0.0, 0.0, 0]}
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for side, gid in self.pad.items():
                if c.geom1 == gid or c.geom2 == gid:
                    mujoco.mj_contactForce(self.model, self.data, i, f6)
                    out[side][0] += abs(f6[0])
                    out[side][1] += math.hypot(f6[1], f6[2])
                    out[side][2] += 1
        return out

    NOISE = dict(force=0.04, torque=0.002, touch=0.08, pos=0.0002, acc=0.05)   # sensor noise (1 sigma)

    def read_sensors(self):
        rn = self._rng.normal
        pc = self.pad_contacts()
        fl = max(0.0, self._s("touch_l")[0] + rn(0, self.NOISE["touch"]))
        fr = max(0.0, self._s("touch_r")[0] + rn(0, self.NOISE["touch"]))
        ql = self._s("pos_l")[0] + rn(0, self.NOISE["pos"])
        qr = self._s("pos_r")[0] + rn(0, self.NOISE["pos"])
        inner0 = 2 * (FINGER_Y0 - FINGER_HALF[1] - 2 * PAD_HALF[1])
        aperture = max(0.0, inner0 - ql - qr)
        normal = pc["l"][0] + pc["r"][0]
        shear = pc["l"][1] + pc["r"][1]
        return dict(
            t=self.time,
            gripper_xyz=self.gripper_pos(),
            touch_l=fl, touch_r=fr,                                   # N (touch sensors)
            pressure_kpa=max(fl, fr) / PAD_AREA / 1000.0,             # kPa on the pads
            shear_ratio=(shear / normal) if normal > 0.5 else 0.0,    # tangential / normal on pads
            contact=(pc["l"][2] > 0, pc["r"][2] > 0),
            aperture_mm=aperture * 1000,                              # hand opening = object width by touch
            wrist_force=self._s("wrist_force") + rn(0, self.NOISE["force"], 3),     # N, 3-axis
            wrist_torque=self._s("wrist_torque") + rn(0, self.NOISE["torque"], 3),  # N*m
            wrist_acc=self._s("wrist_acc") + rn(0, self.NOISE["acc"], 3),           # m/s^2
            grip_cmd=self.grip_cmd,
            hand_collision=self.hand_hit_something(),
        )

    def hand_hit_something(self):
        """Fingers/palm touching anything that is not between the pads (e.g. landing on top of a box)."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g = {c.geom1, c.geom2}
            if g & {self.model.geom("finger_l").id, self.model.geom("finger_r").id, self.model.geom("palm").id}:
                if self.table_geom in g or g & set(self.obj_geom):
                    return True
        return False

    # ---------------- ground truth: physics side-effects + evaluation only ----------------
    def truth_obj_pos(self, idx):
        return self.data.xpos[self.obj_body[idx]].copy()

    def _apply_damage(self):
        """An object squeezed beyond its limit is damaged (MuJoCo can't break things, so we mark it).
        Squeeze = total normal force from one pad on the object, low-passed over 30 ms."""
        per = {}
        f6 = np.zeros(6)
        pads = (self.pad["l"], self.pad["r"])
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in pads or g2 in pads:
                other, pad = (g2, g1) if g1 in pads else (g1, g2)
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                per[(other, pad)] = per.get((other, pad), 0.0) + abs(f6[0])
        a = self.model.opt.timestep / 0.03
        for idx, gid in enumerate(self.obj_geom):
            f = max(per.get((gid, pads[0]), 0.0), per.get((gid, pads[1]), 0.0))
            self._f_filt[idx] += a * (f - self._f_filt[idx])
            if idx in self.crushed:
                continue
            ff = self._f_filt[idx]
            if ff > self.peak_force[idx]:
                self.peak_force[idx] = ff
            if ff > self.slots[idx]["max_force"]:
                self.crushed.add(idx)
                self.model.geom_rgba[gid] = [0.9, 0.05, 0.05, 1]

    # ---------------- rendering ----------------
    def render(self, camera, w, h, depth=False):
        key = (w, h)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, h, w)
        r = self._renderers[key]
        if depth:
            r.enable_depth_rendering()
        r.update_scene(self.data, camera=camera)
        img = r.render().copy()
        if depth:
            r.disable_depth_rendering()
        return img

    DEPTH_NOISE = (0.001, 0.0025)   # depth noise sigma = a + b * depth^2 (m), like an active-stereo camera

    def read_wrist_camera(self, w=640, h=480):
        """Wrist camera as a sensor: RGB (uint8 HxWx3) + planar depth in metres (float32 HxW, with noise)."""
        rgb = self.render("wrist_cam", w, h)
        depth = self.render("wrist_cam", w, h, depth=True).astype(np.float32)
        a, b = self.DEPTH_NOISE
        depth = depth + self._rng.normal(0, 1, depth.shape).astype(np.float32) * (a + b * depth ** 2)
        return rgb, depth

    # ---------------- test-bench reset (not the robot) ----------------
    def reset_object(self, idx):
        """Put an object back on its slot after a trial (a lab technician resetting the bench).
        Crushed objects stay red so you can see what happened."""
        s = self.slots[idx]
        jid = self.model.body(f"body{idx}").jntadr[0]
        qa = self.model.jnt_qposadr[jid]
        va = self.model.jnt_dofadr[jid]
        self.data.qpos[qa:qa + 3] = [s["x"], s["y"], TABLE_Z + s["h"] / 2 + 0.001]
        self.data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
        self.data.qvel[va:va + 6] = 0
        mujoco.mj_forward(self.model, self.data)
