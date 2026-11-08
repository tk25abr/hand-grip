"""MJCF: table, 15 objects, an overhead gantry (XYZ carriage) and a parallel two-finger gripper.

Gripper layout (all sensors are real MuJoCo sensors):
  carriage (slide x/y/z, position-controlled)
   └─ hand  (rigid)      site 'wrist'  -> force + torque sensor (weight, impacts), accelerometer
       ├─ finger_l (slide, force-controlled motor)  pad_l + site 'tact_l' -> touch sensor
       └─ finger_r (slide, force-controlled motor)  pad_r + site 'tact_r' -> touch sensor
  finger joint positions -> aperture (how wide the hand is open = object width by touch)
"""
from objects import slots, TABLE_Z

PAD_HALF = (0.014, 0.003, 0.022)   # pad: 2.8 x 4.4 cm contact face
PAD_AREA = (2 * PAD_HALF[0]) * (2 * PAD_HALF[2])   # m^2
FINGER_Y0 = 0.080                  # finger centre offset from palm centre
FINGER_HALF = (0.014, 0.006, 0.11)   # 22 cm long fingers: palm stays above tall objects
MAX_TRAVEL = 0.066                 # per finger: opening 13.6 cm -> 0.4 cm
PAD_Z = -0.215 + PAD_HALF[2]       # pad centre relative to the hand frame (pad bottom = finger tip)
HOME = (0.0, 0.0, TABLE_Z + 0.55)
MAX_OPENING = 2 * (FINGER_Y0 - FINGER_HALF[1] - 2 * PAD_HALF[1])   # 0.136 m between the pads, fully open
PALM_ABOVE_PAD = (-0.035 - 0.015) - (PAD_Z - PAD_HALF[2])          # palm bottom is 0.165 m above the pad bottom

# wrist camera mount on the hand (robot calibration, known to the controller): RGB + depth
WRIST_CAM_POS = (0.06, 0.0, -0.06)
WRIST_CAM_XYAXES = (0, -1, 0, 0.94, 0, 0.34)    # looks down and ~20 deg forward (+x)
WRIST_CAM_FOVY = 75.0


TIMESTEP = 0.001          # 1 ms: lets contacts be stiff enough for eggs / cans
TC_MIN, TC_DEFAULT = 0.002, 0.004   # contact time constant range (>= 2 x timestep for stability)


def _obj_geom(s):
    sz = " ".join(f"{v}" for v in s["size"])
    return (f'<geom name="obj{s["idx"]}" type="{s["shape"]}" size="{sz}" rgba="{s["rgba"]}" '
            f'mass="{s["mass"]}" friction="0.6 0.005 0.0001" condim="4"/>')


def _pad_pairs(s, tc):
    """Pad <-> object contact: the pair's friction = object's friction against rubber, and its softness
    (solref time constant) is calibrated so the object has its real stiffness in N/mm (calibrate.py)."""
    mu = s["mu"]
    return "".join(
        f'<pair geom1="pad_{side}" geom2="obj{s["idx"]}" condim="4" friction="{mu} {mu} 0.005 0.0001 0.0001" '
        f'solref="{tc} 1" solimp="0.9 0.95 0.001"/>' for side in ("l", "r"))


def build_mjcf(slot_list=None, tc=None):
    """slot_list: objects with concrete properties (objects.slots(params)); tc: contact time constant
    per object (from the stiffness calibration). Defaults = nominal objects, default softness."""
    slot_list = slot_list or slots()
    tc = tc or [TC_DEFAULT] * len(slot_list)
    objs, pairs, excl = [], [], []
    for s in slot_list:
        pairs.append(_pad_pairs(s, tc[s["idx"]]))
        excl.append(f'<exclude body1="finger_l" body2="body{s["idx"]}"/><exclude body1="finger_r" body2="body{s["idx"]}"/>')
    for s in slot_list:
        z = TABLE_Z + s["h"] / 2 + 0.001
        objs.append(f'<body name="body{s["idx"]}" pos="{s["x"]} {s["y"]} {z}">'
                    f'<freejoint name="free{s["idx"]}"/>{_obj_geom(s)}</body>')

    px, py, pz = PAD_HALF
    fx, fy, fz = FINGER_HALF
    pad_y = fy + py  # pad sits on the inner face of the finger
    finger = lambda side, sgn: f"""
        <body name="finger_{side}" pos="0 {sgn * FINGER_Y0} 0">
          <joint name="j_{side}" type="slide" axis="0 {-sgn} 0" limited="true" range="0 {MAX_TRAVEL}" damping="150" armature="0.3" solreflimit="0.004 1"/>
          <geom name="finger_{side}" type="box" size="{fx} {fy} {fz}" pos="0 0 {0.005 - fz}" rgba="0.25 0.27 0.3 1" mass="0.08" friction="0.05"/>
          <geom name="pad_{side}" type="box" size="{px} {py} {pz}" pos="0 {-sgn * pad_y} {PAD_Z}" rgba="0.15 0.15 0.15 1"
                mass="0.01" friction="0.05 0.005 0.0001" condim="4" solref="0.004 1" solimp="0.95 0.99 0.001"/>
          <site name="tact_{side}" type="box" size="{px + 0.001} {py + 0.002} {pz + 0.001}" pos="0 {-sgn * pad_y} {PAD_Z}" rgba="0.1 0.8 0.4 0.25"/>
        </body>"""

    gantry = """
    <geom type="box" pos="0 0.62 1.25" size="0.85 0.03 0.03" rgba="0.75 0.75 0.78 1" contype="0" conaffinity="0"/>
    <geom type="box" pos="0 -0.62 1.25" size="0.85 0.03 0.03" rgba="0.75 0.75 0.78 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" fromto="0.85 0.62 0 0.85 0.62 1.25" size="0.025" rgba="0.6 0.6 0.63 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" fromto="-0.85 0.62 0 -0.85 0.62 1.25" size="0.025" rgba="0.6 0.6 0.63 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" fromto="0.85 -0.62 0 0.85 -0.62 1.25" size="0.025" rgba="0.6 0.6 0.63 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" fromto="-0.85 -0.62 0 -0.85 -0.62 1.25" size="0.025" rgba="0.6 0.6 0.63 1" contype="0" conaffinity="0"/>"""

    return f"""
<mujoco model="grip_lab">
  <compiler angle="radian"/>
  <option timestep="{TIMESTEP}" cone="elliptic" impratio="10" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.4 0.4 0.4"/>
    <map znear="0.005"/>
  </visual>
  <asset>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.3 0.32 0.36" rgb2="0.26 0.28 0.32" width="256" height="256"/>
    <material name="floor" texture="floor" texrepeat="8 8"/>
    <texture name="wood" type="2d" builtin="flat" rgb1="0.62 0.5 0.38" width="64" height="64"/>
    <material name="table" texture="wood"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.4 2.2" dir="-0.1 0.2 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floor"/>
    <geom name="table" type="box" pos="0 0 {TABLE_Z / 2}" size="0.62 0.42 {TABLE_Z / 2}" material="table" friction="0.6"/>
    <camera name="main" pos="0.0 -1.2 1.25" xyaxes="1 0 0 0 0.58 0.81" fovy="52"/>
    {gantry}
    {''.join(objs)}
    <body name="carriage" pos="{HOME[0]} {HOME[1]} {HOME[2]}" gravcomp="1">
      <joint name="gx" type="slide" axis="1 0 0" damping="150"/>
      <joint name="gy" type="slide" axis="0 1 0" damping="150"/>
      <joint name="gz" type="slide" axis="0 0 1" damping="300"/>
      <geom type="box" size="0.06 0.06 0.03" pos="0 0 0.50" rgba="0.85 0.4 0.1 1" mass="1" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.018 0.25" pos="0 0 0.24" rgba="0.7 0.7 0.72 1" mass="0.5" contype="0" conaffinity="0"/>
      <body name="hand" pos="0 0 0" gravcomp="1">
        <site name="wrist" pos="0 0 0" size="0.01"/>
        <geom type="cylinder" size="0.035 0.012" pos="0 0 -0.008" rgba="0.1 0.1 0.12 1" mass="0.2" contype="0" conaffinity="0"/>
        <geom name="palm" type="box" size="0.03 0.105 0.015" pos="0 0 -0.035" rgba="0.85 0.4 0.1 1" mass="0.4"/>
        <camera name="close" pos="0.18 -0.30 -0.10" xyaxes="0.857 0.515 0 -0.128 0.213 0.969" fovy="55"/>
        <camera name="wrist_cam" pos="{' '.join(map(str, WRIST_CAM_POS))}" xyaxes="{' '.join(map(str, WRIST_CAM_XYAXES))}" fovy="{WRIST_CAM_FOVY:g}"/>
        {finger('l', 1)}
        {finger('r', -1)}
      </body>
    </body>
  </worldbody>
  <contact>
    <exclude body1="finger_l" body2="finger_r"/>
    {''.join(excl)}
    {''.join(pairs)}
  </contact>
  <actuator>
    <position name="ax" joint="gx" kp="4000"/>
    <position name="ay" joint="gy" kp="4000"/>
    <position name="az" joint="gz" kp="12000" forcerange="-60 400"/>
    <motor name="grip_l" joint="j_l" gear="1" ctrlrange="-30 120"/>
    <motor name="grip_r" joint="j_r" gear="1" ctrlrange="-30 120"/>
  </actuator>
  <sensor>
    <force name="wrist_force" site="wrist"/>
    <torque name="wrist_torque" site="wrist"/>
    <accelerometer name="wrist_acc" site="wrist"/>
    <touch name="touch_l" site="tact_l"/>
    <touch name="touch_r" site="tact_r"/>
    <jointpos name="pos_l" joint="j_l"/>
    <jointpos name="pos_r" joint="j_r"/>
  </sensor>
</mujoco>
"""
