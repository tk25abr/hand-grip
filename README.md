# Grip Lab — 15-object pick test (AgentVerse 2026, Robotics)

A two-finger gripper on a 3-axis gantry tries to pick up 15 very different objects: an egg, a
strawberry, a wine glass, a 3 kg steel weight, a wet jar, a phone... Every sensor on the gripper
is shown live, and **every object property comes with a source and a range** (`objects_sources.md`).

Two policies run on the same objects and the same Monte-Carlo property draws:

* **baseline**: a "dumb" industrial gripper — same squeeze force and same grasp height for every
  object, **no camera, no AI, no feedback**. The "before".
* **smart**: looks first (wrist depth camera + one Gemini call per object), then feels (stiffness,
  weight, friction, slip) and corrects the guess while grasping. Refuses objects it cannot hold safely.

## Run it on your Mac

```bash
cd grip_lab
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt

mjpython run.py                                   # live window (on macOS it MUST be mjpython)
python run.py --mode bench                        # one run, nominal object properties (~20 s)
python run.py --mode bench --runs 30              # 30 runs, properties drawn from real ranges (~5 min)
python run.py --mode record --out baseline.mp4    # video: all camera views + sensors
python run.py --force 10                          # try a gentler dumb gripper
python run.py --sample 7                          # live run with randomly drawn properties

export GEMINI_API_KEY=...                         # smart policy; key only from this variable
python guess_all.py                               # first guess for all 15 (fills the cache, ~15 calls)
python run.py --mode bench --policy smart         # one smart run
python run.py --mode record --policy smart --out smart.mp4
mjpython run.py --policy smart                    # live window with LOOK / FEEL / DECIDE panel
python -m pytest tests/ -q                        # incl. the "never reads ground truth" test
python calibrate.py                               # redo the stiffness calibration (~30 s)
python objects.py                                 # regenerate objects_sources.md
```

Keys in the live window: **N** next object · **P** pause · **R** restart · **U / J** force ±5 N.

## Where the numbers come from (realism)

1. **Every property has a source or is labelled an estimate.** Mass, friction against the rubber
   pads, damage force and stiffness for all 15 objects — **13 measured, 20 derived, 27 estimated
   out of 60 numbers**. Details and links: `objects_sources.md`. Highlights:
   - egg breaks at 36.4 N on average, range 10–59 N (Nedomova et al. 2009); ~140 N/mm stiff
   - ripe tomato bursts at ~17 N (Uba et al. 2020), firm varieties ~58 N; 3–9 N/mm
   - tennis ball stiffness straight from the ITF rules (5.6–7.4 mm under 80 N)
   - rubber pads grip steel at μ≈0.53 and aluminium at 0.78 (gripper pad manufacturer data);
     dry glass grips rubber *very* well — a dry wine glass is **not** slippery
   - tomato soup can mass from the YCB benchmark set (349 g)
2. **Real stiffness, calibrated.** Each object's pad contact is tuned so that squeezing it gives its
   real N/mm, checked with a virtual compression test against the published data
   (`results/compression_test.png`): tomato, tennis ball and strawberry land inside the measured
   range. That makes "feeling the material" possible — a tomato gives under the fingers, a block doesn't.
3. **Ranges, not one number.** Real eggs don't all break at the same force. Every run draws each
   property from its range; we report the success rate over many runs, not a single tuned result.

## Baseline results (fixed force, no camera, no AI)

Mean objects picked up per run (16–30 runs each, `results/baseline_*N_montecarlo.md`):

| fixed force | picked up | what works | what fails |
|---|---|---|---|
| 10 N | **7.1 / 15** | egg, tomato, glass, can, block, ball | steel weight, wet jar, book slip out |
| 40 N | **8.1 / 15** | can, block, ball, jar, book, plush; egg 50/50 | strawberry, tomato, cup crushed; egg cracks half the time |
| 60 N | **7.6 / 15** | steel weight, jar, book, can... | egg, strawberry, tomato, cup always crushed |

Always failing at any force: the phone (9 mm, below the pads), the 18 cm box (wider than the hand),
the 22 cm bottle (palm hits its top at the fixed grasp height).

**Each force wins on a different set of objects — there is no single right squeeze.** The gripper has
to work out each object. That is the case for the smart version.

## What the mechanism looks like

```
        ┌───────── gantry frame ─────────┐
        │   X/Y carriage  ──►  Z column   │     3-axis Cartesian robot: simple, real, no IK
        │               [wrist F/T sensor]│     ← weight, impacts, acceleration
        │                 ┌────┴────┐     │
        │               finger    finger  │     parallel jaws, force-controlled motors
        │               [pad]      [pad]  │     ← tactile pads (normal + shear force)
        │                                 │     + finger encoders (gap = object width)
        │   wrist camera RGB + depth ──►  │     ← rendered, not used by the baseline
        └─────────────────────────────────┘
```

## Sensors (real MuJoCo sensors, with noise)

| Sensor | Shows | Used for |
|---|---|---|
| tactile pads L/R | normal force (N), pressure (kPa) | squeeze, crush risk |
| pad shear / normal | tangential vs normal force | slip risk |
| finger encoders | gap (mm) | width by touch, softness (gap change vs force), missed grasp |
| wrist force/torque | load change after lifting (g) | weight, impacts |
| wrist accelerometer | m/s² | shake test |
| object lift (ground truth) | cm | scoring only — the robot never sees it |

## Honest limits (say them before a judge asks)

* **27 of 60 numbers are estimates** — mostly frictions and the strawberry/wine-glass/jar damage
  limits, where we found no measurement. They have wide ranges, and the Monte-Carlo spread includes them.
* **The egg is about 2× too soft in simulation** (~65 N/mm vs ~140 N/mm real): that's the stiffest
  contact the solver allows for such a light object at a 1 ms step. The difference (~0.3 mm at 40 N)
  is close to the finger encoder noise. Same cap for other light rigid things.
* MuJoCo can't break objects: crossing the damage force (pad force low-passed over 30 ms) marks the
  object CRUSHED and turns it red. Real damage also depends on direction, speed and contact area.
* Objects are simple shapes (no wine-glass stem, no bottle neck), with uniform mass (no sloshing).
* The tactile pad gives one total force, not a pressure map; no slip vibration sensing.
* The robot knows only each slot's taught position, never the object's name or properties.
* **Smart policy limits.** MuJoCo friction under slow sliding is chattery (pads lose contact for
  a tick), so slip is detected from the shear/normal ratio, not from the wrist load. The depth image
  sees glass and clear plastic like any surface (real depth cameras often do not). The vision model
  sees simple coloured shapes, so its guess leans on the measured size we give it; it cannot tell
  that the jar is wet — the squeeze test finds that out by touch.

## The smart policy (`SmartPolicy` in `policies.py`)

It may only use the taught slot x/y, `sim.read_sensors()` and the wrist camera (RGB + depth). It never
reads `sim.slots` — `tests/test_no_ground_truth.py` fails if it does (static check + a runtime guard,
plus a deliberately cheating policy that the guard must catch).

1. **Look (TARE, once per object).** `perception.py` turns the depth image into a point cloud, cuts out
   the robot's own fingers and finds the object above the table at the taught slot: height, width
   across the jaws, shape (heights within ~1 mm, widths within ~2–6 mm on all 15 objects).
   `vlm.py` then asks the vision model **once**, before touching, with a crop of the object and the
   measured size: Gemini (Interactions API, `gemini-3.8-flash`, `--model` to change) → local Gemma
   via an Ollama-compatible endpoint (`GEMMA_URL`) → depth-only prior if neither is available. The
   answer is parsed defensively and cached by picture in `results/gemini_cache.json`, so 30 runs do
   not make 450 calls. The dashboard shows which source answered and how long it took.
2. **Plan.** Pads at mid-height, never lower than 4 mm, and high enough that the palm (165 mm above
   the pad bottom) clears the top. **Declines** objects wider than the hand opens. Safe cap = lower
   bound of the guessed safe squeeze (x 0.6 if fragile).
3. **Close.** Fast approach to the depth-measured width, gentle touch at 1 N, stiffness probe
   (gap change vs pad force): soft -> cap at 0.6 x the safe lower bound.
4. **Micro-lift + weigh.** Lift 1 cm slowly. Squeeze test: grip 25 % harder — if the pad shear/normal
   ratio falls, the object is held; if it stays at the friction limit, it is sliding (and that ratio
   *is* its friction): tighten and test again. Hold 0.3 s, wrist load = weight ("guessed 2-160 g,
   wrist says 56 g"). Needed squeeze = 1.5 x m x (g + 5 m/s²) / (2 x friction). Above the cap: put it
   back down and **DECLINE** with the reason.
5. **Lift / hold / shake.** Slip reflex: shear/normal above 0.8 x friction -> +15 % (never above the cap).

`adjust()` returns `None`, a force, or a dict (`force`, `abort`, `done`, `lift_speed`, `say`).
DECLINED is its own outcome: not a success, not damage.

**Smart, nominal properties, depth-only fallback (no API key): 11 / 15 picked up, 4 declined, 0
crushed / slipped / dropped** (`results/smart_results.md`). Declined: the 18 cm box (too wide); the
steel weight, wet jar and full bottle (the depth-only prior assumes an unknown material may break
above ~25 N, and these need more). A vision model that recognises steel or a PET bottle raises that cap.

## Files

`objects.py` objects, ranges, sources · `objects_sources.md` source table · `calibrate.py`
stiffness calibration · `calibration.json` · `scene.py` MJCF · `sim.py` physics, sensors, damage ·
`trial.py` pick sequence + judge · `policies.py` baseline + smart · `perception.py` depth -> size ·
`vlm.py` Gemini / Gemma / depth-only first guess + cache · `guess_all.py` first guess for all 15 ·
`dashboard.py` · `run.py` · `tests/` · `results/`
