"""The 15 test objects, their physical properties, and WHERE EACH NUMBER COMES FROM.

Every property is a range (lo, nominal, hi) plus a confidence label:
  measured  - taken from a published measurement / standard / dataset
  derived   - calculated from published data or physics (e.g. density x volume)
  estimate  - no source found; our engineering guess, deliberately given a wide range
Each Monte-Carlo run samples every property from its range, so results come with a spread
instead of one hand-tuned number. `python objects.py` regenerates objects_sources.md.

Properties
  mass_g     mass
  mu         static friction against the gripper's rubber pads
  break_n    squeeze force per pad (N) at which the object is damaged (cracks/bruises/crumples)
  stiff      whole-object stiffness between two flat plates, N/mm (how much it gives when squeezed)

Shapes are MuJoCo primitives: sphere [r], ellipsoid [rx ry rz], cylinder [r, half_h],
box [hx hy hz]. The gripper closes along y. The controller never sees any of this.
"""
import numpy as np

TABLE_Z = 0.40
NO_DAMAGE = 5000.0   # objects that a 120 N gripper cannot damage

SOURCES = {
    "nedomova": ("Nedomova et al. 2009, Int. Agrophysics 23(3) - hen egg breaking force, quasi-static compression "
                 "(via Koprivica et al. 2021, J. Processing & Energy in Agriculture 25(3))",
                 "https://aseestant.ceon.rs/index.php/jpea/article/download/32280/19038"),
    "dikmen": ("Dikmen et al. 2017 - average egg mass 58.35 g (via Koprivica et al. 2021)",
               "https://aseestant.ceon.rs/index.php/jpea/article/download/32280/19038"),
    "mendelu_egg": ("Acta Univ. Agric. Silvic. Mendel. Brun. 2004, LII(5) - egg force vs shortening "
                    "(49.54 N -> 0.347 mm, i.e. ~140 N/mm)", "https://actavia.mendelu.cz/pdfs/acu/2004/05/07.pdf"),
    "uba_tomato": ("Uba et al. 2020, Scientific African 10:e00616 - ripe tomato fracture 16.88 N",
                   "https://doaj.org/article/b8999b086c46407eb1ae436d42832830"),
    "lak_tomato": ("Lak et al. 2018, CIGR Journal 20(3) - Roma tomato rupture force 57.85 N",
                   "https://core.ac.uk/download/427004305.pdf"),
    "ruiz_tomato": ("Ruiz Altisent et al. - processing tomatoes: compression resistance 3-9 N/mm",
                    "https://core.ac.uk/download/148666305.pdf"),
    "pttz_strawberry": ("Strawberry texture study (Instron): puncture force up to ~2.7 N, softest cultivars "
                        "<1.1 N/mm - puncture test, used only as a rough stiffness hint",
                        "https://journal.pttz.org/?p=2689"),
    "itf_tennis": ("ITF ball rules (quoted in US patent 4022469): 56.7-58.47 g, 6.35-6.68 cm, forward deformation "
                   "5.59-7.37 mm under 8.165 kg (80.1 N) -> 10.9-14.3 N/mm",
                   "https://patents.google.com/patent/US4022469"),
    "ycb": ("YCB Object and Model Set - measured masses (tomato soup can 0.349 kg)",
            "https://amarrmb.github.io/robosandbox/guides/bring-your-own-object/"),
    "pfa_pads": ("PFA Inc. gripper pads - knurled rubber pad friction 0.53 on steel, 0.78 on aluminium",
                 "https://www.pfa-inc.com/gripper-pads-overview/"),
    "etb_friction": ("Engineering ToolBox friction table - rubber on cardboard 0.5-0.8",
                     "https://engineeringtoolbox.com/friction-coefficients-d_778.html"),
    "rubber_glass": ("Slabeycius, Engineering Mechanics 2005 - rubber on dry glass, friction ~1.9-2.5 at low speed "
                     "(strongly pressure dependent)", "https://engmechx.it.cas.cz/improc/2005/Slabeycius-PT.pdf"),
    "paper_cup_patent": ("US patent 10358253 - a strengthened paper cup is specified at lateral pressing strength >= 4 N",
                         "https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10358253"),
    "physics": ("Physics / definitions (1 L of water = 1.0 kg; mass = density x volume)", ""),
    "rigid": ("Steel/glass/wood/aluminium are orders of magnitude stiffer than rubber pads - treated as rigid", ""),
    "no_damage": ("The gripper's max force (120 N) is far below what damages this object", ""),
    "estimate": ("No source found - engineering estimate with a deliberately wide range", ""),
}

R = lambda lo, nom, hi: (float(lo), float(nom), float(hi))
FIX = lambda v: (float(v), float(v), float(v))

_CATALOG = [
    dict(name="egg", shape="ellipsoid", size=[0.029, 0.022, 0.022], rgba="0.97 0.93 0.82 1",
         note="fragile shell; squeezed across its short axis",
         mass_g=R(50, 58, 66), mu=R(0.5, 0.65, 0.8), break_n=R(10, 36.4, 59), stiff=R(100, 140, 180),
         src=dict(mass_g="dikmen", mu="estimate", break_n="nedomova", stiff="mendelu_egg"),
         conf=dict(mass_g="measured", mu="estimate", break_n="measured", stiff="derived")),
    dict(name="strawberry", shape="sphere", size=[0.017], rgba="0.85 0.1 0.15 1",
         note="soft, bruises easily - weakest data in the set",
         mass_g=R(12, 18, 25), mu=R(0.6, 0.8, 1.0), break_n=R(2, 4, 6), stiff=R(0.8, 1.5, 3.0),
         src=dict(mass_g="estimate", mu="estimate", break_n="estimate", stiff="pttz_strawberry"),
         conf=dict(mass_g="estimate", mu="estimate", break_n="estimate", stiff="derived")),
    dict(name="tomato", shape="sphere", size=[0.032], rgba="0.9 0.25 0.15 1",
         note="ripe salad tomato; firm varieties are far stronger",
         mass_g=R(90, 110, 130), mu=R(0.5, 0.7, 0.9), break_n=R(10, 17, 58), stiff=R(3, 6, 9),
         src=dict(mass_g="estimate", mu="estimate", break_n="uba_tomato,lak_tomato", stiff="ruiz_tomato"),
         conf=dict(mass_g="estimate", mu="estimate", break_n="measured", stiff="measured")),
    dict(name="paper cup", shape="cylinder", size=[0.036, 0.045], rgba="0.95 0.95 0.95 1",
         note="empty; 'damage' = crumpling",
         mass_g=R(8, 10, 12), mu=R(0.5, 0.65, 0.8), break_n=R(2, 3.5, 5), stiff=R(0.3, 0.6, 1.0),
         src=dict(mass_g="estimate", mu="etb_friction", break_n="paper_cup_patent", stiff="estimate"),
         conf=dict(mass_g="estimate", mu="derived", break_n="derived", stiff="estimate")),
    dict(name="wine glass", shape="cylinder", size=[0.034, 0.07], rgba="0.75 0.88 0.95 0.6",
         note="dry glass grips rubber well; fragility unknown",
         mass_g=R(150, 180, 220), mu=R(0.9, 1.2, 2.0), break_n=R(30, 70, 150), stiff=FIX(1000),
         src=dict(mass_g="estimate", mu="rubber_glass", break_n="estimate", stiff="rigid"),
         conf=dict(mass_g="estimate", mu="measured", break_n="estimate", stiff="derived")),
    dict(name="tomato soup can", shape="cylinder", size=[0.0335, 0.051], rgba="0.8 0.15 0.12 1",
         note="YCB object 005, steel can, full",
         mass_g=FIX(349), mu=R(0.45, 0.53, 0.65), break_n=R(150, 250, 400), stiff=FIX(1000),
         src=dict(mass_g="ycb", mu="pfa_pads", break_n="estimate", stiff="rigid"),
         conf=dict(mass_g="measured", mu="measured", break_n="estimate", stiff="derived")),
    dict(name="water bottle 1L", shape="cylinder", size=[0.040, 0.11], rgba="0.35 0.6 0.95 0.8",
         note="full PET bottle; squeezes but does not break",
         mass_g=R(1015, 1025, 1035), mu=R(0.4, 0.55, 0.7), break_n=FIX(NO_DAMAGE), stiff=R(5, 10, 20),
         src=dict(mass_g="physics", mu="estimate", break_n="no_damage", stiff="estimate"),
         conf=dict(mass_g="derived", mu="estimate", break_n="derived", stiff="estimate")),
    dict(name="wooden block", shape="box", size=[0.03, 0.03, 0.03], rgba="0.75 0.55 0.3 1",
         note="6 cm cube; mass from wood density 500-750 kg/m3",
         mass_g=R(108, 130, 162), mu=R(0.5, 0.65, 0.8), break_n=FIX(NO_DAMAGE), stiff=FIX(1000),
         src=dict(mass_g="physics", mu="estimate", break_n="no_damage", stiff="rigid"),
         conf=dict(mass_g="derived", mu="estimate", break_n="derived", stiff="derived")),
    dict(name="tennis ball", shape="sphere", size=[0.0326], rgba="0.85 0.95 0.2 1",
         note="ITF-spec ball - best-documented object in the set",
         mass_g=R(56.7, 57.6, 58.5), mu=R(0.6, 0.75, 0.9), break_n=FIX(NO_DAMAGE), stiff=R(10.9, 12.5, 14.3),
         src=dict(mass_g="itf_tennis", mu="estimate", break_n="no_damage", stiff="itf_tennis"),
         conf=dict(mass_g="measured", mu="estimate", break_n="derived", stiff="measured")),
    dict(name="smartphone", shape="box", size=[0.075, 0.036, 0.0045], rgba="0.1 0.1 0.12 1",
         note="9 mm thin, lying flat; glass back",
         mass_g=R(170, 195, 230), mu=R(0.9, 1.2, 2.0), break_n=FIX(NO_DAMAGE), stiff=FIX(1000),
         src=dict(mass_g="estimate", mu="rubber_glass", break_n="no_damage", stiff="rigid"),
         conf=dict(mass_g="estimate", mu="measured", break_n="derived", stiff="derived")),
    dict(name="steel weight 3kg", shape="cylinder", size=[0.040, 0.05], rgba="0.3 0.3 0.33 1",
         note="plain steel",
         mass_g=FIX(3000), mu=R(0.45, 0.53, 0.65), break_n=FIX(NO_DAMAGE), stiff=FIX(1000),
         src=dict(mass_g="physics", mu="pfa_pads", break_n="no_damage", stiff="rigid"),
         conf=dict(mass_g="measured", mu="measured", break_n="derived", stiff="derived")),
    dict(name="wet glass jar", shape="cylinder", size=[0.045, 0.06], rgba="0.6 0.8 0.75 0.7",
         note="wet/soapy glass: water drastically lowers rubber friction",
         mass_g=R(500, 600, 700), mu=R(0.1, 0.2, 0.3), break_n=R(150, 250, 400), stiff=FIX(1000),
         src=dict(mass_g="estimate", mu="estimate", break_n="estimate", stiff="rigid"),
         conf=dict(mass_g="estimate", mu="estimate", break_n="estimate", stiff="derived")),
    dict(name="big cardboard box", shape="box", size=[0.07, 0.09, 0.07], rgba="0.72 0.58 0.4 1",
         note="18 cm wide - wider than the hand opens",
         mass_g=R(250, 300, 350), mu=R(0.5, 0.65, 0.8), break_n=R(20, 40, 60), stiff=R(1, 2, 4),
         src=dict(mass_g="estimate", mu="etb_friction", break_n="estimate", stiff="estimate"),
         conf=dict(mass_g="estimate", mu="measured", break_n="estimate", stiff="estimate")),
    dict(name="hardcover book", shape="box", size=[0.10, 0.06, 0.02], rgba="0.2 0.3 0.55 1",
         note="lying flat",
         mass_g=R(900, 1100, 1300), mu=R(0.5, 0.65, 0.8), break_n=FIX(NO_DAMAGE), stiff=R(30, 50, 80),
         src=dict(mass_g="estimate", mu="etb_friction", break_n="no_damage", stiff="estimate"),
         conf=dict(mass_g="estimate", mu="derived", break_n="derived", stiff="estimate")),
    dict(name="plush toy", shape="sphere", size=[0.045], rgba="0.85 0.6 0.75 1",
         note="squishy",
         mass_g=R(60, 80, 100), mu=R(0.8, 0.9, 1.0), break_n=FIX(NO_DAMAGE), stiff=R(0.2, 0.4, 0.8),
         src=dict(mass_g="estimate", mu="estimate", break_n="no_damage", stiff="estimate"),
         conf=dict(mass_g="estimate", mu="estimate", break_n="derived", stiff="estimate")),
]

PROPS = ("mass_g", "mu", "break_n", "stiff")
_COLS = [-0.44, -0.22, 0.0, 0.22, 0.44]
_ROWS = [0.25, 0.0, -0.25]


def _height(shape, size):
    i = {"sphere": 0, "ellipsoid": 2, "cylinder": 1, "box": 2}[shape]
    return 2 * size[i]


def _width_y(shape, size):
    i = {"sphere": 0, "ellipsoid": 1, "cylinder": 0, "box": 1}[shape]
    return 2 * size[i]


def _draw(spec, rng):
    lo, nom, hi = spec
    if lo == hi:
        return lo
    # truncated normal centred on the nominal value, ~95% of draws inside [lo, hi]
    sd = (hi - lo) / 4
    for _ in range(100):
        v = rng.normal(nom, sd)
        if lo <= v <= hi:
            return float(v)
    return float(np.clip(nom, lo, hi))


def nominal_params():
    return [{p: c[p][1] for p in PROPS} for c in _CATALOG]


def sample_params(seed):
    rng = np.random.default_rng(seed)
    return [{p: _draw(c[p], rng) for p in PROPS} for c in _CATALOG]


def slots(params=None):
    params = params or nominal_params()
    out = []
    for i, (c, p) in enumerate(zip(_CATALOG, params)):
        out.append(dict(idx=i, name=c["name"], shape=c["shape"], size=c["size"], rgba=c["rgba"], note=c["note"],
                        mass=p["mass_g"] / 1000, mu=p["mu"], max_force=p["break_n"], stiff=p["stiff"],
                        x=_COLS[i % 5], y=_ROWS[i // 5],
                        h=_height(c["shape"], c["size"]), width=_width_y(c["shape"], c["size"])))
    return out


def public_slot(s):
    """What the robot was 'taught': where slot i is. Nothing about the object itself."""
    return dict(idx=s["idx"], x=s["x"], y=s["y"])


def write_sources_md(path):
    units = dict(mass_g="g", mu="", break_n="N", stiff="N/mm")
    labels = dict(mass_g="mass", mu="friction vs rubber pad", break_n="damage force", stiff="stiffness")
    lines = ["# Where every number comes from", "",
             "Each property is a range **low / nominal / high**. Every simulation run draws a value from the range.",
             "Confidence: **measured** = published measurement or standard · **derived** = calculated from published "
             "data or physics · **estimate** = no source found, our guess with a wide range.", ""]
    counts = {"measured": 0, "derived": 0, "estimate": 0}
    for c in _CATALOG:
        lines += [f"## {c['name']}", f"_{c['note']}_", "", "| property | low / nominal / high | confidence | source |",
                  "|---|---|---|---|"]
        for p in PROPS:
            lo, nom, hi = c[p]
            if p == "break_n" and nom >= NO_DAMAGE:
                rng = "cannot be damaged by this gripper"
            elif p == "stiff" and nom >= 1000:
                rng = "rigid (simulated at the solver's max stiffness)"
            else:
                f = (lambda v: f"{v:.2f}") if p == "mu" else (lambda v: f"{v:g}")
                rng = f"{f(lo)} / {f(nom)} / {f(hi)} {units[p]}" if lo != hi else f"{f(nom)} {units[p]}"
            keys = c["src"][p].split(",")
            refs = "; ".join(f"[{SOURCES[k][0].split(' - ')[0]}]({SOURCES[k][1]})" if SOURCES[k][1] else SOURCES[k][0]
                             for k in keys)
            conf = c["conf"][p]
            counts[conf] += 1
            lines.append(f"| {labels[p]} | {rng} | **{conf}** | {refs} |")
        lines.append("")
    total = sum(counts.values())
    lines.insert(5, f"**Overall: {counts['measured']} measured, {counts['derived']} derived, {counts['estimate']} "
                    f"estimated out of {total} numbers.**\n")
    lines += ["## Full references", ""]
    for k, (label, url) in SOURCES.items():
        lines.append(f"- **{k}**: {label}" + (f" — {url}" if url else ""))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return counts


if __name__ == "__main__":
    import os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "objects_sources.md")
    print(write_sources_md(p), "->", p)
