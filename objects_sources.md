# Where every number comes from

Each property is a range **low / nominal / high**. Every simulation run draws a value from the range.
Confidence: **measured** = published measurement or standard · **derived** = calculated from published data or physics · **estimate** = no source found, our guess with a wide range.

**Overall: 13 measured, 20 derived, 27 estimated out of 60 numbers.**

## egg
_fragile shell; squeezed across its short axis_

| property | low / nominal / high | confidence | source |
|---|---|---|---|
| mass | 50 / 58 / 66 g | **measured** | [Dikmen et al. 2017](https://aseestant.ceon.rs/index.php/jpea/article/download/32280/19038) |
| friction vs rubber pad | 0.50 / 0.65 / 0.80  | **estimate** | No source found - engineering estimate with a deliberately wide range |
| damage force | 10 / 36.4 / 59 N | **measured** | [Nedomova et al. 2009, Int. Agrophysics 23(3)](https://aseestant.ceon.rs/index.php/jpea/article/download/32280/19038) |
| stiffness | 100 / 140 / 180 N/mm | **derived** | [Acta Univ. Agric. Silvic. Mendel. Brun. 2004, LII(5)](https://actavia.mendelu.cz/pdfs/acu/2004/05/07.pdf) |

## strawberry
_soft, bruises easily - weakest data in the set_

| property | low / nominal / high | confidence | source |
|---|---|---|---|
| mass | 12 / 18 / 25 g | **estimate** | No source found - engineering estimate with a deliberately wide range |
| friction vs rubber pad | 0.60 / 0.80 / 1.00  | **estimate** | No source found - engineering estimate with a deliberately wide range |
| damage force | 2 / 4 / 6 N | **estimate** | No source found - engineering estimate with a deliberately wide range |
| stiffness | 0.8 / 1.5 / 3 N/mm | **derived** | [Strawberry texture study (Instron): puncture force up to ~2.7 N, softest cultivars <1.1 N/mm](https://journal.pttz.org/?p=2689) |

## tomato
_ripe salad tomato; firm varieties are far stronger_

| property | low / nominal / high | confidence | source |
|---|---|---|---|
| mass | 90 / 110 / 130 g | **estimate** | No source found - engineering estimate with a deliberately wide range |
| friction vs rubber pad | 0.50 / 0.70 / 0.90  | **estimate** | No source found - engineering estimate with a deliberately wide range |
| damage force | 10 / 17 / 58 N | **measured** | [Uba et al. 2020, Scientific African 10:e00616](https://doaj.org/article/b8999b086c46407eb1ae436d42832830); [Lak et al. 2018, CIGR Journal 20(3)](https://core.ac.uk/download/427004305.pdf) |
| stiffness | 3 / 6 / 9 N/mm | **measured** | [Ruiz Altisent et al.](https://core.ac.uk/download/148666305.pdf) |

## paper cup
_empty; 'damage' = crumpling_

| property | low / nominal / high | confidence | source |
|---|---|---|---|
| mass | 8 / 10 / 12 g | **estimate** | No source found - engineering estimate with a deliberately wide range |
| friction vs rubber pad | 0.50 / 0.65 / 0.80  | **derived** | [Engineering ToolBox friction table](https://engineeringtoolbox.com/friction-coefficients-d_778.html) |
| damage force | 2 / 3.5 / 5 N | **derived** | [US patent 10358253](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10358253) |
| stiffness | 0.3 / 0.6 / 1 N/mm | **estimate** | No source found - engineering estimate with a deliberately wide range |

## wine glass
_dry glass grips rubber well; fragility unknown_

| property | low / nominal / high | confidence | source |
|---|---|---|---|
| mass | 150 / 180 / 220 g | **estimate** | No source found - engineering estimate with a deliberately wide range |
| friction vs rubber pad | 0.90 / 1.20 / 2.00  | **measured** | [Slabeycius, Engineering Mechanics 2005](https://engmechx.it.cas.cz/improc/2005/Slabeycius-PT.pdf) |
| damage force | 30 / 70 / 150 N | **estimate** | No source found - engineering estimate with a deliberately wide range |
| stiffness | rigid (simulated at the solver's max stiffness) | **derived** | Steel/glass/wood/aluminium are orders of magnitude stiffer than rubber pads - treated as rigid |

## tomato soup can
_YCB object 005, steel can, full_

| property | low / nominal / high | confidence | source |
|---|---|---|---|
| mass | 349 g | **measured** | [YCB Object and Model Set](https://amarrmb.github.io/robosandbox/guides/bring-your-own-object/) |
| friction vs rubber pad | 0.45 / 0.53 / 0.65  | **measured** | [PFA Inc. gripper pads](https://www.pfa-inc.com/gripper-pads-overview/) |
| damage force | 150 / 250 / 400 N | **estimate** | No source found - engineering estimate with a deliberately wide range |
| stiffness | rigid (simulated at the solver's max stiffness) | **derived** | Steel/glass/wood/aluminium are orders of magnitude stiffer than rubber pads - treated as rigid |

## water bottle 1L
_full PET bottle; squeezes but does not break_

| property | low / nominal / high | confidence | source |
|---|---|---|---|
| mass | 1015 / 1025 / 1035 g | **derived** | Physics / definitions (1 L of water = 1.0 kg; mass = density x volume) |
| friction vs rubber pad | 0.40 / 0.55 / 0.70  | **estimate** | No source found - engineering estimate with a deliberately wide range |
| damage force | cannot be damaged by this gripper | **derived** | The gripper's max force (120 N) is far below what damages this object |
| stiffness | 5 / 10 / 20 N/mm | **estimate** | No source found - engineering estimate with a deliberately wide range |

## wooden block
_6 cm cube; mass from wood density 500-750 kg/m3_

| property | low / nominal / high | confidence | source |
|---|---|---|---|
| mass | 108 / 130 / 162 g | **derived** | Physics / definitions (1 L of water = 1.0 kg; mass = density x volume) |
| friction vs rubber pad | 0.50 / 0.65 / 0.80  | **estimate** | No source found - engineering estimate with a deliberately wide range |
| damage force | cannot be damaged by this gripper | **derived** | The gripper's max force (120 N) is far below what damages this object |
| stiffness | rigid (simulated at the solver's max stiffness) | **derived** | Steel/glass/wood/aluminium are orders of magnitude stiffer than rubber pads - treated as rigid |

## tennis ball
_ITF-spec ball - best-documented object in the set_

| property | low / nominal / high | confidence | source |
|---|---|---|---|
| mass | 56.7 / 57.6 / 58.5 g | **measured** | [ITF ball rules (quoted in US patent 4022469): 56.7-58.47 g, 6.35-6.68 cm, forward deformation 5.59-7.37 mm under 8.165 kg (80.1 N) -> 10.9-14.3 N/mm](https://patents.google.com/patent/US4022469) |
| friction vs rubber pad | 0.60 / 0.75 / 0.90  | **estimate** | No source found - engineering estimate with a deliberately wide range |
| damage force | cannot be damaged by this gripper | **derived** | The gripper's max force (120 N) is far below what damages this object |
| stiffness | 10.9 / 12.5 / 14.3 N/mm | **measured** | [ITF ball rules (quoted in US patent 4022469): 56.7-58.47 g, 6.35-6.68 cm, forward deformation 5.59-7.37 mm under 8.165 kg (80.1 N) -> 10.9-14.3 N/mm](https://patents.google.com/patent/US4022469) |

## smartphone
_9 mm thin, lying flat; glass back_

| property | low / nominal / high | confidence | source |
|---|---|---|---|
| mass | 170 / 195 / 230 g | **estimate** | No source found - engineering estimate with a deliberately wide range |
| friction vs rubber pad | 0.90 / 1.20 / 2.00  | **measured** | [Slabeycius, Engineering Mechanics 2005](https://engmechx.it.cas.cz/improc/2005/Slabeycius-PT.pdf) |
| damage force | cannot be damaged by this gripper | **derived** | The gripper's max force (120 N) is far below what damages this object |
| stiffness | rigid (simulated at the solver's max stiffness) | **derived** | Steel/glass/wood/aluminium are orders of magnitude stiffer than rubber pads - treated as rigid |

## steel weight 3kg
_plain steel_

| property | low / nominal / high | confidence | source |
|---|---|---|---|
| mass | 3000 g | **measured** | Physics / definitions (1 L of water = 1.0 kg; mass = density x volume) |
| friction vs rubber pad | 0.45 / 0.53 / 0.65  | **measured** | [PFA Inc. gripper pads](https://www.pfa-inc.com/gripper-pads-overview/) |
| damage force | cannot be damaged by this gripper | **derived** | The gripper's max force (120 N) is far below what damages this object |
| stiffness | rigid (simulated at the solver's max stiffness) | **derived** | Steel/glass/wood/aluminium are orders of magnitude stiffer than rubber pads - treated as rigid |

## wet glass jar
_wet/soapy glass: water drastically lowers rubber friction_

| property | low / nominal / high | confidence | source |
|---|---|---|---|
| mass | 500 / 600 / 700 g | **estimate** | No source found - engineering estimate with a deliberately wide range |
| friction vs rubber pad | 0.10 / 0.20 / 0.30  | **estimate** | No source found - engineering estimate with a deliberately wide range |
| damage force | 150 / 250 / 400 N | **estimate** | No source found - engineering estimate with a deliberately wide range |
| stiffness | rigid (simulated at the solver's max stiffness) | **derived** | Steel/glass/wood/aluminium are orders of magnitude stiffer than rubber pads - treated as rigid |

## big cardboard box
_18 cm wide - wider than the hand opens_

| property | low / nominal / high | confidence | source |
|---|---|---|---|
| mass | 250 / 300 / 350 g | **estimate** | No source found - engineering estimate with a deliberately wide range |
| friction vs rubber pad | 0.50 / 0.65 / 0.80  | **measured** | [Engineering ToolBox friction table](https://engineeringtoolbox.com/friction-coefficients-d_778.html) |
| damage force | 20 / 40 / 60 N | **estimate** | No source found - engineering estimate with a deliberately wide range |
| stiffness | 1 / 2 / 4 N/mm | **estimate** | No source found - engineering estimate with a deliberately wide range |

## hardcover book
_lying flat_

| property | low / nominal / high | confidence | source |
|---|---|---|---|
| mass | 900 / 1100 / 1300 g | **estimate** | No source found - engineering estimate with a deliberately wide range |
| friction vs rubber pad | 0.50 / 0.65 / 0.80  | **derived** | [Engineering ToolBox friction table](https://engineeringtoolbox.com/friction-coefficients-d_778.html) |
| damage force | cannot be damaged by this gripper | **derived** | The gripper's max force (120 N) is far below what damages this object |
| stiffness | 30 / 50 / 80 N/mm | **estimate** | No source found - engineering estimate with a deliberately wide range |

## plush toy
_squishy_

| property | low / nominal / high | confidence | source |
|---|---|---|---|
| mass | 60 / 80 / 100 g | **estimate** | No source found - engineering estimate with a deliberately wide range |
| friction vs rubber pad | 0.80 / 0.90 / 1.00  | **estimate** | No source found - engineering estimate with a deliberately wide range |
| damage force | cannot be damaged by this gripper | **derived** | The gripper's max force (120 N) is far below what damages this object |
| stiffness | 0.2 / 0.4 / 0.8 N/mm | **estimate** | No source found - engineering estimate with a deliberately wide range |

## Full references

- **nedomova**: Nedomova et al. 2009, Int. Agrophysics 23(3) - hen egg breaking force, quasi-static compression (via Koprivica et al. 2021, J. Processing & Energy in Agriculture 25(3)) — https://aseestant.ceon.rs/index.php/jpea/article/download/32280/19038
- **dikmen**: Dikmen et al. 2017 - average egg mass 58.35 g (via Koprivica et al. 2021) — https://aseestant.ceon.rs/index.php/jpea/article/download/32280/19038
- **mendelu_egg**: Acta Univ. Agric. Silvic. Mendel. Brun. 2004, LII(5) - egg force vs shortening (49.54 N -> 0.347 mm, i.e. ~140 N/mm) — https://actavia.mendelu.cz/pdfs/acu/2004/05/07.pdf
- **uba_tomato**: Uba et al. 2020, Scientific African 10:e00616 - ripe tomato fracture 16.88 N — https://doaj.org/article/b8999b086c46407eb1ae436d42832830
- **lak_tomato**: Lak et al. 2018, CIGR Journal 20(3) - Roma tomato rupture force 57.85 N — https://core.ac.uk/download/427004305.pdf
- **ruiz_tomato**: Ruiz Altisent et al. - processing tomatoes: compression resistance 3-9 N/mm — https://core.ac.uk/download/148666305.pdf
- **pttz_strawberry**: Strawberry texture study (Instron): puncture force up to ~2.7 N, softest cultivars <1.1 N/mm - puncture test, used only as a rough stiffness hint — https://journal.pttz.org/?p=2689
- **itf_tennis**: ITF ball rules (quoted in US patent 4022469): 56.7-58.47 g, 6.35-6.68 cm, forward deformation 5.59-7.37 mm under 8.165 kg (80.1 N) -> 10.9-14.3 N/mm — https://patents.google.com/patent/US4022469
- **ycb**: YCB Object and Model Set - measured masses (tomato soup can 0.349 kg) — https://amarrmb.github.io/robosandbox/guides/bring-your-own-object/
- **pfa_pads**: PFA Inc. gripper pads - knurled rubber pad friction 0.53 on steel, 0.78 on aluminium — https://www.pfa-inc.com/gripper-pads-overview/
- **etb_friction**: Engineering ToolBox friction table - rubber on cardboard 0.5-0.8 — https://engineeringtoolbox.com/friction-coefficients-d_778.html
- **rubber_glass**: Slabeycius, Engineering Mechanics 2005 - rubber on dry glass, friction ~1.9-2.5 at low speed (strongly pressure dependent) — https://engmechx.it.cas.cz/improc/2005/Slabeycius-PT.pdf
- **paper_cup_patent**: US patent 10358253 - a strengthened paper cup is specified at lateral pressing strength >= 4 N — https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10358253
- **physics**: Physics / definitions (1 L of water = 1.0 kg; mass = density x volume)
- **rigid**: Steel/glass/wood/aluminium are orders of magnitude stiffer than rubber pads - treated as rigid
- **no_damage**: The gripper's max force (120 N) is far below what damages this object
- **estimate**: No source found - engineering estimate with a deliberately wide range
