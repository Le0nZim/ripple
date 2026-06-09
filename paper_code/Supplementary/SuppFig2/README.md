# Supplementary Fig. 2 — resolution dependence of APP (256×256 rescale)

How APP and annotator disagreement change when every dataset is rescaled to a
common 256×256 grid (the resolution the APP metric is adopted from).

- **(a)** Dataset-level APP vs. total annotations after the 256×256 rescale
  (diamond = manual reference, square = RIPPLE).
- **(b)** RIPPLE/Linear APP vs. corrections at 256 scale (shipped SVGs).
- **(c)** Annotator disagreement after the 256×256 rescale (box + CDF).

## Pipeline

```
source_data/  ──(generate_data.py)──►  data/suppfig2_inputs.json  ──(reproduce_suppfig2.py)──►  results/{panel_a,disagreement_*}.svg
```

```bash
python generate_data.py        # optional: rebuilds data/ from source_data/
python reproduce_suppfig2.py   # writes panels (a) and (c) into results/
```

Panels (a) and (c) are recomputed from the matched coordinates and per-anchor
points (`numpy`/`matplotlib`). APP and distances are evaluated after scaling each
coordinate by (256/H, 256/W).

## Before → after APP (panel a)

| Dataset | Before | After |
|---|---:|---:|
| Neural *Clytia* | 97.97 | 99.93 |
| Pinned *Clytia* | 96.93 | **76.01** (decrease) |
| Freely *Clytia* | 75.60 | 96.41 |
| Sperm QPM | 25.17 | 59.45 |
| Homeostatic *Clytia* | 64.08 | 99.92 |

APP uses fixed thresholds {1,2,4,8,16} px, so it depends on image scale. Datasets
larger than 256 px increase after rescaling; **Pinned decreases** because its
native 100×100 grid is magnified to 256×256, enlarging the same physical error.

## Panel (b)

`results/scaling_flow_blend.svg` and `scaling_linear_interp.svg` replay the
APP-vs-corrections curves in the 256×256 space, which requires propagating the
corrections through the per-dataset optical-flow volumes (15+ GB, not in
`source_data/`). They are therefore shipped as outputs rather than recomputed by
`reproduce_suppfig2.py`.
