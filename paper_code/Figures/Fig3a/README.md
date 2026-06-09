# Fig. 3a — annotation effort vs. accuracy

Per dataset, a **diamond** marks exhaustive manual annotation (total annotations
at 100% APP) and a **square** marks RIPPLE (correction count vs. average point
precision, APP); an arrow links the two. The annotated variant labels each
square with the APP and the effort-reduction factor.

## Pipeline

```
source_data/  ──(generate_data.py)──►  data/fig3a_values.json  ──(reproduce_fig3a.py)──►  results/panel_B*.svg
```

```bash
python generate_data.py        # optional: rebuilds data/ from source_data/
python reproduce_fig3a.py      # writes the panels into results/
```

`generate_data.py` matches each manual track to its RIPPLE track and records the
plotted values (manual frame count, RIPPLE correction count, RIPPLE APP).
`reproduce_fig3a.py` draws the panels with the published styling.

## Panels (`results/`)

| File | Description |
|------|-------------|
| `panel_B_annotated.svg` | published Fig. 3a (with APP labels and effort factors) |
| `panel_B_minimal.svg` | unlabelled variant |
| `panel_B.svg` | with arrows, no labels |

## Values

| Dataset | Manual frames | RIPPLE corrections | APP (%) |
|---|---:|---:|---:|
| Neural *Clytia* | 4,000 | 158 | 97.97 |
| Pinned *Clytia* | 1,199 | 116 | 96.93 |
| Freely *Clytia* | 1,008 | 352 | 75.60 |
| Sperm QPM | 1,847 | 529 | 25.17 |
| Homeostatic *Clytia* | 480 | 47 | 64.08 |

The APP values reproduce Table 1; the manual frame and correction counts are
read from the matched ground-truth and RIPPLE annotations in `source_data/`.
