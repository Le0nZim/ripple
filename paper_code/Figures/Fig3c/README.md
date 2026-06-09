# Fig. 3c — APP vs. correction count, RIPPLE vs. TAP-Vid

Average point precision as a function of the number of corrections, for RIPPLE's
flow-blend interpolation and for the TAP-Vid algorithm. Each dataset curve is the
mean over tracks with a per-track min/max envelope; the dashed line marks
k = 25 corrections.

## Pipeline

```
data/optimal_corrections_*.json  ──(reproduce_fig3c.py)──►  results/scaling_*.svg
```

```bash
python reproduce_fig3c.py      # writes both scaling panels into results/
```

Requires `numpy` and `matplotlib`.

## Panels (`results/`)

| File | Method |
|------|--------|
| `scaling_flow_blend.svg` | RIPPLE flow-blend interpolation |
| `scaling_tapvid_original.svg` | TAP-Vid algorithm |

Mean APP at k = 25: **RIPPLE 70.78%**, **TAP-Vid 69.43%** (Table 3).

## Data provenance

`data/optimal_corrections_flow_blend.json` and
`optimal_corrections_tapvid_original.json` are outputs of the sparse-correction
scaling experiment (shared with [Table 3](../../Tables/Table3) and
[Supplementary Fig. 1](../../Supplementary/SuppFig1)); each stores per track the
APP curve as a function of correction count. They are also provided in
`source_data/benchmark_outputs/`. Regenerating them requires the optical-flow
volumes and the RIPPLE engine.
