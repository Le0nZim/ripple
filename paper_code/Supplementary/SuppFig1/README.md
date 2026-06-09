# Supplementary Fig. 1 — interpolation-strategy comparison

Accuracy and computational cost of four interpolation strategies — Linear, RIPPLE
flow-blend, Corridor-DP and TAP-Vid original — across the five datasets.

## Pipeline

```
data/optimal_corrections_*.json  ──(reproduce_suppfig1.py)──►  results/*.svg
```

```bash
python reproduce_suppfig1.py     # writes all panels into results/
```

Requires `numpy` and `matplotlib`. Both the accuracy curves and the timing
summary are computed from the four scaling-experiment outputs in `data/` (per
track: the APP curve and the wall-clock rebuild-time curve vs. correction count).

## Panels (`results/`)

**(a) Accuracy vs. corrections** — one panel per strategy:

| File | Strategy |
|------|----------|
| `scaling_linear_interp.svg` | Linear interpolation |
| `scaling_flow_blend.svg` | RIPPLE flow-blend |
| `scaling_corridor_dp.svg` | Corridor-DP |
| `scaling_tapvid_original.svg` | TAP-Vid original |

**(b) Runtime summary at k = 25:**

| File | Description |
|------|-------------|
| `01_grouped_log_bars.svg` | per-dataset build time (log scale) |
| `02_time_vs_pixels.svg` | build time vs. frame size |
| `03_speedup_heatmap.svg` | speed-up over TAP-Vid original |

## Speed-ups

Mean rebuild times at k = 25 give the speed-up ladder TAP-Vid → Corridor-DP →
RIPPLE flow-blend → Linear. The published figure reports 883× (TAP-Vid →
Corridor-DP), 12.1× (Corridor-DP → flow-blend), 10,684× overall, and 29.1×
(flow-blend → Linear) as a multiplicative chain (883 × 12.1 = 10,684); the direct
per-build ratios computed here reproduce these to within a few percent.

## Data provenance

The `optimal_corrections_*.json` files are outputs of the sparse-correction
scaling experiment (shared with [Fig. 3c](../../Figures/Fig3c) and
[Table 3](../../Tables/Table3)); they are also in `source_data/benchmark_outputs/`.
