# Table 3 — rebuild time and average APP at k = 25 (RIPPLE vs. TAP-Vid)

Track-rebuild time per dataset and the average point precision (APP) at
k = 25 corrections, comparing RIPPLE's CPU flow-blend interpolation against the
GPU TAP-Vid algorithm.

## Pipeline

```
data/optimal_corrections_*.json  ──(reproduce_table3.py)──►  results/table3.md
```

- **`reproduce_table3.py`** reads the two sparse-correction scaling-experiment
  outputs and computes the per-dataset rebuild time at k = 25, the
  across-dataset mean, and the average APP at k = 25. Requires only `numpy`.

```bash
python reproduce_table3.py     # writes results/table3.md
```

## Data provenance

`data/optimal_corrections_flow_blend.json` (RIPPLE) and
`optimal_corrections_tapvid_original.json` (TAP-Vid) are outputs of the
sparse-correction scaling experiment — the same artefacts used for
[Fig. 3c](../../Figures/Fig3c) and
[Supplementary Fig. 1](../../Supplementary/SuppFig1). Each stores, per track, the
APP curve and the wall-clock rebuild-time curve as a function of correction
count. They are also provided in `source_data/benchmark_outputs/`.

## Values

The per-dataset rebuild times at k = 25 and the average APP reproduce the
published table:

| Method | Pinned | Neural | Sperm | Freely | Homeo | Avg APP (k=25) |
|---|---:|---:|---:|---:|---:|---:|
| RIPPLE | 12.7 | 12.9 | 11.1 | 12.8 | 2.1 | 70.78 |
| TAP-Vid | 2,912 | 35,773 | 65,104 | 121,286 | 328,718 | 69.43 |

Rebuild times are **wall-clock measurements** and depend on the host machine.
The Mean column here is the per-dataset times averaged across datasets (per the
table caption: 10.3 ms / 110,759 ms). The published table reports a pooled mean
over individual builds (9.9 ms / 105,577 ms). Either way RIPPLE rebuilds tracks
more than 10,000× faster than TAP-Vid, which is the hardware-independent result.
The average-APP column is deterministic and reproduces exactly.
