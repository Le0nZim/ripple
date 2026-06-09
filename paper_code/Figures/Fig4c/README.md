# Fig. 4c — coordinate-replacement (disagreement-elimination) analysis

How much of each dataset's residual error comes from the comparison annotator's
coordinate disagreement (Fig. 4b) rather than from interpolation? The RIPPLE
anchor coordinates are replaced with the exhaustive ground-truth coordinates at
the same frames, the flow-blend interpolation is re-run, and the APP is recomputed
(AFTER) and compared with the original RIPPLE APP (BEFORE).

## Pipeline

```
source_data/ + data/fig4c_after.json  ──(generate_data.py)──►  data/fig4c_results.json  ──(reproduce_fig4c.py)──►  results/*.svg
```

```bash
python generate_data.py        # optional: recomputes BEFORE + counts from source_data
python reproduce_fig4c.py      # writes the panel and bar charts into results/
```

## Results

| Dataset | BEFORE | AFTER | ΔAPP |
|---|---:|---:|---:|
| Neural *Clytia* | 97.97 | 97.00 | −0.97 |
| Pinned *Clytia* | 96.93 | 96.20 | −0.73 |
| Freely *Clytia* | 75.60 | 79.76 | +4.16 |
| Sperm QPM | 25.17 | 62.71 | **+37.54** |
| Homeostatic *Clytia* | 64.08 | 65.08 | +1.00 |

Substituting the exhaustive ground-truth coordinates lifts Sperm from 25% to 63%
APP, while the already-accurate datasets change by <1.2 pp — the Sperm gap is
dominated by annotator coordinate disagreement, not interpolation.

## Data provenance

- **BEFORE** (= Table 1 APP), the manual-frame counts, and the RIPPLE-correction
  counts are recomputed from `source_data/` by `generate_data.py`.
- **AFTER** (GT-anchor flow-blend) needs the per-dataset optical-flow volumes
  (15+ GB; not in `source_data/`), so it is shipped as a checkpoint in
  `data/fig4c_after.json`. `generate_data.py` merges the two into
  `data/fig4c_results.json`.
