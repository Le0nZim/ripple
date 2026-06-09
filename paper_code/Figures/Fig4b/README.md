# Fig. 4b — annotator-disagreement distribution and CDF

At each corrected (anchor) frame, the disagreement is the Euclidean distance in
native pixels between the exhaustive manual ground-truth point and a second
annotator's RIPPLE correction. Panel 4b shows the per-dataset distribution
(box + strip) and the cumulative distribution.

## Pipeline

```
source_data/  ──(generate_data.py)──►  data/fig4b_inputs.json  ──(reproduce_fig4b.py)──►  results/disagreement_{box,cdf}.svg
```

```bash
python generate_data.py        # optional: rebuilds data/ from source_data/
python reproduce_fig4b.py      # writes the box + CDF panels into results/
```

`generate_data.py` matches each ground-truth track to its RIPPLE track and
records, per anchor frame, the ground-truth and RIPPLE coordinates (needs the
NRRD/JSON annotations in `source_data/`). `reproduce_fig4b.py` computes the
distances and draws the panels (needs only `numpy`/`matplotlib`).

## Panels (`results/`)

| File | Description |
|------|-------------|
| `disagreement_box_log.svg` | box + strip, **log** y-axis (the published panel) |
| `disagreement_cdf_log.svg` | CDF, **log** x-axis (the published panel) |
| `disagreement_box.svg` | box + strip, linear y-axis |
| `disagreement_cdf.svg` | CDF, linear x-axis |

The published figure uses the **log-scale** panels, because the disagreement
spans sub-pixel (Neural/Pinned) to ~25 px (Sperm). Zeros are clamped to 0.1 px
(box) / 0.05 px (CDF) so they remain visible on the log axis. The linear panels
are kept for reference.

## Values

| Dataset | n | median (px) | mean (px) |
|---|---:|---:|---:|
| Neural *Clytia* | 158 | 0.00 | 0.27 |
| Pinned *Clytia* | 115 | 0.00 | 0.40 |
| Freely *Clytia* | 351 | 1.00 | 1.04 |
| Homeostatic *Clytia* | 47 | 2.00 | 3.15 |
| Sperm QPM | 526 | 19.95 | 25.54 |

Neural and Pinned are sub-pixel, Freely ~1 px, Homeostatic ~3 px, Sperm ~25 px —
the disagreement figures quoted in the Discussion.
