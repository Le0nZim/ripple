# Fig. 3d — Neural *Clytia* GCaMP6s downstream analysis

A downstream analysis enabled by RIPPLE tracking: the GCaMP6s brightness of every
tracked neuron is read out per frame to produce a neuron-by-frame z-scored
activity heatmap and an event-triggered population calcium response.

## Pipeline

```
source_data/  ──(generate_data.py)──►  data/neural_brightness_71.csv  ──(reproduce_fig3d.py)──►  results/*.svg
```

1. **`generate_data.py`** samples the GCaMP brightness of each tracked neuron from
   the raw volume `source_data/neural_activity/gcamp_volume.tif` at the positions
   in `gcamp_annotations.json` (Track23 excluded → 71 neurons). Requires
   `tifffile`.
2. **`reproduce_fig3d.py`** computes the z-scored matrix and the event-triggered
   trace and draws both panels with the published styling (Tol-sunset colormap,
   7 pt Arial). Requires `numpy`, `scipy`, `matplotlib`.

```bash
python generate_data.py        # optional: rebuilds data/ from source_data/
python reproduce_fig3d.py      # writes the panels into results/
```

`data/neural_brightness_71.csv` is included, so `reproduce_fig3d.py` runs without
the raw volume.

## Panels (`results/`)

| File | Description |
|------|-------------|
| `neural_heatmap_zscore.svg` | 71 neurons × 400 frames, z-scored |
| `gcamp_event_triggered_zscore_trace.svg` | population response aligned to detected events |

## Method

Per neuron: `F0 = mean(first 11 frames)`, `dF/F`, then z-score. Calcium events
are the peaks of the population-mean z-score
(`find_peaks(height=0.8, distance=15, prominence=0.5)` → 2 events at frames 25
and 111); peri-event window −20…+60 frames.

## Source data

The 277 MB GCaMP volume (`gcamp_volume.tif`) is not committed; place it under
`source_data/neural_activity/` (see [`source_data/README.md`](../../source_data/README.md)).
