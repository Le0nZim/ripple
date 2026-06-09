# Fig. 1c — 3D visualization of tracked data

Two views of the tracked Neural *Clytia* data: the trajectories alone, and the
trajectories overlaid on a volume rendering (grayscale slices at the first,
middle and last frame). Axes: X = frame (time), Y = x (px), Z = y (px); each
track is drawn in its stored colour.

## Pipeline

```
source_data/  ──(generate_data.py)──►  data/tracks_3d.json  ──(reproduce_fig1c.py)──►  results/*.png/svg
```

1. **`generate_data.py`** slims the full 3D-visualization annotation file
   (`source_data/neural_activity/annotations_3d_viz.json`) to the track
   coordinates needed for the render (`data/tracks_3d.json`).
2. **`reproduce_fig1c.py`** renders the trajectories-alone view from the JSON, and
   — if `source_data/neural_activity/raw_volume.tif` is present — the
   volume-overlay view. Requires `numpy`, `matplotlib` (and `tifffile` for the
   overlay).

```bash
python generate_data.py        # optional: rebuilds data/ from source_data/
python reproduce_fig1c.py      # writes the renders into results/
```

`data/tracks_3d.json` is included, so the trajectories-alone view runs without
`source_data/`.

## Panels (`results/`)

| File | Description |
|------|-------------|
| `tracks_3d_trajectories_only.{png,svg}` | 87 trajectories alone |
| `tracks_3d_volume_overlay.png` | trajectories on grayscale volume slices |

## Source data

The volume-overlay view samples the 277 MB raw volume
(`raw_volume.tif`, brightness window 275–814); place it under
`source_data/neural_activity/` (see [`source_data/README.md`](../../source_data/README.md)).
