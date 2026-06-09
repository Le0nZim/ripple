# Fig. 3e — Sperm QPM downstream analysis

For two RIPPLE-tracked sperm cells (Track1, Track4), the quantitative
phase-microscopy (QPM) phase trace and its spectrogram. The spectral band reveals
the periodic flagellar/rolling beat; each trace is truncated at the occlusion
cutoff where the cell rolls out of view.

## Pipeline

```
source_data/  ──(generate_data.py)──►  data/sperm_brightness.csv  ──(reproduce_fig3e.py)──►  results/*.svg
```

1. **`generate_data.py`** samples the QPM phase of each cell from the raw volume
   `source_data/sperm/qpm_volume.tif` at the tracked positions, and records each
   cell's occlusion cutoff. Requires `tifffile`.
2. **`reproduce_fig3e.py`** draws the phase traces and spectrograms with the
   published styling (Tol-sunset colormap, 7 pt Arial). Requires `numpy`,
   `scipy`, `matplotlib`.

```bash
python generate_data.py        # optional: rebuilds data/ from source_data/
python reproduce_fig3e.py      # writes the panels into results/
```

`data/sperm_brightness.csv` and `data/sperm_meta.json` are included, so
`reproduce_fig3e.py` runs without the raw volume.

## Panels (`results/`)

| File | Description |
|------|-------------|
| `sperm_trace_s1.svg`, `sperm_trace_s4.svg` | phase traces (Track1, Track4) |
| `sperm_spectrogram_s1.svg`, `sperm_spectrogram_s4.svg` | spectrograms |

Occlusion cutoffs: Track1 (S1) frame 351, Track4 (S4) frame 197. Spectrograms use
`scipy.signal.spectrogram(fs=1.0, nperseg=min(64, n//2), noverlap=nperseg−2)`.

## Source data

The ~1.9 GB QPM volume (`qpm_volume.tif`) is not committed; place it under
`source_data/sperm/` (see [`source_data/README.md`](../../source_data/README.md)).
