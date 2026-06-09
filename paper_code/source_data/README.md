# `source_data/` — raw inputs for the analysis

This folder holds the **raw microscopy data and annotations** that the
`generate_data.py` scripts in each figure/table folder consume to build their
lightweight `data/` inputs. Every `reproduce_*.py` script ships with its `data/`
already populated, so **you do not need `source_data/` to reproduce the figures
and tables** — it is only needed to rebuild the inputs from scratch (or to
regenerate the image-based panels in Fig. 1c, Fig. 3d/3e and Fig. 4a).

## Obtaining the data

Most annotation files (RIPPLE corrections, GCaMP/3D tracks, TrackMate exports,
benchmark outputs) and the small pinned-jelly segmentation masks **are committed
here**. The large raw files — the **source videos / image volumes** (`*.tif`) and
the **oversized segmentation masks** (the neural, sperm and freely `*.nrrd`, each
well over GitHub's 100 MB per-file limit) — are **not** redistributed. In their
place the repository ships a small `<name>.PLACEHOLDER` text file.

These files **can be provided by communicating with the authors** (see the
contact details in the top-level [README](../README.md)). To use one, obtain it,
drop it in next to its placeholder with the original name (without the
`.PLACEHOLDER` suffix), and run the relevant `generate_data.py`.

## Layout

```
source_data/
  neural_activity/
    manual_segmentation/segmentation_id1..10.nrrd   # exhaustive manual GT (NRRD)
    ripple_annotations.json                         # RIPPLE corrections
    video.tif                                        # raw video (Fig. 4a)
    gcamp_volume.tif                                 # GCaMP channel (Fig. 3d)
    raw_volume.tif                                   # volume render (Fig. 1c)
    gcamp_annotations.json                           # neuron tracks (Fig. 3d)
    annotations_3d_viz.json                          # 3D tracks (Fig. 1c)
  pinned_jelly/      manual_segmentation/*.nrrd  ripple_annotations.json  video.tif
  sperm/             manual_segmentation/*.nrrd  ripple_annotations.json  qpm_volume.tif  video.tif
  freely/            manual_segmentation/*.nrrd  ripple_annotations.json  video.tif
  homeostasis/       manual_segmentation.json    ripple_annotations.json  video.tif
  trackmate/         {neural,pinned,sperm,freely,homeo}.xml   # TrackMate exports (Supp. Note 1)
  benchmark_outputs/
    neural_benchmark_results.json                    # Table 2 / Fig. 3b
    optimal_corrections_{flow_blend,corridor_dp,linear_interp,tapvid_original}.json  # Fig. 3c / Table 3 / Supp. Fig. 1
    optical_flow_benchmark/*.json                    # Supp. Tables 1–2
```

## Which component uses what

| Component | Needs from `source_data/` |
|---|---|
| Table 1, Fig. 3a, Fig. 4b, Fig. 4c, Supp. Fig. 2 | `*/manual_segmentation/*` + `*/ripple_annotations.json` |
| Table 2, Fig. 3b | `benchmark_outputs/neural_benchmark_results.json` |
| Table 3, Fig. 3c, Supp. Fig. 1 | `benchmark_outputs/optimal_corrections_*.json` |
| Fig. 3d | `neural_activity/gcamp_volume.tif` + `gcamp_annotations.json` |
| Fig. 3e | `sperm/qpm_volume.tif` + `sperm/ripple_annotations.json` |
| Fig. 1c | `neural_activity/annotations_3d_viz.json` (+ `raw_volume.tif` for the overlay) |
| Fig. 4a | `*/video.tif` + `*/manual_segmentation/*` + `*/ripple_annotations.json` |
| Supp. Tables 1–2 | `benchmark_outputs/optical_flow_benchmark/*.json` |
| Supp. Note 1 | `trackmate/*.xml` + `*/manual_segmentation/*` |

A few panels (Fig. 4a image crops, Fig. 4c AFTER curve, Supp. Fig. 2 panel b)
also depend on multi-gigabyte optical-flow volumes that are **not** redistributed;
those panels are shipped as outputs and the corresponding `README.md` explains how
to regenerate them.
