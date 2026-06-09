# Fig. 4a — representative matched manual-insertion crops

A qualitative montage. For every matched track, the anchor frame with the largest
ground-truth-vs-RIPPLE disagreement is cropped from the raw video and overlaid
with the exhaustive ground-truth marker (yellow diamond) and the comparison
annotator's RIPPLE marker (magenta square).

## Pipeline

```
source_data/<dataset>/video.tif + annotations  ──(generate_frames.py)──►  results/frames/*.svg
```

Because the panels are image crops, they are produced directly from the raw
videos — there is no lightweight intermediate. `generate_frames.py` locates the
worst-disagreement frame per track (from the NRRD/JSON annotations), crops the
raw video, and renders the markers. It also writes the selected frames to
`data/fig4a_worst_frames.json`.

```bash
python generate_frames.py      # writes the crops into results/frames/
```

The 32 crops (one per matched track: Neural 10, Pinned 3, Freely 3, Sperm 6,
Homeostatic 10) are committed in `results/frames/`, so the figure is available
without the videos.

## Required videos

`generate_frames.py` reads one video per dataset:

| Dataset | Place at |
|---|---|
| Neural *Clytia* | `source_data/neural_activity/video.tif` |
| Pinned *Clytia* | `source_data/pinned_jelly/video.tif` |
| Freely *Clytia* | `source_data/freely/video.tif` |
| Sperm QPM | `source_data/sperm/video.tif` |
| Homeostatic *Clytia* | `source_data/homeostasis/video.tif` |

plus the NRRD/JSON annotations already under each `source_data/<dataset>/` folder.
See [`source_data/README.md`](../../source_data/README.md) for how to obtain and
place these files.

## Rendering

Crop sizes, marker sizes and brightness/contrast windows are set per dataset (and
per track where noted) to match the published montage; each crop carries a `N×`
magnification label (full field of view ÷ crop size). The two markers are the
ground-truth and RIPPLE coordinates whose distance is the per-anchor disagreement
quantified in [Fig. 4b](../Fig4b/README.md).
