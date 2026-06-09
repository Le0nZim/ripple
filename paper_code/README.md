# RIPPLE — paper code and results

Analysis code and result files for the figures and tables in:

> **Motion-guided sparse correction enables expert-quality point tracking across
> diverse microscopy regimes**
> Leonidas Zimianitis, Pasindu Thenahandi, Kai Buckhalter, Dineth Jayakody,
> Julian O. Kimura, Xinyue Liang, Karen Cunningham, Azeem Ahmad,
> Balpreet S. Ahluwalia, Sampath Jayarathna, Nikos Chrisochoides,
> Brandon Weissbourd, Dushan N. Wadduwage.

This folder holds the **analysis layer** of the RIPPLE project: the scripts that
turn tracking outputs and ground truth into the paper's figures and tables. It
lives inside the RIPPLE software repository; the interactive annotation tool and
the motion-guided sparse-correction engine are in the repository root.

- RIPPLE software (this repository): <https://github.com/Le0nZim/ripple>
- Interactive demo: <https://huggingface.co/spaces/Le0nZim/ripple-demo>

## Layout

Every figure panel, figure component and table has its own folder:

```
Figures/         Fig1c  Fig3a Fig3b Fig3c Fig3d Fig3e  Fig4a Fig4b Fig4c
Tables/          Table1 Table2 Table3
Supplementary/   SuppFig1 SuppFig2  SuppTable1 SuppTable2  SuppNote1
common/          shared loaders (ripple_io.py)
source_data/     raw inputs (large media not committed; see source_data/README.md)
```

Each component folder contains:

| Item | Role |
|------|------|
| `reproduce_*.py` | Builds the panel/table from the local `data/` and writes it to `results/`. Self-contained — run it directly. |
| `generate_data.py` | (where applicable) Rebuilds `data/` from the raw inputs in `source_data/`. |
| `data/` | The committed lightweight inputs the reproduce script reads. |
| `results/` | The figure panels / tables, exactly as in the paper. |
| `README.md` | The pipeline, the command, and where the raw data comes from. |

The two-stage design is:

```
source_data/  ──(generate_data.py)──►  data/  ──(reproduce_*.py)──►  results/
```

`data/` is committed, so every `reproduce_*.py` runs **without** `source_data/`.
`source_data/` is only needed to rebuild `data/` from scratch or to regenerate the
image-based panels (Fig. 1c, Fig. 3d/3e, Fig. 4a).

## Running

```bash
pip install numpy scipy matplotlib pandas tifffile pynrrd

# reproduce a single component
cd Figures/Fig3a && python reproduce_fig3a.py

# rebuild its inputs from the raw data first (needs source_data/)
python generate_data.py && python reproduce_fig3a.py
```

The `generate_data.py` scripts import shared loaders from `common/`, so run them
from inside their component folder (with the repository root on `sys.path`, which
they set up automatically).

## Components

### Main text

| Item | Folder | What it produces |
|------|--------|------------------|
| Fig. 1c | [`Figures/Fig1c`](Figures/Fig1c) | 3D track visualization (trajectories + volume overlay) |
| Fig. 3a | [`Figures/Fig3a`](Figures/Fig3a) | effort-vs-accuracy scatter |
| Fig. 3b | [`Figures/Fig3b`](Figures/Fig3b) | RIPPLE vs. baselines on four cost axes |
| Fig. 3c | [`Figures/Fig3c`](Figures/Fig3c) | APP vs. corrections, RIPPLE vs. TAP-Vid |
| Fig. 3d | [`Figures/Fig3d`](Figures/Fig3d) | Neural GCaMP downstream analysis |
| Fig. 3e | [`Figures/Fig3e`](Figures/Fig3e) | Sperm QPM downstream analysis |
| Fig. 4a | [`Figures/Fig4a`](Figures/Fig4a) | representative disagreement crops |
| Fig. 4b | [`Figures/Fig4b`](Figures/Fig4b) | disagreement distribution + CDF |
| Fig. 4c | [`Figures/Fig4c`](Figures/Fig4c) | coordinate-replacement analysis |
| Table 1 | [`Tables/Table1`](Tables/Table1) | accuracy + effort across datasets |
| Table 2 | [`Tables/Table2`](Tables/Table2) | RIPPLE vs. baselines |
| Table 3 | [`Tables/Table3`](Tables/Table3) | rebuild time vs. TAP-Vid |

### Supplementary

| Item | Folder | What it produces |
|------|--------|------------------|
| Supp. Fig. 1 | [`Supplementary/SuppFig1`](Supplementary/SuppFig1) | interpolation-strategy comparison |
| Supp. Fig. 2 | [`Supplementary/SuppFig2`](Supplementary/SuppFig2) | 256×256 resolution scaling |
| Supp. Table 1 | [`Supplementary/SuppTable1`](Supplementary/SuppTable1) | candidate optical-flow algorithms |
| Supp. Table 2 | [`Supplementary/SuppTable2`](Supplementary/SuppTable2) | corrections to reach APP ≥ 90% |
| Supp. Note 1 | [`Supplementary/SuppNote1`](Supplementary/SuppNote1) | TrackMate intervention cost |

## RIPPLE software

The interactive RIPPLE annotation tool and the motion-guided sparse-correction
engine live in the root of this repository (this `paper_code/` folder is the
analysis layer that accompanies them). See the repository's top-level README and
<https://github.com/Le0nZim/ripple> for the software itself.

## Data availability

The committed `data/` folders are sufficient to reproduce every figure and table.
The large raw media — the source **videos / image volumes** and the oversized
segmentation masks — are **not** committed (they exceed GitHub's per-file limit);
the repository ships `*.PLACEHOLDER` stand-ins in their place. **These files can be
provided by communicating with the authors.** See
[`source_data/README.md`](source_data/README.md) for the full layout and the
per-file list.

## Third-party baselines

The comparison methods are external tools, not redistributed here:
LocoTrack (<https://github.com/cvlab-kaist/locotrack>), SLEAP (<https://sleap.ai>),
TrackMate (<https://imagej.net/plugins/trackmate/>).
