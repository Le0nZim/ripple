# Table 1 — RIPPLE accuracy and annotation effort

Average point precision (APP) of RIPPLE against exhaustive manual annotation
across the five datasets, alongside the logged annotation effort.

## Pipeline

```
source_data/  ──(generate_data.py)──►  data/app_inputs.json  ──(reproduce_table1.py)──►  results/table1.md
```

1. **`generate_data.py`** — matches each manual ground-truth track to its RIPPLE
   track and stores the aligned per-frame coordinate arrays
   (`data/app_inputs.json`). Reads the raw NRRD segmentations and RIPPLE
   annotation JSONs from `source_data/`. Requires `numpy`, `scipy`, `pynrrd`.

2. **`reproduce_table1.py`** — computes the APP column from
   `data/app_inputs.json` and writes the table. Requires only `numpy`. The
   effort columns (tracks, time, clicks) are logged annotation-session
   measurements.

```bash
python generate_data.py        # optional: rebuilds data/ from source_data/
python reproduce_table1.py     # writes results/table1.md
```

`data/app_inputs.json` is included, so `reproduce_table1.py` runs without
`source_data/`.

## APP (computed)

| Dataset | APP (%) |
|---|---:|
| Neural *Clytia* | 97.97 |
| Pinned *Clytia* | 96.93 |
| Freely swimming *Clytia* | 75.60 |
| Homeostatic *Clytia* | 64.08 |
| Sperm QPM | 25.17 |

The published table lists the Neural value as 97.90 and the Table 2 benchmark of
the same dataset gives 97.98; the value computed here (97.97) is within rounding
of both. All other datasets match the published table exactly.

## Source data

`generate_data.py` reads, per dataset, the manual NRRD segmentations
(`source_data/<dataset>/manual_segmentation/`) and the RIPPLE annotation JSON
(`source_data/<dataset>/ripple_annotations.json`). See
[`source_data/README.md`](../../source_data/README.md) for how to obtain and
place these files.
