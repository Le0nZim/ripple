# Supplementary Table 2 — corrections to reach APP ≥ 90%

For each candidate backbone optical-flow algorithm, the number of corrections
needed to reach 90% average point precision on the Pinned, Neural, Sperm and
Freely datasets.

## Pipeline

```
benchmark_optical_flows.py  ──►  data/<algo>.json  ──(reproduce_supptable2.py)──►  results/corrections_to_app90.md
```

1. **`benchmark_optical_flows.py`** runs each algorithm as the RIPPLE motion
   backbone and records, per track, the number of corrections to reach each APP
   threshold — producing the 48 per-algorithm files in `data/` (plus
   `errors.json` for the models that failed to run). This is the heavy GPU/CPU
   step; its outputs are provided directly.
2. **`reproduce_supptable2.py`** reads `data/*.json`, takes the median over tracks
   of the optimal correction count at APP ≥ 0.90, averages over the four datasets
   (round-half-up), ranks the algorithms, and writes the table. Requires `numpy`.

```bash
python reproduce_supptable2.py     # writes results/corrections_to_app90.md
```

## Notes

- Each cell = median over tracks of `optimal_k` at APP ≥ 0.90; `—` marks a
  dataset with no result for that algorithm. `Avg` is filled only when all four
  datasets are present.
- `dis_medium` (the backbone RIPPLE uses) reaches APP ≥ 90% in 25 / 2 / 212 / 225
  corrections (Avg 116).
- The algorithm registry behind these results is listed in
  [`../SuppTable1`](../SuppTable1).
