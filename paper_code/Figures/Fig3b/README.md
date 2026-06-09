# Fig. 3b — RIPPLE vs. baselines on four practical-cost axes

Average point precision (APP) of RIPPLE, TrackMate, LocoTrack and SLEAP on the
Neural *Clytia* dataset, plotted against four axes of practical cost: manual
annotations, total elapsed time, computation time, and hyper-parameter search
space. The exhaustive-manual reference sits at 100% APP.

## Pipeline

```
data/benchmark_results.json  ──(reproduce_fig3b.py)──►  results/aggregate_*_aggregated.svg
```

```bash
python reproduce_fig3b.py      # writes the four panels into results/
```

`reproduce_fig3b.py` reads the benchmark output and draws the four scatter
panels with the published styling. Requires `numpy` and `matplotlib`.

## Panels (`results/`)

| File | x-axis |
|------|--------|
| `aggregate_annotations_aggregated.svg` | manual annotations |
| `aggregate_real_time_aggregated.svg` | total elapsed time (log) |
| `aggregate_comp_time_no_cache_aggregated.svg` | computation time (log) |
| `aggregate_params_aggregated.svg` | hyper-parameter combinations |

## Data provenance

`data/benchmark_results.json` is the output of the baseline-evaluation pipeline
(`benchmark.py`, kept in [`Tables/Table2`](../../Tables/Table2)), which runs
RIPPLE and the baselines on the Neural *Clytia* video. It is the same artefact
behind Table 2, and is also provided in `source_data/benchmark_outputs/`.
Regenerating it requires the raw video and the external tracking tools.
