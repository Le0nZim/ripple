# Table 2 — RIPPLE vs. baseline trackers (Neural *Clytia*)

Accuracy (APP) and practical cost of RIPPLE compared with TrackMate, LocoTrack
and SLEAP on the Neural *Clytia* dataset.

## Pipeline

```
benchmark.py  ──►  data/benchmark_results.json  ──(reproduce_table2.py)──►  results/table2.md
```

- **`reproduce_table2.py`** reads the APP and manual-annotation columns from
  `data/benchmark_results.json` and combines them with the logged time and
  hyper-parameter measurements to write the table. Requires only the standard
  library.

```bash
python reproduce_table2.py     # writes results/table2.md
```

## Data provenance

`data/benchmark_results.json` is the output of the full baseline-evaluation
pipeline in **`benchmark.py`**, which runs RIPPLE and the three baselines on the
Neural *Clytia* video and evaluates every method against the manual ground
truth. Reproducing it from scratch requires the raw video and the external
tracking tools (the RIPPLE engine, TrackMate, LocoTrack and SLEAP — see the main
repository README), so the benchmark output is provided directly as the
reproducibility checkpoint (also in `source_data/benchmark_outputs/`).

## Values

| Method | APP (%) | Manual annotations |
|---|---:|---:|
| RIPPLE | 97.98 | 158 |
| TrackMate | 70.15 | 0 |
| LocoTrack | 71.18 | 0 |
| SLEAP | 73.18 | 158 |
| SLEAP-op | 74.76 | 210 |

Elapsed-time, computation-time and hyper-parameter columns are logged
measurements reproduced from the recorded values.
