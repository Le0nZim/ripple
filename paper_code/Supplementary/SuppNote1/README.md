# Supplementary Note 1 — TrackMate vs. RIPPLE intervention cost

How many manual interventions a user would need to reconstruct the tracks of
interest starting from TrackMate's output, compared with RIPPLE.

## Pipeline

```
source_data/  ──(generate_data.py)──►  data/intervention_inputs.json  ──(reproduce_suppnote1.py)──►  results/intervention_cost.md
```

1. **`generate_data.py`** parses each dataset's TrackMate XML
   (`source_data/trackmate/<dataset>.xml`) and the matched ground-truth tracks,
   and records each GT track's visible coordinates plus the TrackMate detections
   pruned to within tolerance of the GT (`data/intervention_inputs.json`).
   Requires `numpy` (+ `scipy`/`pynrrd` for the NRRD ground truth).
2. **`reproduce_suppnote1.py`** replays the per-track intervention count
   (initial pick + relink + manual click) and compares it with the RIPPLE
   correction count. Requires only the standard library.

```bash
python generate_data.py        # optional: rebuilds data/ from source_data/
python reproduce_suppnote1.py  # writes results/intervention_cost.md
```

`data/intervention_inputs.json` is included, so `reproduce_suppnote1.py` runs
without `source_data/`.

## Result

| Dataset | TrackMate | RIPPLE | TM / RIPPLE |
|---|---:|---:|---:|
| Neural *Clytia* | 251 | 158 | 1.6× |
| Pinned *Clytia* | 27 | 116 | 0.2× |
| Freely *Clytia* | 267 | 352 | 0.8× |
| Sperm QPM | 1007 | 529 | 1.9× |
| Homeostatic *Clytia* | 90 | 47 | 1.9× |
| **All** | **1642** | **1202** | **1.4×** |

Overall, TrackMate requires 1.4× more manual interventions than RIPPLE to
reconstruct the same tracks.

## Source data

`generate_data.py` reads the TrackMate XML exports in `source_data/trackmate/`
and the GT annotations under each `source_data/<dataset>/` folder. See
[`source_data/README.md`](../../source_data/README.md).
